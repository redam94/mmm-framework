"""LLM provider configuration and construction for the MMM agent.

This module decouples the LangGraph agent (:func:`create_agent_graph`) from any
specific LLM vendor. It reads a user-editable **model configuration file**
(YAML) and builds the matching LangChain chat model.

It supports two classes of provider:

* **Direct API providers** — ``anthropic``, ``openai``, ``google_genai`` —
  authenticated with an API key (from the config file, the ``X-API-Key``
  request header, or the provider's usual environment variable).
* **Google Vertex AI** — ``vertex_anthropic`` (Claude via Model Garden) and
  ``vertex_gemini`` (Gemini) — authenticated through **Application Default
  Credentials (ADC)**. On a GCP VM the attached service account is picked up
  automatically with no key in the config; set ``credentials_path`` only if you
  want to point at an explicit service-account JSON instead.

Config file resolution order:

1. an explicit ``path`` argument
2. ``$MMM_MODEL_CONFIG``
3. ``./model_config.yaml``
4. ``./config/model_config.yaml``

If none is found, a backward-compatible default (direct Anthropic Claude) is
used so existing deployments keep working. Individual fields can additionally be
overridden by environment variables prefixed ``MMM_LLM_`` (e.g.
``MMM_LLM_PROVIDER``, ``MMM_LLM_MODEL``, ``MMM_LLM_PROJECT``,
``MMM_LLM_LOCATION``), which is convenient on a VM configured purely via env.

See ``config/model_config.example.yaml`` for an annotated template.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

Provider = Literal[
    "vertex_anthropic",
    "vertex_gemini",
    "anthropic",
    "openai",
    "google_genai",
]

_VERTEX_PROVIDERS: frozenset[str] = frozenset({"vertex_anthropic", "vertex_gemini"})

# Default config-file search locations, relative to the current working directory.
_DEFAULT_CONFIG_PATHS: tuple[str, ...] = (
    "model_config.yaml",
    "config/model_config.yaml",
)

# Environment-variable prefix for per-field overrides (e.g. MMM_LLM_PROVIDER).
_ENV_PREFIX = "MMM_LLM_"

# Placeholder "keys" the frontend may send when the server manages credentials
# itself (Vertex/ADC or a server-side env key). They must NOT override server
# config — a blank or sentinel client key is treated as "no client key".
_SENTINEL_API_KEYS: frozenset[str] = frozenset(
    {"", "server-managed", "__server_managed__"}
)

# Only simple scalar fields may be overridden from the environment.
_ENV_OVERRIDABLE: frozenset[str] = frozenset(
    {
        "provider",
        "model",
        "temperature",
        "max_tokens",
        "project",
        "location",
        "credentials_path",
        "api_key",
    }
)

# Model family used to decide whether a client-supplied model name is compatible
# with the configured provider, and to infer a direct provider from a model name.
_PROVIDER_FAMILY: dict[str, str] = {
    "vertex_anthropic": "claude",
    "vertex_gemini": "gemini",
    "anthropic": "claude",
    "openai": "openai",
    "google_genai": "gemini",
}

# Environment variables that already carry a key for each direct provider.
_PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}

# Vertex regions that ship the model when the user leaves ``location`` blank.
# Claude on Model Garden is only served in a subset of regions, so its default
# differs from Gemini's.
_DEFAULT_VERTEX_LOCATION: dict[str, str] = {
    "vertex_anthropic": "us-east5",
    "vertex_gemini": "us-central1",
}

# Applied to Vertex providers when ``max_tokens`` is unset, so long report turns
# aren't silently truncated (ChatAnthropicVertex otherwise defaults to 4096).
_DEFAULT_VERTEX_MAX_TOKENS = 8192


class ModelConfig(BaseModel):
    """User-facing model configuration.

    Exactly one provider is active at a time. ``api_key`` is only used by the
    direct (non-Vertex) providers; Vertex providers authenticate through
    Application Default Credentials, or an explicit ``credentials_path``.
    """

    provider: Provider = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int | None = None

    # --- Google Vertex AI / GCP settings (ignored by direct providers) ---
    project: str | None = None
    location: str | None = None
    credentials_path: str | None = None  # service-account JSON; None => ADC

    # --- Direct provider settings (ignored by Vertex providers) ---
    api_key: str | None = None

    # Free-form passthrough to the underlying LangChain constructor.
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def uses_vertex(self) -> bool:
        """True when the active provider is served through Google Vertex AI."""
        return self.provider in _VERTEX_PROVIDERS

    @property
    def uses_adc(self) -> bool:
        """True when authentication comes from Application Default Credentials.

        This is the GCP-VM case: a Vertex provider with no explicit
        service-account JSON, so ``google.auth.default()`` resolves the VM's
        attached service account.
        """
        return self.uses_vertex and not self.credentials_path


# ── Config loading ─────────────────────────────────────────────────────────


def _resolve_config_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Return the first existing config-file path per the documented order."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("MMM_MODEL_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(Path(p) for p in _DEFAULT_CONFIG_PATHS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``MMM_LLM_*`` environment variables onto a config dict in place."""
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        field = key[len(_ENV_PREFIX) :].lower()
        if field in _ENV_OVERRIDABLE and value != "":
            data[field] = value  # pydantic coerces scalars (temperature, max_tokens)
    return data


def load_model_config(path: str | os.PathLike[str] | None = None) -> ModelConfig:
    """Load the model configuration from a YAML file plus environment overrides.

    Args:
        path: Explicit path to a YAML config file. If ``None``, the standard
            search order (``$MMM_MODEL_CONFIG`` → ``./model_config.yaml`` →
            ``./config/model_config.yaml``) is used.

    Returns:
        A validated :class:`ModelConfig`. Falls back to the built-in default
        (direct Anthropic Claude) if no file is found and no env overrides
        are set.

    Raises:
        ValueError: If the file is not a YAML mapping, or the resulting
            configuration is invalid (e.g. an unknown ``provider``).
    """
    data: dict[str, Any] = {}
    resolved = _resolve_config_path(path)
    if resolved is not None:
        import yaml

        with open(resolved, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Model config file {resolved} must contain a YAML mapping, "
                f"got {type(loaded).__name__}."
            )
        # Drop keys explicitly set to null so they fall back to model defaults
        # rather than overriding them with None (e.g. a blank `temperature:`).
        data.update({k: v for k, v in loaded.items() if v is not None})

    _apply_env_overrides(data)

    try:
        return ModelConfig(**data)
    except ValidationError as exc:
        valid = ", ".join(sorted(_PROVIDER_FAMILY))
        src = f" (from {resolved})" if resolved else ""
        raise ValueError(
            f"Invalid model configuration{src}: {exc}. "
            f"Valid providers are: {valid}."
        ) from exc


# ── Provider inference (header-key flow) ───────────────────────────────────


def _model_family(model_name: str | None) -> str:
    """Best-effort family for a model name: claude / openai / gemini / unknown."""
    name = (model_name or "").lower()
    if "gpt" in name or name.startswith(("o1", "o3", "o4")):
        return "openai"
    if "gemini" in name:
        return "gemini"
    if "claude" in name:
        return "claude"
    return "unknown"


def infer_provider_from_model(model_name: str | None) -> Provider:
    """Infer a *direct* API provider from a model name (used for the UI key flow)."""
    family = _model_family(model_name)
    if family == "openai":
        return "openai"
    if family == "gemini":
        return "google_genai"
    return "anthropic"


# ── Construction ───────────────────────────────────────────────────────────


def _load_credentials(credentials_path: str | None):
    """Return google credentials for an explicit JSON key, or ``None`` for ADC."""
    if not credentials_path:
        return None  # ADC: google.auth.default() (VM service account, gcloud, etc.)
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def _build_from_config(cfg: ModelConfig):
    """Construct the LangChain chat model for a fully-resolved config.

    Provider classes are imported lazily so that, e.g., a pure-Vertex
    deployment never imports ``langchain_openai`` (matching the framework's
    lazy-import convention and keeping cold-start light).
    """
    extra = dict(cfg.model_kwargs)

    if cfg.provider == "vertex_anthropic":
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        kwargs: dict[str, Any] = {
            "model_name": cfg.model,
            "temperature": cfg.temperature,
            "location": cfg.location or _DEFAULT_VERTEX_LOCATION["vertex_anthropic"],
            "credentials": _load_credentials(cfg.credentials_path),
        }
        if cfg.project:
            kwargs["project"] = cfg.project
        kwargs["max_output_tokens"] = (
            cfg.max_tokens if cfg.max_tokens is not None else _DEFAULT_VERTEX_MAX_TOKENS
        )
        kwargs.update(extra)
        return ChatAnthropicVertex(**kwargs)

    if cfg.provider == "vertex_gemini":
        # The non-deprecated Vertex path: ChatGoogleGenerativeAI(vertexai=True)
        # (langchain's ChatVertexAI is deprecated in favour of this).
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": cfg.model,
            "vertexai": True,
            "temperature": cfg.temperature,
            "location": cfg.location or _DEFAULT_VERTEX_LOCATION["vertex_gemini"],
            "credentials": _load_credentials(cfg.credentials_path),
        }
        if cfg.project:
            kwargs["project"] = cfg.project
        kwargs["max_output_tokens"] = (
            cfg.max_tokens if cfg.max_tokens is not None else _DEFAULT_VERTEX_MAX_TOKENS
        )
        kwargs.update(extra)
        return ChatGoogleGenerativeAI(**kwargs)

    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": cfg.model, "temperature": cfg.temperature}
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        kwargs.update(extra)
        return ChatAnthropic(**kwargs)

    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": cfg.model, "temperature": cfg.temperature}
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        kwargs.update(extra)
        return ChatOpenAI(**kwargs)

    if cfg.provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": cfg.model, "temperature": cfg.temperature}
        if cfg.max_tokens is not None:
            kwargs["max_output_tokens"] = cfg.max_tokens
        if cfg.api_key:
            kwargs["google_api_key"] = cfg.api_key
        kwargs.update(extra)
        return ChatGoogleGenerativeAI(**kwargs)

    raise ValueError(f"Unknown provider: {cfg.provider!r}")


def build_llm(
    config: ModelConfig | None = None,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
):
    """Construct a LangChain chat model from configuration + per-request overrides.

    Precedence is designed for the headline deployment — *ADC on a GCP VM,
    served through the React UI* — where the browser still sends ``X-API-Key`` /
    ``X-Model-Name`` headers (the login gate only does an unauthenticated health
    check, so a user may type a dummy key just to get in):

    * **Vertex provider configured** → the server's ADC config is authoritative.
      A client-supplied ``api_key`` is **ignored**. A client-supplied
      ``model_name`` is honored only if it is the same model family as the
      configured provider (e.g. swapping one Claude id for another); a
      cross-family name is ignored.
    * **Direct provider configured** → a client-supplied ``api_key`` /
      ``model_name`` selects a direct provider inferred from the model name.
      This preserves the multi-provider key-entry flow the UI uses on
      non-Vertex deployments.

    Args:
        config: A pre-loaded :class:`ModelConfig`. If ``None``, it is loaded via
            :func:`load_model_config`.
        model_name: Optional per-request model override (e.g. ``X-Model-Name``).
        api_key: Optional per-request API key (e.g. ``X-API-Key``).

    Returns:
        A LangChain chat model ready for ``.bind_tools(...)``.
    """
    cfg = config or load_model_config()

    # Treat a blank/sentinel client key as absent so it can't clobber a
    # server-side credential (Vertex/ADC or a provider env key).
    if api_key is not None and api_key.strip().lower() in _SENTINEL_API_KEYS:
        api_key = None

    if cfg.uses_vertex:
        # ADC is authoritative: never let a stale/dummy client key hijack Vertex.
        if model_name and _model_family(model_name) == _PROVIDER_FAMILY[cfg.provider]:
            cfg = cfg.model_copy(update={"model": model_name})
        return _build_from_config(cfg)

    # Direct provider: honor per-request UI overrides.
    if api_key or model_name:
        effective_model = model_name or cfg.model
        cfg = cfg.model_copy(
            update={
                "provider": infer_provider_from_model(effective_model),
                "model": effective_model,
                "api_key": api_key or cfg.api_key,
            }
        )
    return _build_from_config(cfg)


# ── Introspection (for a non-secret /model-config endpoint) ─────────────────


def _provider_env_key_present(provider: str) -> bool:
    return any(os.environ.get(var) for var in _PROVIDER_ENV_KEYS.get(provider, ()))


def describe_active_config(config: ModelConfig | None = None) -> dict[str, Any]:
    """Return a non-secret summary of the active model configuration.

    Safe to expose over HTTP: it never includes the API key or the contents of a
    credentials file. ``requires_api_key`` tells a frontend whether to prompt
    the user for a key — ``False`` when the server authenticates via Vertex/ADC,
    a config-file key, or a provider environment variable.
    """
    cfg = config or load_model_config()
    requires_api_key = (
        not cfg.uses_vertex
        and not cfg.api_key
        and not _provider_env_key_present(cfg.provider)
    )
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "uses_vertex": cfg.uses_vertex,
        "uses_adc": cfg.uses_adc,
        "project": cfg.project,
        "location": cfg.location,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "requires_api_key": requires_api_key,
    }
