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

import logging
import os
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

Provider = Literal[
    "vertex_anthropic",
    "vertex_gemini",
    "anthropic",
    "openai",
    "google_genai",
    "lmstudio",
]

_VERTEX_PROVIDERS: frozenset[str] = frozenset({"vertex_anthropic", "vertex_gemini"})

# All valid provider strings (for validating a client X-Provider override).
_VALID_PROVIDERS: frozenset[str] = frozenset(get_args(Provider))

# OpenAI-compatible local/self-hosted endpoints. For these the model name is
# arbitrary (whatever the user loaded), so a per-request model override must NOT
# re-infer a different cloud provider — it just swaps the model on this endpoint.
_ENDPOINT_PROVIDERS: frozenset[str] = frozenset({"lmstudio"})

# Default base URL for LM Studio's OpenAI-compatible server.
_DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# Default config-file search locations, relative to the current working directory.
_DEFAULT_CONFIG_PATHS: tuple[str, ...] = (
    "model_config.yaml",
    "config/model_config.yaml",
)

# Environment-variable prefix for per-field overrides (e.g. MMM_LLM_PROVIDER).
_ENV_PREFIX = "MMM_LLM_"

# Prefix for overrides targeting the nested "expert" (strong) tier, e.g.
# MMM_LLM_EXPERT_PROVIDER / MMM_LLM_EXPERT_MODEL.
_EXPERT_ENV_PREFIX = "MMM_LLM_EXPERT_"

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
        "base_url",
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

# Within Vertex, a selected model's family maps to the Vertex provider that
# serves it. Used so a user can pick a Gemini *or* Claude id from the discovered
# list and have build_llm route to the right Vertex backend (never to a direct,
# key-based provider — the ADC-only security property is preserved).
_VERTEX_PROVIDER_FOR_FAMILY: dict[str, str] = {
    "claude": "vertex_anthropic",
    "gemini": "vertex_gemini",
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

    # --- OpenAI-compatible endpoint settings (lmstudio / self-hosted) ---
    # Base URL of the local server, e.g. LM Studio's http://localhost:1234/v1.
    base_url: str | None = None

    # Optional convenience model ids surfaced in the Vertex model picker in
    # addition to live-discovered models (e.g. Claude/Model-Garden ids, which
    # can't be reliably enumerated). These are user-curated — verify them in
    # your Vertex console; they are never auto-populated with guessed ids.
    extra_models: list[str] = Field(default_factory=list)

    # Free-form passthrough to the underlying LangChain constructor.
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    # --- Optional strong/"expert" tier ---
    # A second, fully-specified provider block used by the agent's
    # ``delegate_to_expert`` tool for code generation, model fitting,
    # optimization, and other hard tasks. The top-level block above stays the
    # fast "chat" tier. When this is ``None`` the expert tier reuses the chat
    # model, so single-model deployments are unaffected. The expert block may
    # mix providers freely (e.g. a Gemini-flash chat tier + a Claude-Sonnet
    # expert tier). Unset Vertex fields (project/location/credentials_path) are
    # inherited from the chat tier at build time (see :func:`build_expert_llm`).
    expert: "ModelConfig | None" = None

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


# Resolve the ``expert: "ModelConfig | None"`` forward reference (self-recursive).
ModelConfig.model_rebuild()


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
    """Overlay ``MMM_LLM_*`` environment variables onto a config dict in place.

    ``MMM_LLM_EXPERT_*`` keys map onto the nested ``expert`` block (the strong
    tier), e.g. ``MMM_LLM_EXPERT_PROVIDER`` / ``MMM_LLM_EXPERT_MODEL``. They are
    handled first so the generic loop's stripped field name (``expert_provider``)
    — which isn't in ``_ENV_OVERRIDABLE`` — is simply ignored there.
    """
    expert: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(_EXPERT_ENV_PREFIX):
            continue
        field = key[len(_EXPERT_ENV_PREFIX) :].lower()
        if field in _ENV_OVERRIDABLE and value != "":
            expert[field] = value
    if expert:
        merged = {**(data.get("expert") or {}), **expert}
        data["expert"] = merged

    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX) or key.startswith(_EXPERT_ENV_PREFIX):
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

    if cfg.provider == "lmstudio":
        # LM Studio exposes an OpenAI-compatible server, so we drive it through
        # ChatOpenAI pointed at its base URL. The api_key is unused by LM Studio
        # but the OpenAI client requires a non-empty string. Tool calling works
        # for models that support it.
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "base_url": cfg.base_url or _DEFAULT_LMSTUDIO_BASE_URL,
            "api_key": cfg.api_key or "lm-studio",
        }
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        kwargs.update(extra)
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unknown provider: {cfg.provider!r}")


def build_llm(
    config: ModelConfig | None = None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
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
    cfg = _apply_request_overrides(
        cfg,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    return _build_from_config(cfg)


def _apply_request_overrides(
    cfg: ModelConfig,
    *,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ModelConfig:
    """Fold per-request UI overrides into a base config, returning the result.

    This is the shared precedence logic for BOTH model tiers — :func:`build_llm`
    (chat tier, ``X-*`` headers) and :func:`build_expert_llm` (expert tier,
    ``X-Expert-*`` headers) — so the security properties are identical:

    * **Vertex provider** → ADC is authoritative; a client key is ignored, and a
      client model is honored only when its family maps to a Vertex backend
      (routes ``vertex_gemini`` ↔ ``vertex_anthropic``, never off Vertex).
    * **Non-Vertex** → a valid ``X-Provider`` may switch provider entirely; an
      endpoint provider (LM Studio) swaps only the model/base_url on the same
      endpoint; otherwise a direct provider is inferred from the model name.
    """
    # Treat a blank/sentinel client key as absent so it can't clobber a
    # server-side credential (Vertex/ADC or a provider env key).
    if api_key is not None and api_key.strip().lower() in _SENTINEL_API_KEYS:
        api_key = None

    if cfg.uses_vertex:
        # ADC is authoritative: never let a stale/dummy client key hijack Vertex.
        # A selected model may belong to either Vertex family — route to the
        # matching Vertex provider (vertex_gemini / vertex_anthropic), keeping
        # the same project/location/credentials. Cross-family selection never
        # leaves Vertex (no direct, key-based provider), so ADC still governs.
        if model_name:
            target = _VERTEX_PROVIDER_FOR_FAMILY.get(_model_family(model_name))
            if target:
                cfg = cfg.model_copy(update={"provider": target, "model": model_name})
            # Unknown family (e.g. an OpenAI id) is ignored: keep server config.
        return cfg

    # Non-Vertex server: a client may switch the provider entirely (X-Provider),
    # e.g. to reach LM Studio or OpenAI from a default-Anthropic deployment. The
    # Vertex branch above runs first, so a Vertex/ADC deployment ignores this and
    # stays locked (preserving its ADC-only credential property). An invalid
    # provider string falls through to the existing inference behaviour.
    client_provider = (provider or "").strip().lower() or None
    if client_provider and client_provider in _VALID_PROVIDERS:
        update = {"provider": client_provider}
        if model_name:
            update["model"] = model_name
        if api_key:
            update["api_key"] = api_key
        if base_url:
            update["base_url"] = base_url
        return cfg.model_copy(update=update)

    if cfg.provider in _ENDPOINT_PROVIDERS:
        # OpenAI-compatible local endpoint (LM Studio): a model_name override
        # just swaps the loaded model on the SAME endpoint — never re-route to a
        # cloud provider. A client api_key is honored only if the endpoint needs
        # one (LM Studio ignores it). A client base_url override is honored ONLY
        # here (an endpoint provider) so it can't redirect a cloud deployment.
        update: dict[str, Any] = {}
        if model_name:
            update["model"] = model_name
        if api_key:
            update["api_key"] = api_key
        if base_url:
            update["base_url"] = base_url
        if update:
            cfg = cfg.model_copy(update=update)
        return cfg

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
    return cfg


# Vertex/GCP fields the expert block inherits from the chat tier when left unset,
# so a Vertex deployment needn't repeat project/region/credentials twice.
_EXPERT_INHERITED_FIELDS: tuple[str, ...] = ("project", "location", "credentials_path")


def resolve_expert_config(config: ModelConfig | None = None) -> ModelConfig:
    """Return the fully-resolved strong/"expert" tier configuration.

    Falls back to the chat tier (the top-level block) when no ``expert`` block is
    configured, so single-model deployments keep a working delegate path. When an
    ``expert`` block is present, any unset Vertex/GCP fields are inherited from the
    chat tier (see :data:`_EXPERT_INHERITED_FIELDS`).
    """
    cfg = config or load_model_config()
    if cfg.expert is None:
        # No second tier: the expert reuses the chat model. Drop the (absent)
        # nested field so we don't carry a self-reference around.
        return cfg.model_copy(update={"expert": None})
    inherited = {
        field: getattr(cfg, field)
        for field in _EXPERT_INHERITED_FIELDS
        if getattr(cfg.expert, field) is None and getattr(cfg, field) is not None
    }
    expert = cfg.expert.model_copy(update={**inherited, "expert": None})
    return expert


def build_expert_llm(
    config: ModelConfig | None = None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Construct the LangChain chat model for the strong/"expert" tier.

    Used by the agent's ``delegate_to_expert`` tool. Starts from the configured
    expert block (or the chat tier when none is configured) and folds in any
    per-request overrides from the ``X-Expert-*`` headers via the SAME precedence
    rules as the chat tier (:func:`_apply_request_overrides`) — so a Vertex/ADC
    deployment stays ADC-authoritative for the expert too, while a direct-provider
    deployment may point the expert at a different provider + key.

    Returns a model ready for ``.bind_tools(...)``. With no expert block and no
    overrides this returns the chat model, so delegation degrades gracefully to a
    single-model deployment.
    """
    cfg = resolve_expert_config(config)
    cfg = _apply_request_overrides(
        cfg,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
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
    # Local OpenAI-compatible endpoints (LM Studio) need no key.
    is_local_endpoint = cfg.provider in _ENDPOINT_PROVIDERS
    requires_api_key = (
        not cfg.uses_vertex
        and not is_local_endpoint
        and not cfg.api_key
        and not _provider_env_key_present(cfg.provider)
    )
    # Non-secret summary of the strong tier (provider + model only). When no
    # expert block is configured the delegate path reuses the chat model, so we
    # report that resolved (chat) provider/model.
    expert_cfg = resolve_expert_config(cfg)
    expert_summary = {
        "provider": expert_cfg.provider,
        "model": expert_cfg.model,
        "configured": cfg.expert is not None,
    }
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "uses_vertex": cfg.uses_vertex,
        "uses_adc": cfg.uses_adc,
        "is_local_endpoint": is_local_endpoint,
        "base_url": (
            (cfg.base_url or _DEFAULT_LMSTUDIO_BASE_URL) if is_local_endpoint else None
        ),
        "project": cfg.project,
        "location": cfg.location,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "requires_api_key": requires_api_key,
        "expert": expert_summary,
    }


# ── Vertex model discovery ──────────────────────────────────────────────────


def _normalize_model_id(name: str | None) -> str:
    """Reduce a Vertex resource name to the bare id the constructor expects.

    ``publishers/google/models/gemini-2.5-pro`` -> ``gemini-2.5-pro`` and
    ``models/gemini-2.5-pro`` -> ``gemini-2.5-pro``. The bare form is exactly
    what ``build_llm`` passes to ``ChatGoogleGenerativeAI`` / ``ChatAnthropicVertex``.
    """
    if not name:
        return ""
    if "/models/" in name:
        return name.split("/models/")[-1]
    if name.startswith("models/"):
        return name.split("/", 1)[1]
    return name


def _discover_gemini_vertex_models(
    project: str | None, location: str | None, credentials_path: str | None
) -> list[dict[str, Any]]:
    """Live-list Gemini base models available to the project/region via ADC.

    Scoped to ``location`` so it never offers models the region can't serve.
    Returns ``[]`` on any failure (missing creds, no project, API error) so the
    caller degrades gracefully to manual entry.
    """
    try:
        from google import genai

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=_load_credentials(credentials_path),
        )
        out: list[dict[str, Any]] = []
        for m in client.models.list(config={"query_base": True}):
            mid = _normalize_model_id(getattr(m, "name", None))
            if not mid or "gemini" not in mid.lower():
                continue
            actions = [a.lower() for a in (getattr(m, "supported_actions", None) or [])]
            # Keep only chat-capable models (skip embeddings, etc.). If the model
            # doesn't report actions, include it rather than hide it.
            if actions and not any("generatecontent" in a for a in actions):
                continue
            out.append(
                {
                    "id": mid,
                    "provider": "vertex_gemini",
                    "family": "gemini",
                    "display_name": getattr(m, "display_name", None) or mid,
                    "source": "live",
                    "location": location,
                }
            )
        return out
    except Exception as exc:
        logger.debug("Gemini Vertex model discovery failed: %s", exc)
        return []


def _discover_anthropic_vertex_models(
    project: str | None, location: str | None, credentials_path: str | None
) -> list[dict[str, Any]]:
    """Best-effort list of Claude models from the Model Garden catalog.

    This is the global publisher catalog (not project/region enablement), so it
    is advisory: entries are tagged ``source="catalog"`` and the picker always
    keeps a free-text field. Returns ``[]`` on any failure — Claude selection
    then relies on ``extra_models`` and manual entry, never on guessed ids.
    """
    try:
        from google.cloud import aiplatform_v1beta1 as ga

        client_options = (
            {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
            if location
            else None
        )
        client = ga.ModelGardenServiceClient(
            credentials=_load_credentials(credentials_path),
            client_options=client_options,
        )
        out: list[dict[str, Any]] = []
        for pm in client.list_publisher_models(parent="publishers/anthropic"):
            mid = _normalize_model_id(getattr(pm, "name", None))
            if "claude" not in mid.lower():
                continue
            out.append(
                {
                    "id": mid,
                    "provider": "vertex_anthropic",
                    "family": "claude",
                    "display_name": mid,
                    "source": "catalog",
                    "location": location,
                }
            )
        return out
    except Exception as exc:
        logger.debug("Anthropic Vertex model discovery failed: %s", exc)
        return []


def _dedupe_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate ids, keeping the first (live > catalog > config order)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in models:
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        out.append(m)
    return out


def list_vertex_models(
    config: ModelConfig | None = None,
    *,
    project: str | None = None,
    location: str | None = None,
    credentials_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return selectable Vertex models for the given (or configured) project/region.

    Combines: live-discovered Gemini models (project/region-scoped), a
    best-effort Claude catalog from Model Garden, and any user-curated
    ``extra_models`` from the config. Each entry is
    ``{id, provider, family, display_name, source, location}``. The list may be
    empty (e.g. ADC not configured); callers should always also allow free-text
    model entry. Never raises — discovery failures degrade to fewer entries.
    """
    cfg = config or load_model_config()
    project = project if project is not None else cfg.project
    location = location if location is not None else cfg.location
    credentials_path = (
        credentials_path if credentials_path is not None else cfg.credentials_path
    )

    models: list[dict[str, Any]] = []
    models += _discover_gemini_vertex_models(project, location, credentials_path)
    models += _discover_anthropic_vertex_models(project, location, credentials_path)

    for mid in cfg.extra_models or []:
        family = _model_family(mid)
        provider = _VERTEX_PROVIDER_FOR_FAMILY.get(family)
        if provider:
            models.append(
                {
                    "id": mid,
                    "provider": provider,
                    "family": family,
                    "display_name": mid,
                    "source": "config",
                    "location": location,
                }
            )

    return _dedupe_models(models)


def lmstudio_base_url(config: ModelConfig | None = None) -> str:
    """The LM Studio base URL: config/env override, else the default port."""
    cfg = config or load_model_config()
    return (
        cfg.base_url or os.environ.get("MMM_LLM_BASE_URL") or _DEFAULT_LMSTUDIO_BASE_URL
    )


def list_lmstudio_models(
    config: ModelConfig | None = None, *, base_url: str | None = None
) -> list[dict[str, Any]]:
    """List the models currently loaded in LM Studio via its OpenAI-compatible
    ``GET /v1/models`` endpoint. Returns ``[]`` if the server is unreachable so
    the caller degrades gracefully to manual model entry. Never raises.
    """
    import json as _json
    import urllib.request

    cfg = config or load_model_config()
    url = (base_url or cfg.base_url or _DEFAULT_LMSTUDIO_BASE_URL).rstrip(
        "/"
    ) + "/models"
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:  # noqa: S310 (local URL)
            payload = _json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("LM Studio model discovery failed (%s): %s", url, exc)
        return []

    out: list[dict[str, Any]] = []
    for m in payload.get("data", []) or []:
        mid = m.get("id") if isinstance(m, dict) else None
        if not mid:
            continue
        out.append(
            {
                "id": mid,
                "provider": "lmstudio",
                "family": "lmstudio",
                "display_name": mid,
                "source": "live",
            }
        )
    return out
