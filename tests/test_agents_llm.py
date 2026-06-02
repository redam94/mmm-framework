"""Tests for the agent LLM configuration and construction layer.

These cover config-file loading, env overrides, the ADC-vs-client-key
precedence rules (the behaviour that makes "Vertex AI + ADC on a VM, served
through the UI" work correctly), and that each provider dispatches to the right
LangChain class with the expected kwargs — all without any network calls.
"""

import textwrap

import pytest

from mmm_framework.agents import llm as llm_mod
from mmm_framework.agents.llm import (
    ModelConfig,
    build_llm,
    describe_active_config,
    infer_provider_from_model,
    list_vertex_models,
    load_model_config,
)


# ── ModelConfig ─────────────────────────────────────────────────────────────


def test_default_config_is_backward_compatible():
    cfg = ModelConfig()
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.uses_vertex is False
    assert cfg.uses_adc is False


@pytest.mark.parametrize(
    "provider,expected_vertex,credentials,expected_adc",
    [
        ("vertex_anthropic", True, None, True),
        ("vertex_gemini", True, None, True),
        ("vertex_anthropic", True, "/path/key.json", False),  # explicit key, not ADC
        ("anthropic", False, None, False),
        ("openai", False, None, False),
    ],
)
def test_uses_vertex_and_adc(provider, expected_vertex, credentials, expected_adc):
    cfg = ModelConfig(provider=provider, credentials_path=credentials)
    assert cfg.uses_vertex is expected_vertex
    assert cfg.uses_adc is expected_adc


def test_invalid_provider_raises_clear_error():
    with pytest.raises(ValueError):
        ModelConfig(provider="not_a_provider")


# ── Config loading ──────────────────────────────────────────────────────────


def _write_cfg(tmp_path, body: str):
    p = tmp_path / "model_config.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_from_explicit_path(tmp_path):
    p = _write_cfg(
        tmp_path,
        """
        provider: vertex_anthropic
        model: claude-sonnet-4-5@20250929
        project: my-proj
        location: us-east5
        max_tokens: 8192
        """,
    )
    cfg = load_model_config(p)
    assert cfg.provider == "vertex_anthropic"
    assert cfg.model == "claude-sonnet-4-5@20250929"
    assert cfg.project == "my-proj"
    assert cfg.location == "us-east5"
    assert cfg.max_tokens == 8192
    assert cfg.uses_adc is True  # no credentials_path => ADC


def test_blank_yaml_values_fall_back_to_defaults(tmp_path):
    # A blank `temperature:` is null in YAML and must NOT override the default.
    p = _write_cfg(
        tmp_path,
        """
        provider: openai
        model: gpt-4o
        temperature:
        project:
        """,
    )
    cfg = load_model_config(p)
    assert cfg.temperature == 0.0
    assert cfg.project is None


def test_env_overrides_take_precedence(tmp_path, monkeypatch):
    p = _write_cfg(
        tmp_path,
        """
        provider: anthropic
        model: claude-sonnet-4-6
        """,
    )
    monkeypatch.setenv("MMM_LLM_PROVIDER", "vertex_gemini")
    monkeypatch.setenv("MMM_LLM_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("MMM_LLM_PROJECT", "env-proj")
    monkeypatch.setenv("MMM_LLM_MAX_TOKENS", "4096")
    cfg = load_model_config(p)
    assert cfg.provider == "vertex_gemini"
    assert cfg.model == "gemini-2.5-pro"
    assert cfg.project == "env-proj"
    assert cfg.max_tokens == 4096  # coerced from str


def test_missing_file_uses_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no model_config.yaml here
    monkeypatch.delenv("MMM_MODEL_CONFIG", raising=False)
    cfg = load_model_config()
    assert cfg.provider == "anthropic"


def test_non_mapping_yaml_raises(tmp_path):
    p = tmp_path / "model_config.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_model_config(p)


# ── Provider inference ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("gpt-4o", "openai"),
        ("o3-mini", "openai"),
        ("gemini-2.5-pro", "google_genai"),
        ("claude-sonnet-4-6", "anthropic"),
        ("something-weird", "anthropic"),  # default family
    ],
)
def test_infer_provider_from_model(name, expected):
    assert infer_provider_from_model(name) == expected


# ── build_llm precedence (the critical ADC-vs-client-key logic) ─────────────
# We stub _build_from_config so we can assert on the resolved config without
# constructing any real client.


@pytest.fixture
def capture_cfg(monkeypatch):
    monkeypatch.setattr(llm_mod, "_build_from_config", lambda cfg: cfg)
    return None


def test_vertex_ignores_client_api_key(capture_cfg):
    cfg = ModelConfig(provider="vertex_anthropic", model="claude-x", project="p")
    out = build_llm(cfg, model_name="gpt-4o", api_key="sk-bogus")
    # api_key ignored; cross-family model name (gpt vs claude) ignored.
    assert out.provider == "vertex_anthropic"
    assert out.model == "claude-x"
    assert out.api_key is None


def test_vertex_allows_same_family_model_swap(capture_cfg):
    cfg = ModelConfig(provider="vertex_anthropic", model="claude-a")
    out = build_llm(cfg, model_name="claude-b", api_key="sk-bogus")
    assert out.provider == "vertex_anthropic"
    assert out.model == "claude-b"  # same family => honored
    assert out.api_key is None


def test_vertex_cross_family_switches_provider(capture_cfg):
    # Picking a Claude id while configured for vertex_gemini routes to
    # vertex_anthropic (same project/location/ADC), and vice versa.
    cfg = ModelConfig(provider="vertex_gemini", model="gemini-2.5-pro", project="p")
    out = build_llm(cfg, model_name="claude-sonnet-4-5@20250929")
    assert out.provider == "vertex_anthropic"
    assert out.model == "claude-sonnet-4-5@20250929"
    assert out.project == "p"  # project/location preserved
    assert out.api_key is None  # never leaves Vertex / never takes a key


def test_vertex_unknown_family_keeps_config(capture_cfg):
    # An OpenAI id has no Vertex provider -> ignore it, keep server config.
    cfg = ModelConfig(provider="vertex_gemini", model="gemini-2.5-pro")
    out = build_llm(cfg, model_name="gpt-4o")
    assert out.provider == "vertex_gemini"
    assert out.model == "gemini-2.5-pro"


def test_direct_provider_honors_client_override(capture_cfg):
    cfg = ModelConfig(provider="anthropic", model="claude-x")
    out = build_llm(cfg, model_name="gpt-4o", api_key="sk-1")
    assert out.provider == "openai"
    assert out.model == "gpt-4o"
    assert out.api_key == "sk-1"


def test_direct_provider_model_only_override(capture_cfg):
    cfg = ModelConfig(provider="anthropic", model="claude-x")
    out = build_llm(cfg, model_name="gemini-2.5-pro")
    assert out.provider == "google_genai"
    assert out.model == "gemini-2.5-pro"


@pytest.mark.parametrize("sentinel", ["", "  ", "server-managed", "__server_managed__"])
def test_sentinel_key_does_not_clobber_server_env_key(capture_cfg, sentinel):
    # Frontend "server-managed" mode sends a placeholder key; it must be ignored
    # so the server's env/config credential (here: ANTHROPIC_API_KEY) is used.
    cfg = ModelConfig(provider="anthropic", model="claude-x")
    out = build_llm(cfg, model_name="claude-sonnet-4-6", api_key=sentinel)
    assert out.provider == "anthropic"
    assert out.model == "claude-sonnet-4-6"
    assert out.api_key is None  # sentinel normalized away => falls back to env


def test_sentinel_key_ignored_for_vertex(capture_cfg):
    cfg = ModelConfig(provider="vertex_anthropic", model="claude-x", project="p")
    out = build_llm(cfg, model_name="claude-sonnet-4-6", api_key="server-managed")
    assert out.provider == "vertex_anthropic"
    assert out.api_key is None


# ── Real construction (offline) — guards against provider kwarg renames ─────


def test_build_direct_anthropic_constructs():
    cfg = ModelConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="x")
    obj = build_llm(cfg)
    assert type(obj).__name__ == "ChatAnthropic"


def test_build_direct_openai_constructs():
    cfg = ModelConfig(provider="openai", model="gpt-4o", api_key="x")
    obj = build_llm(cfg)
    assert type(obj).__name__ == "ChatOpenAI"


def test_build_vertex_anthropic_constructs(monkeypatch):
    # Avoid an ADC lookup by injecting anonymous credentials.
    from google.auth.credentials import AnonymousCredentials

    monkeypatch.setattr(llm_mod, "_load_credentials", lambda _p: AnonymousCredentials())
    cfg = ModelConfig(
        provider="vertex_anthropic",
        model="claude-x",
        project="p",
        location="us-east5",
        max_tokens=8192,
    )
    obj = build_llm(cfg)
    assert type(obj).__name__ == "ChatAnthropicVertex"
    assert obj.model_name == "claude-x"
    assert obj.location == "us-east5"


def test_build_vertex_gemini_constructs(monkeypatch):
    from google.auth.credentials import AnonymousCredentials

    monkeypatch.setattr(llm_mod, "_load_credentials", lambda _p: AnonymousCredentials())
    cfg = ModelConfig(
        provider="vertex_gemini",
        model="gemini-2.5-pro",
        project="p",
        max_tokens=8192,
    )
    obj = build_llm(cfg)
    assert type(obj).__name__ == "ChatGoogleGenerativeAI"
    assert obj.vertexai is True
    assert obj.location == "us-central1"  # default applied


def test_vertex_default_location_for_claude(monkeypatch):
    from google.auth.credentials import AnonymousCredentials

    monkeypatch.setattr(llm_mod, "_load_credentials", lambda _p: AnonymousCredentials())
    cfg = ModelConfig(provider="vertex_anthropic", model="claude-x", project="p")
    obj = build_llm(cfg)
    assert obj.location == "us-east5"  # Claude default, not us-central1


def test_vertex_default_max_tokens_applied(monkeypatch):
    # When max_tokens is unset, Vertex providers get a generous default rather
    # than ChatAnthropicVertex's low 4096, to avoid truncating long report turns.
    from google.auth.credentials import AnonymousCredentials

    monkeypatch.setattr(llm_mod, "_load_credentials", lambda _p: AnonymousCredentials())
    cfg = ModelConfig(provider="vertex_anthropic", model="claude-x", project="p")
    obj = build_llm(cfg)  # no max_tokens
    assert obj.max_output_tokens == llm_mod._DEFAULT_VERTEX_MAX_TOKENS


# ── describe_active_config ──────────────────────────────────────────────────


def test_describe_hides_secrets_and_flags_adc():
    cfg = ModelConfig(
        provider="vertex_anthropic",
        model="claude-x",
        project="p",
        location="us-east5",
        api_key="should-not-leak",
    )
    info = describe_active_config(cfg)
    assert "api_key" not in info
    assert "credentials_path" not in info
    assert info["uses_vertex"] is True
    assert info["uses_adc"] is True
    assert info["requires_api_key"] is False  # Vertex never needs a client key


def test_describe_requires_key_for_direct_without_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = ModelConfig(provider="anthropic", model="claude-sonnet-4-6")
    info = describe_active_config(cfg)
    assert info["requires_api_key"] is True


def test_describe_no_key_needed_when_env_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = ModelConfig(provider="anthropic", model="claude-sonnet-4-6")
    info = describe_active_config(cfg)
    assert info["requires_api_key"] is False


# ── Vertex model discovery ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("publishers/google/models/gemini-2.5-pro", "gemini-2.5-pro"),
        ("publishers/anthropic/models/claude-3-5-sonnet", "claude-3-5-sonnet"),
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_model_id(raw, expected):
    assert llm_mod._normalize_model_id(raw) == expected


def test_list_vertex_models_combines_sources(monkeypatch):
    # Stub the two discovery helpers so no network is hit.
    monkeypatch.setattr(
        llm_mod,
        "_discover_gemini_vertex_models",
        lambda p, loc, c: [
            {
                "id": "gemini-2.5-pro",
                "provider": "vertex_gemini",
                "family": "gemini",
                "display_name": "Gemini 2.5 Pro",
                "source": "live",
                "location": loc,
            },
        ],
    )
    monkeypatch.setattr(
        llm_mod,
        "_discover_anthropic_vertex_models",
        lambda p, loc, c: [
            {
                "id": "claude-3-5-sonnet",
                "provider": "vertex_anthropic",
                "family": "claude",
                "display_name": "claude-3-5-sonnet",
                "source": "catalog",
                "location": loc,
            },
        ],
    )
    cfg = ModelConfig(
        provider="vertex_anthropic",
        model="claude-x",
        project="p",
        location="us-east5",
        extra_models=[
            "claude-sonnet-4-5@20250929",
            "gpt-4o",
        ],  # gpt-4o has no vertex provider
    )
    models = list_vertex_models(cfg)
    ids = [m["id"] for m in models]
    assert "gemini-2.5-pro" in ids
    assert "claude-3-5-sonnet" in ids
    assert "claude-sonnet-4-5@20250929" in ids  # extra_models (claude family)
    assert "gpt-4o" not in ids  # non-vertex family dropped
    # config extra carries the right provider/source
    extra = next(m for m in models if m["id"] == "claude-sonnet-4-5@20250929")
    assert extra["provider"] == "vertex_anthropic"
    assert extra["source"] == "config"


def test_list_vertex_models_dedupes(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "_discover_gemini_vertex_models",
        lambda p, loc, c: [
            {
                "id": "gemini-2.5-pro",
                "provider": "vertex_gemini",
                "family": "gemini",
                "display_name": "g",
                "source": "live",
                "location": loc,
            },
        ],
    )
    monkeypatch.setattr(
        llm_mod, "_discover_anthropic_vertex_models", lambda p, loc, c: []
    )
    # extra_models repeats the live id -> should be deduped (live kept)
    cfg = ModelConfig(
        provider="vertex_gemini",
        model="gemini-2.5-pro",
        extra_models=["gemini-2.5-pro"],
    )
    models = list_vertex_models(cfg)
    assert [m["id"] for m in models].count("gemini-2.5-pro") == 1
    assert models[0]["source"] == "live"


def test_discovery_helpers_never_raise(monkeypatch):
    # With no usable credentials/SDK path, helpers must return [] not raise.
    assert llm_mod._discover_gemini_vertex_models(None, None, None) == []
    assert llm_mod._discover_anthropic_vertex_models(None, None, None) == []
