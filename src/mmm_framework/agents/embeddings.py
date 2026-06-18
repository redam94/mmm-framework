"""Embedding-model resolution for the knowledge base.

The chat LLM and the embedding model are resolved **independently**: the most
common configured chat provider here is ``vertex_anthropic`` (Claude on Vertex),
and **Anthropic has no embedding model**. So we map the chat provider to a
sensible embedder that shares its credentials where possible:

* Vertex providers  → ``VertexAIEmbeddings`` (ADC, same GCP ``project``).
* google_genai      → ``GoogleGenerativeAIEmbeddings``.
* openai            → ``OpenAIEmbeddings``.
* anthropic (direct)→ Vertex if a GCP project is configured, else OpenAI if a
  key is present, else a clear error.

Everything is overridable via ``MMM_EMBED_PROVIDER`` / ``MMM_EMBED_MODEL`` /
``MMM_EMBED_LOCATION``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from mmm_framework.agents.llm import ModelConfig, load_model_config

# Default embedding model per embedder backend.
_DEFAULT_MODELS = {
    "vertex": "text-embedding-005",
    "google_genai": "models/text-embedding-004",
    "openai": "text-embedding-3-small",
    # LM Studio model ids are arbitrary; this is a common default, but the user
    # should set MMM_EMBED_MODEL to whichever embedding model they loaded.
    "lmstudio": "text-embedding-nomic-embed-text-v1.5",
}

_DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# Vertex text-embedding models are served in us-central1 (independent of the
# us-east5 region used for Claude on Vertex).
_DEFAULT_VERTEX_EMBED_LOCATION = "us-central1"


class EmbeddingConfigError(RuntimeError):
    """Raised when no embedding backend can be resolved for the active config."""


def _backend_for(provider: str) -> str:
    if provider in ("vertex_anthropic", "vertex_gemini"):
        return "vertex"
    if provider == "google_genai":
        return "google_genai"
    if provider == "openai":
        return "openai"
    if provider == "lmstudio":
        # Local-first: embed with LM Studio too (user must load an embedding model).
        return "lmstudio"
    # direct anthropic — pick whatever credentials are around
    if os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("MMM_EMBED_LOCATION"):
        return "vertex"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "google_genai"
    raise EmbeddingConfigError(
        "No embedding backend available for provider 'anthropic'. Anthropic has "
        "no embeddings; set MMM_EMBED_PROVIDER (vertex/openai/google_genai) and "
        "the matching credentials (GOOGLE_CLOUD_PROJECT for Vertex, OPENAI_API_KEY "
        "for OpenAI, GOOGLE_API_KEY for Gemini)."
    )


def _build(cfg: ModelConfig) -> Any:
    backend = os.environ.get("MMM_EMBED_PROVIDER") or _backend_for(cfg.provider)
    model = os.environ.get("MMM_EMBED_MODEL") or _DEFAULT_MODELS.get(backend)

    if backend == "vertex":
        from langchain_google_vertexai import VertexAIEmbeddings

        project = cfg.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = (
            os.environ.get("MMM_EMBED_LOCATION") or _DEFAULT_VERTEX_EMBED_LOCATION
        )
        kwargs: dict[str, Any] = {"model_name": model, "location": location}
        if project:
            kwargs["project"] = project
        return VertexAIEmbeddings(**kwargs)

    if backend == "google_genai":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        kwargs = {"model": model}
        key = (
            cfg.api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if key:
            kwargs["google_api_key"] = key
        return GoogleGenerativeAIEmbeddings(**kwargs)

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs = {"model": model}
        key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            kwargs["api_key"] = key
        return OpenAIEmbeddings(**kwargs)

    if backend == "lmstudio":
        # LM Studio's OpenAI-compatible /v1/embeddings. check_embedding_ctx_length
        # is disabled so langchain sends raw strings (not tiktoken token arrays),
        # which LM Studio expects.
        from langchain_openai import OpenAIEmbeddings

        base_url = (
            os.environ.get("MMM_EMBED_LOCATION")
            or cfg.base_url
            or os.environ.get("MMM_LLM_BASE_URL")
            or _DEFAULT_LMSTUDIO_BASE_URL
        )
        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key="lm-studio",
            check_embedding_ctx_length=False,
        )

    raise EmbeddingConfigError(f"Unknown embedding backend: {backend!r}")


@lru_cache(maxsize=4)
def _cached_embeddings(provider: str, model_key: str) -> Any:
    cfg = load_model_config()
    return _build(cfg)


def build_embeddings(cfg: ModelConfig | None = None) -> Any:
    """Return a LangChain ``Embeddings`` instance for the active configuration.

    The instance is cached per (provider, override-model) so we don't re-create
    a Vertex/OpenAI client on every ingest call.
    """
    cfg = cfg or load_model_config()
    model_key = (
        os.environ.get("MMM_EMBED_MODEL", "")
        + "|"
        + os.environ.get("MMM_EMBED_PROVIDER", "")
    )
    return _cached_embeddings(cfg.provider, model_key)


def embed_documents(
    texts: list[str], cfg: ModelConfig | None = None
) -> list[list[float]]:
    return build_embeddings(cfg).embed_documents(list(texts))


def embed_query(text: str, cfg: ModelConfig | None = None) -> list[float]:
    return build_embeddings(cfg).embed_query(text)


def describe_embedder() -> dict[str, Any]:
    """Non-secret description of the active embedder (for diagnostics)."""
    cfg = load_model_config()
    try:
        backend = os.environ.get("MMM_EMBED_PROVIDER") or _backend_for(cfg.provider)
        model = os.environ.get("MMM_EMBED_MODEL") or _DEFAULT_MODELS.get(backend)
        return {"backend": backend, "model": model, "chat_provider": cfg.provider}
    except EmbeddingConfigError as exc:
        return {
            "backend": None,
            "model": None,
            "chat_provider": cfg.provider,
            "error": str(exc),
        }
