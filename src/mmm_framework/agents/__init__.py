"""Agentic Framework for Marketing Mix Modeling using LangGraph.

The LLM stack (langgraph / langchain-*) is an OPTIONAL install::

    pip install mmm-framework[agents]

This ``__init__`` is deliberately lazy (PEP 562): the langchain-free service
modules in this package — ``fitting``, ``workspace``, ``tables``,
``estimand_rows``, ``spec_locks``, ``spec_normalize``, ``model_ops``,
``report_builder``, ``branding`` — are imported from core code (validation
spec-curve, garden compat, data studio, reporting triangulation) and MUST stay
importable in a lean ``pip install mmm-framework`` environment. Importing the
package therefore must not eagerly pull ``graph``/``llm``/``state``, which
require the [agents] extra. tests/test_lean_imports.py pins this invariant.
"""

from typing import TYPE_CHECKING

# name -> submodule that provides it (all lazily resolved)
_LAZY_EXPORTS = {
    "AgentState": "state",
    "ModelSpec": "state",
    "create_agent_graph": "graph",
    "ModelConfig": "llm",
    "build_llm": "llm",
    "describe_active_config": "llm",
    "list_vertex_models": "llm",
    "list_lmstudio_models": "llm",
    "lmstudio_base_url": "llm",
    "load_model_config": "llm",
}

__all__ = list(_LAZY_EXPORTS)

if TYPE_CHECKING:  # static analyzers see the real symbols
    from mmm_framework.agents.graph import create_agent_graph
    from mmm_framework.agents.llm import (
        ModelConfig,
        build_llm,
        describe_active_config,
        list_lmstudio_models,
        list_vertex_models,
        lmstudio_base_url,
        load_model_config,
    )
    from mmm_framework.agents.state import AgentState, ModelSpec


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        try:
            module = importlib.import_module(
                f"mmm_framework.agents.{_LAZY_EXPORTS[name]}"
            )
        except ModuleNotFoundError as exc:  # missing optional LLM stack
            raise ModuleNotFoundError(
                f"{exc.name!r} is required for mmm_framework.agents.{name}. "
                "The LLM agent stack is an optional install: "
                "pip install 'mmm-framework[agents]'"
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
