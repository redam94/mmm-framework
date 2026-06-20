"""Model Garden — author, test, version, and share bespoke MMM models.

A *garden model* is a custom ``BayesianMMM`` subclass (see :class:`CustomMMM`)
that an expert authors, tests for compatibility, and registers under a name +
version + documentation so the agent ("the oracle") can load and re-fit it on
any project's data, and so it can be reused across projects and future sessions.

Light symbols (the contract + its checks) are importable directly. The heavier
pieces — :class:`CustomMMM` (pulls in the PyMC model stack), the source
:func:`load_garden_class_from_path`, and the :func:`run_compatibility_check`
suite — are resolved lazily via module ``__getattr__`` so ``import
mmm_framework.garden`` stays cheap and free of import cycles.
"""

from __future__ import annotations

from typing import Any

from .contract import (
    GARDEN_CONTRACT_VERSION,
    PARAM_PREFIXES,
    RECOMMENDED_METHODS,
    REQUIRED_ATTRS,
    REQUIRED_METHODS,
    describe_contract,
    find_garden_class,
    is_bayesian_mmm_subclass,
    validate_class,
    validate_fitted,
    validate_instance,
)

_LAZY = {
    "CustomMMM": (".base", "CustomMMM"),
    "load_garden_class_from_path": (".loader", "load_garden_class_from_path"),
    "class_qualname": (".loader", "class_qualname"),
    "clear_cache": (".loader", "clear_cache"),
    "run_compatibility_check": (".compat", "run_compatibility_check"),
    "BLOCKING_TIERS": (".compat", "BLOCKING_TIERS"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(target[0], __name__)
    return getattr(mod, target[1])


__all__ = [
    "GARDEN_CONTRACT_VERSION",
    "REQUIRED_ATTRS",
    "REQUIRED_METHODS",
    "RECOMMENDED_METHODS",
    "PARAM_PREFIXES",
    "validate_class",
    "validate_instance",
    "validate_fitted",
    "find_garden_class",
    "is_bayesian_mmm_subclass",
    "describe_contract",
    # lazy
    "CustomMMM",
    "load_garden_class_from_path",
    "class_qualname",
    "clear_cache",
    "run_compatibility_check",
    "BLOCKING_TIERS",
]
