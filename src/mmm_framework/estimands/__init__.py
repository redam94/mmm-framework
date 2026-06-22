"""First-class, declarative estimands — the counterfactual causal lens.

A model declares named, serializable :class:`~mmm_framework.estimands.spec.Estimand`
objects; the framework realizes them from the posterior as mean + HDI, either
post-hoc (:class:`~mmm_framework.estimands.evaluate.EstimandEvaluator`) or in the
PyMC graph (:func:`~mmm_framework.estimands.graph.build_estimand_expr`). This is
the single registry that subsumes the framework's previously-scattered estimand
logic (decomposition ROI, counterfactual ROI, marginal ROAS, calibration
estimands) while keeping every existing number bit-stable.

``spec`` is dependency-light (pure Pydantic); the realization engines and the
registry import numpy/pytensor lazily, so importing this package is cheap.
"""

from __future__ import annotations

from .spec import (
    ALL_CHANNELS,
    ESTIMAND_SCHEMA_VERSION,
    Constant,
    Contrast,
    Contribution,
    CustomIntervention,
    Estimand,
    EstimandResult,
    Intervention,
    LatentVar,
    MarginalSpend,
    Observed,
    ObservedInput,
    Outcome,
    Quantity,
    Realization,
    ScaleInput,
    SetInput,
    Summary,
    SupportsEstimands,
    TimeWindow,
    ZeroInput,
)

__all__ = [
    "ALL_CHANNELS",
    "ESTIMAND_SCHEMA_VERSION",
    "Constant",
    "Contrast",
    "Contribution",
    "CustomIntervention",
    "Estimand",
    "EstimandResult",
    "Intervention",
    "LatentVar",
    "MarginalSpend",
    "Observed",
    "ObservedInput",
    "Outcome",
    "Quantity",
    "Realization",
    "ScaleInput",
    "SetInput",
    "Summary",
    "SupportsEstimands",
    "TimeWindow",
    "ZeroInput",
    # Lazy attributes (see __getattr__): EstimandEvaluator, registry, capabilities,
    # model_capabilities, build_estimand_expr.
]


def __getattr__(name: str):
    """Lazily expose the heavier members (numpy/pytensor) on first access.

    Uses ``import_module`` (not ``from . import``) so binding the submodule name
    does not re-enter this hook (``_handle_fromlist`` → ``__getattr__``).
    """
    import importlib

    if name in ("registry", "capabilities", "evaluate", "graph"):
        return importlib.import_module(f"{__name__}.{name}")
    if name == "EstimandEvaluator":
        return importlib.import_module(f"{__name__}.evaluate").EstimandEvaluator
    if name == "model_capabilities":
        return importlib.import_module(f"{__name__}.capabilities").model_capabilities
    if name == "build_estimand_expr":
        return importlib.import_module(f"{__name__}.graph").build_estimand_expr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
