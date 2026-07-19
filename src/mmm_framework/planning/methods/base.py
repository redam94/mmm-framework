"""Experiment-method registry — the unifying descriptor for every named
experiment methodology (synthetic control, TBR, GBR, DiD-MMT, and — Phase 2 —
ghost ads and switchback).

Each method declares four capabilities so the DesignStudio, the agent, and the
measurement loop treat them uniformly:

1. **design**   — build a runnable design (assignment / schedule) from the data,
2. **estimate** — an ``EstimatorResult`` from a panel + assignment + window
   (geo/national families; user-level families supply their own analysis),
3. **power**    — a method-specific analytic MDE model (the A/A·A/B harness in
   :mod:`planning.simulation` layers the calibrated + empirical MDE on top),
4. **data requirement** — what the method needs (≥N geos / a national series /
   user-level counts), so ``design_options`` can gate cleanly.

This mirrors the existing ``estimands`` and ``garden`` registries: built-ins
register themselves at import (see :mod:`planning.methods`), and callers
enumerate via :func:`list_methods` / :func:`methods_for_data`.

Pure Python (no numpy/pandas/PyMC at import) so importing the registry is cheap
and side-effect-free; the estimator modules carry their own heavier imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DataRequirement:
    """What data a method needs to be offered. ``design_options`` filters on this."""

    family: str  # 'geo' | 'national' | 'user' | 'switchback'
    min_geos: int = 0  # geo methods: minimum distinct geos (treatment + donor pool)
    needs_panel: bool = False  # requires a geo x week KPI panel (kpi_wide)
    needs_pre_period: bool = True  # requires a clean pre-window to fit a counterfactual
    min_pre_weeks: int = 8
    notes: str = ""

    def supported(
        self,
        *,
        n_geos: int = 0,
        n_weeks: int = 0,
        has_user_counts: bool = False,
    ) -> tuple[bool, str]:
        """Return ``(ok, reason)``. ``reason`` explains a gate failure for the UI."""
        if self.family == "user":
            if not has_user_counts:
                return False, "needs user/impression-level counts"
            return True, ""
        if self.family in ("geo",) and n_geos < max(self.min_geos, 1):
            return False, f"needs ≥ {max(self.min_geos, 1)} geos (have {n_geos})"
        if self.needs_panel and n_geos < 2:
            return False, "needs a geo panel (≥ 2 geos)"
        if self.needs_pre_period and n_weeks < self.min_pre_weeks:
            return (
                False,
                f"needs ≥ {self.min_pre_weeks} pre-period weeks (have {n_weeks})",
            )
        return True, ""


@dataclass(frozen=True)
class MethodSpec:
    """A named experiment methodology.

    ``estimator_fn`` follows the :mod:`planning.simulation` contract
    ``fn(panel, assignment, window, **kw) -> EstimatorResult`` for the geo /
    national families; ``user`` / ``switchback`` families may leave it ``None``
    and supply their own analysis (e.g. ghost ads' two-proportion calculator).
    """

    key: str
    name: str
    requirement: DataRequirement
    design_fn: Callable[..., dict] | None = None
    estimator_fn: Callable[..., Any] | None = None
    power_fn: Callable[..., dict] | None = None
    references: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary for the endpoint / agent (loose-JSON safe)."""
        return {
            "key": self.key,
            "name": self.name,
            "family": self.requirement.family,
            "min_geos": self.requirement.min_geos,
            "needs_panel": self.requirement.needs_panel,
            "min_pre_weeks": self.requirement.min_pre_weeks,
            "references": list(self.references),
            "description": self.description,
        }


_METHODS: dict[str, MethodSpec] = {}


def register(spec: MethodSpec) -> MethodSpec:
    """Register (or replace) a method. Returns the spec for decorator-style use."""
    _METHODS[spec.key] = spec
    return spec


def get_method(key: str) -> MethodSpec:
    try:
        return _METHODS[key]
    except KeyError:
        raise KeyError(
            f"unknown experiment method {key!r}; known: {sorted(_METHODS)}"
        ) from None


def has_method(key: str) -> bool:
    return key in _METHODS


def list_methods(*, family: str | None = None) -> list[MethodSpec]:
    out = [
        m for m in _METHODS.values() if family is None or m.requirement.family == family
    ]
    return sorted(out, key=lambda m: (m.requirement.family, m.key))


def methods_for_data(
    *,
    n_geos: int = 0,
    n_weeks: int = 0,
    has_user_counts: bool = False,
    family: str | None = None,
) -> list[dict[str, Any]]:
    """Enumerate methods with a ``supported`` flag + gate reason for the given
    data shape — the payload ``design_options`` returns to the UI."""
    out: list[dict[str, Any]] = []
    for m in list_methods(family=family):
        ok, reason = m.requirement.supported(
            n_geos=n_geos, n_weeks=n_weeks, has_user_counts=has_user_counts
        )
        row = m.to_dict()
        row["supported"] = ok
        row["reason"] = reason
        out.append(row)
    return out


__all__ = [
    "DataRequirement",
    "MethodSpec",
    "register",
    "get_method",
    "has_method",
    "list_methods",
    "methods_for_data",
]
