"""Convert synthetic DGP scenarios to Master Flat File (MFF) datasets.

The scenario factories in :mod:`mmm_framework.synth.dgp` (national worlds) and
:mod:`mmm_framework.synth.dgp_geo` (geo / geo x product panels) return wide,
model-ready frames plus causal ground truth. This module flattens a scenario
into the 8-column long MFF layout the data loader and the app ingest
(``Period, Geography, Product, Campaign, Outlet, Creative, VariableName,
VariableValue``) and emits a JSON-safe "answer key" so a fitted model can be
graded against the world's known causal truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import dgp, dgp_geo

MFF_COLUMNS = [
    "Period",
    "Geography",
    "Product",
    "Campaign",
    "Outlet",
    "Creative",
    "VariableName",
    "VariableValue",
]

#: Minimum series length the violation worlds support (seasonality cycles,
#: mid-series breaks, and seeded error placement all assume at least a year).
MIN_WEEKS = 52


def _blocks(
    period: np.ndarray,
    variables: dict[str, np.ndarray],
    geography: np.ndarray | None = None,
    product: np.ndarray | None = None,
) -> pd.DataFrame:
    """One MFF block per variable, stacked then ordered period-major."""
    frames = []
    for name, values in variables.items():
        frames.append(
            pd.DataFrame(
                {
                    "Period": period,
                    "Geography": geography if geography is not None else None,
                    "Product": product if product is not None else None,
                    "Campaign": None,
                    "Outlet": None,
                    "Creative": None,
                    "VariableName": name,
                    "VariableValue": np.asarray(values, dtype=float),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    # Period-major, variables in insertion order (KPI, media, controls).
    return out.sort_values("Period", kind="stable", ignore_index=True)[MFF_COLUMNS]


def scenario_to_mff(sc: dgp.Scenario) -> pd.DataFrame:
    """Flatten a national :class:`~mmm_framework.synth.dgp.Scenario` to MFF."""
    period = np.asarray(sc.weeks.strftime("%Y-%m-%d"))
    variables: dict[str, np.ndarray] = {"Sales": sc.y.to_numpy(float)}
    for c in sc.spend.columns:
        variables[c] = sc.spend[c].to_numpy(float)
    for c in sc.controls.columns:
        variables[c] = sc.controls[c].to_numpy(float)
    return _blocks(period, variables)


def geo_scenario_to_mff(sc: dgp_geo.GeoScenario) -> pd.DataFrame:
    """Flatten a panel :class:`~mmm_framework.synth.dgp_geo.GeoScenario` to MFF."""
    idx = sc.spend.index
    period = np.asarray(
        pd.DatetimeIndex(idx.get_level_values("Period")).strftime("%Y-%m-%d")
    )
    geography = np.asarray(idx.get_level_values("Geography"))
    product = np.asarray(idx.get_level_values("Product")) if sc.products else None
    variables: dict[str, np.ndarray] = {"Sales": sc.y.to_numpy(float)}
    for c in sc.spend.columns:
        variables[c] = sc.spend[c].to_numpy(float)
    for c in sc.controls.columns:
        variables[c] = sc.controls[c].to_numpy(float)
    return _blocks(period, variables, geography=geography, product=product)


class _Dropped:
    """Sentinel for non-serializable values pruned from the answer key."""


def _json_safe(value: Any) -> Any:
    """Recursively keep JSON-serializable content; drop arrays/Series."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return _Dropped()


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if not isinstance(v, _Dropped)}
    if isinstance(value, list):
        return [_prune(v) for v in value if not isinstance(v, _Dropped)]
    return value


def truth_summary(sc: dgp.Scenario | dgp_geo.GeoScenario) -> dict:
    """JSON-safe answer key: scenario metadata + causal ground truth."""
    out: dict[str, Any] = {
        "scenario": sc.name,
        "description": sc.description,
        "violates": sc.violates,
        "representable": sc.representable,
        "channels": list(sc.channels),
        "true_contribution": {c: float(v) for c, v in sc.true_contribution.items()},
        "true_roas": {c: float(v) for c, v in sc.true_roas.items()},
    }
    if isinstance(sc, dgp_geo.GeoScenario):
        out["geographies"] = list(sc.geos)
        out["products"] = list(sc.products) if sc.products else None
        out["true_contribution_by_geo"] = {
            cell: {c: float(v) for c, v in row.items()}
            for cell, row in sc.true_contribution_by_geo.iterrows()
        }
        out["true_roas_by_geo"] = {
            cell: {c: float(v) for c, v in row.items()}
            for cell, row in sc.true_roas_by_geo.iterrows()
        }
    if getattr(sc, "control_roles", None):
        out["control_roles"] = {
            k: getattr(v, "value", str(v)) for k, v in sc.control_roles.items()
        }
    out["notes"] = _prune(_json_safe(sc.notes))
    return out


def generate_mff(
    scenario: str = "realistic",
    *,
    seed: int | None = None,
    n_weeks: int | None = None,
    geographies: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate a long-format MFF dataset from a named synthetic world.

    Parameters
    ----------
    scenario : str
        A national scenario from ``dgp.SCENARIOS`` (e.g. ``"realistic"``,
        ``"clean"``, ``"unobserved_confounding"``) or a panel scenario from
        ``dgp_geo.SCENARIOS`` (``"geo_clean"``, ``"geo_heterogeneous"``,
        ``"geo_product"``). When ``geographies`` is given with a national
        name, the world is upgraded to a panel: ``"clean"`` maps to
        ``"geo_clean"`` and everything else to ``"geo_heterogeneous"``.
    seed : int, optional
        Random seed (each factory's default when omitted).
    n_weeks : int, optional
        Series length in weeks (scenario default when omitted; minimum 52).
    geographies : list of str, optional
        Geography names for a panel world. Custom names get seeded baseline
        offsets, budget shares, and (heterogeneous world) effectiveness
        multipliers.

    Returns
    -------
    (DataFrame, dict)
        The MFF-format data and the JSON-safe ground-truth answer key.
    """
    if n_weeks is not None and int(n_weeks) < MIN_WEEKS:
        raise ValueError(
            f"n_weeks must be at least {MIN_WEEKS} (got {n_weeks}): the "
            "synthetic worlds need a full seasonal cycle to be identifiable."
        )
    if scenario in dgp_geo.SCENARIOS or geographies:
        if scenario in dgp_geo.SCENARIOS:
            name = scenario
        elif scenario == "clean":
            name = "geo_clean"
        else:
            name = "geo_heterogeneous"
        sc = dgp_geo.build(name, seed, geos=geographies, n_weeks=n_weeks)
        return geo_scenario_to_mff(sc), truth_summary(sc)
    if scenario not in dgp.SCENARIOS:
        raise KeyError(
            f"Unknown scenario {scenario!r}. National: "
            f"{sorted(dgp.SCENARIOS)}; panel: {sorted(dgp_geo.SCENARIOS)}."
        )
    sc = dgp.build(scenario, seed, n_weeks=n_weeks)
    return scenario_to_mff(sc), truth_summary(sc)


__all__ = [
    "MFF_COLUMNS",
    "MIN_WEEKS",
    "scenario_to_mff",
    "geo_scenario_to_mff",
    "truth_summary",
    "generate_mff",
]
