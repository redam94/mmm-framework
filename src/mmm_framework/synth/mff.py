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


def make_awareness_survey(
    *,
    n_weeks: int = 104,
    n_trials: int = 500,
    retention: float = 0.8,
    intercept: float = -0.3,
    amplitude: float = 1.2,
    seed: int | None = 7,
) -> tuple[pd.DataFrame, dict]:
    """A synthetic **brand-awareness survey** dataset for the awareness model.

    The KPI is a weekly **aware-count** ``n_aware ~ Binomial(n_trials, p_t)`` from
    a survey of ``n_trials`` people, where the aware-rate ``p_t = sigmoid(intercept
    + goodwill_t)`` and ``goodwill_t`` is a **media goodwill stock** that decays at
    ``retention`` ρ each week (Σ_c Σ_{τ≤t} ρ^(t-τ)·βc·sat(spend_τ,c)) — i.e. the
    exact generative structure the ``AwarenessStructuralMMM`` recovers. Two media
    channels (TV, Search). Returned as an MFF long-format dataframe (KPI
    ``Awareness`` = the count) + an answer key carrying the true ρ / half-life /
    ``n_trials`` / per-channel goodwill coefficients.

    Fit it with ``likelihood={"family": "binomial"}`` and
    ``model_params={"number_of_trials": n_trials}`` — see the awareness model's
    Atelier demo notebook.
    """
    rng = np.random.default_rng(seed)
    if not 0.0 < retention < 1.0:
        raise ValueError(f"retention must be in (0, 1), got {retention!r}")

    periods = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON").strftime(
        "%Y-%m-%d"
    )
    channels = {
        "TV": (rng.normal(110, 28, n_weeks), 1.4),
        "Search": (rng.normal(70, 18, n_weeks), 0.9),
    }

    t = np.arange(n_weeks)
    lag = t[:, None] - t[None, :]
    decay = np.where(lag >= 0, retention ** np.maximum(lag, 0), 0.0)  # un-normalized

    goodwill = np.zeros(n_weeks)
    spend_cols: dict[str, np.ndarray] = {}
    betas: dict[str, float] = {}
    for name, (raw, beta) in channels.items():
        spend = np.clip(raw, 0.0, None)
        spend_cols[name] = spend
        x_norm = spend / (spend.max() + 1e-9)
        sat = 1.0 - np.exp(-3.0 * x_norm)  # diminishing returns (1-exp saturation)
        goodwill = goodwill + decay @ (beta * sat)  # geometric goodwill stock at ρ
        betas[name] = beta

    # Center + scale the goodwill so the aware-rate VARIES in a realistic band
    # (no saturation at 0/100%); the temporal decay structure — hence ρ — is
    # preserved (amplitude scaling only rescales the effective coefficients).
    gw = (goodwill - goodwill.mean()) / (goodwill.std() + 1e-9)
    aware_logit = intercept + amplitude * gw
    aware_rate = np.clip(1.0 / (1.0 + np.exp(-aware_logit)), 0.03, 0.97)
    n_aware = rng.binomial(int(n_trials), aware_rate).astype(float)

    variables = {"Awareness": n_aware, **spend_cols}
    df = _blocks(np.asarray(periods), variables)
    half_life = float(np.log(0.5) / np.log(retention))
    answer = {
        "scenario": "awareness_survey",
        "kpi": "Awareness",
        "kpi_kind": "binomial_count",
        "n_trials": int(n_trials),
        "channels": list(channels),
        "true_retention": float(retention),
        "true_half_life_weeks": half_life,
        "true_intercept_logit": float(intercept),
        "true_goodwill_betas": betas,
        "mean_aware_rate": float(aware_rate.mean()),
        "notes": (
            "Awareness as a survey aware-count: n_aware ~ Binomial(n_trials, "
            "sigmoid(intercept + media goodwill stock)); goodwill decays at the "
            "retention ρ. Fit with likelihood=binomial + model_params.number_of_trials."
        ),
    }
    return df, answer


__all__ = [
    "MFF_COLUMNS",
    "MIN_WEEKS",
    "scenario_to_mff",
    "geo_scenario_to_mff",
    "truth_summary",
    "generate_mff",
    "make_awareness_survey",
]


def brand_funnel_mff(
    seed: int = 21, *, n_weeks: int = 156
) -> tuple[pd.DataFrame, dict]:
    """MFF long table for the :func:`~mmm_framework.synth.dgp.make_brand_funnel`
    world, INCLUDING its mediator surveys as extra variables — the format the
    StructuralNestedMMM fit path consumes.

    Beyond the standard Sales/media/Price blocks, the table carries:

    - ``awareness_count`` / ``awareness_trials`` — the weekly binary tracker
      (rows only for observed weeks; a missing week = no survey, which the
      structural builder loads as NaN = unobserved)
    - ``consideration_cat_1`` .. ``_5`` — the weekly Likert category counts
      (low → high), again with unobserved weeks omitted

    Returns ``(mff_df, answer_key)``; the key carries the structural truth
    (rho, betas, mediated shares) plus the survey variable names, so a
    recovery harness can assemble the DAG spec without magic strings.
    """
    sc = dgp.make_brand_funnel(seed=seed, n_weeks=n_weeks)
    mff = scenario_to_mff(sc)
    period = np.asarray(sc.weeks.strftime("%Y-%m-%d"))

    frames = [mff]

    def _sparse_block(name: str, values: np.ndarray) -> pd.DataFrame:
        obs = np.isfinite(values)
        return pd.DataFrame(
            {
                "Period": period[obs],
                "Geography": None,
                "Product": None,
                "Campaign": None,
                "Outlet": None,
                "Creative": None,
                "VariableName": name,
                "VariableValue": np.asarray(values, dtype=float)[obs],
            }
        )

    frames.append(_sparse_block("awareness_count", sc.notes["awareness_counts"]))
    frames.append(_sparse_block("awareness_trials", sc.notes["awareness_trials"]))
    cons = np.asarray(sc.notes["consideration_counts"], dtype=float)
    category_variables = [f"consideration_cat_{k + 1}" for k in range(cons.shape[1])]
    for k, cname in enumerate(category_variables):
        frames.append(_sparse_block(cname, cons[:, k]))

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["Period", "VariableName"], kind="stable").reset_index(
        drop=True
    )

    key = truth_summary(sc)
    key["mediator_variables"] = {
        "awareness": {
            "counts": "awareness_count",
            "trials": "awareness_trials",
        },
        "consideration": {"category_variables": category_variables},
    }
    return out, key
