"""Pre-fit model-design facts for the Model Design Readout.

Everything a *pre-registration* document needs to know about a configured (but
not yet fitted) :class:`~mmm_framework.model.base.BayesianMMM`:

- ``enumerate_model_priors`` — every free random variable in the graph as a
  table row (group, family, implied prior mean/sd/interval from prior samples).
- ``model_assumptions`` — the structural choices (likelihood, carryover,
  saturation, trend, seasonality, pooling, controls, fit plan) as rows of
  *choices, not findings*.
- ``prior_predictive_facts`` — the prior-predictive distribution of the KPI
  aggregated to the period axis (fan-chart bands, coverage of the observed
  series, negativity share, replicate mean/sd distributions).
- ``prior_response_curves`` — the saturation and carryover shapes the priors
  imply per channel (median + credible band), so the response-curve assumptions
  are visible before any data has spoken.

All functions are read-only on the model and JSON-safe in their outputs; prior
samples are drawn once (``sample_prior_predictive``) and shared between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

__all__ = [
    "PriorRow",
    "AssumptionRow",
    "enumerate_model_priors",
    "model_assumptions",
    "prior_predictive_facts",
    "prior_component_facts",
    "prior_estimand_facts",
    "prior_response_curves",
    "sample_prior",
]


# ─────────────────────────────────────────────────────────────────────────────
# Row dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PriorRow:
    """One free random variable of the model graph, as a priors-table row."""

    group: str
    name: str
    family: str
    dims: str  # "" for scalars, e.g. "5 geos" for vectors
    mean: float | None = None
    sd: float | None = None
    lower: float | None = None
    upper: float | None = None
    calibrated: bool = False  # experiment-calibrated prior (roi_prior set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "name": self.name,
            "family": self.family,
            "dims": self.dims,
            "mean": self.mean,
            "sd": self.sd,
            "lower": self.lower,
            "upper": self.upper,
            "calibrated": self.calibrated,
        }


@dataclass
class AssumptionRow:
    """One structural modeling choice, stated as an assumption."""

    topic: str
    setting: str
    detail: str
    channels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "setting": self.setting,
            "detail": self.detail,
            "channels": list(self.channels),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Prior sampling (shared)
# ─────────────────────────────────────────────────────────────────────────────
def sample_prior(model: Any, n_samples: int = 500, random_seed: int = 42) -> Any:
    """Draw prior (+ prior predictive) samples from the model graph.

    Returns an arviz ``InferenceData`` with ``.prior`` and ``.prior_predictive``
    groups. Uses the model's own ``sample_prior_predictive`` when present so the
    version-drift shims in :mod:`mmm_framework.utils.arviz_compat` apply.
    """
    if hasattr(model, "sample_prior_predictive"):
        try:
            return model.sample_prior_predictive(n_samples, random_seed)
        except TypeError:
            return model.sample_prior_predictive(samples=n_samples)
    from ...utils import arviz_compat

    pymc_model = getattr(model, "model", None)
    with pymc_model:
        return arviz_compat.sample_prior_predictive(n_samples, random_seed)


def _prior_group(idata: Any) -> Any | None:
    return getattr(idata, "prior", None) if idata is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Prior enumeration
# ─────────────────────────────────────────────────────────────────────────────
def _rv_family(pymc_model: Any, name: str) -> str:
    """Best-effort human name of an RV's distribution family."""
    try:
        rv = pymc_model[name]
        op = rv.owner.op
        pn = getattr(op, "_print_name", None)
        if pn and pn[0]:
            return str(pn[0])
        raw = str(getattr(op, "name", "") or "")
        if raw:
            return raw.replace("_rv", "").replace("_", " ").title()
    except Exception:  # noqa: BLE001
        pass
    return "—"


def _classify_param(name: str, channels: list[str], controls: list[str]) -> str:
    """Map a free-RV name to a priors-table group via the graph's conventions."""
    n = name.lower()
    if n.startswith("adstock_"):
        return "Carryover (adstock)"
    if n.startswith("sat_"):
        return "Saturation"
    if n.startswith("roi_"):
        # media_prior_mode="roi": the free RV is the channel's prior ROI itself
        return "Media effects"
    if n.startswith("beta_") or n.startswith("logmu_") or n.endswith("_z"):
        for ch in channels:
            cl = ch.lower()
            if cl and cl in n:
                return "Media effects"
        if "control" in n or any(c.lower() in n for c in controls):
            return "Controls"
        return "Media effects"
    if "control" in n or "gamma" in n or "horseshoe" in n or "lambda_local" in n:
        return "Controls"
    if "season" in n or "fourier" in n:
        return "Seasonality"
    if n.startswith(("trend", "spline", "gp_", "changepoint")) or n in {"k", "m"}:
        return "Trend"
    if n.startswith(("geo_", "product_")):
        return "Pooling (geo/product)"
    if n in {"sigma", "nu", "phi"} or n.startswith(("sigma", "noise")):
        return "Observation noise"
    if "intercept" in n or n == "alpha":
        return "Baseline"
    return "Other"


# Display order of the groups in the priors table.
PRIOR_GROUP_ORDER = [
    "Media effects",
    "Carryover (adstock)",
    "Saturation",
    "Baseline",
    "Trend",
    "Seasonality",
    "Controls",
    "Pooling (geo/product)",
    "Observation noise",
    "Other",
]


def enumerate_model_priors(
    model: Any,
    prior: Any = None,
    *,
    interval: float = 0.90,
) -> list[PriorRow]:
    """Every free RV in the model graph as a priors-table row.

    ``prior`` is the ``.prior`` group of a prior-predictive ``InferenceData``
    (or the InferenceData itself); when given, each row carries the empirical
    prior mean/sd and the central ``interval`` range so a reader sees what the
    prior *implies*, not just its name.
    """
    pymc_model = getattr(model, "model", None)
    if pymc_model is None:
        return []

    prior_ds = getattr(prior, "prior", prior)  # accept idata or Dataset
    channels = [str(c) for c in getattr(model, "channel_names", [])]
    controls = [str(c) for c in getattr(model, "control_names", [])]

    mff = getattr(model, "mff_config", None)
    calibrated_channels: set[str] = set()
    if mff is not None:
        for ch in channels:
            try:
                cfg = mff.get_media_config(ch)
                if cfg is not None and getattr(cfg, "roi_prior", None) is not None:
                    calibrated_channels.add(ch)
            except Exception:  # noqa: BLE001
                continue

    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100
    rows: list[PriorRow] = []
    try:
        free_names = [rv.name for rv in pymc_model.free_RVs]
    except Exception:  # noqa: BLE001
        free_names = []

    for name in free_names:
        group = _classify_param(name, channels, controls)
        family = _rv_family(pymc_model, name)

        dims = ""
        mean = sd = lower = upper = None
        if prior_ds is not None and name in getattr(prior_ds, "data_vars", {}):
            vals = np.asarray(prior_ds[name].values, dtype=float)
            # (chain, draw, *shape) -> per-draw flattened
            extra = int(np.prod(vals.shape[2:])) if vals.ndim > 2 else 1
            if extra > 1:
                dims = f"{extra} elements"
            flat = vals.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size:
                mean = float(np.mean(flat))
                sd = float(np.std(flat))
                lower = float(np.percentile(flat, lo_q))
                upper = float(np.percentile(flat, hi_q))

        calibrated = group == "Media effects" and any(
            ch in calibrated_channels and ch.lower() in name.lower() for ch in channels
        )
        rows.append(
            PriorRow(
                group=group,
                name=name,
                family=family,
                dims=dims,
                mean=mean,
                sd=sd,
                lower=lower,
                upper=upper,
                calibrated=calibrated,
            )
        )

    order = {g: i for i, g in enumerate(PRIOR_GROUP_ORDER)}
    rows.sort(key=lambda r: (order.get(r.group, 99), r.name))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Assumptions
# ─────────────────────────────────────────────────────────────────────────────
def _group_channels_by(mff: Any, channels: list[str], keyfn) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ch in channels:
        try:
            cfg = mff.get_media_config(ch)
            key = keyfn(cfg) if cfg is not None else "unknown"
        except Exception:  # noqa: BLE001
            key = "unknown"
        out.setdefault(str(key), []).append(ch)
    return out


def model_assumptions(model: Any) -> list[AssumptionRow]:
    """The model's structural choices as explicit, reviewable assumptions."""
    rows: list[AssumptionRow] = []
    mc = getattr(model, "model_config", None)
    mff = getattr(model, "mff_config", None)
    trend = getattr(model, "trend_config", None)
    channels = [str(c) for c in getattr(model, "channel_names", [])]

    # Observation model
    lik = getattr(mc, "likelihood", None)
    fam = getattr(getattr(lik, "family", None), "value", None) or "normal"
    link = getattr(getattr(lik, "link", None), "value", None) or "identity"
    rows.append(
        AssumptionRow(
            topic="Observation model",
            setting=f"{fam} likelihood, {link} link",
            detail=(
                "The KPI is modeled as this distribution around the structural "
                "mean; residual scale is estimated jointly with the effects."
            ),
        )
    )

    # Carryover / adstock
    if mff is not None and channels:
        by = _group_channels_by(
            mff,
            channels,
            lambda c: f"{getattr(getattr(c, 'adstock', None), 'type', None) and c.adstock.type.value} "
            f"(l_max={getattr(getattr(c, 'adstock', None), 'l_max', '?')})",
        )
        for setting, chs in by.items():
            rows.append(
                AssumptionRow(
                    topic="Carryover (adstock)",
                    setting=setting,
                    detail=(
                        "Media effect persists beyond the exposure period with "
                        "this kernel; the decay rate is estimated per channel "
                        "from its prior."
                    ),
                    channels=chs,
                )
            )
        by = _group_channels_by(
            mff,
            channels,
            lambda c: getattr(getattr(c, "saturation", None), "type", None)
            and c.saturation.type.value,
        )
        for setting, chs in by.items():
            rows.append(
                AssumptionRow(
                    topic="Saturation",
                    setting=str(setting),
                    detail=(
                        "Response to media is concave: each additional unit of "
                        "adstocked exposure buys less. The curve family is a "
                        "choice; its parameters are estimated."
                    ),
                    channels=chs,
                )
            )

    rows.append(
        AssumptionRow(
            topic="Media effect sign",
            setting="non-negative",
            detail=(
                "Channel coefficients use positive-support priors — media can "
                "fail to pay back, but cannot causally *reduce* the KPI in this "
                "specification."
            ),
        )
    )

    # Trend
    if trend is not None:
        t = getattr(getattr(trend, "type", None), "value", None) or str(
            getattr(trend, "type", "none")
        )
        rows.append(
            AssumptionRow(
                topic="Baseline trend",
                setting=str(t),
                detail=(
                    "Long-run movement of the KPI not attributable to media or "
                    "controls is absorbed by this trend form."
                ),
            )
        )

    # Seasonality
    seas = getattr(mc, "seasonality", None)
    if seas is not None:
        yearly = getattr(seas, "yearly", None)
        setting = f"yearly Fourier order {yearly}" if yearly else "none"
        rows.append(
            AssumptionRow(
                topic="Seasonality",
                setting=setting,
                detail=(
                    "Recurring within-year cycles are modeled with smooth "
                    "Fourier terms rather than month dummies."
                ),
            )
        )

    # Pooling / hierarchy
    n_geos = len(getattr(model, "geo_names", []) or [])
    if n_geos > 1:
        rows.append(
            AssumptionRow(
                topic="Pooling",
                setting=f"partial pooling across {n_geos} geographies",
                detail=(
                    "Geography-level deviations share a common prior — regions "
                    "inform each other rather than being fit independently."
                ),
            )
        )

    # Controls
    controls = [str(c) for c in getattr(model, "control_names", [])]
    if controls:
        sel = getattr(getattr(mc, "control_selection", None), "method", None)
        sel_s = getattr(sel, "value", None) or str(sel or "none")
        rows.append(
            AssumptionRow(
                topic="Controls",
                setting=f"{len(controls)} control(s), selection: {sel_s}",
                detail=(
                    "Controls absorb demand drivers that would otherwise be "
                    "credited to media. Confounder-role controls receive wide, "
                    "un-shrunk priors. Controls: " + ", ".join(controls) + "."
                ),
            )
        )

    # Fit plan — reflect the PLANNED inference method (NUTS vs an approximate
    # method), so the readout does not promise calibrated uncertainty / post-fit
    # diagnostics for a MAP/ADVI/Pathfinder fit. Emitted even when there is no
    # ModelConfig (extended models) so the plan is never silently omitted.
    _fm = getattr(mc, "fit_method", None) if mc is not None else None
    method = str(getattr(_fm, "value", _fm) or "nuts").lower()
    if method == "nuts":
        if mc is not None:
            setting = (
                f"NUTS — {getattr(mc, 'n_chains', '?')} chains × "
                f"{getattr(mc, 'n_draws', '?')} draws "
                f"(tune {getattr(mc, 'n_tune', '?')}, "
                f"target_accept {getattr(mc, 'target_accept', '?')})"
            )
        else:
            setting = "NUTS — full MCMC"
        detail = (
            "The sampling plan the final fit will use. Convergence "
            "(R-hat, ESS, divergences) will be reported post-fit."
        )
    else:
        setting = f"{method.upper()} — approximate (fast check)"
        detail = (
            "An APPROXIMATE fit (not full MCMC): fast to run for a plausibility "
            "check, but its uncertainty is NOT calibrated — R-hat/ESS are not "
            "assessable. Re-fit with NUTS before trusting intervals or decisions."
        )
    rows.append(AssumptionRow(topic="Inference plan", setting=setting, detail=detail))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Prior predictive facts
# ─────────────────────────────────────────────────────────────────────────────
def prior_predictive_facts(
    model: Any,
    idata: Any = None,
    *,
    n_samples: int = 500,
    random_seed: int = 42,
    n_traces: int = 12,
) -> dict[str, Any]:
    """Prior-predictive KPI facts on the period axis (original scale).

    Returns dates, the observed KPI (summed across geo/product per period),
    prior-predictive quantile bands, 90%-band coverage of the observed series,
    the share of negative draws, per-replicate mean/sd distributions, and
    ``n_traces`` individual simulated series (spaghetti) so the reader sees
    what a *single* dataset from the priors actually looks like — not just the
    envelope.
    """
    if idata is None:
        idata = sample_prior(model, n_samples, random_seed)

    pp = getattr(idata, "prior_predictive", None)
    if pp is None or not list(getattr(pp, "data_vars", [])):
        raise ValueError("Prior predictive group is empty — cannot build facts.")
    var = "y_obs" if "y_obs" in pp.data_vars else list(pp.data_vars)[0]
    draws = np.asarray(pp[var].values, dtype=float)
    n_obs_axis = draws.shape[-1]
    draws = draws.reshape(-1, n_obs_axis)  # (S, n_obs)

    # Back to the original KPI scale (Gaussian families standardize y).
    y_mean = float(getattr(model, "y_mean", 0.0) or 0.0)
    y_std = float(getattr(model, "y_std", 1.0) or 1.0)
    if y_std != 1.0 or y_mean != 0.0:
        draws = draws * y_std + y_mean

    observed_obs = np.asarray(
        getattr(model, "y_raw", getattr(model, "y", np.zeros(n_obs_axis))),
        dtype=float,
    )
    time_idx = np.asarray(getattr(model, "time_idx", np.arange(n_obs_axis)), dtype=int)
    n_periods = int(time_idx.max()) + 1 if time_idx.size else 0

    # Aggregate observation rows to the period axis (sum across geo/product).
    observed = np.bincount(time_idx, weights=observed_obs, minlength=n_periods)
    S = draws.shape[0]
    period_draws = np.zeros((S, n_periods), dtype=float)
    np.add.at(period_draws, (np.arange(S)[:, None], time_idx[None, :]), draws)

    qs = [5, 25, 50, 75, 95]
    bands = {f"p{q:02d}": np.percentile(period_draws, q, axis=0) for q in qs}
    inside = (observed >= bands["p05"]) & (observed <= bands["p95"])
    coverage_90 = float(np.mean(inside)) if n_periods else float("nan")

    dates: list[str] = []
    try:
        periods = model.panel.coords.periods
        dates = [str(p)[:10] for p in periods]
        if len(dates) != n_periods:
            dates = []
    except Exception:  # noqa: BLE001
        dates = []
    if not dates:
        dates = [str(i) for i in range(n_periods)]

    kpi_label = "KPI"
    try:
        kpi_label = str(model.mff_config.kpi.name)
    except Exception:  # noqa: BLE001
        pass

    # A handful of individual simulated series (evenly thinned across draws) —
    # single realizations reveal wiggliness/scale pathologies the bands hide.
    k = max(0, min(int(n_traces), S))
    traces = period_draws[np.linspace(0, S - 1, k).astype(int)] if k else None

    # How far off-scale is the prior, beyond in/out of band: the observed
    # series' z-position within the prior predictive per period.
    p50 = bands["p50"]
    spread = (bands["p95"] - bands["p05"]) / 3.29  # ≈ prior predictive sd
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(spread > 0, (observed - p50) / spread, np.nan)
    z = z[np.isfinite(z)]
    scale_z_abs_mean = float(np.mean(np.abs(z))) if z.size else float("nan")

    return {
        "kpi_label": kpi_label,
        "dates": dates,
        "observed": observed,
        "bands": bands,
        "traces": traces,
        "coverage_90": coverage_90,
        "scale_z_abs_mean": scale_z_abs_mean,
        "frac_negative": float(np.mean(draws < 0)),
        "rep_means": draws.mean(axis=1),
        "rep_sds": draws.std(axis=1),
        "obs_mean": float(observed_obs.mean()),
        "obs_sd": float(observed_obs.std()),
        "n_draws": int(S),
        "var_name": var,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prior structural components over time (original scale)
# ─────────────────────────────────────────────────────────────────────────────
# The graph's registered component deterministics (name → display label). Each
# is an additive term of the standardized mean, so original-scale contribution
# per observation is ``component * y_std``; per period we sum across geo rows,
# matching the KPI fan's aggregation.
_COMPONENT_VARS = (
    ("trend", "trend_component", "Baseline trend"),
    ("seasonality", "seasonality_component", "Seasonality"),
    ("controls", "controls_total", "Control contributions"),
    ("media", "media_total", "Media contributions"),
)


def prior_component_facts(
    model: Any,
    idata: Any,
    *,
    interval: float = 0.90,
    n_traces: int = 6,
) -> dict[str, dict[str, Any]]:
    """Prior draws of the structural components in time, on the KPI scale.

    For each component present in the prior group (trend / seasonality /
    controls / media), returns per-period quantile bands, a few individual
    prior traces, and the component's prior share of total KPI variation —
    so the reader can see what the priors *already commit to* about
    seasonality shape, trend direction and control/media magnitudes before
    any data has spoken.
    """
    prior_ds = getattr(idata, "prior", None)
    if prior_ds is None:
        return {}

    y_std = float(getattr(model, "y_std", 1.0) or 1.0)
    n_obs = int(getattr(model, "n_obs", 0) or 0)
    time_idx = np.asarray(getattr(model, "time_idx", np.arange(n_obs)), dtype=int)
    n_periods = int(time_idx.max()) + 1 if time_idx.size else 0
    if n_periods == 0:
        return {}

    dates: list[str] = []
    try:
        periods = model.panel.coords.periods
        dates = [str(p)[:10] for p in periods]
        if len(dates) != n_periods:
            dates = []
    except Exception:  # noqa: BLE001
        dates = []
    if not dates:
        dates = [str(i) for i in range(n_periods)]

    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100
    out: dict[str, dict[str, Any]] = {}
    for key, var, label in _COMPONENT_VARS:
        if var not in getattr(prior_ds, "data_vars", {}):
            continue
        arr = np.asarray(prior_ds[var].values, dtype=float)
        if arr.shape[-1] != time_idx.size:
            continue  # not obs-indexed (unexpected shape) — skip, don't guess
        arr = arr.reshape(-1, time_idx.size) * y_std  # (S, n_obs), KPI units
        S = arr.shape[0]
        period = np.zeros((S, n_periods), dtype=float)
        np.add.at(period, (np.arange(S)[:, None], time_idx[None, :]), arr)

        k = max(0, min(int(n_traces), S))
        traces = period[np.linspace(0, S - 1, k).astype(int)] if k else None
        out[key] = {
            "label": label,
            "dates": dates,
            "bands": {
                "lower": np.percentile(period, lo_q, axis=0),
                "median": np.percentile(period, 50, axis=0),
                "upper": np.percentile(period, hi_q, axis=0),
            },
            "traces": traces,
            # typical per-period magnitude the prior implies for this component
            "abs_scale": float(np.mean(np.abs(period))),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Prior-implied estimands (what the priors say about ROI before data)
# ─────────────────────────────────────────────────────────────────────────────
def prior_estimand_facts(
    model: Any,
    idata: Any,
    *,
    interval: float = 0.90,
    max_density_draws: int = 400,
) -> dict[str, Any]:
    """The prior distribution of the flagship estimands, in original units.

    Mirrors the ``contribution_roi`` estimand's semantics on the *prior*
    samples: per channel, total prior contribution (``channel_contributions``
    summed over observations, rescaled by ``y_std``) divided by the channel's
    measurement-aware divisor (spend, or volume for impression/click channels —
    the same :func:`resolve_channel_divisor` the fitted report uses). Also
    returns the blended value across monetary channels and marketing's prior
    share of the observed KPI.
    """
    prior_ds = getattr(idata, "prior", None)
    if prior_ds is None or "channel_contributions" not in getattr(
        prior_ds, "data_vars", {}
    ):
        return {}

    from .measurement import resolve_channel_divisor

    channels = [str(c) for c in getattr(model, "channel_names", [])]
    y_std = float(getattr(model, "y_std", 1.0) or 1.0)
    arr = np.asarray(prior_ds["channel_contributions"].values, dtype=float)
    # (chain, draw, obs, channel) -> (S, obs, channel)
    arr = arr.reshape(-1, *arr.shape[-2:])
    if arr.shape[-1] != len(channels):
        return {}
    contrib = arr.sum(axis=1) * y_std  # (S, channel), KPI units

    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100
    rows: list[dict[str, Any]] = []
    blended_num = np.zeros(contrib.shape[0])
    blended_den = 0.0
    blended_all_monetary = True
    thin = np.linspace(
        0, contrib.shape[0] - 1, min(max_density_draws, contrib.shape[0])
    ).astype(int)

    for c, ch in enumerate(channels):
        try:
            div = resolve_channel_divisor(model, ch)
        except Exception:  # noqa: BLE001
            div = None
        if div is None or not div.found or div.total <= 0:
            continue
        draws = contrib[:, c] / div.total
        meta = div.meta
        ref = float(getattr(meta, "reference", 1.0))
        rows.append(
            {
                "channel": ch,
                "label": str(getattr(meta, "roi_label", "ROI")),
                "is_monetary": bool(getattr(meta, "is_monetary", True)),
                "reference": ref,
                "mean": float(np.mean(draws)),
                "lower": float(np.percentile(draws, lo_q)),
                "upper": float(np.percentile(draws, hi_q)),
                "p_above_reference": float(np.mean(draws > ref)),
                "draws": draws[thin],
            }
        )
        if getattr(meta, "is_monetary", True):
            blended_num += contrib[:, c]
            blended_den += float(div.total)
        else:
            blended_all_monetary = False

    if not rows:
        return {}

    out: dict[str, Any] = {"channels": rows, "interval": interval}
    if blended_den > 0:
        blended = blended_num / blended_den
        out["blended"] = {
            "mean": float(np.mean(blended)),
            "lower": float(np.percentile(blended, lo_q)),
            "upper": float(np.percentile(blended, hi_q)),
            "partial": not blended_all_monetary,
        }
    total_kpi = float(
        np.sum(np.asarray(getattr(model, "y_raw", np.zeros(1)), dtype=float))
    )
    if total_kpi > 0:
        share = contrib.sum(axis=1) / total_kpi
        out["marketing_share"] = {
            "mean": float(np.mean(share)),
            "lower": float(np.percentile(share, lo_q)),
            "upper": float(np.percentile(share, hi_q)),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Prior-implied response curves
# ─────────────────────────────────────────────────────────────────────────────
def _sat_curve(
    family: str, x: np.ndarray, s: dict[str, np.ndarray]
) -> np.ndarray | None:
    """Evaluate one saturation family over grid ``x`` for each prior draw."""
    if family == "logistic" and "lam" in s:
        return 1.0 - np.exp(-np.outer(s["lam"], x))
    if family == "hill" and "half" in s and "slope" in s:
        k = s["half"][:, None]
        sl = s["slope"][:, None]
        xs = np.power(x[None, :], sl)
        return xs / (np.power(k, sl) + xs + 1e-12)
    if family == "michaelis_menten" and "half" in s:
        return x[None, :] / (x[None, :] + s["half"][:, None] + 1e-12)
    if family == "tanh" and "half" in s:
        return np.tanh(x[None, :] / (s["half"][:, None] + 1e-12))
    return None


def _adstock_weights(
    family: str, l_max: int, s: dict[str, np.ndarray], normalize: bool
) -> np.ndarray | None:
    """Adstock weight curves (draws × lags) for one channel's prior draws."""
    lags = np.arange(l_max + 1, dtype=float)
    if family == "geometric" and "alpha" in s:
        w = np.power(s["alpha"][:, None], lags[None, :])
    elif family == "delayed" and "alpha" in s and "theta" in s:
        a = np.clip(s["alpha"][:, None], 1e-6, 1 - 1e-6)
        w = np.power(a, (lags[None, :] - s["theta"][:, None]) ** 2)
    elif family == "weibull" and "shape" in s and "scale" in s:
        k = np.clip(s["shape"][:, None], 1e-3, None)
        lam = np.clip(s["scale"][:, None], 1e-3, None)
        t = lags[None, :] + 1.0
        w = (k / lam) * np.power(t / lam, k - 1.0) * np.exp(-np.power(t / lam, k))
    else:
        return None
    if normalize:
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    return w


def prior_response_curves(
    model: Any,
    prior: Any = None,
    *,
    n_grid: int = 80,
    interval: float = 0.90,
    max_draws: int = 300,
) -> dict[str, dict[str, Any]]:
    """Prior-implied saturation + carryover shapes per channel.

    For each channel returns (when the prior samples allow) the median curve and
    the central ``interval`` band of the saturation response over normalized
    spend ``[0, 1]``, and of the adstock weights over the channel's lag window.
    """
    prior_ds = getattr(prior, "prior", prior)
    if prior_ds is None:
        return {}
    mff = getattr(model, "mff_config", None)
    channels = [str(c) for c in getattr(model, "channel_names", [])]
    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100
    x = np.linspace(0.0, 1.0, n_grid)

    def _samples(name: str) -> np.ndarray | None:
        if name in getattr(prior_ds, "data_vars", {}):
            v = np.asarray(prior_ds[name].values, dtype=float).reshape(-1)
            v = v[np.isfinite(v)]
            if v.size:
                return v[:max_draws]
        return None

    out: dict[str, dict[str, Any]] = {}
    for ch in channels:
        sat_family = "logistic"
        ad_family = "geometric"
        l_max = 8
        normalize = True
        if mff is not None:
            try:
                cfg = mff.get_media_config(ch)
                if cfg is not None:
                    sat_family = cfg.saturation.type.value
                    ad_family = cfg.adstock.type.value
                    l_max = int(cfg.adstock.l_max)
                    normalize = bool(cfg.adstock.normalize)
            except Exception:  # noqa: BLE001
                pass

        entry: dict[str, Any] = {
            "sat_family": sat_family,
            "adstock_family": ad_family,
        }

        sat_params: dict[str, np.ndarray] = {}
        for key, rv in (
            ("lam", f"sat_lam_{ch}"),
            ("half", f"sat_half_{ch}"),
            ("slope", f"sat_slope_{ch}"),
        ):
            v = _samples(rv)
            if v is not None:
                sat_params[key] = v
        if sat_params:
            n = min(v.size for v in sat_params.values())
            curves = _sat_curve(
                sat_family, x, {k: v[:n] for k, v in sat_params.items()}
            )
            if curves is not None and curves.size:
                entry["saturation"] = {
                    "x": x,
                    "median": np.median(curves, axis=0),
                    "lower": np.percentile(curves, lo_q, axis=0),
                    "upper": np.percentile(curves, hi_q, axis=0),
                }

        ad_params: dict[str, np.ndarray] = {}
        for key, rv in (
            ("alpha", f"adstock_alpha_{ch}"),
            ("theta", f"adstock_theta_{ch}"),
            ("shape", f"adstock_shape_{ch}"),
            ("scale", f"adstock_scale_{ch}"),
        ):
            v = _samples(rv)
            if v is not None:
                ad_params[key] = v
        if ad_params:
            n = min(v.size for v in ad_params.values())
            w = _adstock_weights(
                ad_family, l_max, {k: v[:n] for k, v in ad_params.items()}, normalize
            )
            if w is not None and w.size:
                entry["adstock"] = {
                    "lags": np.arange(l_max + 1),
                    "median": np.median(w, axis=0),
                    "lower": np.percentile(w, lo_q, axis=0),
                    "upper": np.percentile(w, hi_q, axis=0),
                }

        if "saturation" in entry or "adstock" in entry:
            out[ch] = entry
        else:
            logger.debug(f"prefit: no prior curve samples for channel {ch}")
    return out
