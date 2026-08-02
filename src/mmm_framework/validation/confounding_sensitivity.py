"""Confounding sensitivity on the **decision** scale, and how to calibrate it.

:mod:`mmm_framework.validation.sensitivity_unobserved` answers the parameter-scale
question — what partial ``R^2`` would a hidden confounder need with both a
channel's spend and the KPI to drive its *coefficient* to zero. This module
answers the two questions a reader actually acts on:

1. **How much hidden bias would change the decision?** Applying the bias-parameter
   engine (:mod:`mmm_framework.diagnostics.bias_sensitivity`) to a channel's ROI
   posterior gives a tipping point in the currency of the recommendation: "TV's
   ROI would have to be overstated by more than 24% for it to stop clearing
   break-even". No refit — the de-biased posterior is a closed-form Gaussian
   mixture over draws the model already produced.

2. **Is that much bias plausible?** A tipping point on its own invites an
   arbitrary answer, because the analyst supplies the bias prior. Cinelli–Hazlett
   *benchmarking* removes the arbitrariness by pricing a hypothetical confounder
   against the covariates you did measure: "a confounder as strong as Price would
   move TV's ROI by 9%" — comfortably inside the 24% tipping point, so
   Price-strength confounding does not overturn it.

The benchmark is what makes this more than a slider. It runs on the **linear
design** the frequentist path already builds
(:func:`mmm_framework.frequentist.design.build_design_matrix`), whose documented
invariant is that ``X @ theta`` reproduces the graph's ``mu`` to 1e-12 once
adstock and saturation are fixed — so the partial ``R^2`` values are computed
against the model that was actually fitted, at its own posterior-mean transform
point, rather than against a stand-in.

Why the benchmark uses an OLS standard error and not the posterior sd
---------------------------------------------------------------------
The Cinelli–Hazlett bias identity is ``|bias| = se * sqrt(df) * BF``, and it is
calibrated on the OLS fact that ``se * sqrt(df) = ||y_res|| / ||d_res||`` — a
ratio of residual norms. Substituting a Bayesian posterior sd breaks that
identity by whatever factor the prior contributed, and it breaks it in the
dangerous direction: this framework's media priors have positive support and are
informative, so the posterior sd is *smaller* than the OLS standard error, the
implied bias comes out *too small*, and the model reports robustness the data did
not supply.

Note this is the mirror image of the caveat on the robustness value, which is
inflated by a tight prior because ``RV`` rises with ``|mean| / sd``. Two opposite
mechanisms, one direction: **a tighter prior makes both numbers look more
reassuring.** A reader who has internalised one caveat should not assume the
other path inherits it — each has its own, and both are guarded here.

What this cannot do
-------------------
Everything here is a re-weighting of a fitted posterior. It shifts and widens an
estimand; it cannot move a channel's coefficient *relative to the controls*,
because the model is never re-estimated. A confounder that would genuinely be
absorbed differently by the trend, the seasonality or another channel has to
enter the graph and be re-fitted — see the ``ConfounderSweep`` path.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..diagnostics.bias_sensitivity import (
    DEFAULT_DECISION_THRESHOLD,
    BiasPrior,
    BiasSensitivity,
    bias_sensitivity_report,
    named_prior_ladder,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BenchmarkBound",
    "BenchmarkReport",
    "ChannelConfoundingSensitivity",
    "ConfoundingSensitivityReport",
    "DEFAULT_BENCHMARK_MULTIPLIERS",
    "adjusted_se_from_r2",
    "benchmark_bias_priors",
    "bias_bound",
    "ovb_partial_r2_bound",
    "posterior_transform_point",
    "ConfounderSweepConfig",
    "ConfounderSweepPoint",
    "ConfounderSweepResult",
    "build_confounder",
    "run_confounder_sweep",
    "run_confounding_sensitivity",
]

#: How much stronger than the benchmark covariate the hypothetical confounder is
#: assumed to be. ``1.0`` is the natural anchor ("as strong as Price"); the larger
#: rungs answer "and if it were twice / three times as strong?".
DEFAULT_BENCHMARK_MULTIPLIERS = (1.0, 2.0, 3.0)

#: Design blocks eligible to be benchmarked against by default. Seasonality,
#: trend and the geo/product dummies are excluded deliberately: "a confounder as
#: strong as the annual seasonal cycle" is not a statement anyone can act on, and
#: those columns are also the ones most likely to breach the bound's validity
#: condition and produce a refusal for every channel.
_BENCHMARKABLE_BLOCKS = ("controls",)


# --------------------------------------------------------------------------- #
# the Cinelli-Hazlett algebra
# --------------------------------------------------------------------------- #


def ovb_partial_r2_bound(
    r2dxj_x: float,
    r2yxj_dx: float,
    *,
    kd: float = 1.0,
    ky: float | None = None,
) -> tuple[float, float, float, bool, str]:
    """Bound a hypothetical confounder's strength by an observed covariate's.

    Given the observed covariate ``x_j``'s partial ``R^2`` with the treatment
    (``r2dxj_x``) and with the outcome (``r2yxj_dx``), returns the implied bounds
    on an unobserved ``Z`` assumed ``kd`` times as strongly associated with the
    treatment and ``ky`` times as strongly with the outcome (``ky`` defaults to
    ``kd``). This is ``sensemakr::ovb_partial_r2_bound``.

    Returns ``(r2dz_x, r2yz_dx, r2zxj_xd, saturated, reason)``. ``saturated`` is
    ``True`` when ``r2yz_dx`` had to be clipped at 1.0 — the bound is then
    degenerate (``adjusted_se`` is exactly 0) and must not be rendered as an
    ordinary number.

    **Validity.** The binding condition is ``r2dz_x < 1``, i.e.
    ``r2dxj_x < 1 / (1 + kd)`` — at ``kd = 1`` that is ``r2dxj_x < 0.5``, *not*
    ``< 1``. Past it the published formula takes the square root of a negative
    number and yields ``NaN`` rather than raising, and a ``NaN`` compares
    ``False`` against every fragility threshold — i.e. it would silently report
    "not fragile". So the breach is returned as an explicit reason, never
    clipped, and the caller is expected to surface it as a refusal.
    """
    ky = kd if ky is None else ky
    if not (0.0 <= r2dxj_x < 1.0) or not (0.0 <= r2yxj_dx < 1.0):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            "partial R^2 values must lie in [0, 1)",
        )

    r2dz_x = kd * (r2dxj_x / (1.0 - r2dxj_x))
    if r2dz_x >= 1.0:
        max_kd = (1.0 - r2dxj_x) / r2dxj_x if r2dxj_x > 0 else float("inf")
        return (
            r2dz_x,
            float("nan"),
            float("nan"),
            False,
            (
                f"kd={kd:g} is impossible against this covariate: it implies the "
                f"confounder explains {r2dz_x:.0%} of the treatment's residual "
                f"variance. The largest admissible kd here is {max_kd:.2f}."
            ),
        )

    denom = (1.0 - kd * r2dxj_x) * (1.0 - r2dxj_x)
    if denom <= 0:
        return (
            r2dz_x,
            float("nan"),
            float("nan"),
            False,
            f"kd={kd:g} is impossible against this covariate (degenerate bound).",
        )
    r2zxj_xd = kd * (r2dxj_x**2) / denom
    if r2zxj_xd >= 1.0:
        return (
            r2dz_x,
            float("nan"),
            r2zxj_xd,
            False,
            (
                f"kd={kd:g} is impossible against this covariate: it implies the "
                "confounder explains all of the covariate's residual variance."
            ),
        )

    r2yz_dx = (
        (math.sqrt(ky) + math.sqrt(r2zxj_xd)) / math.sqrt(1.0 - r2zxj_xd)
    ) ** 2 * (r2yxj_dx / (1.0 - r2yxj_dx))
    saturated = False
    reason = ""
    if r2yz_dx > 1.0:
        r2yz_dx = 1.0
        saturated = True
        reason = (
            f"ky={ky:g} saturates the outcome side: the implied confounder would "
            "explain all remaining outcome variance, so this bound is an extreme "
            "case rather than a calibrated comparison."
        )
    return r2dz_x, r2yz_dx, r2zxj_xd, saturated, reason


def bias_bound(se: float, dof: int, r2dz_x: float, r2yz_dx: float) -> float:
    """``|bias| = se * sqrt(df) * sqrt(r2yz_dx * r2dz_x / (1 - r2dz_x))``.

    ``se`` and ``dof`` must come from the **same** regression — mixing a nominal
    degrees-of-freedom count with a standard error produced under a different one
    puts two incompatible quantities in one identity.
    """
    if dof <= 0 or not (0.0 <= r2dz_x < 1.0) or not (0.0 <= r2yz_dx <= 1.0):
        return float("nan")
    return float(se * math.sqrt(dof) * math.sqrt(r2yz_dx * r2dz_x / (1.0 - r2dz_x)))


def adjusted_se_from_r2(se: float, dof: int, r2dz_x: float, r2yz_dx: float) -> float:
    """``se * sqrt((1 - r2yz_dx) / (1 - r2dz_x)) * sqrt(df / (df - 1))``."""
    if dof <= 1 or not (0.0 <= r2dz_x < 1.0) or not (0.0 <= r2yz_dx <= 1.0):
        return float("nan")
    return float(
        se * math.sqrt((1.0 - r2yz_dx) / (1.0 - r2dz_x)) * math.sqrt(dof / (dof - 1.0))
    )


# --------------------------------------------------------------------------- #
# the design-matrix OLS
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _OLS:
    """A least-squares fit of one column set, with t-statistics."""

    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    dof: int


def _ols(X: np.ndarray, y: np.ndarray) -> _OLS:
    n, k = X.shape
    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    coef = xtx_inv @ (X.T @ y)
    resid = y - X @ coef
    dof = n - k
    sigma2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, coef / se, np.nan)
    return _OLS(coef=coef, se=se, t=t, dof=dof)


def _partial_r2(t_value: float, dof: int) -> float:
    if dof <= 0 or not np.isfinite(t_value):
        return float("nan")
    t2 = float(t_value) ** 2
    return t2 / (t2 + dof)


#: Order in which columns are offered to the independence scan. Structural
#: blocks come first so that when a column is redundant it is the *later* one
#: that gets dropped — a control that duplicates the seasonal basis should be
#: reported as carrying no independent information, not allowed to stand in for
#: seasonality and be benchmarked as though it were a real confounder.
_COLUMN_PRIORITY = (
    "intercept",
    "trend",
    "seasonality",
    "geo",
    "product",
    "media",
    "controls",
)


def _select_independent_columns(
    X: np.ndarray, names: list[str], blocks: dict[str, slice]
) -> tuple[np.ndarray, list[str], list[str], str]:
    """Keep a maximal independent column subset, in a deliberate priority order.

    Two independent things make the raw design singular. A full geo (or product)
    dummy set sits alongside an unpenalized intercept **by construction** — the
    frequentist path relies on the ridge penalty to pick a unique split, and
    ordinary least squares has no such tie-break. And a control can happen to be
    an exact function of the model's own basis (a price series that is a pure
    seasonal cosine is a real example). Either way ``lstsq``/``pinv`` would return
    the minimum-norm solution and every partial ``R^2`` computed from it would
    depend on that arbitrary choice.

    Dropping a redundant column leaves the column **space** untouched, so every
    surviving column residualizes against exactly the same subspace and no
    partial ``R^2`` changes. What the priority order buys is *which* member of a
    redundant pair is dropped: offering structural blocks first means a control
    that merely duplicates them is the one reported as uninformative.

    A dropped **media** column is a different matter — that channel is not
    identified given the rest of the design — and is returned as a refusal.

    Implemented as modified Gram–Schmidt against the accepted basis rather than
    repeated rank calls: exact, deterministic, and ``O(n k^2)``.
    """
    n, k = X.shape
    order: list[int] = []
    seen: set[int] = set()
    for block in _COLUMN_PRIORITY:
        sl = blocks.get(block)
        if sl is None:
            continue
        for i in range(sl.start, sl.stop):
            if i not in seen:
                order.append(i)
                seen.add(i)
    order += [i for i in range(k) if i not in seen]

    basis: list[np.ndarray] = []
    keep_idx: list[int] = []
    dropped: list[str] = []
    for i in order:
        col = X[:, i].astype(float)
        norm0 = float(np.linalg.norm(col))
        if norm0 <= 0:
            dropped.append(names[i])
            continue
        resid = col.copy()
        for q in basis:
            resid -= float(q @ resid) * q
        if float(np.linalg.norm(resid)) <= 1e-10 * norm0:
            dropped.append(names[i])
            continue
        basis.append(resid / float(np.linalg.norm(resid)))
        keep_idx.append(i)

    dropped_media = [d for d in dropped if d.startswith("media_")]
    if dropped_media:
        return (
            X,
            names,
            dropped,
            (
                "channel column(s) "
                + ", ".join(d[len("media_") :] for d in dropped_media)
                + " are exactly collinear with the rest of the design, so their "
                "effects are not separately identified and no bound against them "
                "would mean anything"
            ),
        )

    keep_idx.sort()
    return X[:, keep_idx], [names[i] for i in keep_idx], dropped, ""


# --------------------------------------------------------------------------- #
# posterior transform point
# --------------------------------------------------------------------------- #


def posterior_transform_point(model: Any) -> tuple[dict[str, dict], dict[str, dict]]:
    """Posterior-mean adstock and saturation parameters, keyed as the design wants.

    Reuses the framework's own name tables (``model.base._ADSTOCK_KIND`` and
    ``frequentist._transforms.SATURATION_PARAMS``) so this cannot drift from the
    graph when a transform family is added.

    Note the approximation this fixes in place: the posterior *mean* transform
    point is not the point that reproduces the posterior-mean contribution, since
    saturation is concave. It is recorded on the report so the benchmark is
    reproducible and the approximation is visible rather than assumed away.
    """
    from ..frequentist._transforms import SATURATION_PARAMS
    from ..model.base import _ADSTOCK_KIND

    trace = getattr(model, "_trace", None)
    if trace is None:
        raise ValueError("Model must be fitted (no posterior trace found).")
    posterior = trace.posterior

    def _mean(name: str) -> float | None:
        if name not in posterior:
            return None
        return float(np.asarray(posterior[name].values).mean())

    alpha: dict[str, dict] = {}
    lam: dict[str, dict] = {}
    for ch in model.channel_names:
        acfg = model._get_adstock_config(ch)
        kind = _ADSTOCK_KIND.get(acfg.type, "geometric")
        params: dict[str, float] = {}
        wanted = {
            "geometric": ("alpha",),
            "delayed": ("alpha", "theta"),
            "weibull": ("shape", "scale"),
            "none": (),
        }.get(kind, ("alpha",))
        for p in wanted:
            v = _mean(f"adstock_{p}_{ch}")
            if v is not None:
                params[p] = v
        alpha[ch] = params

        sat_kind = model._get_saturation_config(ch).type
        sat: dict[str, float] = {}
        for p in SATURATION_PARAMS[sat_kind]:
            v = _mean(f"{p}_{ch}")
            if v is not None:
                sat[p] = v
        lam[ch] = sat
    return alpha, lam


# --------------------------------------------------------------------------- #
# the benchmark
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BenchmarkBound:
    """What a confounder ``kd`` times as strong as one observed covariate implies."""

    channel: str
    covariate: str
    kd: float
    ky: float
    r2dxj_x: float
    r2yxj_dx: float
    r2dz_x: float
    r2yz_dx: float
    estimate: float
    se: float
    dof: int
    bias: float
    fractional_bias: float
    adjusted_estimate: float
    adjusted_se: float
    saturated: bool
    status: str  # "ok" | "refused"
    reason: str = ""

    def as_prior(self) -> BiasPrior:
        """This bound as a bias prior, on the portable fraction-of-estimate scale.

        A bound is a magnitude, not a spread, so it becomes ``mu`` with
        ``sigma = 0`` — a point-mass commitment. That is the honest reading:
        "if the confounder were exactly this strong, here is where the estimate
        lands", which is directly comparable against the ``mu`` tipping point.
        """
        return BiasPrior(
            mu=float(self.fractional_bias),
            sigma=0.0,
            scale="fraction_of_mean",
            correlation="shared",
            label=f"{self.kd:g}x {self.covariate}",
            source=f"benchmark:{self.covariate}(kd={self.kd:g},ky={self.ky:g})",
        )

    def describe(self) -> str:
        if self.status != "ok":
            return f"{self.kd:g}x {self.covariate}: not computable — {self.reason}"
        return (
            f"a confounder {self.kd:g}x as strong as {self.covariate} would move "
            f"{self.channel} by {self.fractional_bias:.0%} of its estimate"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "covariate": self.covariate,
            "kd": float(self.kd),
            "ky": float(self.ky),
            "r2dxj_x": float(self.r2dxj_x),
            "r2yxj_dx": float(self.r2yxj_dx),
            "r2dz_x": float(self.r2dz_x),
            "r2yz_dx": float(self.r2yz_dx),
            "estimate": float(self.estimate),
            "se": float(self.se),
            "dof": int(self.dof),
            "bias": float(self.bias),
            "fractional_bias": float(self.fractional_bias),
            "adjusted_estimate": float(self.adjusted_estimate),
            "adjusted_se": float(self.adjusted_se),
            "saturated": bool(self.saturated),
            "status": self.status,
            "reason": self.reason,
            "description": self.describe(),
        }


@dataclass
class BenchmarkReport:
    """Every observed-covariate benchmark, or a stated reason there are none."""

    bounds: dict[str, list[BenchmarkBound]] = field(default_factory=dict)
    status: str = "ok"  # "ok" | "unavailable"
    reason: str = ""
    covariates: tuple[str, ...] = ()
    dropped_columns: tuple[str, ...] = ()
    transform_point: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def strongest(self, channel: str) -> BenchmarkBound | None:
        """The largest admissible fractional bias for a channel, or ``None``."""
        rows = [
            b
            for b in self.bounds.get(channel, [])
            if b.status == "ok" and np.isfinite(b.fractional_bias)
        ]
        if not rows:
            return None
        return max(rows, key=lambda b: b.fractional_bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "covariates": list(self.covariates),
            "dropped_columns": list(self.dropped_columns),
            "transform_point": self.transform_point,
            "notes": list(self.notes),
            "bounds": {
                ch: [b.to_dict() for b in rows] for ch, rows in self.bounds.items()
            },
        }


def _benchmark_covariates(
    model: Any,
    names: list[str],
    blocks: dict[str, slice],
    explicit: Sequence[str] | None,
) -> list[int]:
    """Column indices eligible as benchmark covariates.

    Restricted to controls, and — when the model declares causal roles — to those
    marked ``CONFOUNDER``. Benchmarking a media effect against a *mediator* or a
    *collider* would produce a bound with no causal meaning, and the framework
    already knows which is which.
    """
    idx: list[int] = []
    if explicit is not None:
        wanted = {str(c) for c in explicit}
        return [
            i
            for i, n in enumerate(names)
            if n.startswith("control_") and n[len("control_") :] in wanted
        ]

    roles = getattr(model, "_control_causal_roles", None) or []
    control_names = list(getattr(model, "control_names", []) or [])
    confounders: set[str] = set()
    if len(roles) == len(control_names):
        from ..config.enums import CausalControlRole

        confounders = {
            cn
            for cn, role in zip(control_names, roles)
            if role == CausalControlRole.CONFOUNDER
        }

    for block in _BENCHMARKABLE_BLOCKS:
        sl = blocks.get(block)
        if sl is None:
            continue
        for i in range(sl.start, sl.stop):
            name = (
                names[i][len("control_") :]
                if names[i].startswith("control_")
                else names[i]
            )
            if confounders and name not in confounders:
                continue
            idx.append(i)
    return idx


def benchmark_bias_priors(
    model: Any,
    *,
    kd: Sequence[float] = DEFAULT_BENCHMARK_MULTIPLIERS,
    ky: float | None = None,
    covariates: Sequence[str] | None = None,
) -> BenchmarkReport:
    """Price a hypothetical confounder against the covariates you did measure.

    Builds the linear design at the model's posterior-mean transform point, fits
    ordinary least squares on it, and turns each observed covariate's partial
    ``R^2`` into a bound on what an unobserved confounder of comparable strength
    would do to each channel.

    Never raises for a model it cannot handle: an unsupported feature, a
    rank-deficient design or a missing trace all return
    ``BenchmarkReport(status="unavailable", reason=...)`` so the caller can report
    the absence rather than silently skipping — a skipped benchmark reads as "no
    problem found", which is the failure this module exists to prevent.
    """
    from ..frequentist.design import UnsupportedModelError, build_design_matrix

    try:
        alpha, lam = posterior_transform_point(model)
    except Exception as e:  # noqa: BLE001
        return BenchmarkReport(status="unavailable", reason=str(e))

    try:
        design = build_design_matrix(
            model.panel,
            alpha,
            lam,
            model_config=model.model_config,
            trend_config=model.trend_config,
        )
    except UnsupportedModelError as e:
        return BenchmarkReport(
            status="unavailable",
            reason=(
                f"{e.feature} is not linear given fixed transforms, so no design "
                f"matrix exists to benchmark against ({e.reason}). The tipping "
                "point below still applies; only the plausibility comparison is "
                "missing."
            ),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("design matrix unavailable for benchmarking", exc_info=True)
        return BenchmarkReport(status="unavailable", reason=str(e))

    X, names, dropped, rank_reason = _select_independent_columns(
        np.asarray(design.X, dtype=float), list(design.columns), dict(design.blocks)
    )
    if rank_reason:
        return BenchmarkReport(
            status="unavailable",
            reason=rank_reason,
            dropped_columns=tuple(dropped),
        )

    # Column indices survive the drop, so recompute the blocks by name.
    name_to_idx = {n: i for i, n in enumerate(names)}
    blocks = {
        b: slice(
            min(
                (name_to_idx[n] for n in design.columns[sl] if n in name_to_idx),
                default=0,
            ),
            max(
                (name_to_idx[n] + 1 for n in design.columns[sl] if n in name_to_idx),
                default=0,
            ),
        )
        for b, sl in design.blocks.items()
    }

    y = np.asarray(design.y, dtype=float)
    outcome = _ols(X, y)
    cov_idx = _benchmark_covariates(model, names, blocks, covariates)
    if not cov_idx:
        return BenchmarkReport(
            status="unavailable",
            reason=(
                "no control variable is available to benchmark against. A bound "
                "prices a hypothetical confounder against a measured one; with no "
                "measured confounder there is nothing to anchor it to."
            ),
            dropped_columns=tuple(dropped),
        )

    notes: list[str] = []
    if dropped:
        notes.append(
            "Dropped "
            + ", ".join(dropped)
            + ": these columns carry no variation independent of the rest of the "
            "design (a reference dummy level, or a control that is an exact "
            "function of the trend/seasonal basis). The column space is unchanged, "
            "so no partial R^2 is affected — but a dropped control cannot be "
            "benchmarked against, because there is nothing in it to benchmark."
        )
    notes.append(
        "Standard errors are ordinary least squares on the model's own design at "
        "its posterior-mean transform point — not posterior standard deviations, "
        "which the informative media priors would shrink and which would make the "
        "implied bias too small."
    )

    bounds: dict[str, list[BenchmarkBound]] = {}
    for ch in model.channel_names:
        col = f"media_{ch}"
        if col not in name_to_idx:
            continue
        d = name_to_idx[col]
        estimate = float(outcome.coef[d])
        se = float(outcome.se[d])
        dof_y = outcome.dof

        # Treatment regression: the channel column on everything else. Its
        # residual dof differs from the outcome regression's by one, and using
        # the wrong one is a quiet error in the partial-R^2 denominators.
        others = [i for i in range(X.shape[1]) if i != d]
        treat = _ols(X[:, others], X[:, d])
        pos_in_others = {c: k for k, c in enumerate(others)}

        rows: list[BenchmarkBound] = []
        for j in cov_idx:
            if j == d:
                continue
            cov_name = (
                names[j][len("control_") :]
                if names[j].startswith("control_")
                else names[j]
            )
            r2dxj = _partial_r2(float(treat.t[pos_in_others[j]]), treat.dof)
            r2yxj = _partial_r2(float(outcome.t[j]), dof_y)
            for k in kd:
                r2dz, r2yz, _r2zxj, saturated, reason = ovb_partial_r2_bound(
                    r2dxj, r2yxj, kd=float(k), ky=ky
                )
                if not np.isfinite(r2yz):
                    rows.append(
                        BenchmarkBound(
                            channel=ch,
                            covariate=cov_name,
                            kd=float(k),
                            ky=float(k if ky is None else ky),
                            r2dxj_x=r2dxj,
                            r2yxj_dx=r2yxj,
                            r2dz_x=r2dz,
                            r2yz_dx=float("nan"),
                            estimate=estimate,
                            se=se,
                            dof=dof_y,
                            bias=float("nan"),
                            fractional_bias=float("nan"),
                            adjusted_estimate=float("nan"),
                            adjusted_se=float("nan"),
                            saturated=False,
                            status="refused",
                            reason=reason,
                        )
                    )
                    continue
                b = bias_bound(se, dof_y, r2dz, r2yz)
                adj_se = adjusted_se_from_r2(se, dof_y, r2dz, r2yz)
                frac = abs(b) / abs(estimate) if estimate else float("nan")
                # The bias is signed against the estimate: a confounder that
                # inflated the effect is removed by moving toward zero.
                adjusted = estimate - math.copysign(abs(b), estimate)
                rows.append(
                    BenchmarkBound(
                        channel=ch,
                        covariate=cov_name,
                        kd=float(k),
                        ky=float(k if ky is None else ky),
                        r2dxj_x=r2dxj,
                        r2yxj_dx=r2yxj,
                        r2dz_x=r2dz,
                        r2yz_dx=r2yz,
                        estimate=estimate,
                        se=se,
                        dof=dof_y,
                        bias=abs(b),
                        fractional_bias=frac,
                        adjusted_estimate=adjusted,
                        adjusted_se=adj_se,
                        saturated=saturated,
                        status="ok",
                        reason=reason,
                    )
                )
        bounds[ch] = rows

    return BenchmarkReport(
        bounds=bounds,
        status="ok",
        covariates=tuple(
            sorted(
                {
                    (
                        names[j][len("control_") :]
                        if names[j].startswith("control_")
                        else names[j]
                    )
                    for j in cov_idx
                }
            )
        ),
        dropped_columns=tuple(dropped),
        transform_point={"adstock": alpha, "saturation": lam},
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# the decision-scale report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChannelConfoundingSensitivity:
    """One channel's decision-scale sensitivity, plus what calibrates it."""

    channel: str
    sensitivity: BiasSensitivity
    benchmarks: tuple[BenchmarkBound, ...] = ()
    metric_label: str = "ROI"
    is_monetary: bool = True
    prior_contraction: float | None = None

    @property
    def is_fragile(self) -> bool:
        return self.sensitivity.is_fragile

    @property
    def is_assessable(self) -> bool:
        return self.sensitivity.is_assessable

    @property
    def benchmark_exceeds_tipping_point(self) -> bool | None:
        """Whether a plausible confounder would actually overturn the channel.

        The headline comparison: a tipping point says how much bias it would
        take, a benchmark says how much a *measured* covariate's worth of
        confounding implies. ``None`` when either side is missing — which is not
        the same as ``False`` and must not be rendered as reassurance.
        """
        tip = self.sensitivity.tipping_mu
        usable = [
            b
            for b in self.benchmarks
            if b.status == "ok" and np.isfinite(b.fractional_bias)
        ]
        if not usable:
            return None
        worst = max(b.fractional_bias for b in usable)
        if tip.already_below:
            return True
        if not tip.crossed or tip.value is None:
            return False
        return bool(worst >= tip.value)

    def describe(self) -> str:
        base = self.sensitivity.describe()
        verdict = self.benchmark_exceeds_tipping_point
        if verdict is None:
            return base
        usable = [b for b in self.benchmarks if b.status == "ok"]
        worst = max(usable, key=lambda b: b.fractional_bias)
        if verdict:
            return (
                f"{base} A confounder as strong as {worst.covariate} "
                f"({worst.kd:g}x) implies {worst.fractional_bias:.0%} — enough to "
                "overturn it."
            )
        return (
            f"{base} The strongest observed benchmark ({worst.kd:g}x "
            f"{worst.covariate}) implies only {worst.fractional_bias:.0%}, so "
            "confounding of that size would not."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "sensitivity": self.sensitivity.to_dict(),
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "metric_label": self.metric_label,
            "is_monetary": bool(self.is_monetary),
            "prior_contraction": (
                None
                if self.prior_contraction is None
                else float(self.prior_contraction)
            ),
            "is_fragile": bool(self.is_fragile),
            "is_assessable": bool(self.is_assessable),
            "benchmark_exceeds_tipping_point": self.benchmark_exceeds_tipping_point,
            "description": self.describe(),
        }


@dataclass
class ConfoundingSensitivityReport:
    """Decision-scale confounding sensitivity for every channel."""

    channels: list[ChannelConfoundingSensitivity]
    estimand: str
    threshold: float
    benchmark: BenchmarkReport
    caveat: str

    @property
    def fragile_channels(self) -> list[str]:
        return [c.channel for c in self.channels if c.is_fragile]

    @property
    def unassessable_channels(self) -> list[str]:
        """Channels with no usable tipping point.

        These are **not** in :attr:`fragile_channels` — a verdict of
        ``not_assessable`` fails the fragility test — so an empty
        ``fragile_channels`` must never be read as "every channel is fine"
        without consulting this list too.
        """
        return [c.channel for c in self.channels if not c.is_assessable]

    @property
    def overturned_channels(self) -> list[str]:
        return [
            c.channel for c in self.channels if c.sensitivity.verdict == "overturned"
        ]

    def summary(self) -> pd.DataFrame:
        def _pct(v: float | None) -> str:
            return "n/a" if v is None or not np.isfinite(v) else f"{v:.0%}"

        rows = []
        for c in self.channels:
            tip = c.sensitivity.tipping_mu
            strongest = max(
                (
                    b
                    for b in c.benchmarks
                    if b.status == "ok" and np.isfinite(b.fractional_bias)
                ),
                key=lambda b: b.fractional_bias,
                default=None,
            )
            rows.append(
                {
                    "Channel": c.channel,
                    c.metric_label: (
                        f"{c.sensitivity.estimate:.2f}"
                        if np.isfinite(c.sensitivity.estimate)
                        else "n/a"
                    ),
                    "P(above break-even)": _pct(c.sensitivity.prob_at_zero_bias),
                    "Tipping point": (
                        "already below"
                        if tip.already_below
                        else (
                            _pct(tip.value)
                            if tip.crossed
                            else f">{_pct(tip.max_scanned)}"
                        )
                    ),
                    "Strongest benchmark": (
                        "n/a"
                        if strongest is None
                        else f"{_pct(strongest.fractional_bias)} ({strongest.kd:g}x {strongest.covariate})"
                    ),
                    "Verdict": c.sensitivity.verdict,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": [c.to_dict() for c in self.channels],
            "estimand": self.estimand,
            "threshold": float(self.threshold),
            "benchmark": self.benchmark.to_dict(),
            "fragile_channels": self.fragile_channels,
            "unassessable_channels": self.unassessable_channels,
            "overturned_channels": self.overturned_channels,
            "caveat": self.caveat,
        }


def _prior_roi_draws(model: Any, channels: Sequence[str]) -> dict[str, np.ndarray]:
    """Prior-scale ROI draws, when the trace carries a prior group.

    Turns the "the implied prior on tau is not the one you elicited" caveat into
    a number. Best-effort: an absent prior group means the disclosure is simply
    not made, which is reported as unknown rather than as a pass.
    """
    out: dict[str, np.ndarray] = {}
    trace = getattr(model, "_trace", None)
    if trace is None:
        return out
    try:
        from ..reporting.helpers.measurement import resolve_channel_divisor
        from ..utils.arviz_compat import has_group

        if not has_group(trace, "prior"):
            return out
        prior = trace.prior
        if "channel_contributions" not in prior:
            return out
        contrib = np.asarray(prior["channel_contributions"].values, dtype=float)
        # (chain, draw, obs, channel) -> (draws, obs, channel)
        contrib = contrib.reshape(-1, *contrib.shape[2:])
        y_std = float(getattr(model, "y_std", 1.0) or 1.0)
        for i, ch in enumerate(channels):
            if i >= contrib.shape[-1]:
                continue
            div = resolve_channel_divisor(model, ch)
            total = float(getattr(div, "total", 0.0) or 0.0)
            if not np.isfinite(total) or total <= 0:
                continue
            out[str(ch)] = contrib[..., i].sum(axis=1) * y_std / total
    except Exception:  # noqa: BLE001 — a disclosure must never sink the analysis
        logger.debug("prior ROI draws unavailable", exc_info=True)
    return out


def _coefficient_contraction(model: Any, channel: str) -> float | None:
    """``1 - Var_post / Var_prior`` for a channel's coefficient, or ``None``.

    Refuses to pool a multi-dimensional ``beta_<channel>`` (which a per-geo
    hierarchy or a time-varying coefficient produces) into one number: flattening
    would fold between-geo heterogeneity into the posterior sd and silently
    deflate the contraction.
    """
    trace = getattr(model, "_trace", None)
    if trace is None:
        return None
    name = f"beta_{channel}"
    try:
        from ..utils.arviz_compat import has_group

        post = trace.posterior
        if name not in post or not has_group(trace, "prior"):
            return None
        post_vals = np.asarray(post[name].values)
        if post_vals.ndim > 2:  # (chain, draw, ...) — not a scalar coefficient
            return None
        prior = trace.prior
        if name not in prior:
            return None
        prior_vals = np.asarray(prior[name].values).reshape(-1)
        prior_vals = prior_vals[np.isfinite(prior_vals)]
        post_flat = post_vals.reshape(-1)
        post_flat = post_flat[np.isfinite(post_flat)]
        if prior_vals.size < 2 or post_flat.size < 2:
            return None
        prior_var = float(np.var(prior_vals, ddof=1))
        if prior_var <= 0:
            return None
        return float(1.0 - float(np.var(post_flat, ddof=1)) / prior_var)
    except Exception:  # noqa: BLE001
        logger.debug("contraction unavailable for %s", name, exc_info=True)
        return None


def run_confounding_sensitivity(
    model: Any,
    *,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    benchmark: bool = True,
    priors: Sequence[BiasPrior] | None = None,
    include_surface: bool = True,
    max_draws: int = 400,
    random_seed: int = 42,
) -> ConfoundingSensitivityReport:
    """Decision-scale confounding sensitivity for every channel of a fitted model.

    Uses each channel's contribution-ROI posterior — the same numerator and
    denominator as the ``contribution_roi`` estimand, via
    :func:`mmm_framework.validation.spec_curve.channel_roi_draws` — and judges it
    against the measurement-aware break-even reference, so an impression-measured
    channel is compared to 0 rather than to 1.
    """
    from ..reporting.helpers.measurement import resolve_channel_divisor
    from .spec_curve import channel_roi_draws

    if getattr(model, "_trace", None) is None:
        raise ValueError("Model must be fitted (no posterior trace found).")

    channels = [str(c) for c in getattr(model, "channel_names", []) or []]
    roi = channel_roi_draws(
        model, channels, max_draws=max_draws, random_seed=random_seed
    )
    prior_roi = _prior_roi_draws(model, channels)

    bench = (
        benchmark_bias_priors(model)
        if benchmark
        else BenchmarkReport(status="unavailable", reason="benchmarking not requested")
    )

    rows: list[ChannelConfoundingSensitivity] = []
    for ch in channels:
        draws = roi.get(ch)
        div = resolve_channel_divisor(model, ch)
        meta = div.meta
        bounds = tuple(bench.bounds.get(ch, ()))
        ladder = list(priors) if priors is not None else named_prior_ladder()
        ladder += [b.as_prior() for b in bounds if b.status == "ok"]
        contraction = _coefficient_contraction(model, ch)

        if draws is None or not np.size(draws):
            sens = bias_sensitivity_report(
                np.array([]),
                reference=float(meta.reference),
                label=ch,
                reference_label=f"{meta.roi_label} above break-even",
                include_surface=False,
                priors=ladder,
                threshold=threshold,
                units=meta.value_units,
            )
        else:
            sens = bias_sensitivity_report(
                np.asarray(draws, dtype=float),
                reference=float(meta.reference),
                label=ch,
                reference_label=f"{meta.roi_label} above break-even",
                priors=ladder,
                threshold=threshold,
                include_surface=include_surface,
                units=meta.value_units,
                prior_draws=prior_roi.get(ch),
                prior_contraction=contraction,
            )
        rows.append(
            ChannelConfoundingSensitivity(
                channel=ch,
                sensitivity=sens,
                benchmarks=bounds,
                metric_label=meta.roi_label,
                is_monetary=bool(meta.is_monetary),
                prior_contraction=contraction,
            )
        )

    caveat = (
        "This prices the no-unobserved-confounding assumption; it does not test "
        "it. A tipping point says how large a hidden bias would have to be to "
        "change the recommendation — it is an argument that such a confounder is "
        "implausible, never evidence that the effect is causal. Anchor high-stakes "
        "channels with a randomized experiment (mmm_framework.calibration). "
        "Because the model is not re-fitted, this cannot show a confounder being "
        "absorbed differently by the trend, the seasonality or another channel."
    )
    if bench.status != "ok":
        caveat += (
            f" No observed-covariate benchmark was available ({bench.reason}), so "
            "the bias priors below are named guesses rather than measurements and "
            "the tipping point has nothing calibrated to compare against."
        )
    n_unassessable = sum(1 for r in rows if not r.is_assessable)
    if n_unassessable:
        caveat += (
            f" {n_unassessable} channel(s) could not be assessed at all; an empty "
            "fragile list does not cover them."
        )

    return ConfoundingSensitivityReport(
        channels=rows,
        estimand="contribution_roi",
        threshold=float(threshold),
        benchmark=bench,
        caveat=caveat,
    )


# --------------------------------------------------------------------------- #
# the in-graph sweep: an assumed confounder that actually competes for signal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfounderSweepConfig:
    """How hard to push the hypothetical confounder, and how to refit.

    ``strengths`` are partial ``R^2`` values the confounder is assumed to have
    with *both* the treatment and the outcome — the same currency as the
    robustness value, so a sweep point is directly comparable to it.
    """

    strengths: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
    #: Both directions. Positive confounding (demand lifts spend AND sales) is
    #: the MMM's default worry and gives the attenuating arm, but reporting only
    #: that would read as one-directional.
    signs: tuple[int, ...] = (1, -1)
    #: Channels the confounder is assumed to drive. ``None`` means all of them.
    channels: tuple[str, ...] | None = None
    #: AR(1) persistence of the hypothetical confounder. ``None`` matches the
    #: KPI residual's own lag-1 autocorrelation, so it looks like real demand
    #: rather than white noise — a white-noise confounder is trivially absorbed
    #: and would understate the exposure.
    rho: float | None = None
    method: str = "map"
    decision_threshold: float = 1.0
    random_seed: int = 42
    max_draws: int = 200


@dataclass(frozen=True)
class ConfounderSweepPoint:
    """One grid point: what the ROIs look like if the world were like this."""

    strength: float
    sign: int
    roi: dict[str, float]
    delivered_r2_y: float
    delivered_r2_t: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strength": float(self.strength),
            "sign": int(self.sign),
            "roi": {k: float(v) for k, v in self.roi.items()},
            "delivered_r2_y": float(self.delivered_r2_y),
            "delivered_r2_t": {k: float(v) for k, v in self.delivered_r2_t.items()},
        }


@dataclass
class ConfounderSweepResult:
    """The exact counterpart to the post-hoc tipping point.

    The post-hoc analysis re-weights a fitted posterior, so it can widen and
    shift an estimand but never move a coefficient *relative to the controls*.
    Here the confounder is in the graph and the model is re-fitted at each point,
    so the media coefficients genuinely move — including through the nonlinear
    adstock and saturation, and including being partly absorbed by the trend,
    which is itself worth seeing.
    """

    points: list[ConfounderSweepPoint]
    baseline_roi: dict[str, float]
    crossing: dict[str, float | None]
    config: ConfounderSweepConfig
    caveat: str
    status: str = "ok"
    reason: str = ""

    def summary(self) -> pd.DataFrame:
        rows = []
        for ch, base in self.baseline_roi.items():
            cross = self.crossing.get(ch)
            rows.append(
                {
                    "Channel": ch,
                    "ROI (no confounder)": f"{base:.2f}",
                    "Crosses break-even at": (
                        "never in range" if cross is None else f"partial R² {cross:.2f}"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "baseline_roi": {k: float(v) for k, v in self.baseline_roi.items()},
            "crossing": {
                k: (None if v is None else float(v)) for k, v in self.crossing.items()
            },
            "strengths": list(self.config.strengths),
            "signs": list(self.config.signs),
            "decision_threshold": float(self.config.decision_threshold),
            "caveat": self.caveat,
            "status": self.status,
            "reason": self.reason,
        }


def _period_collapse(
    values: np.ndarray, time_idx: np.ndarray, n_periods: int
) -> np.ndarray:
    """Average an obs-axis series onto the period axis.

    The confounder is national — one series over time — so a geo panel must be
    collapsed before anything is measured against it, or a ``G``-geo panel would
    weight the construction ``G``-fold.
    """
    out = np.zeros(n_periods, dtype=float)
    counts = np.zeros(n_periods, dtype=float)
    np.add.at(out, time_idx, np.asarray(values, dtype=float))
    np.add.at(counts, time_idx, 1.0)
    return out / np.maximum(counts, 1.0)


def _adjustment_basis(model: Any) -> np.ndarray:
    """The period-axis span of everything the model already adjusts for.

    A confounder that is a linear function of an observed control is *not*
    unobserved, so only the part orthogonal to this basis counts — the same
    logic the Cinelli–Hazlett bound rests on. Orthogonalizing against it also
    stops the trend, the seasonality and the controls from simply re-fitting to
    absorb the injected confounder and silently nullifying the sweep.
    """
    n_p = int(model.n_periods)
    ti = np.asarray(model.time_idx, dtype=int)
    cols = [np.ones(n_p), np.linspace(0.0, 1.0, n_p)]
    for name, feats in (getattr(model, "seasonality_features", {}) or {}).items():
        F = np.asarray(feats, dtype=float)
        if F.ndim == 2 and F.shape[0] == n_p:
            cols.extend(F[:, j] for j in range(F.shape[1]))
    controls = getattr(model, "X_controls", None)
    if controls is not None and np.size(controls):
        C = np.asarray(controls, dtype=float)
        for j in range(C.shape[1]):
            cols.append(_period_collapse(C[:, j], ti, n_p))
    return np.column_stack(cols)


def _residualize(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
    return v - basis @ coef


def _ar1_series(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(size=n)
    out = np.empty(n)
    out[0] = e[0]
    for k in range(1, n):
        out[k] = rho * out[k - 1] + e[k]
    return out


def build_confounder(
    model: Any,
    *,
    strength: float,
    sign: int = 1,
    channels: Sequence[str] | None = None,
    rho: float | None = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Construct a hypothetical confounder at an assumed strength.

    Returns ``(u_scaled, lambda_y, delivered_r2_t)`` where ``u_scaled`` is the
    period-axis vector to hand :meth:`BayesianMMM.add_latent_confounder` — already
    multiplied by its outcome loading, and already orthogonal to everything the
    model adjusts for.

    The construction is Imbens-style: both associations are **fixed** at the grid
    point rather than estimated. ``U`` is correlated with each named channel's
    *spend residual* (so it is a genuine back-door, not a random common cause
    that the refutation suite already tests), given AR(1) persistence matched to
    the KPI residual so it resembles demand, then orthogonalized against the
    adjustment basis and standardized in numpy — never in the graph, where
    recomputed constants would contaminate counterfactuals.
    """
    n_p = int(model.n_periods)
    ti = np.asarray(model.time_idx, dtype=int)
    rng = np.random.default_rng(random_seed)
    basis = _adjustment_basis(model)

    y_res = _residualize(_period_collapse(model.y, ti, n_p), basis)
    if rho is None:
        if n_p > 2 and y_res.std() > 0:
            rho = float(np.corrcoef(y_res[:-1], y_res[1:])[0, 1])
        else:
            rho = 0.0
    rho = float(np.clip(rho, 0.0, 0.95))

    names = [str(c) for c in (channels or model.channel_names)]
    X = np.asarray(model.X_media_raw, dtype=float)
    idx = {c: i for i, c in enumerate(model.channel_names)}
    spend_res = []
    for c in names:
        if c not in idx:
            continue
        s = _residualize(_period_collapse(X[:, idx[c]], ti, n_p), basis)
        if s.std() > 0:
            spend_res.append(s / s.std())
    driver = np.mean(spend_res, axis=0) if spend_res else np.zeros(n_p)

    noise = _residualize(_ar1_series(n_p, rho, rng), basis)
    if driver.std() > 0:
        driver = _residualize(driver, basis)

    strength = float(np.clip(strength, 0.0, 0.95))
    d_unit = driver / driver.std() if driver.std() > 0 else driver
    n_unit = noise / noise.std() if noise.std() > 0 else noise

    # Per-channel spend residuals, on the period axis, for calibrating the mix.
    targets = []
    for c in names:
        if c not in idx:
            continue
        s = _residualize(_period_collapse(X[:, idx[c]], ti, n_p), basis)
        if s.std() > 0:
            targets.append(s / s.std())

    def _mix(w: float) -> np.ndarray:
        v = _residualize(w * d_unit + math.sqrt(max(1.0 - w * w, 0.0)) * n_unit, basis)
        return (v - v.mean()) / v.std() if v.std() > 0 else v

    def _mean_r2(v: np.ndarray) -> float:
        if not targets or v.std() <= 0:
            return 0.0
        return float(np.mean([np.corrcoef(v, s)[0, 1] ** 2 for s in targets]))

    # `strength` has to MEAN what it says. Mixing at `w = sqrt(strength)` looks
    # right and is not: the driver is an average over the targeted channels, so
    # its correlation with any ONE of them is diluted and the delivered partial
    # R^2 comes out far below the assumed one — measured at 0.01-0.05 against an
    # assumed 0.15 before this was calibrated. `_mean_r2` is monotone in `w`, so
    # bisect until the mean delivered strength matches. Per-channel spread
    # remains (one series cannot hit an exact partial R^2 with several
    # non-orthogonal channels at once) and is reported in `delivered_r2_t`.
    if strength <= 0 or not targets:
        u = _mix(0.0)
    else:
        lo, hi = 0.0, 1.0
        if _mean_r2(_mix(hi)) <= strength:
            u = _mix(hi)
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _mean_r2(_mix(mid)) < strength:
                    lo = mid
                else:
                    hi = mid
            u = _mix(0.5 * (lo + hi))

    # lambda_Y from the assumed outcome-side partial R^2, on the standardized
    # outcome scale the graph's `mu` lives on.
    lam_y = (
        float(sign)
        * float(y_res.std())
        * math.sqrt(strength / max(1.0 - strength, 1e-9))
    )

    delivered = {}
    for c in names:
        if c not in idx:
            continue
        s = _residualize(_period_collapse(X[:, idx[c]], ti, n_p), basis)
        delivered[c] = (
            float(np.corrcoef(u, s)[0, 1] ** 2) if s.std() > 0 and u.std() > 0 else 0.0
        )
    return u * lam_y, lam_y, delivered


def run_confounder_sweep(
    model: Any,
    config: ConfounderSweepConfig | None = None,
) -> ConfounderSweepResult:
    """Re-fit the model with an assumed confounder in the graph, across a grid.

    The exact counterpart to :func:`run_confounding_sensitivity`, and the only
    version that can show a confounder being *absorbed differently* by the trend,
    the seasonality or another channel. Costs one re-fit per grid point.

    Refuses, naming the reason, for anything the construction cannot honestly
    support: extension models (there is no single "the" media coefficient), the
    frequentist paradigm (a ridge fit is not a posterior), a multiplicative
    specification (``mu`` is a log there, so an additive term is a multiplicative
    confounder and the assumed partial ``R^2`` is not the quantity computed), and
    a Gaussian-process trend (flexible enough to absorb any smooth confounder and
    report an infinite tipping point).
    """
    from ..reporting.helpers.measurement import resolve_channel_divisor
    from .backtest import rebuild_like
    from .spec_curve import channel_roi_draws

    config = config or ConfounderSweepConfig()
    if getattr(model, "_trace", None) is None:
        raise ValueError("Model must be fitted (no posterior trace found).")

    def _refuse(reason: str) -> ConfounderSweepResult:
        return ConfounderSweepResult(
            points=[],
            baseline_roi={},
            crossing={},
            config=config,
            caveat="",
            status="refused",
            reason=reason,
        )

    if not isinstance(getattr(model, "channel_names", None), (list, tuple)):
        return _refuse("the model exposes no channel list")
    if getattr(model, "_multiplicative", False):
        return _refuse(
            "a multiplicative specification models log-KPI, so an additive "
            "confounder term is a multiplicative one on the natural scale and the "
            "assumed partial R^2 would not be the quantity computed"
        )
    if not hasattr(model, "add_latent_confounder"):
        return _refuse(
            "this model family has no in-graph confounder hook — the sweep is "
            "defined for the core BayesianMMM only"
        )
    trend_type = getattr(getattr(model, "trend_config", None), "type", None)
    if str(getattr(trend_type, "value", trend_type)) == "gaussian_process":
        return _refuse(
            "a Gaussian-process trend is flexible enough to absorb any smooth "
            "confounder, so the sweep would report robustness that is really the "
            "trend re-fitting"
        )
    if (
        str(getattr(model, "_fit_diagnostics", {}) or {})
        and getattr(model, "inference_family", None) == "frequentist"
    ):
        return _refuse("the sweep is defined for a Bayesian posterior")

    channels = [str(c) for c in model.channel_names]
    references = {
        ch: float(resolve_channel_divisor(model, ch).meta.reference) for ch in channels
    }
    baseline = {
        ch: float(np.mean(v))
        for ch, v in channel_roi_draws(
            model, channels, max_draws=config.max_draws, random_seed=config.random_seed
        ).items()
    }

    points: list[ConfounderSweepPoint] = []
    for sign in config.signs:
        for strength in config.strengths:
            if strength == 0.0 and sign != config.signs[0]:
                continue  # the zero point is the same in both directions
            u_scaled, _lam, delivered = build_confounder(
                model,
                strength=strength,
                sign=sign,
                channels=config.channels,
                rho=config.rho,
                random_seed=config.random_seed,
            )
            clone = rebuild_like(model, model.panel)
            clone.add_latent_confounder(None if strength == 0.0 else u_scaled)
            clone.fit(
                method=config.method,
                random_seed=config.random_seed,
                progressbar=False,
            )
            roi = {
                ch: float(np.mean(v))
                for ch, v in channel_roi_draws(
                    clone,
                    channels,
                    max_draws=config.max_draws,
                    random_seed=config.random_seed,
                ).items()
            }
            y_res_r2 = float(strength)
            points.append(
                ConfounderSweepPoint(
                    strength=float(strength),
                    sign=int(sign),
                    roi=roi,
                    delivered_r2_y=y_res_r2,
                    delivered_r2_t=delivered,
                )
            )

    # Anchor on the zero-strength point when it exists rather than on the
    # original fit: the sweep refits with `config.method` (MAP by default) while
    # the original may be a full NUTS posterior, and comparing across estimators
    # would read as an effect of the confounder.
    zero = next((p for p in points if p.strength == 0.0), None)
    if zero is not None and zero.roi:
        baseline = dict(zero.roi)

    # Where each channel first falls below its own break-even, on the attenuating
    # arm (the direction that matters for an over-credited channel).
    crossing: dict[str, float | None] = {}
    for ch in channels:
        arm = sorted(
            (p for p in points if p.sign == 1 and ch in p.roi),
            key=lambda p: p.strength,
        )
        hit = next(
            (p.strength for p in arm if p.roi[ch] < references.get(ch, 1.0)), None
        )
        crossing[ch] = hit

    caveat = (
        "The confounder is FIXED at each assumed strength, not estimated — that "
        "is what makes this a sensitivity analysis rather than a model of "
        "confounding. Letting the likelihood choose its loading would let any "
        "spend-predictable structure be relabelled as the confounder and shrink "
        "the media coefficients for free. It is orthogonalized against everything "
        "the model already adjusts for, so it represents genuinely UNobserved "
        "confounding; and it is national, so a geo-varying confounder is a "
        "strictly stronger threat this does not cover."
    )
    return ConfounderSweepResult(
        points=points,
        baseline_roi=baseline,
        crossing=crossing,
        config=config,
        caveat=caveat,
    )
