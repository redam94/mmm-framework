"""Choosing the transforms the frequentist path holds fixed.

The frequentist estimator fixes each channel's adstock and saturation and solves
the resulting linear problem, so *something* has to choose them. This is that
something — and it is the layer that separates the approach from
``fit(method="map")``, where the transforms are estimated jointly with the
coefficients under their priors.

**In-sample fit cannot select these.** Adstock and saturation trade off against
each other and against the baseline: a long-carryover / weak-saturation fit and a
short-carryover / strong-saturation fit are nearly indistinguishable in-sample
(``transforms/adstock.py`` documents the same equifinality for the Bayesian
path). Maximizing in-sample R² picks whatever bends hardest to the noise and the
resulting ROI is confidently wrong. The criterion here is therefore
**rolling-origin out-of-sample error**, evaluated with the same machinery
``validation/backtest.py`` uses to grade the Bayesian path, so the two are
measured the same way.

What is reused, and what is not
-------------------------------
``run_backtest`` is **not** pluggable — ``_clone_for_prefix`` reconstructs the
model's own class and calls ``.fit(draws=, tune=, chains=)``. The reusable pieces
are the pure ones, and they are reused directly: :func:`rolling_origins` for the
splits, :func:`_slice_panel_prefix` for the geo-correct panel prefix, and
:func:`_point_metrics` for the error measures.

The honest caveat, carried in the result
----------------------------------------
This selects on **predictive skill**, and predictive skill is not the causal
estimand. A specification that forecasts better can attribute worse — the same
objection raised against LOO-stacking in the v1.1.0 ``spec_curve`` fix. That is
not a reason to select in-sample instead; it is a reason to say so wherever the
winner's ROI is read, which is why :attr:`SearchResult.caveat` exists and is
meant to be rendered, not just logged.

For the same reason :attr:`SearchResult.candidates` carries the **entire**
evaluated set rather than the winner alone. The spread of ROI across
near-optimal candidates is the honest uncertainty about the transforms, and it
is what #186's bootstrap needs in order to stop pretending the selected
``(alpha, lambda)`` was known in advance.

What this recovers, measured
----------------------------
Graded against ``synth.dgp.make_clean``, whose planted truth this model can
represent exactly (``tests/frequentist/test_search.py``):

* **Carryover is recovered better than chance.** Mean absolute error on
  ``alpha`` is ~0.17 against ~0.26 for an uninformed draw from the same bounds.
* **Saturation is not identified by this criterion.** Within the set of
  candidates scoring inside 10% of the best, TV's ``sat_lam`` ranges over
  roughly **0.16 to 7.8** — nearly the whole search range — while the planted
  value is 1.6. Prediction cannot separate them.
* **Searching harder moves the winner away from the truth.** At budget 1000 the
  best candidate scores an out-of-sample MAPE of 0.0328 against the *planted
  parameters'* own 0.0337 — it forecasts better than the truth while sitting
  further from it in ``lambda``.

That last point is the caveat above, measured rather than asserted. Read the
winner as one draw from :meth:`SearchResult.spread`, not as an estimate of the
transforms.

Containment, which is not identification
----------------------------------------
None of this is particular to the frequentist path or to this codebase. Jin et
al. (Google, 2017) showed the Hill parameters are "essentially unidentifiable in
some scenarios" and that the half-saturation posterior median averages roughly
twice the truth at two years of weekly data, concentrating only at *sixty*.
Dew et al. (2024) show predictive fit — cross-validation included — cannot
arbitrate between observationally equivalent response specifications. So no
production MMM identifies saturation from the sales likelihood; each one
accommodates the fact.

The accommodation that transfers here is **bounding in data units**, which is
what :data:`HALF_SATURATION_FRACTION` does. Measured on ``make_clean`` over four
seeds, moving from an absolute ``lam`` bound to the half-saturation fraction cuts
mean absolute ``lam`` error from **2.08 to 0.42** — a 5x improvement — at no cost
in out-of-sample score, and the run-to-run spread collapses from 1.77–2.42 to
0.39–0.44.

**It does not identify the parameter.** Inside the new bound the near-optimal set
still spans essentially the whole window (0.70–2.28 of a 0.69–2.31 range): the
criterion still cannot order ``lam``, it simply can no longer propose a curve no
analyst would entertain. What genuinely identifies curvature is **dose spread** —
a rank condition on the spend design, not a property of any estimator. A single
lift test at one spend level yields one equation in two unknowns and leaves
``lam`` unidentified at any sample size; ``planning/identification.py`` reaches
the same conclusion from a Laplace bound and refuses to claim identification
below three in-support levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.enums import AdstockType, SaturationType
from .design import UnsupportedModelError, build_design_matrix
from .ridge import fit_ridge

if TYPE_CHECKING:  # pragma: no cover
    from ..data_loader import PanelDataset

__all__ = [
    "ADSTOCK_BOUNDS",
    "SATURATION_BOUNDS",
    "SearchCandidate",
    "SearchResult",
    "search_transforms",
]

#: Search bounds per adstock family, keyed as the graph names the parameters.
#: ``alpha`` stops short of 1.0 because a unit-decay geometric kernel has
#: infinite memory and is not identifiable from a finite panel.
ADSTOCK_BOUNDS: dict[AdstockType, dict[str, tuple[float, float]]] = {
    AdstockType.GEOMETRIC: {"alpha": (0.0, 0.95)},
    AdstockType.DELAYED: {"alpha": (0.0, 0.95), "theta": (0.0, 6.0)},
    AdstockType.WEIBULL: {"shape": (0.3, 5.0), "scale": (0.5, 12.0)},
    AdstockType.NONE: {},
}

#: Half-saturation point as a fraction of the channel's observed maximum. This
#: is the interval every production MMM bounds saturation over, and bounding in
#: *data units* rather than in the parameter's own units is the single change
#: that most improves recovery (see the module docstring's measurement). Robyn
#: hard-codes ``inflexion = max(x) * gamma`` and recommends ``gamma`` in
#: ``[0.3, 1.0]``; Meridian scales so its ``ec`` prior is centred on the median
#: non-zero spend. Media here is already normalized by the channel max, so this
#: fraction *is* the half-saturation point on the modelled scale.
HALF_SATURATION_FRACTION: tuple[float, float] = (0.3, 1.0)

#: Search bounds per saturation family, on the normalized (~[0, 1]) media scale.
#: Every entry is derived from :data:`HALF_SATURATION_FRACTION` so the families
#: agree about where the curve is allowed to bend:
#:
#: * ``logistic`` — ``sat(x) = 1 - exp(-lam*x)`` has its half-saturation point at
#:   ``ln(2)/lam``, so a fraction ``g`` maps to ``lam = ln(2)/g``. The resulting
#:   ``[0.69, 2.31]`` is a 3.3x window; the absolute ``[0.1, 8.0]`` it replaced
#:   was 80x and let the search wander to curves no analyst would entertain.
#: * ``michaelis_menten`` / ``tanh`` / ``hill`` — ``sat_half`` already *is* the
#:   half-saturation point, so the fraction applies directly.
#: * ``root`` — has no half-saturation point (no asymptote); the exponent's own
#:   concavity range is the natural bound.
#:
#: The Hill *slope* stays free but narrow. Meridian fixes it outright
#: (``Deterministic(1)``, "difficult to learn because of identifiability
#: reasons"); this keeps it estimable while refusing the extremes.
SATURATION_BOUNDS: dict[SaturationType, dict[str, tuple[float, float]]] = {
    SaturationType.LOGISTIC: {
        "sat_lam": (
            float(np.log(2) / HALF_SATURATION_FRACTION[1]),
            float(np.log(2) / HALF_SATURATION_FRACTION[0]),
        )
    },
    SaturationType.HILL: {
        "sat_half": HALF_SATURATION_FRACTION,
        "sat_slope": (0.5, 3.0),
    },
    SaturationType.MICHAELIS_MENTEN: {"sat_half": HALF_SATURATION_FRACTION},
    SaturationType.TANH: {"sat_half": HALF_SATURATION_FRACTION},
    SaturationType.ROOT: {"sat_exponent": (0.1, 1.0)},
    SaturationType.NONE: {},
}

#: Penalty grid searched jointly with the transforms. Selecting the penalty by the
#: same out-of-sample criterion is deliberate: an in-sample rule (AIC, GCV) would
#: reintroduce exactly the bias the out-of-sample criterion exists to avoid.
DEFAULT_PENALTIES: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)

CAVEAT = (
    "Transforms were selected by out-of-sample predictive error "
    "({criterion}), not by a causal criterion. A specification that predicts "
    "better can attribute worse, so ROI read off the winning candidate is "
    "conditional on this selection. The spread across near-optimal candidates "
    "(SearchResult.spread) is the honest uncertainty about the transforms."
)


@dataclass(frozen=True)
class SearchCandidate:
    """One evaluated ``(adstock, saturation, penalty)`` point.

    Attributes:
        alpha: Per-channel adstock parameters.
        lam: Per-channel saturation parameters.
        penalty: The ridge penalty selected for this transform point.
        score: The out-of-sample criterion, **lower is better**.
        fold_scores: The per-origin criterion values behind ``score``.
        metrics: All point metrics at the selected penalty (mape / smape / rmse /
            mae / bias), averaged over origins.
    """

    alpha: dict[str, dict[str, float]]
    lam: dict[str, dict[str, float]]
    penalty: float
    score: float
    fold_scores: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """The outcome of a transform search, including everything it rejected.

    Attributes:
        best: The winning candidate.
        candidates: **Every** evaluated candidate, best first. Deliberately not
            just the winner — see the module docstring.
        criterion: The selection criterion's name.
        origins: The rolling-origin training cutoffs used.
        horizon: Forecast length per origin, in periods.
        strategy: ``"random"`` or ``"grid"``.
        seed: The seed the candidate draw used, so a search is reproducible.
        caveat: Plain-language statement of what the criterion does and does not
            justify. Intended to be **rendered** wherever the fit is reported.
    """

    best: SearchCandidate
    candidates: list[SearchCandidate]
    criterion: str
    origins: list[int]
    horizon: int
    strategy: str
    seed: int
    caveat: str

    def top(self, k: int = 5) -> list[SearchCandidate]:
        """The ``k`` best candidates."""
        return self.candidates[:k]

    def spread(self, within: float = 0.05) -> list[SearchCandidate]:
        """Candidates scoring within ``within`` (relative) of the best.

        These are the specifications the data cannot distinguish. Their
        disagreement — not the winner's interval — is the honest measure of how
        much the transform choice was determined by evidence.
        """
        if not np.isfinite(self.best.score) or self.best.score <= 0:
            return [self.best]
        limit = self.best.score * (1.0 + within)
        return [c for c in self.candidates if c.score <= limit]


def _sample_params(
    bounds: dict[str, tuple[float, float]], rng: np.random.Generator
) -> dict[str, float]:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in bounds.items()}


def _candidate_points(
    mmm: Any, budget: int, strategy: str, rng: np.random.Generator
) -> list[tuple[dict, dict]]:
    """Draw ``budget`` (adstock, saturation) points inside the family bounds."""
    a_bounds, s_bounds = {}, {}
    for ch in mmm.channel_names:
        a_bounds[ch] = ADSTOCK_BOUNDS[mmm._get_adstock_config(ch).type]
        s_bounds[ch] = SATURATION_BOUNDS[mmm._get_saturation_config(ch).type]

    if strategy == "grid":
        # A full product over channels x parameters explodes combinatorially, so
        # the grid sweeps every parameter together along one shared fraction of
        # its own range. That traces a diagonal through the space rather than
        # covering it -- adequate for a small, well-understood problem and a
        # reproducible smoke test, but "random" is the default for a reason.
        points = []
        for i in range(budget):
            frac = i / max(budget - 1, 1)
            at = lambda b: float(b[0] + frac * (b[1] - b[0]))  # noqa: E731
            points.append(
                (
                    {
                        ch: {k: at(b) for k, b in a_bounds[ch].items()}
                        for ch in mmm.channel_names
                    },
                    {
                        ch: {k: at(b) for k, b in s_bounds[ch].items()}
                        for ch in mmm.channel_names
                    },
                )
            )
        return points

    return [
        (
            {ch: _sample_params(a_bounds[ch], rng) for ch in mmm.channel_names},
            {ch: _sample_params(s_bounds[ch], rng) for ch in mmm.channel_names},
        )
        for _ in range(budget)
    ]


def search_transforms(
    panel: "PanelDataset",
    *,
    model_config: Any,
    trend_config: Any,
    objective: str = "mape",
    budget: int = 256,
    strategy: str = "random",
    penalties: "tuple[float, ...]" = DEFAULT_PENALTIES,
    min_train_size: int | None = None,
    horizon: int = 13,
    step: int | None = None,
    max_origins: int | None = 4,
    seed: int = 0,
) -> SearchResult:
    """Search per-channel (adstock, saturation) by rolling-origin out-of-sample error.

    Args:
        panel: The panel to fit and score on.
        model_config: The :class:`ModelConfig` the fit would use.
        trend_config: The :class:`TrendConfig` the fit would use. Must be
            ``none`` or ``linear`` — the spline and piecewise bases do not
            extrapolate past the fitted period range, so they cannot be scored
            out of time.
        objective: Which of ``_point_metrics``' measures to minimize —
            ``mape``, ``smape``, ``rmse``, ``mae`` or ``bias``.
        budget: Number of transform points to evaluate. The default of 256
            costs roughly two seconds on a 156-week national panel. Lower
            budgets are demonstrably under-powered — at 60 the winner does not
            even match the planted truth's own out-of-sample score.
        strategy: ``"random"`` (default) or ``"grid"``. The interface admits an
            evolutionary strategy later without a signature change.
        penalties: Ridge penalties searched jointly with the transforms, by the
            same out-of-sample criterion.
        min_train_size: Periods in the first training window. Defaults to
            ``max(2, n_periods - horizon * (max_origins or 1))``, so a short
            panel still yields origins instead of raising.
        horizon: Forecast length per origin, in periods.
        step: Spacing between origins. ``None`` uses ``horizon``.
        max_origins: Cap on refits. The default of 4 keeps a search tractable;
            ``None`` uses every origin the data allows.
        seed: Seed for the candidate draw.

    Returns:
        The :class:`SearchResult`, carrying every evaluated candidate.

    Raises:
        ValueError: If the panel is too short to yield a single origin, or
            ``objective`` is not a known metric.
        UnsupportedModelError: If the model configuration is not linear given
            fixed transforms, or the trend cannot be extrapolated.
    """
    # Imported lazily: `validation.backtest` imports this package's numpy
    # transforms, so a module-level import here would close the cycle
    # (frequentist.__init__ -> search -> validation.backtest -> frequentist).
    # `frequentist` is the lower-level package and must not reach up eagerly.
    from ..model.base import BayesianMMM
    from ..validation.backtest import (
        _point_metrics,
        _slice_panel_prefix,
        rolling_origins,
    )

    if objective not in {"mape", "smape", "rmse", "mae", "bias"}:
        raise ValueError(
            f"objective {objective!r} is not one of mape/smape/rmse/mae/bias"
        )

    mmm = BayesianMMM(panel, model_config, trend_config)
    trend_type = str(getattr(trend_config.type, "value", trend_config.type))
    if trend_type not in ("none", "linear"):
        raise UnsupportedModelError(
            f"Transform search with a {trend_type} trend",
            "the basis does not extrapolate past the fitted period range, so "
            "out-of-time scoring is undefined; use trend 'none' or 'linear'",
        )

    n_periods = mmm.n_periods
    if min_train_size is None:
        min_train_size = max(2, n_periods - horizon * (max_origins or 1))
    origins = rolling_origins(
        n_periods,
        min_train_size=min_train_size,
        horizon=horizon,
        step=step,
        max_origins=max_origins,
    )
    if not origins:
        raise ValueError(
            f"panel has {n_periods} periods, too few for horizon={horizon} and "
            f"min_train_size={min_train_size} — no rolling origin fits"
        )

    rng = np.random.default_rng(seed)
    points = _candidate_points(mmm, budget, strategy, rng)

    # One prefix panel per origin, reused across every candidate.
    prefixes = [(T, _slice_panel_prefix(panel, T)) for T in origins]
    y_raw = np.asarray(mmm.y_raw, dtype=float)
    # Rows are period-major, so select by PERIOD rather than by row index: on a
    # geo panel every cell in the horizon window must be scored.
    ti_all = np.asarray(mmm.time_idx)

    evaluated: list[SearchCandidate] = []
    for alpha, lam in points:
        per_penalty: dict[float, list[dict[str, float]]] = {p: [] for p in penalties}
        for T, prefix in prefixes:
            try:
                design = build_design_matrix(
                    prefix,
                    alpha,
                    lam,
                    model_config=model_config,
                    trend_config=trend_config,
                    evaluate_panel=panel,
                )
            except (UnsupportedModelError, ValueError, KeyError):
                raise
            train = ti_all < T
            test = (ti_all >= T) & (ti_all < min(T + horizon, n_periods))
            if not test.any() or not train.any():
                continue
            for p in penalties:
                fit = _fit_on(design, train, p)
                pred_std = design.X[test] @ fit.theta
                pred = pred_std * design.scaling["y_std"] + design.scaling["y_mean"]
                per_penalty[p].append(_point_metrics(y_raw[test], pred))

        scored = []
        for p, folds in per_penalty.items():
            if not folds:
                continue
            mean = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
            scored.append(
                (abs(mean[objective]), p, mean, [abs(f[objective]) for f in folds])
            )
        if not scored:
            continue
        score, best_p, mean, fold_scores = min(scored, key=lambda r: r[0])
        evaluated.append(
            SearchCandidate(
                alpha=alpha,
                lam=lam,
                penalty=float(best_p),
                score=float(score),
                fold_scores=fold_scores,
                metrics=mean,
            )
        )

    if not evaluated:
        raise ValueError(
            "no candidate could be scored — every rolling origin produced an "
            "empty training or test window"
        )

    evaluated.sort(key=lambda c: c.score)
    return SearchResult(
        best=evaluated[0],
        candidates=evaluated,
        criterion=f"rolling_origin_{objective}",
        origins=list(origins),
        horizon=horizon,
        strategy=strategy,
        seed=seed,
        caveat=CAVEAT.format(criterion=f"rolling-origin {objective.upper()}"),
    )


def _fit_on(design, rows, penalty):
    """Ridge on a row subset, keeping the design's own penalty mask."""
    from .design import DesignMatrix

    sub = DesignMatrix(
        X=design.X[rows],
        y=design.y[rows],
        columns=design.columns,
        blocks=design.blocks,
        penalize=design.penalize,
        param_map=design.param_map,
        roi_scale=design.roi_scale,
        scaling=design.scaling,
    )
    return fit_ridge(sub, penalty=penalty)
