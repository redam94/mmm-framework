"""Within-wave stationarity guard — Bayesian online changepoint detection.

The geo-week likelihood (math page Eq. 7) and the summary-observation bridge
(Eq. 7a) both assume the response regime is **stable within the test
window**: the information-decay clock (Eq. 22) models drift *between* waves,
but a competitor action, a promo, or a seasonality shock landing *mid-wave*
contaminates exactly the between-cell contrast the design is built to
measure. The randomized holdout / center / shut-off cells already provide the
contemporaneous control that cancels *common* shocks (Vaver & Koehler 2011;
Kerman, Wang & Vaver 2017) — what was missing is the **detection and
censoring** step for shocks that do NOT cancel.

This module supplies it:

* :func:`bocd` — Bayesian online changepoint detection (Adams & MacKay 2007)
  on a univariate series, with the Normal–Inverse-Gamma conjugate model
  (unknown mean AND variance -> Student-t predictive), returning the
  run-length posterior and the per-step changepoint probability.
* :func:`wave_stationarity_check` — the wave-level guard: builds each design
  cell's **treatment-minus-control** per-period contrast series (holdout geos
  when present, else the center cell) and runs :func:`bocd` on each. Under
  stationarity every contrast is constant + noise; a mid-wave regime change
  that hits treatment differently from control shows up as a level shift.
  Common shocks (a national demand swing) cancel in the contrast and do NOT
  fire the alarm — by construction, the same reason the design is causal.
* :func:`censor_periods` — drop the flagged periods from a wave dict before
  :meth:`~mmm_framework.continuous_learning.loop.LearningState.ingest`, so a
  contaminated segment is excised rather than folded into the posterior.
  (The alternative, per the geo-experiment literature, is to extend or repeat
  the wave.)

The check is retrospective (run at readout, before ingesting) but the
detector itself is online, so it can also monitor a live wave period by
period. ``LearningState.ingest(wave, check_stationarity=True)`` runs the
guard automatically and warns + records the report on the state.

References: Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
(arXiv:0710.3742); Fearnhead & Liu (2007) JRSS-B 69(4); Brodersen et al.
(2015) *Annals of Applied Statistics* 9(1) (the counterfactual-divergence
framing); Vaver & Koehler (2011) / Kerman, Wang & Vaver (2017), Google
Research (control-based common-shock cancellation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as _student_t

__all__ = [
    "BocdResult",
    "StationarityReport",
    "bocd",
    "wave_stationarity_check",
    "censor_periods",
]


@dataclass
class BocdResult:
    """Run-length posterior summary from :func:`bocd`.

    ``cp_prob[t]`` is the posterior probability that a **sustained** restart
    happened one step ago — ``P(r_t = 2 | x_{1:t})``, the hypothesis that the
    current regime contains exactly the last two observations. Two naive
    alternatives fail: ``P(r_t = 0)`` is *identically* the hazard under a
    constant-hazard BOCD (the changepoint and growth branches share the same
    predictive weights, so their mass split is ``H : 1-H`` regardless of the
    data), and ``P(r_t = 1)`` fires on any single-point outlier (the fresh
    run's wide prior predictive absorbs a spike that the tight long-run model
    rejects). Requiring the restarted run to explain **two consecutive**
    points is the minimum evidence for a regime rather than a blip — a spike
    reverts at the next step and the long run recovers the mass. The price is
    a one-period detection lag: a flag at ``t`` means the break onset was at
    ``t - 1``. ``run_length_map[t]`` is the MAP run length after step ``t``;
    ``changepoints`` are the steps (``t >= 2``) whose ``cp_prob`` cleared the
    threshold.
    """

    cp_prob: np.ndarray
    run_length_map: np.ndarray
    changepoints: list[int]
    threshold: float


def bocd(
    x: np.ndarray,
    *,
    hazard: float = 0.04,
    mu0: float | None = None,
    kappa0: float = 1.0,
    alpha0: float = 3.0,
    beta0: float | None = None,
    threshold: float = 0.5,
) -> BocdResult:
    """Bayesian online changepoint detection (Adams & MacKay 2007).

    Normal observations with unknown mean and variance: the conjugate
    Normal–Inverse-Gamma model, whose posterior predictive is a Student-t —
    so the detector needs no known noise scale. ``hazard`` is the constant
    per-step changepoint prior ``P(r_t = 0)`` (``1/hazard`` = expected run
    length; the default expects a run of ~25 periods, conservative for a
    10-week wave). ``mu0``/``beta0`` default empirically: the series median,
    and a noise variance from the MAD of the **first differences** (robust
    to the very level shift being tested for — a full-series spread estimate
    would fold the shift into the noise prior and mask it, while differences
    see only the step at the break). An empirical-Bayes convenience that is
    appropriate for the retrospective wave check; pass explicit priors for
    genuinely online use.

    Returns a :class:`BocdResult`; ``changepoints`` flags every step ``t >= 2``
    with ``P(a sustained restart happened one step ago) > threshold`` — see
    :class:`BocdResult` for why the flag is ``P(r_t = 2)``, not
    ``P(r_t = 0)``. A flag at ``t`` means the break onset was at ``t - 1``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        return BocdResult(np.zeros(0), np.zeros(0, dtype=int), [], threshold)
    if not 0.0 < hazard < 1.0:
        raise ValueError(f"hazard must be in (0, 1), got {hazard}")
    if mu0 is None:
        mu0 = float(np.median(x))
    if beta0 is None:
        if n >= 3:
            d = np.diff(x)
            mad = float(np.median(np.abs(d - np.median(d))))
            s2 = (1.4826 * mad) ** 2 / 2.0  # var(x_t - x_{t-1}) = 2 var(x)
        else:
            mad = float(np.median(np.abs(x - np.median(x))))
            s2 = (1.4826 * mad) ** 2
        beta0 = float(alpha0) * max(s2, 1e-12)

    log_h = np.log(hazard)
    log_1mh = np.log1p(-hazard)

    mu = np.array([mu0])
    kappa = np.array([float(kappa0)])
    alpha = np.array([float(alpha0)])
    beta = np.array([float(beta0)])
    log_rl = np.array([0.0])  # log P(run length | x_{1:t}), length t+1

    cp_prob = np.zeros(n)
    rl_map = np.zeros(n, dtype=int)
    for t in range(n):
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        log_pred = _student_t.logpdf(x[t], df=2.0 * alpha, loc=mu, scale=scale)
        log_growth = log_rl + log_pred + log_1mh
        log_cp = logsumexp(log_rl + log_pred + log_h)
        log_rl = np.concatenate([[log_cp], log_growth])
        log_rl = log_rl - logsumexp(log_rl)
        cp_prob[t] = float(np.exp(log_rl[2])) if log_rl.size > 2 else 0.0
        rl_map[t] = int(np.argmax(log_rl))
        # conjugate NIG update; run length 0 restarts from the prior
        mu_upd = (kappa * mu + x[t]) / (kappa + 1.0)
        beta_upd = beta + kappa * (x[t] - mu) ** 2 / (2.0 * (kappa + 1.0))
        mu = np.concatenate([[mu0], mu_upd])
        kappa = np.concatenate([[float(kappa0)], kappa + 1.0])
        alpha = np.concatenate([[float(alpha0)], alpha + 0.5])
        beta = np.concatenate([[float(beta0)], beta_upd])

    changepoints = [t for t in range(2, n) if cp_prob[t] > threshold]
    return BocdResult(cp_prob, rl_map, changepoints, threshold)


@dataclass
class StationarityReport:
    """Result of :func:`wave_stationarity_check` (JSON-safe via
    :meth:`to_dict`)."""

    flagged: bool
    break_periods: list[int]  # global period_idx values with a detected break
    cell_breaks: dict[int, list[int]]  # design-cell -> flagged period values
    control: str  # "holdout" or "center-cell"
    n_test_periods: int
    n_series: int
    threshold: float
    max_cp_prob: float
    note: str = ""
    cp_prob: dict[int, list[float]] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cell_breaks"] = {str(k): v for k, v in d["cell_breaks"].items()}
        d["cp_prob"] = {str(k): v for k, v in d["cp_prob"].items()}
        return d


def _test_period_mask(
    spend: np.ndarray, period_idx: np.ndarray, periods: np.ndarray
) -> np.ndarray:
    """Which periods belong to the test window?

    A pre-period row block has every geo at the same allocation (the
    status-quo center), so a period with a single distinct spend row is
    treated as pre-period and skipped — no ``t_pre`` bookkeeping needed.
    """
    is_test = np.zeros(periods.size, dtype=bool)
    for i, p in enumerate(periods):
        rows = spend[period_idx == p]
        is_test[i] = np.unique(rows, axis=0).shape[0] > 1
    return is_test


def wave_stationarity_check(
    wave_data: dict[str, Any],
    *,
    hazard: float = 0.04,
    threshold: float = 0.5,
    min_geos_per_cell: int = 1,
) -> StationarityReport:
    """Detect a mid-wave regime change in a designed wave's contrasts.

    For every design cell, build the per-period **treatment-minus-control**
    series (mean outcome of the cell's geos minus the control's — holdout
    geos ``cell_idx == -1`` when the wave has them, else the center cell,
    design row 0) and run :func:`bocd` on it. Common shocks cancel in the
    contrast; a shock that hits treatment differently from control — the kind
    that biases the readout — appears as a level shift and fires the flag.

    ``wave_data`` needs the simulate/ingest contract keys ``spend``,
    ``geo_idx``, ``y``, ``period_idx`` and the per-geo ``cell_idx``. Raises
    ``ValueError`` when a required key is missing (the guard cannot run
    blind). Returns a :class:`StationarityReport`; when ``flagged``, either
    :func:`censor_periods` the affected periods before ingesting or extend /
    repeat the wave.
    """
    missing = [
        k
        for k in ("spend", "geo_idx", "y", "period_idx", "cell_idx")
        if wave_data.get(k) is None
    ]
    if missing:
        raise ValueError(
            f"wave_stationarity_check needs wave keys {missing} — the guard "
            "requires the designed wave's row-level panel (spend/geo_idx/y/"
            "period_idx) and the per-geo cell assignment (cell_idx)"
        )
    spend = np.asarray(wave_data["spend"], dtype=float)
    geo_idx = np.asarray(wave_data["geo_idx"], dtype=int)
    y = np.asarray(wave_data["y"], dtype=float)
    period_idx = np.asarray(wave_data["period_idx"], dtype=int)
    cell_idx = np.asarray(wave_data["cell_idx"], dtype=int)

    row_cell = cell_idx[geo_idx]
    periods = np.unique(period_idx)
    is_test = _test_period_mask(spend, period_idx, periods)
    test_periods = periods[is_test]
    if test_periods.size < 4:
        return StationarityReport(
            flagged=False,
            break_periods=[],
            cell_breaks={},
            control="none",
            n_test_periods=int(test_periods.size),
            n_series=0,
            threshold=threshold,
            max_cp_prob=0.0,
            note="fewer than 4 test periods — too short for changepoint "
            "detection; guard skipped",
        )

    has_holdout = bool(np.any(cell_idx == -1))
    control_cell = -1 if has_holdout else 0
    control = "holdout" if has_holdout else "center-cell"

    cells = [int(c) for c in np.unique(cell_idx) if int(c) != control_cell]
    cell_breaks: dict[int, list[int]] = {}
    cp_traces: dict[int, list[float]] = {}
    max_cp = 0.0
    n_series = 0
    for c in cells:
        series = np.empty(test_periods.size)
        ok = True
        for i, p in enumerate(test_periods):
            m_c = (row_cell == c) & (period_idx == p)
            m_0 = (row_cell == control_cell) & (period_idx == p)
            if m_c.sum() < min_geos_per_cell or m_0.sum() < min_geos_per_cell:
                ok = False
                break
            series[i] = float(y[m_c].mean() - y[m_0].mean())
        if not ok:
            continue
        n_series += 1
        res = bocd(series, hazard=hazard, threshold=threshold)
        cp_traces[c] = [float(p) for p in res.cp_prob]
        max_cp = max(max_cp, float(res.cp_prob[2:].max(initial=0.0)))
        if res.changepoints:
            cell_breaks[c] = [int(test_periods[t]) for t in res.changepoints]

    break_periods = sorted({p for ps in cell_breaks.values() for p in ps})
    return StationarityReport(
        flagged=bool(break_periods),
        break_periods=break_periods,
        cell_breaks=cell_breaks,
        control=control,
        n_test_periods=int(test_periods.size),
        n_series=n_series,
        threshold=threshold,
        max_cp_prob=max_cp,
        cp_prob=cp_traces,
    )


def censor_periods(
    wave_data: dict[str, Any], periods: list[int] | np.ndarray
) -> dict[str, Any]:
    """Drop every row of the given (global) periods from a wave dict.

    Returns a shallow copy with ``spend``/``geo_idx``/``y``/``period_idx``
    filtered row-wise (and any other key left untouched) — feed the result to
    :meth:`LearningState.ingest` in place of the contaminated wave. Censoring
    from a detected break **onward** is the conservative choice when the
    regime did not revert (a level shift, not a spike); censoring just the
    flagged periods suits a transient.
    """
    period_idx = np.asarray(wave_data["period_idx"], dtype=int)
    keep = ~np.isin(period_idx, np.asarray(list(periods), dtype=int))
    out = dict(wave_data)
    out["spend"] = np.asarray(wave_data["spend"], dtype=float)[keep]
    out["geo_idx"] = np.asarray(wave_data["geo_idx"], dtype=int)[keep]
    out["y"] = np.asarray(wave_data["y"], dtype=float)[keep]
    out["period_idx"] = period_idx[keep]
    return out
