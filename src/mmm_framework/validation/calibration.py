"""Inference-calibration verification: LOO-PIT (predictive) + SBC (structural).

The framework's headline claim is *genuine uncertainty quantification* — that the
posterior intervals it reports have nominal coverage. This module machine-verifies
that claim instead of merely asserting it:

* :func:`loo_pit_check` — leave-one-out probability-integral-transform. From an
  existing fit's posterior-predictive draws, the LOO-PIT values should be
  Uniform(0, 1) if the predictive distribution is calibrated. Cheap.

* :func:`simulation_based_calibration` — SBC (Talts et al. 2018). Draw "true"
  parameters from the prior, simulate data, fit, and record the rank of each true
  value within its posterior draws. Under a correctly-implemented inference engine
  those ranks are uniform. This validates the *inference engine itself*, on data
  generated from the model's own prior — the strongest calibration check. It is
  EXPENSIVE (one fit per simulation), so it is a verification tool, not a per-fit
  step. The harness is model-agnostic (callbacks for prior/simulate/fit), so it
  can be unit-tested on a fast conjugate model and applied to the MMM offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# LOO-PIT (predictive calibration)
# ---------------------------------------------------------------------------
@dataclass
class LooPitResult:
    pit: np.ndarray
    ks_stat: float
    ks_pvalue: float
    calibrated: bool
    n: int

    def summary(self) -> str:
        verdict = "calibrated" if self.calibrated else "MISCALIBRATED"
        return (
            f"LOO-PIT {verdict}: KS={self.ks_stat:.3f}, p={self.ks_pvalue:.3f} "
            f"(n={self.n}); uniform PIT ⇒ calibrated predictive distribution."
        )


def loo_pit_check(
    *,
    y: np.ndarray,
    y_hat: np.ndarray,
    log_weights: np.ndarray | None = None,
    alpha: float = 0.05,
) -> LooPitResult:
    """LOO-PIT calibration check from observed values + posterior-predictive draws.

    Parameters
    ----------
    y : array (n_obs,)
        Observed outcomes.
    y_hat : array (n_obs, n_samples) or (n_samples, n_obs)
        Posterior-predictive draws (orientation auto-detected).
    log_weights : array, optional
        PSIS log-weights aligned to ``y_hat``. If omitted, uniform weights are
        used (this reduces LOO-PIT to ordinary PIT — still a useful check).
    alpha : float
        Significance level for the KS uniformity test. ``calibrated`` is
        ``ks_pvalue > alpha``.
    """
    from scipy import stats

    y = np.asarray(y, dtype=float).ravel()
    y_hat = np.asarray(y_hat, dtype=float)
    n_obs = y.shape[0]

    # Normalize y_hat to (n_obs, n_samples).
    if y_hat.ndim != 2:
        raise ValueError(f"y_hat must be 2-D, got shape {y_hat.shape}")
    if y_hat.shape[0] == n_obs and y_hat.shape[1] != n_obs:
        yh = y_hat
    elif y_hat.shape[1] == n_obs:
        yh = np.moveaxis(y_hat, -1, 0)
    elif y_hat.shape[0] == n_obs:
        yh = y_hat  # square; assume (n_obs, n_samples)
    else:
        raise ValueError(
            f"y_hat shape {y_hat.shape} is incompatible with y length {n_obs}"
        )

    if log_weights is None:
        lw = np.full(yh.shape, -np.log(yh.shape[1]))
    else:
        lw = np.asarray(log_weights, dtype=float)
        if lw.shape != yh.shape:
            lw = np.moveaxis(lw, -1, 0)
        if lw.shape != yh.shape:
            raise ValueError("log_weights shape does not match y_hat")

    # arviz 1.x removed the array-based ``loo_pit(y=, y_hat=)`` API (the new
    # one wants a full posterior-predictive DataTree). The weighted PIT is
    # direct numpy — the same formula legacy arviz used: PIT_i =
    # Σ_s w_is · 1[y_hat_is <= y_i], with per-observation weights normalized
    # from the PSIS log-weights (softmax; uniform lw reduces to ordinary PIT).
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    pit = np.asarray((w * (yh <= y[:, None])).sum(axis=1), dtype=float)
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        raise ValueError("No finite LOO-PIT values computed")
    ks = stats.kstest(pit, "uniform")
    return LooPitResult(
        pit=pit,
        ks_stat=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        calibrated=bool(ks.pvalue > alpha),
        n=int(pit.size),
    )


# ---------------------------------------------------------------------------
# Simulation-based calibration (structural calibration of the inference engine)
# ---------------------------------------------------------------------------
@dataclass
class SBCResult:
    param_names: list[str]
    n_sims: int
    normalized_ranks: dict[str, np.ndarray] = field(default_factory=dict)
    ks_pvalue: dict[str, float] = field(default_factory=dict)
    calibrated: dict[str, bool] = field(default_factory=dict)

    @property
    def all_calibrated(self) -> bool:
        return bool(self.calibrated) and all(self.calibrated.values())

    def summary(self) -> str:
        lines = [f"SBC over {self.n_sims} simulations:"]
        for k in self.param_names:
            verdict = "ok" if self.calibrated.get(k) else "MISCALIBRATED"
            lines.append(
                f"  - {k}: KS p={self.ks_pvalue.get(k, float('nan')):.3f} [{verdict}]"
            )
        return "\n".join(lines)


def simulation_based_calibration(
    sample_prior: Callable[[np.random.Generator], dict],
    simulate: Callable[[dict, np.random.Generator], object],
    fit: Callable[[object, np.random.Generator], dict],
    *,
    n_sims: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> SBCResult:
    """Simulation-based calibration (Talts et al. 2018), model-agnostic.

    For each of ``n_sims`` iterations: draw a "true" parameter set from the prior,
    simulate a dataset, fit it, and record where each true value falls within its
    posterior draws (the normalized rank). Under correctly-calibrated inference
    those normalized ranks are Uniform(0, 1); a per-parameter KS test flags
    miscalibration.

    Callbacks (each receives the shared ``Generator`` for reproducibility):

    * ``sample_prior(rng) -> {param: scalar}`` — one draw of the true parameters.
    * ``simulate(theta, rng) -> data`` — synthetic data given the true params.
    * ``fit(data, rng) -> {param: 1-D posterior draws}`` — keys must match
      ``sample_prior``.

    Scalar parameters only (the common SBC case). Returns an :class:`SBCResult`.
    """
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    rng = np.random.default_rng(seed)
    names: list[str] | None = None
    u_ranks: dict[str, list[float]] = {}

    for _ in range(n_sims):
        theta = sample_prior(rng)
        data = simulate(theta, rng)
        post = fit(data, rng)
        if names is None:
            names = list(theta.keys())
            u_ranks = {k: [] for k in names}
        for k in names:
            draws = np.asarray(post[k], dtype=float).ravel()
            true = float(np.asarray(theta[k], dtype=float).ravel()[0])
            n_draws = draws.size
            # Normalized rank in (0, 1); uniform under calibrated inference.
            u = (np.sum(draws < true) + 0.5) / (n_draws + 1.0)
            u_ranks[k].append(float(u))

    from scipy import stats

    assert names is not None
    result = SBCResult(param_names=names, n_sims=n_sims)
    for k in names:
        arr = np.asarray(u_ranks[k], dtype=float)
        ks = stats.kstest(arr, "uniform")
        result.normalized_ranks[k] = arr
        result.ks_pvalue[k] = float(ks.pvalue)
        result.calibrated[k] = bool(ks.pvalue > alpha)
    return result
