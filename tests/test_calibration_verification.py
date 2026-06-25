"""Tests for inference-calibration verification (Phase 2 / I2).

LOO-PIT and SBC machine-verify that the inference produces calibrated intervals.
The SBC harness is validated on a conjugate Normal model (fast, exact), where a
correct posterior must pass and a deliberately too-narrow posterior must fail.
"""

from __future__ import annotations

import numpy as np

from mmm_framework.validation import (
    loo_pit_check,
    simulation_based_calibration,
)

# --- conjugate Normal model for SBC ----------------------------------------
SIGMA = 1.0        # known likelihood noise
PRIOR_SD = 2.0     # prior sd on mu
N_OBS = 20
L = 500            # posterior draws per fit


def _sample_prior(rng):
    return {"mu": float(rng.normal(0.0, PRIOR_SD))}


def _simulate(theta, rng):
    return theta["mu"] + rng.normal(0.0, SIGMA, N_OBS)


def _posterior_draws(y, rng, shrink):
    y = np.asarray(y, dtype=float)
    n = y.size
    prec = 1.0 / PRIOR_SD**2 + n / SIGMA**2
    mean = (0.0 / PRIOR_SD**2 + y.sum() / SIGMA**2) / prec
    sd = np.sqrt(1.0 / prec) * shrink
    return {"mu": rng.normal(mean, sd, L)}


def _fit_correct(data, rng):
    return _posterior_draws(data, rng, shrink=1.0)


def _fit_too_narrow(data, rng):
    return _posterior_draws(data, rng, shrink=0.3)  # overconfident posterior


# --- SBC --------------------------------------------------------------------
def test_sbc_correct_inference_is_calibrated():
    res = simulation_based_calibration(
        _sample_prior, _simulate, _fit_correct, n_sims=200, seed=7
    )
    assert res.param_names == ["mu"]
    assert res.ks_pvalue["mu"] > 0.05
    assert res.calibrated["mu"] is True
    assert res.all_calibrated


def test_sbc_overconfident_posterior_flagged():
    res = simulation_based_calibration(
        _sample_prior, _simulate, _fit_too_narrow, n_sims=200, seed=7
    )
    # A too-narrow posterior over-concentrates ranks -> non-uniform -> flagged.
    assert res.calibrated["mu"] is False
    assert not res.all_calibrated


# --- LOO-PIT ----------------------------------------------------------------
def test_loo_pit_calibrated_predictive():
    rng = np.random.default_rng(1)
    n_obs, n_s = 200, 600
    y = rng.normal(0, 1, n_obs)
    y_hat = rng.normal(0, 1, (n_obs, n_s))  # predictive matches the data
    res = loo_pit_check(y=y, y_hat=y_hat)
    assert res.calibrated is True
    assert res.n == n_obs


def test_loo_pit_flags_overconfident_predictive():
    rng = np.random.default_rng(2)
    n_obs, n_s = 200, 600
    y = rng.normal(0, 1, n_obs)
    y_hat = rng.normal(0, 0.3, (n_obs, n_s))  # far too narrow
    res = loo_pit_check(y=y, y_hat=y_hat)
    assert res.calibrated is False


def test_loo_pit_accepts_sample_first_orientation():
    rng = np.random.default_rng(3)
    n_obs, n_s = 150, 400
    y = rng.normal(0, 1, n_obs)
    y_hat = rng.normal(0, 1, (n_s, n_obs))  # (sample, obs) -> auto-transposed
    res = loo_pit_check(y=y, y_hat=y_hat)
    assert res.n == n_obs
