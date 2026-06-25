"""IV/2SLS + front-door estimators recover known causal truth (deferred V2-full).

Each builds data with UNOBSERVED confounding so naive OLS is biased, and checks
the estimator recovers the planted effect where OLS does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.estimators import (
    frontdoor_estimate,
    two_stage_least_squares,
)
from mmm_framework.estimators.causal import _ols


def _ols_effect(y, t):
    X = np.column_stack([np.asarray(t, float), np.ones(len(y))])
    beta, _, _ = _ols(np.asarray(y, float), X)
    return float(beta[0])


def test_2sls_recovers_effect_under_confounding():
    rng = np.random.default_rng(0)
    n = 6000
    tau = 2.0
    u = rng.normal(0, 1, n)  # unobserved confounder
    z = rng.normal(0, 1, n)  # instrument (affects T only)
    t = 0.9 * z + 1.3 * u + rng.normal(0, 0.5, n)
    y = tau * t + 2.5 * u + rng.normal(0, 0.5, n)

    iv = two_stage_least_squares(y, t, z)
    ols = _ols_effect(y, t)

    assert abs(iv.effect - tau) < 0.1  # IV recovers the truth
    assert abs(ols - tau) > 0.3  # OLS is biased by the confounder
    assert iv.ci_low <= tau <= iv.ci_high
    assert not iv.weak_instrument  # strong instrument here (F >> 10)


def test_2sls_flags_weak_instrument():
    rng = np.random.default_rng(1)
    n = 4000
    u = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)  # nearly irrelevant to T
    t = 0.02 * z + 1.0 * u + rng.normal(0, 1, n)
    y = 1.5 * t + 2.0 * u + rng.normal(0, 1, n)
    iv = two_stage_least_squares(y, t, z)
    assert iv.weak_instrument  # first-stage F < 10


def test_2sls_with_controls():
    rng = np.random.default_rng(2)
    n = 5000
    tau = 1.2
    u = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)  # observed control
    z = rng.normal(0, 1, n)
    t = 0.8 * z + 1.0 * u + 0.5 * x + rng.normal(0, 0.5, n)
    y = tau * t + 1.5 * u + 0.7 * x + rng.normal(0, 0.5, n)
    iv = two_stage_least_squares(y, t, z, controls=x)
    assert abs(iv.effect - tau) < 0.12


def test_frontdoor_recovers_mediated_effect():
    rng = np.random.default_rng(3)
    n = 8000
    a, b = 1.5, 0.7  # T->M and M->Y ; total mediated effect = a*b = 1.05
    u = rng.normal(0, 1, n)  # confounds T and Y (unobserved), but NOT via M
    t = 1.2 * u + rng.normal(0, 0.5, n)
    m = a * t + rng.normal(0, 0.5, n)
    y = b * m + 3.0 * u + rng.normal(0, 0.5, n)

    fd = frontdoor_estimate(y, t, m)
    ols = _ols_effect(y, t)

    assert abs(fd.effect - a * b) < 0.1  # front-door recovers a*b
    assert abs(ols - a * b) > 0.3  # OLS is biased by U
    assert fd.ci_low <= a * b <= fd.ci_high


def test_input_validation():
    y = np.zeros(10)
    with pytest.raises(ValueError):
        two_stage_least_squares(y, np.zeros((10, 2)), np.zeros(10))  # 2 treatments
    with pytest.raises(ValueError):
        two_stage_least_squares(y, np.zeros(10), np.empty((10, 0)))  # no instrument
