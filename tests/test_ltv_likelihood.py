"""BG/NBD + Gamma-Gamma likelihood correctness (ltv/likelihood.py): pytensor
expressions vs an independent scipy reference, finiteness at the x=0 boundary,
and the switch-branch NaN-gradient guard."""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.special import betaln, gammaln, hyp2f1

from mmm_framework.ltv.likelihood import (
    bgnbd_expected_purchases,
    bgnbd_loglik,
    bgnbd_p_alive,
    gamma_gamma_expected_value,
    gamma_gamma_loglik,
)

# fixture: mixed population incl. a zero-repeat customer and large T
X = np.array([0.0, 1.0, 3.0, 7.0, 12.0])
TX = np.array([0.0, 4.0, 20.0, 30.0, 47.0])
T = np.array([30.0, 26.0, 40.0, 38.0, 52.0])
M = np.array([np.nan, 25.0, 31.0, 28.5, 33.0])
PARAMS = dict(r=0.8, alpha=6.0, a=1.3, b=3.0)
GG = dict(p=6.0, q=4.0, gamma=15.0)


def _ref_bgnbd(r, alpha, a, b, x, t_x, T):
    ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_a2 = betaln(a, b + x) - betaln(a, b)
    ln_a3 = -(r + x) * np.log(alpha + T)
    out = np.empty_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            ln_a4 = (
                np.log(a) - np.log(b + x[i] - 1) - (r + x[i]) * np.log(alpha + t_x[i])
            )
            out[i] = ln_a1[i] + ln_a2[i] + np.logaddexp(ln_a3[i], ln_a4)
        else:
            out[i] = ln_a1[i] + ln_a2[i] + ln_a3[i]
    return out


def _ref_expected(r, alpha, a, b, x, t_x, T, t):
    z = t / (alpha + T + t)
    hyp = np.array([hyp2f1(r + xi, b + xi, a + b + xi - 1, zi) for xi, zi in zip(x, z)])
    top = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp
    first = (a + b + x - 1) / (a - 1)
    denom = np.ones_like(x)
    mask = x > 0
    denom[mask] = 1 + (a / (b + x[mask] - 1)) * (
        (alpha + T[mask]) / (alpha + t_x[mask])
    ) ** (r + x[mask])
    return first * top / denom


def test_bgnbd_matches_scipy_reference():
    got = bgnbd_loglik(**PARAMS, x=X, t_x=TX, T=T).eval()
    ref = _ref_bgnbd(x=X, t_x=TX, T=T, **PARAMS)
    np.testing.assert_allclose(got, ref, rtol=1e-6)
    assert np.isfinite(got).all()


def test_bgnbd_expected_purchases_matches_reference():
    got = bgnbd_expected_purchases(**PARAMS, x=X, t_x=TX, T=T, horizon=26.0).eval()
    ref = _ref_expected(x=X, t_x=TX, T=T, t=26.0, **PARAMS)
    np.testing.assert_allclose(got, ref, rtol=1e-6)
    assert (got >= 0).all()


def test_p_alive_bounds_and_zero_repeat():
    pa = bgnbd_p_alive(**PARAMS, x=X, t_x=TX, T=T).eval()
    assert ((pa >= 0) & (pa <= 1)).all()
    assert pa[0] == pytest.approx(1.0)  # x=0 → no evidence of death (A4 absent)
    # a recently-active frequent buyer (x=12, silent 5w) is likelier ALIVE than
    # a customer silent for 22 of 26 weeks after a single repeat
    assert pa[4] > pa[1]


def test_gamma_gamma_matches_scipy_reference():
    got = gamma_gamma_loglik(**GG, x=X, m=M).eval()
    p, q, g = GG["p"], GG["q"], GG["gamma"]
    mask = X > 0
    ref = np.zeros_like(X)
    xm, mm = X[mask], M[mask]
    ref[mask] = (
        gammaln(p * xm + q)
        - gammaln(p * xm)
        - gammaln(q)
        + q * np.log(g)
        + (p * xm - 1) * np.log(mm)
        + p * xm * np.log(xm)
        - (p * xm + q) * np.log(g + xm * mm)
    )
    np.testing.assert_allclose(got, ref, rtol=1e-6)
    assert got[0] == 0.0  # x=0 contributes nothing


def test_gamma_gamma_expected_value_shrinks_to_population():
    ev = gamma_gamma_expected_value(**GG, x=X, m=M).eval()
    pop_mean = GG["p"] * GG["gamma"] / (GG["q"] - 1)  # 30
    assert ev[0] == pytest.approx(pop_mean)  # one-timer → population mean
    # high-frequency customers sit closer to their own mean than to the pop mean
    assert abs(ev[4] - M[4]) < abs(ev[1] - M[1])
    assert np.isfinite(ev).all() and (ev > 0).all()


def test_gradients_finite_at_x_zero_boundary():
    """The pt.switch NaN-gradient trap: with b < 1 the masked x=0 branch's
    log(b + x - 1) would be NaN — the sanitized inputs must keep the TOTAL
    log-likelihood gradient finite."""
    r = pt.dscalar("r")
    alpha = pt.dscalar("alpha")
    a = pt.dscalar("a")
    b = pt.dscalar("b")
    total = pt.sum(bgnbd_loglik(r, alpha, a, b, X, TX, T))
    grads = pytensor.function([r, alpha, a, b], pt.grad(total, [r, alpha, a, b]))(
        0.8, 6.0, 1.3, 0.5
    )  # b < 1: the unsanitized version NaNs here
    assert all(np.isfinite(g) for g in grads)


def test_bgnbd_higher_at_truth_than_perturbed():
    ll_true = float(pt.sum(bgnbd_loglik(**PARAMS, x=X, t_x=TX, T=T)).eval())
    ll_off = float(
        pt.sum(bgnbd_loglik(r=3.0, alpha=1.0, a=0.4, b=8.0, x=X, t_x=TX, T=T)).eval()
    )
    assert ll_true > ll_off
