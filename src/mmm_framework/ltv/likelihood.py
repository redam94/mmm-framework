"""BG/NBD + Gamma-Gamma closed-form log-likelihoods and conditional
expectations, as pytensor expressions (NUTS-able via ``pm.Potential`` — no
discrete latents).

BG/NBD (Fader–Hardie–Lee 2005): each customer purchases at latent Poisson rate
``lam ~ Gamma(r, alpha)`` (rate parameterization) and dies with probability
``p ~ Beta(a, b)`` after every purchase. Integrating the latents gives the
individual likelihood in terms of ``(x, t_x, T)``:

    ln L = ln A1 + ln A2 + ln(A3 + 1{x>0} A4)
    A1 = Gamma(r+x) alpha^r / Gamma(r)          A2 = B(a, b+x) / B(a, b)
    A3 = (alpha+T)^-(r+x)                       A4 = a/(b+x-1) (alpha+t_x)^-(r+x)

Gamma-Gamma (Fader–Hardie 2013): transaction values ``z ~ Gamma(p, nu)`` with
``nu ~ Gamma(q, gamma)``; the mean of ``x`` repeat values ``m`` has

    ln L = ln Gamma(p x + q) - ln Gamma(p x) - ln Gamma(q)
         + q ln gamma + (p x - 1) ln m + p x ln x - (p x + q) ln(gamma + x m)

**pytensor gotcha (load-bearing):** ``pt.switch`` evaluates BOTH branches'
gradients, so any NaN in the "unused" branch (e.g. ``log(b + x - 1)`` at
``x = 0`` with ``b < 1``) poisons the gradient. Every masked term here first
SANITIZES its inputs (``pt.switch`` on the argument, not just the result).
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


def bgnbd_loglik(r, alpha, a, b, x, t_x, T):
    """Per-customer BG/NBD log-likelihood — returns a length-``n`` tensor.

    ``x``/``t_x``/``T`` are data arrays (frequency, recency, age); ``r``/
    ``alpha``/``a``/``b`` are (scalar) population parameters.
    """
    x = pt.as_tensor_variable(np.asarray(x, dtype="float64"))
    t_x = pt.as_tensor_variable(np.asarray(t_x, dtype="float64"))
    T = pt.as_tensor_variable(np.asarray(T, dtype="float64"))

    ln_a1 = pt.gammaln(r + x) - pt.gammaln(r) + r * pt.log(alpha)
    ln_a2 = _betaln(a, b + x) - _betaln(a, b)
    ln_a3 = -(r + x) * pt.log(alpha + T)
    # A4 exists only for repeat buyers; sanitize b+x-1 BEFORE the log so the
    # x==0 branch can't emit NaN gradients (see module docstring).
    has_repeat = pt.gt(x, 0)
    bx1 = pt.switch(has_repeat, b + x - 1.0, 1.0)
    ln_a4 = pt.log(a) - pt.log(bx1) - (r + x) * pt.log(alpha + t_x)
    ln_a4 = pt.switch(has_repeat, ln_a4, -np.inf)
    return ln_a1 + ln_a2 + pt.logaddexp(ln_a3, ln_a4)


def bgnbd_p_alive(r, alpha, a, b, x, t_x, T):
    """P(customer still alive | x, t_x, T) = A3 / (A3 + 1{x>0} A4)."""
    x = pt.as_tensor_variable(np.asarray(x, dtype="float64"))
    t_x = pt.as_tensor_variable(np.asarray(t_x, dtype="float64"))
    T = pt.as_tensor_variable(np.asarray(T, dtype="float64"))

    ln_a3 = -(r + x) * pt.log(alpha + T)
    has_repeat = pt.gt(x, 0)
    bx1 = pt.switch(has_repeat, b + x - 1.0, 1.0)
    ln_a4 = pt.log(a) - pt.log(bx1) - (r + x) * pt.log(alpha + t_x)
    ln_a4 = pt.switch(has_repeat, ln_a4, -np.inf)
    return pt.exp(ln_a3 - pt.logaddexp(ln_a3, ln_a4))


def bgnbd_expected_purchases(r, alpha, a, b, x, t_x, T, horizon):
    """E[# purchases in (T, T + horizon] | x, t_x, T] (Fader–Hardie).

        E = (a+b+x-1)/(a-1)
            * [1 - ((alpha+T)/(alpha+T+t))^(r+x) * 2F1(r+x, b+x; a+b+x-1; z)]
            / (1 + 1{x>0} (a/(b+x-1)) ((alpha+T)/(alpha+t_x))^(r+x))
        z = t / (alpha + T + t)

    The ``(a-1)`` pole cancels against the bracket as ``a -> 1`` analytically,
    but the expression is numerically delicate near ``a = 1`` — priors should
    not concentrate there.
    """
    x = pt.as_tensor_variable(np.asarray(x, dtype="float64"))
    t_x = pt.as_tensor_variable(np.asarray(t_x, dtype="float64"))
    T = pt.as_tensor_variable(np.asarray(T, dtype="float64"))
    t = pt.as_tensor_variable(float(horizon))

    z = t / (alpha + T + t)
    hyp = pt.hyp2f1(r + x, b + x, a + b + x - 1.0, z)
    top = 1.0 - pt.power((alpha + T) / (alpha + T + t), r + x) * hyp
    first = (a + b + x - 1.0) / (a - 1.0)
    has_repeat = pt.gt(x, 0)
    bx1 = pt.switch(has_repeat, b + x - 1.0, 1.0)
    ln_a4_over_a3 = (
        pt.log(a) - pt.log(bx1) + (r + x) * pt.log((alpha + T) / (alpha + t_x))
    )
    denom = 1.0 + pt.switch(has_repeat, pt.exp(ln_a4_over_a3), 0.0)
    return first * top / denom


def gamma_gamma_loglik(p, q, gamma, x, m):
    """Per-customer Gamma-Gamma log-likelihood of the mean repeat value ``m``
    given ``x`` repeat purchases. Rows with ``x == 0`` (no monetary signal)
    contribute 0 — pass the FULL arrays, masking is internal."""
    x = pt.as_tensor_variable(np.asarray(x, dtype="float64"))
    m_arr = np.asarray(m, dtype="float64")
    has_value = pt.gt(x, 0) & pt.as_tensor_variable(np.isfinite(m_arr))
    # sanitize: x=0 / NaN-m rows get neutral inputs so gradients stay finite
    xs = pt.switch(has_value, x, 1.0)
    ms = pt.as_tensor_variable(np.where(np.isfinite(m_arr) & (m_arr > 0), m_arr, 1.0))
    ll = (
        pt.gammaln(p * xs + q)
        - pt.gammaln(p * xs)
        - pt.gammaln(q)
        + q * pt.log(gamma)
        + (p * xs - 1.0) * pt.log(ms)
        + p * xs * pt.log(xs)
        - (p * xs + q) * pt.log(gamma + xs * ms)
    )
    return pt.switch(has_value, ll, 0.0)


def gamma_gamma_expected_value(p, q, gamma, x, m):
    """E[transaction value | x repeat purchases with mean m] — the conditional
    Gamma-Gamma mean ``(gamma + x m) p / (p x + q - 1)``; one-time buyers get
    the population mean ``p gamma / (q - 1)``. Requires ``q > 1`` (guarded by a
    floor on the denominator; priors should keep q away from 1)."""
    x = pt.as_tensor_variable(np.asarray(x, dtype="float64"))
    m_arr = np.asarray(m, dtype="float64")
    has_value = pt.gt(x, 0) & pt.as_tensor_variable(np.isfinite(m_arr))
    ms = pt.as_tensor_variable(np.where(np.isfinite(m_arr), m_arr, 0.0))
    denom_ind = pt.maximum(p * x + q - 1.0, 1e-2)
    denom_pop = pt.maximum(q - 1.0, 1e-2)
    conditional = (gamma + x * ms) * p / denom_ind
    population = p * gamma / denom_pop
    return pt.switch(has_value, conditional, population)


__all__ = [
    "bgnbd_loglik",
    "bgnbd_p_alive",
    "bgnbd_expected_purchases",
    "gamma_gamma_loglik",
    "gamma_gamma_expected_value",
]
