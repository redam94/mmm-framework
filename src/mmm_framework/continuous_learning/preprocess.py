"""Baseline realism — adstock pre-pass and CUPED variance reduction (guide §9.3/9.4).

The reference loop treats spend as the lever directly. Two cheap upgrades make it
realistic without touching the response surface:

* **Adstock pre-pass.** If carryover matters, apply geometric adstock to each
  channel's spend *time series* and fit the surface on the **adstocked** series
  (adstock before saturation). Fitting on raw spend when the true response has
  carryover biases the curve; pre-adstocking with the right decay recovers it.
* **CUPED.** Use the pre-period KPI as a covariate to strip baseline variance
  from the test-period outcome. The lift estimate's variance shrinks by
  ``1 - rho^2`` (``rho`` = corr of test outcome with the pre-period covariate),
  a direct testing-budget lever — it cuts the geo count needed per MDE.

Both reuse the framework's own kernels (``transforms.adstock``) and operate on
the loop's data contract, so the continuous-learning surface stays identical.

The panel layout is the one :func:`mmm_framework.continuous_learning.dgp.simulate_panel`
produces: rows are ``t_pre`` weeks (all geos) followed by ``t_test`` weeks (all
geos), each week block ordered ``geo = 0 .. n_geo-1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm_framework.config.enums import AdstockType
from mmm_framework.transforms.adstock import adstock_weights, apply_adstock


def _to_time_geo_channel(
    spend: np.ndarray, n_geo: int, t_pre: int, t_test: int
) -> np.ndarray:
    """Reshape stacked rows to ``(T, n_geo, K)`` with weeks in pre→test order."""
    k = spend.shape[1]
    pre = spend[: t_pre * n_geo].reshape(t_pre, n_geo, k)
    test = spend[t_pre * n_geo :].reshape(t_test, n_geo, k)
    return np.concatenate([pre, test], axis=0)


def _from_time_geo_channel(arr: np.ndarray, t_pre: int) -> np.ndarray:
    """Inverse of :func:`_to_time_geo_channel`."""
    t_total, n_geo, k = arr.shape
    pre = arr[:t_pre].reshape(t_pre * n_geo, k)
    test = arr[t_pre:].reshape((t_total - t_pre) * n_geo, k)
    return np.vstack([pre, test])


def adstock_panel(
    spend: np.ndarray,
    n_geo: int,
    t_pre: int,
    t_test: int,
    *,
    alpha: float,
    l_max: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """Geometric-adstock each channel's spend series within each geo.

    Reuses ``transforms.adstock`` (``normalize=True`` keeps the kernel a weighted
    moving average so total magnitude stays in the coefficient). Returns a spend
    array of the same shape, in the same row order.
    """
    s = np.asarray(spend, dtype=float)
    cube = _to_time_geo_channel(s, n_geo, t_pre, t_test)  # (T, n_geo, K)
    w = adstock_weights(AdstockType.GEOMETRIC, l_max, alpha=alpha, normalize=normalize)
    out = np.empty_like(cube)
    for g in range(n_geo):
        for c in range(cube.shape[2]):
            out[:, g, c] = apply_adstock(cube[:, g, c], w)
    return _from_time_geo_channel(out, t_pre)


def adstock_prepass(
    data: dict[str, Any],
    t_pre: int,
    t_test: int,
    *,
    alpha: float,
    l_max: int = 8,
) -> dict[str, Any]:
    """Return a copy of the data contract with the spend adstocked (guide §9.4).

    Fit the surface on this instead of the raw panel when the response has
    carryover; the decay ``alpha`` is a hyperparameter (sweep it, or set it from
    a known channel half-life).
    """
    out = dict(data)
    out["spend"] = adstock_panel(
        np.asarray(data["spend"], dtype=float),
        int(data["n_geo"]),
        t_pre,
        t_test,
        alpha=alpha,
        l_max=l_max,
    )
    return out


# ── CUPED ─────────────────────────────────────────────────────────────────────


def cuped_covariate(data: dict[str, Any], t_pre: int) -> np.ndarray:
    """Per-geo, mean-centered pre-period KPI — the CUPED control covariate."""
    y = np.asarray(data["y"], dtype=float)
    geo = np.asarray(data["geo_idx"], dtype=int)
    n_geo = int(data["n_geo"])
    n_pre = t_pre * n_geo
    pre_y, pre_geo = y[:n_pre], geo[:n_pre]
    x = np.array([pre_y[pre_geo == g].mean() for g in range(n_geo)])
    return x - x.mean()


def cuped_adjust(
    data: dict[str, Any], t_pre: int
) -> tuple[dict[str, Any], dict[str, float]]:
    """CUPED-adjust the outcome and report the variance reduction.

    ``y_adj = y - theta * x_pre[geo]`` with ``theta = Cov(y_test, x_pre) /
    Var(x_pre)`` (the regression of the geo's test-period mean on its pre-period
    covariate). Returns ``(adjusted_data, info)`` where ``info`` carries
    ``theta``, ``rho`` (the correlation CUPED exploits) and ``var_reduction``
    (``1 - rho^2``, the fraction of geo-level outcome variance removed).

    **Incompatible with** ``likelihood="negbinomial"``: the adjustment mutates
    ``y`` to non-integer (and possibly negative) values, which the count
    likelihood's validation rejects. For count KPIs fit the raw counts — the
    geo intercept plus the national time effect (``time_effect="national"``)
    covers most of what CUPED buys here.
    """
    x_pre = cuped_covariate(data, t_pre)
    y = np.asarray(data["y"], dtype=float)
    geo = np.asarray(data["geo_idx"], dtype=int)
    n_geo = int(data["n_geo"])
    n_pre = t_pre * n_geo
    test_y, test_geo = y[n_pre:], geo[n_pre:]
    y_test_geo = np.array([test_y[test_geo == g].mean() for g in range(n_geo)])

    var_x = float(np.var(x_pre))
    if var_x <= 1e-12:  # no usable pre-period signal -> CUPED is a no-op
        return dict(data), {"theta": 0.0, "rho": 0.0, "var_reduction": 0.0}
    theta = float(np.cov(y_test_geo, x_pre)[0, 1] / var_x)
    rho = float(np.corrcoef(y_test_geo, x_pre)[0, 1])

    out = dict(data)
    out["y"] = y - theta * x_pre[geo]
    return out, {"theta": theta, "rho": rho, "var_reduction": 1.0 - rho**2}
