"""
Time-series decomposition and stationarity checks for EDA.

STL is used when the series spans at least two full seasonal cycles;
otherwise a centered rolling-median detrend stands in (no seasonal part).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EDAConfig
from .results import DecompositionResult


def _strength(resid: np.ndarray, component: np.ndarray) -> float:
    """Component strength: 1 - Var(resid) / Var(component + resid), in [0, 1]."""
    denom = float(np.var(component + resid))
    if denom <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(resid) / denom))


def decompose_series(
    series: pd.Series,
    period: int | None = None,
    *,
    robust: bool = False,
    rolling_window: int = 13,
    variable: str | None = None,
) -> DecompositionResult:
    """Decompose one series into trend / seasonal / residual.

    Parameters
    ----------
    series : pd.Series
        Period-indexed values (NaNs are interpolated for decomposition only).
    period : int, optional
        Seasonal period (e.g. 52 for weekly). When None or when fewer than
        two full cycles are available, falls back to a rolling-median detrend.
    robust : bool
        Passed to STL. Default False: with only a few seasonal cycles,
        statsmodels' robust mode degenerates to near-interpolation (residual
        MAD ~ 0), which breaks residual-based screening. Outlier resistance
        comes from MAD-based z-scores on the residual instead.
    """
    name = variable or str(series.name or "series")
    values = series.astype(float).interpolate(limit_direction="both")

    use_stl = period is not None and period >= 2 and len(values) >= 2 * period
    if use_stl:
        from statsmodels.tsa.seasonal import STL

        res = STL(values.to_numpy(), period=period, robust=robust).fit()
        trend = pd.Series(res.trend, index=series.index)
        seasonal = pd.Series(res.seasonal, index=series.index)
        resid = pd.Series(res.resid, index=series.index)
        return DecompositionResult(
            variable=name,
            method="stl",
            period=period,
            trend=trend,
            seasonal=seasonal,
            resid=resid,
            trend_strength=_strength(resid.to_numpy(), trend.to_numpy()),
            seasonal_strength=_strength(resid.to_numpy(), seasonal.to_numpy()),
        )

    window = max(3, min(rolling_window, len(values)))
    trend = values.rolling(window, center=True, min_periods=1).median()
    resid = values - trend
    return DecompositionResult(
        variable=name,
        method="rolling_median",
        period=None,
        trend=trend,
        seasonal=None,
        resid=resid,
        trend_strength=_strength(resid.to_numpy(), trend.to_numpy()),
        seasonal_strength=None,
    )


def stationarity_tests(
    series: pd.Series, *, significance: float = 0.05
) -> dict[str, object]:
    """ADF + KPSS stationarity tests with a combined verdict.

    ADF null: unit root (non-stationary). KPSS null: stationary.
    """
    values = series.astype(float).dropna().to_numpy()
    out: dict[str, object] = {
        "adf_pvalue": None,
        "kpss_pvalue": None,
        "verdict": "insufficient_data",
    }
    if len(values) < 12 or np.allclose(values, values[0]):
        return out

    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    try:
        out["adf_pvalue"] = float(adfuller(values, autolag="AIC")[1])
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            # KPSS warns when the statistic is outside the lookup table; the
            # clamped p-value is still usable for a verdict.
            warnings.simplefilter("ignore")
            out["kpss_pvalue"] = float(kpss(values, regression="c", nlags="auto")[1])
    except Exception:
        pass

    adf_p, kpss_p = out["adf_pvalue"], out["kpss_pvalue"]
    if adf_p is None or kpss_p is None:
        out["verdict"] = "inconclusive"
    elif adf_p < significance and kpss_p >= significance:
        out["verdict"] = "stationary"
    elif adf_p >= significance and kpss_p < significance:
        out["verdict"] = "non_stationary"
    elif adf_p < significance and kpss_p < significance:
        out["verdict"] = "trend_stationary"  # tests disagree: differencing-vs-trend
    else:
        out["verdict"] = "inconclusive"
    return out


def decomposition_summary(
    panel_series: dict[str, pd.Series],
    period: int | None,
    config: EDAConfig | None = None,
) -> list[DecompositionResult]:
    """Decompose several series (e.g. KPI + media) with shared settings."""
    cfg = config or EDAConfig()
    return [
        decompose_series(s, period, robust=cfg.stl_robust, variable=name)
        for name, s in panel_series.items()
    ]


__all__ = ["decompose_series", "stationarity_tests", "decomposition_summary"]
