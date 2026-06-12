"""Rolling-origin backtesting and forecast-accuracy evaluation for BayesianMMM.

Answers the question every client asks and most MMM documentation dodges:
*"how accurate were the model's forecasts on data it had not seen?"*

The harness refits the model on an expanding training window, forecasts a
fixed horizon past each training cutoff (a genuine out-of-time forecast, not
an in-sample re-prediction), and grades the forecasts against the held-out
actuals: point accuracy (MAPE, sMAPE, RMSE, MASE), interval calibration
(empirical coverage of the central 50/80/95% prediction intervals), and skill
relative to naive baselines (last-value and seasonal-naive).

Out-of-time prediction cannot go through ``BayesianMMM.predict``: the model
graph bakes the seasonality lookup and trend scale over the *training*
periods, so future time indices are out of range. Instead
:class:`PosteriorForecaster` replays the model's forward pass in NumPy from
the posterior draws -- the same approach as the cross-validation path in
:mod:`mmm_framework.validation.validator`, generalized to both adstock
parameterizations (the legacy fixed-alpha blend and the parametric
geometric/delayed/Weibull kernels) and all configured saturation types.
Adstock is convolved over the *full* spend history so carryover from the
training period flows into the forecast window correctly.

Scope (enforced with explicit errors, not silent wrong answers):

* national data only (one geo x product cell);
* trend ``NONE`` or ``LINEAR`` (extrapolating spline/GP/piecewise trends is
  ill-defined without additional assumptions);
* experiment-calibration likelihood terms are dropped in backtest refits
  (their estimands reference the full-period spend).

A good backtest validates the *predictive* model, not the causal one: a model
can forecast well while attributing wrongly (and vice versa). Use this
alongside -- never instead of -- the pressure-testing and calibration
machinery.

Examples
--------
>>> from mmm_framework.validation import BacktestConfig, run_backtest
>>> config = BacktestConfig(min_train_size=104, horizon=13, step=13)
>>> result = run_backtest(mmm, config)          # mmm: an (unfitted) BayesianMMM
>>> result.summary()                            # model vs naive baselines
>>> result.by_horizon()                         # accuracy decay with lead time
>>> result.coverage()                           # interval calibration
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config import SaturationType
from ..transforms.adstock import adstock_weights, geometric_adstock_2d
from ..transforms.seasonality import create_fourier_features

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "PosteriorForecaster",
    "rolling_origins",
    "run_backtest",
]

# Same component-period table as BayesianMMM._prepare_seasonality, so the
# forecaster evaluates the Fourier features at exactly the training phase.
_PERIODS_BY_FREQ: dict[str, dict[str, float]] = {
    "W": {"yearly": 52.0, "monthly": 52.0 / 12.0},
    "D": {"yearly": 365.25, "monthly": 365.25 / 12.0, "weekly": 7.0},
    "M": {"yearly": 12.0},
}

# Default seasonal-naive lag per data frequency (used for the baseline and
# the MASE denominator).
_SEASON_BY_FREQ: dict[str, int] = {"W": 52, "D": 7, "M": 12}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a rolling-origin backtest.

    Attributes
    ----------
    min_train_size : int
        Periods in the first training window. Two seasonal cycles of weekly
        data (104) is a sensible floor; below ~78 the seasonality and trend
        posteriors are barely informed and forecast intervals balloon.
    horizon : int
        Forecast length (periods) past each training cutoff.
    step : int or None
        Spacing between consecutive training cutoffs. ``None`` uses
        ``horizon`` (non-overlapping forecast windows).
    max_origins : int or None
        Cap on the number of refits (``None`` = as many as the data allows).
    coverage_levels : tuple[float, ...]
        Central prediction-interval levels to record and grade.
    draws, tune, chains : int
        MCMC budget per refit. Backtests refit the model once per origin, so
        these default lower than a production fit; convergence is still
        recorded per origin in :attr:`BacktestResult.fits`.
    include_noise : bool
        Forecast the observable (mean + observation noise) rather than the
        latent mean. Keep ``True`` for honest interval coverage.
    season_period : int or None
        Seasonal lag for the seasonal-naive baseline and the MASE scale.
        ``None`` derives it from the data frequency (weekly -> 52).
    random_seed : int
        Base seed; each origin offsets it deterministically.
    """

    min_train_size: int = 104
    horizon: int = 13
    step: int | None = None
    max_origins: int | None = None
    coverage_levels: tuple[float, ...] = (0.5, 0.8, 0.95)
    draws: int = 500
    tune: int = 500
    chains: int = 4
    include_noise: bool = True
    season_period: int | None = None
    random_seed: int = 42


# ---------------------------------------------------------------------------
# origins
# ---------------------------------------------------------------------------


def rolling_origins(
    n_periods: int,
    *,
    min_train_size: int,
    horizon: int,
    step: int | None = None,
    max_origins: int | None = None,
) -> list[int]:
    """Training cutoffs for a rolling-origin (expanding-window) backtest.

    Each returned cutoff ``T`` means: train on periods ``[0, T)``, forecast
    periods ``[T, min(T + horizon, n_periods))``. Only full forecast windows
    are emitted so every origin is graded on the same horizons.
    """
    if min_train_size < 2:
        raise ValueError("min_train_size must be at least 2")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if step is None:
        step = horizon
    if step < 1:
        raise ValueError("step must be at least 1")

    origins = list(range(min_train_size, n_periods - horizon + 1, step))
    if max_origins is not None:
        origins = origins[:max_origins]
    return origins


# ---------------------------------------------------------------------------
# posterior forward pass
# ---------------------------------------------------------------------------


class PosteriorForecaster:
    """Out-of-time forecasts from a fitted BayesianMMM's posterior draws.

    Replays the model's structural forward pass (trend, seasonality, adstock,
    saturation, controls, observation noise) in NumPy at arbitrary future
    period positions, using the full spend history for adstock carryover.

    Parameters
    ----------
    model : BayesianMMM
        A *fitted* model, possibly trained on a prefix of the full series.
    """

    def __init__(self, model: Any):
        if model._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if model.n_cells > 1:
            raise NotImplementedError(
                "PosteriorForecaster supports national (single-cell) data only; "
                f"got {model.n_cells} geo x product cells."
            )
        trend_type = getattr(model.trend_config.type, "value", model.trend_config.type)
        if str(trend_type).lower() not in ("none", "linear"):
            raise NotImplementedError(
                f"Out-of-time extrapolation of trend type {trend_type!r} is not "
                "supported (only 'none' and 'linear'); refit the backtest clone "
                "with a linear trend or disable the trend."
            )

        self.model = model
        posterior = model._trace.posterior
        self._n_samples = int(posterior.sizes["chain"] * posterior.sizes["draw"])

        def get(name: str) -> np.ndarray | None:
            if name in posterior:
                arr = posterior[name].values
                return arr.reshape(self._n_samples, *arr.shape[2:])
            return None

        self._get = get
        self._intercept = get("intercept")
        self._sigma = get("sigma")
        self._trend_slope = get("trend_slope")
        self._beta_controls = get("beta_controls")
        self._season = {
            name: get(f"season_{name}") for name in model.seasonality_features
        }

    @property
    def n_samples(self) -> int:
        return self._n_samples

    # -- components ---------------------------------------------------------

    def _trend_at(self, positions: np.ndarray, train_offset: int = 0) -> np.ndarray:
        """Linear trend extrapolated on the training time scale.

        ``train_offset`` is the absolute position of the trained model's first
        period — nonzero for rolling-window training (e.g. validator CV), where
        the clone's ``t_scaled = 0`` corresponds to that offset, not position 0.
        """
        if self._trend_slope is None:
            return np.zeros((len(positions), self._n_samples))
        # Training used t_scaled = linspace(0, 1, n_train) = pos / (n_train - 1);
        # future positions extrapolate past 1 on the same scale.
        denom = max(self.model.n_periods - 1, 1)
        t_scaled = (positions - train_offset) / denom
        return t_scaled[:, None] * self._trend_slope[None, :]

    def _seasonality_at(self, positions: np.ndarray) -> np.ndarray:
        """Fourier seasonality evaluated at absolute period positions.

        The features are periodic, so evaluating them at future positions on
        the same integer time axis the model trained on keeps the phase
        aligned.
        """
        out = np.zeros((len(positions), self._n_samples))
        freq = getattr(self.model.mff_config, "frequency", "W") or "W"
        component_periods = _PERIODS_BY_FREQ.get(freq, _PERIODS_BY_FREQ["W"])
        for name, train_features in self.model.seasonality_features.items():
            coef = self._season.get(name)
            if coef is None:
                continue
            order = train_features.shape[1] // 2
            period = component_periods.get(name)
            if period is None:  # pragma: no cover - mirrored model-build guard
                continue
            features = create_fourier_features(positions.astype(float), period, order)
            out += features @ coef.T  # (n_pos, 2*order) @ (2*order, n_samples)
        return out

    def _media_at(
        self, X_media_full_raw: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Sum of saturated channel contributions, (n_pos, n_samples)."""
        model = self.model
        out = np.zeros((len(positions), self._n_samples))
        for c, ch in enumerate(model.channel_names):
            x_raw = np.asarray(X_media_full_raw[:, c], dtype=float)
            if model.use_parametric_adstock:
                x_ad = self._parametric_adstock(ch, x_raw)[positions]
            else:
                x_ad = self._legacy_adstock(ch, x_raw, positions)
            x_sat = self._saturate(ch, x_ad)
            beta = self._get(f"beta_{ch}")
            if beta is None:
                continue
            out += x_sat * beta[None, :]
        return out

    def _legacy_adstock(
        self, channel: str, x_raw: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Blend of two fixed-alpha geometric adstocks, (n_pos, n_samples)."""
        model = self.model
        alpha_low = model.adstock_alphas[0]
        alpha_high = model.adstock_alphas[-1]
        max_val = model._media_max[channel] + 1e-8
        x2d = x_raw[:, None]
        x_low = geometric_adstock_2d(x2d, alpha_low)[positions, 0] / max_val
        x_high = geometric_adstock_2d(x2d, alpha_high)[positions, 0] / max_val
        mix = self._get(f"adstock_{channel}")
        if mix is None:
            mix = np.full(self._n_samples, 0.5)
        return (1 - mix)[None, :] * x_low[:, None] + mix[None, :] * x_high[:, None]

    def _parametric_adstock(self, channel: str, x_raw: np.ndarray) -> np.ndarray:
        """Per-draw FIR adstock of the full series, (n_full, n_samples)."""
        from ..model.base import _ADSTOCK_KIND

        model = self.model
        cfg = model._get_adstock_config(channel)
        kind = _ADSTOCK_KIND.get(cfg.type, "geometric")
        x_norm = x_raw / (model._media_raw_max[channel] + 1e-8)

        if kind == "none":
            return np.repeat(x_norm[:, None], self._n_samples, axis=1)

        l_max = cfg.l_max
        n = len(x_norm)
        # One windows matrix for the channel, then all draws in one matmul:
        # windows[t] = [x[t - l_max + 1], ..., x[t]] (zero-padded).
        x_padded = np.concatenate([np.zeros(l_max - 1), x_norm])
        row_idx = np.arange(n)[:, None] + np.arange(l_max)[None, :]
        windows = x_padded[row_idx]  # (n_full, l_max)

        params: dict[str, np.ndarray | None] = {
            "alpha": self._get(f"adstock_alpha_{channel}"),
            "theta": self._get(f"adstock_theta_{channel}"),
            "shape": self._get(f"adstock_shape_{channel}"),
            "scale": self._get(f"adstock_scale_{channel}"),
        }
        kernels = np.empty((l_max, self._n_samples))
        for s in range(self._n_samples):
            kw = {k: float(v[s]) for k, v in params.items() if v is not None}
            kernels[:, s] = adstock_weights(kind, l_max, normalize=cfg.normalize, **kw)[
                ::-1
            ]
        return windows @ kernels  # (n_full, n_samples)

    def _saturate(self, channel: str, x_ad: np.ndarray) -> np.ndarray:
        """Apply the channel's saturation; mirrors ``_apply_saturation_pt``."""
        model = self.model
        kind = model._get_saturation_config(channel).type
        if kind == SaturationType.LOGISTIC:
            lam = self._get(f"sat_lam_{channel}")
            exponent = np.clip(-lam[None, :] * x_ad, -20, 0)
            return 1 - np.exp(exponent)
        if kind == SaturationType.HILL:
            half = self._get(f"sat_half_{channel}")
            slope = self._get(f"sat_slope_{channel}")
            x_safe = np.maximum(x_ad, 1e-9)
            x_pow = x_safe ** slope[None, :]
            return x_pow / (x_pow + half[None, :] ** slope[None, :])
        if kind == SaturationType.MICHAELIS_MENTEN:
            half = self._get(f"sat_half_{channel}")
            return x_ad / (x_ad + half[None, :])
        if kind == SaturationType.TANH:
            half = self._get(f"sat_half_{channel}")
            return np.tanh(x_ad / half[None, :])
        return x_ad  # SaturationType.NONE

    # -- forecast -----------------------------------------------------------

    def forecast(
        self,
        X_media_full_raw: np.ndarray,
        X_controls_full_raw: np.ndarray | None,
        positions: np.ndarray,
        *,
        include_noise: bool = True,
        random_seed: int | None = None,
        train_offset: int = 0,
    ) -> np.ndarray:
        """Posterior predictive draws at absolute period ``positions``.

        Parameters
        ----------
        X_media_full_raw : np.ndarray
            Raw media, shape ``(n_full, n_channels)`` -- the FULL history
            (training + forecast periods) so adstock carryover is correct.
        X_controls_full_raw : np.ndarray or None
            Raw controls over the full history (required if the model has
            controls; future control values are assumed known/planned).
        positions : np.ndarray
            Absolute period positions (0-based on the full axis) to forecast.
        train_offset : int
            Absolute position of the trained model's first period. 0 for
            prefix training (the backtest); the window start for
            rolling-window clones (validator cross-validation).

        Returns
        -------
        np.ndarray
            Samples in original KPI scale, shape ``(n_samples, len(positions))``.
        """
        model = self.model
        positions = np.asarray(positions, dtype=int)

        mu = np.zeros((len(positions), self._n_samples))
        if self._intercept is not None:
            mu += self._intercept[None, :]
        mu += self._trend_at(positions, train_offset)
        mu += self._seasonality_at(positions)
        mu += self._media_at(X_media_full_raw, positions)

        if model.n_controls > 0:
            if X_controls_full_raw is None:
                raise ValueError(
                    "Model was fitted with controls; pass X_controls_full_raw."
                )
            x_ctrl = (
                np.asarray(X_controls_full_raw, dtype=float)[positions]
                - model.control_mean
            ) / model.control_std
            if self._beta_controls is not None:
                mu += x_ctrl @ self._beta_controls.T

        if include_noise and self._sigma is not None:
            rng = np.random.default_rng(random_seed)
            mu = mu + rng.normal(0.0, 1.0, size=mu.shape) * self._sigma[None, :]

        y = mu * model.y_std + model.y_mean
        return y.T  # (n_samples, n_pos)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _seasonal_naive_scale(y_train: np.ndarray, season: int) -> float:
    """In-sample one-step seasonal-naive MAE (the MASE denominator)."""
    if len(y_train) <= season:
        diffs = np.abs(np.diff(y_train))  # fall back to the naive scale
    else:
        diffs = np.abs(y_train[season:] - y_train[:-season])
    return float(diffs.mean()) if len(diffs) else float("nan")


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(err) / np.abs(y_true)
        ape = ape[np.isfinite(ape)]
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sape = np.abs(err) / denom
        sape = sape[np.isfinite(sape)]
    return {
        "mape": float(np.mean(ape)) if len(ape) else float("nan"),
        "smape": float(np.mean(sape)) if len(sape) else float("nan"),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Rolling-origin backtest records plus summary views.

    Attributes
    ----------
    records : pd.DataFrame
        One row per (origin, horizon) forecast: ``origin``, ``position``,
        ``date``, ``horizon``, ``y_true``, ``y_pred`` (posterior-mean),
        ``pred_naive``, ``pred_snaive``, per-level interval bounds
        (``lo_80``/``hi_80``, ...) and coverage flags (``cov_80``, ...).
    fits : pd.DataFrame
        One row per refit origin: train size, wall-clock seconds, R-hat max,
        divergences. Check this before trusting the records.
    """

    config: BacktestConfig
    records: pd.DataFrame
    fits: pd.DataFrame
    season_period: int
    mase_scales: dict[int, float] = field(default_factory=dict)

    @property
    def n_origins(self) -> int:
        return int(self.records["origin"].nunique())

    def _mase(self, sub: pd.DataFrame, pred_col: str) -> float:
        """MASE with per-origin in-sample seasonal-naive scaling."""
        ratios = []
        for origin, grp in sub.groupby("origin"):
            scale = self.mase_scales.get(int(origin))
            if not scale or not np.isfinite(scale) or scale <= 0:
                continue
            ratios.append(np.abs(grp[pred_col] - grp["y_true"]).mean() / scale)
        return float(np.mean(ratios)) if ratios else float("nan")

    def summary(self) -> pd.DataFrame:
        """Headline accuracy: the model vs naive baselines, all records."""
        rows = []
        for label, col in [
            ("mmm", "y_pred"),
            ("seasonal_naive", "pred_snaive"),
            ("naive_last_value", "pred_naive"),
        ]:
            m = _point_metrics(
                self.records["y_true"].to_numpy(), self.records[col].to_numpy()
            )
            m["mase"] = self._mase(self.records, col)
            if label == "mmm":
                for level in self.config.coverage_levels:
                    pct = int(round(level * 100))
                    m[f"coverage_{pct}"] = float(self.records[f"cov_{pct}"].mean())
            rows.append({"model": label, **m})
        return pd.DataFrame(rows).set_index("model")

    def by_horizon(self) -> pd.DataFrame:
        """Accuracy and coverage as a function of forecast lead time."""
        rows = []
        for h, grp in self.records.groupby("horizon"):
            m = _point_metrics(grp["y_true"].to_numpy(), grp["y_pred"].to_numpy())
            m_snaive = _point_metrics(
                grp["y_true"].to_numpy(), grp["pred_snaive"].to_numpy()
            )
            row = {"horizon": int(h), **m, "snaive_mape": m_snaive["mape"]}
            for level in self.config.coverage_levels:
                pct = int(round(level * 100))
                row[f"coverage_{pct}"] = float(grp[f"cov_{pct}"].mean())
            rows.append(row)
        return pd.DataFrame(rows).set_index("horizon")

    def by_origin(self) -> pd.DataFrame:
        """Accuracy per refit origin (forecast-window heterogeneity)."""
        rows = []
        for origin, grp in self.records.groupby("origin"):
            m = _point_metrics(grp["y_true"].to_numpy(), grp["y_pred"].to_numpy())
            row = {"origin": int(origin), "n": len(grp), **m}
            for level in self.config.coverage_levels:
                pct = int(round(level * 100))
                row[f"coverage_{pct}"] = float(grp[f"cov_{pct}"].mean())
            rows.append(row)
        return pd.DataFrame(rows).set_index("origin")

    def coverage(self) -> pd.DataFrame:
        """Interval calibration: nominal vs empirical coverage + sharpness."""
        y_scale = float(np.abs(self.records["y_true"]).mean())
        rows = []
        for level in self.config.coverage_levels:
            pct = int(round(level * 100))
            width = (self.records[f"hi_{pct}"] - self.records[f"lo_{pct}"]).mean()
            rows.append(
                {
                    "nominal": level,
                    "empirical": float(self.records[f"cov_{pct}"].mean()),
                    "mean_width": float(width),
                    "mean_width_pct_of_kpi": float(width / y_scale),
                }
            )
        return pd.DataFrame(rows).set_index("nominal")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _slice_panel_prefix(panel: Any, n_train: int) -> Any:
    """First ``n_train`` periods of a national (single-cell) panel."""
    from ..data_loader import PanelCoordinates, PanelDataset

    idx = np.arange(n_train)
    y = panel.y.iloc[idx]
    X_media = panel.X_media.iloc[idx]
    X_controls = panel.X_controls.iloc[idx] if panel.X_controls is not None else None
    new_index = panel.index[idx]

    coords = PanelCoordinates(
        periods=panel.coords.periods[:n_train],
        geographies=panel.coords.geographies,
        products=panel.coords.products,
        channels=panel.coords.channels,
        controls=panel.coords.controls,
    )
    return PanelDataset(
        y=y,
        X_media=X_media,
        X_controls=X_controls,
        index=new_index,
        config=panel.config,
        coords=coords,
    )


def _clone_for_prefix(model: Any, n_train: int) -> Any:
    """A fresh, unfitted BayesianMMM on the first ``n_train`` periods."""
    from ..model import BayesianMMM

    sliced = _slice_panel_prefix(model.panel, n_train)
    # Experiment-calibration likelihoods are intentionally dropped: their
    # estimands are defined on the full-period spend, not a training prefix.
    return BayesianMMM(
        panel=sliced,
        model_config=model.model_config,
        trend_config=model.trend_config,
        adstock_alphas=model.adstock_alphas,
    )


def _seasonal_naive_pred(
    y: np.ndarray, origin: int, positions: np.ndarray, season: int
) -> np.ndarray:
    """Seasonal-naive forecast using only data before ``origin``."""
    out = np.empty(len(positions))
    for i, p in enumerate(positions):
        q = p - season
        while q >= origin:  # never peek past the training cutoff
            q -= season
        out[i] = y[q] if q >= 0 else y[origin - 1]
    return out


def run_backtest(
    model: Any,
    config: BacktestConfig | None = None,
    **fit_kwargs: Any,
) -> BacktestResult:
    """Rolling-origin backtest of a BayesianMMM specification.

    Refits the model's exact configuration on an expanding training window,
    forecasts ``config.horizon`` periods past each cutoff with
    :class:`PosteriorForecaster`, and grades against held-out actuals and
    naive baselines.

    Parameters
    ----------
    model : BayesianMMM
        The full-data model (does not need to be fitted); supplies the panel,
        configuration, and the raw media/control history.
    config : BacktestConfig, optional
        Backtest settings; defaults to :class:`BacktestConfig()`.
    **fit_kwargs
        Extra arguments forwarded to each refit's ``fit()`` (e.g.
        ``progressbar=False``).

    Returns
    -------
    BacktestResult
    """
    config = config or BacktestConfig()
    if model.n_cells > 1:
        raise NotImplementedError(
            "run_backtest supports national (single-cell) data only; "
            f"got {model.n_cells} geo x product cells."
        )

    n_periods = model.n_periods
    origins = rolling_origins(
        n_periods,
        min_train_size=config.min_train_size,
        horizon=config.horizon,
        step=config.step,
        max_origins=config.max_origins,
    )
    if not origins:
        raise ValueError(
            f"No backtest origins fit: n_periods={n_periods}, "
            f"min_train_size={config.min_train_size}, horizon={config.horizon}. "
            "Reduce min_train_size or horizon."
        )

    freq = getattr(model.mff_config, "frequency", "W") or "W"
    season = config.season_period or _SEASON_BY_FREQ.get(freq, 52)

    y_full = np.asarray(model.y_raw, dtype=float)
    X_media_full = np.asarray(model.X_media_raw, dtype=float)
    X_controls_full = (
        np.asarray(model.X_controls_raw, dtype=float)
        if model.X_controls_raw is not None
        else None
    )
    periods = pd.Index(model.panel.coords.periods)

    record_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    mase_scales: dict[int, float] = {}

    for k, origin in enumerate(origins):
        positions = np.arange(origin, min(origin + config.horizon, n_periods))
        logger.info(
            f"Backtest origin {k + 1}/{len(origins)}: "
            f"train=[0, {origin}), forecast={len(positions)} periods"
        )

        clone = _clone_for_prefix(model, origin)
        seed = config.random_seed + k
        t0 = time.time()
        fit = clone.fit(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            random_seed=seed,
            **fit_kwargs,
        )
        elapsed = time.time() - t0
        fit_rows.append(
            {
                "origin": origin,
                "train_size": origin,
                "fit_seconds": elapsed,
                "rhat_max": fit.diagnostics.get("rhat_max", float("nan")),
                "divergences": fit.diagnostics.get("divergences", 0),
            }
        )

        forecaster = PosteriorForecaster(clone)
        samples = forecaster.forecast(
            X_media_full,
            X_controls_full,
            positions,
            include_noise=config.include_noise,
            random_seed=seed,
        )
        y_pred = samples.mean(axis=0)

        y_true = y_full[positions]
        pred_naive = np.full(len(positions), y_full[origin - 1])
        pred_snaive = _seasonal_naive_pred(y_full, origin, positions, season)
        mase_scales[origin] = _seasonal_naive_scale(y_full[:origin], season)

        bounds = {}
        for level in config.coverage_levels:
            pct = int(round(level * 100))
            alpha = (1 - level) / 2
            bounds[pct] = (
                np.percentile(samples, alpha * 100, axis=0),
                np.percentile(samples, (1 - alpha) * 100, axis=0),
            )

        for i, p in enumerate(positions):
            row: dict[str, Any] = {
                "origin": origin,
                "position": int(p),
                "date": periods[p],
                "horizon": i + 1,
                "y_true": float(y_true[i]),
                "y_pred": float(y_pred[i]),
                "pred_naive": float(pred_naive[i]),
                "pred_snaive": float(pred_snaive[i]),
            }
            for pct, (lo, hi) in bounds.items():
                row[f"lo_{pct}"] = float(lo[i])
                row[f"hi_{pct}"] = float(hi[i])
                row[f"cov_{pct}"] = bool(lo[i] <= y_true[i] <= hi[i])
            record_rows.append(row)

    return BacktestResult(
        config=config,
        records=pd.DataFrame(record_rows),
        fits=pd.DataFrame(fit_rows).set_index("origin"),
        season_period=season,
        mase_scales=mase_scales,
    )
