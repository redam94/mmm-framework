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

Scope (enforced with explicit errors, not silent wrong answers).

The forward pass reproduces the fitted mean exactly for the terms it supports,
and :class:`ForecastUnsupportedError` refuses the rest **at construction time**,
so a caller cannot obtain a forecaster it is not allowed to trust:

* supported — national and geo/product panels, every trend family, seasonality,
  parametric and legacy adstock, all saturation families, controls, per-geo
  media coefficients (``vary_media_by_geo``), and the geo/product level offsets;
* refused — price and promotion levers, event/holiday effects, cross-channel
  interactions, reach/frequency channels, time-varying coefficients, and the
  multiplicative specification. Each of these contributes to the fitted mean
  and the forward pass cannot replay it, so dropping it would shift the
  forecast level while leaving the interval width untouched;
* trend extrapolation is a stated *policy*, not a silent default — read
  ``forecaster.trend_extrapolation``. ``LINEAR`` extrapolates in closed form;
  spline/GP/piecewise **hold the last fitted level flat**, so their intervals do
  not widen with horizon;
* experiment-calibration likelihood terms are dropped by the backtest refit
  (their estimands reference the full-period spend), so :func:`run_backtest`
  refuses a calibrated model rather than reporting an uncalibrated model's
  accuracy under the calibrated model's name.

Before v1.3.1 the forward pass summed five of the fitted mean's ten terms and
raised nothing, and the refit downcast any garden/custom model class to a plain
additive ``BayesianMMM``. See the CHANGELOG entry for v1.3.1.

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

from ..frequentist._transforms import SATURATION_PARAMS, saturate
from ..transforms.adstock import adstock_weights, geometric_adstock_2d
from ..transforms.seasonality import create_fourier_features

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "ForecastUnsupportedError",
    "PosteriorForecaster",
    "TrendExtrapolation",
    "rebuild_like",
    "rolling_origins",
    "run_backtest",
]


class ForecastUnsupportedError(NotImplementedError):
    """The fitted model carries a term the forward pass cannot replay.

    Raised instead of silently dropping the term, mirroring
    :class:`mmm_framework.frequentist.design.UnsupportedModelError`. Carries
    ``feature`` so callers (the backtest harness, the agent, the REST job) can
    report *which* configuration blocked the forecast rather than a generic
    refusal, and ``all_unsupported`` so a UI can list every blocker at once
    instead of making the user fix them one at a time.
    """

    def __init__(
        self,
        feature: str,
        reason: str,
        all_unsupported: "list[tuple[str, str]] | None" = None,
    ):
        self.feature = feature
        self.reason = reason
        self.all_unsupported = all_unsupported or [(feature, reason)]
        extra = ""
        if len(self.all_unsupported) > 1:
            others = ", ".join(f for f, _ in self.all_unsupported[1:])
            extra = f" (this model also carries: {others})"
        super().__init__(
            f"{feature} cannot be replayed by the out-of-time forward pass: "
            f"{reason}. Refusing rather than dropping the term, which would "
            f"shift the forecast level while leaving the interval width "
            f"unchanged{extra}."
        )


@dataclass(frozen=True)
class TrendExtrapolation:
    """How the forecaster continues the trend past the training window.

    Recorded rather than assumed, because the three policies carry very
    different forecast semantics and only one of them is model-defined.

    Attributes
    ----------
    policy : {'none', 'linear', 'held_flat'}
        ``held_flat`` is a *heuristic*: a spline/GP/piecewise basis has no
        out-of-time forecast, so the last fitted level is carried forward and
        the interval consequently does not widen with horizon.
    trend_type : str
        The configured trend family.
    n_train_periods : int
        Training length. Load-bearing for ``linear``: the fitted slope is on a
        ``t_scaled = pos / (n_train - 1)`` axis, so the same posterior implies a
        different slope *per period* depending on how long the training panel
        was. Reporting it makes that visible rather than surprising.
    """

    policy: str
    trend_type: str
    n_train_periods: int

    @property
    def is_model_defined(self) -> bool:
        """False when the continuation is a heuristic rather than the model's."""
        return self.policy in ("none", "linear")

    def describe(self) -> str:
        if self.policy == "none":
            return "No trend term; nothing is extrapolated."
        if self.policy == "linear":
            return (
                f"Linear trend extrapolated in closed form on the training time "
                f"scale ({self.n_train_periods} periods)."
            )
        return (
            f"{self.trend_type} trend has no out-of-time forecast; the last "
            f"fitted level ({self.n_train_periods} periods) is HELD FLAT. The "
            "forecast interval therefore does not widen with horizon and "
            "understates long-horizon uncertainty."
        )

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


def audit_forward_pass(model: Any) -> list[tuple[str, str]]:
    """Every fitted-mean term :class:`PosteriorForecaster` cannot replay.

    Returns ``(feature, reason)`` pairs, most-likely-to-be-hit first, so the
    first message a user sees names the thing they actually configured. Empty
    means the forward pass reproduces the fitted mean exactly.

    The reference for "every term" is the mu construction in
    ``model/base.py::_build_model``: ``intercept + trend + seasonality + geo +
    product + media + controls`` plus the conditional ``event``, ``interaction``
    and ``lever`` blocks. Each conditional block below is one of those.
    """
    out: list[tuple[str, str]] = []

    # A subclass that writes its own graph defines its own mu, and this forward
    # pass reproduces only the BASE additive mu. Before class-preserving cloning
    # this could not arise (the clone was always a plain BayesianMMM, so the
    # replay matched the fitted graph even though the graph was the wrong one);
    # now that the real class is rebuilt, an overridden `_build_model` is a live
    # silent-drop path. Example: the awareness garden model registers its own
    # `trend_component` for an organic-level state and never creates
    # `trend_slope`, so the base replay silently forecasts it trend-free.
    #
    # Opt out by declaring `__forecast_forward_pass__ = "base"` on the class,
    # which asserts its mu is exactly the base mu.
    from ..model.base import BayesianMMM

    cls = type(model)
    if (
        getattr(cls, "_build_model", None) is not BayesianMMM._build_model
        and getattr(cls, "__forecast_forward_pass__", None) != "base"
    ):
        out.append(
            (
                f"Custom model graph ({cls.__name__}._build_model)",
                "this class builds its own PyMC graph, so its fitted mean may "
                "carry terms the base additive forward pass does not reproduce. "
                'Declare `__forecast_forward_pass__ = "base"` on the class to '
                "assert its mean is exactly the base mean",
            )
        )

    likelihood = getattr(getattr(model, "model_config", None), "likelihood", None)
    family = getattr(getattr(likelihood, "family", None), "value", None)
    if family is not None and family not in ("normal", "student_t"):
        out.append(
            (
                f"{family} likelihood",
                "the forward pass bridges back to KPI units with "
                "`mu * y_std + y_mean` and adds additive noise, both of which "
                "are the wrong shape under a non-identity link",
            )
        )

    if getattr(model, "_multiplicative", False):
        out.append(
            (
                "Multiplicative specification",
                "the model fits log(y) and combines channels multiplicatively, "
                "so the additive forward pass and its `mu * y_std + y_mean` "
                "bridge back to KPI units are both the wrong shape",
            )
        )

    for ch in getattr(model, "channel_names", []) or []:
        try:
            cfg = model.mff_config.get_media_config(ch)
        except Exception:  # pragma: no cover - defensive
            continue
        if getattr(cfg, "time_varying", False):
            out.append(
                (
                    f"Time-varying coefficient on {ch!r}",
                    "the graph carries `beta_tv_<ch>` (a per-period random "
                    "walk) while the trace's `beta_<ch>` is only its TIME "
                    "AVERAGE, so the forward pass would apply an average "
                    "effectiveness to a channel whose whole point is that its "
                    "effectiveness moved",
                )
            )
            break

    # Gated on the parametric path: the frequency gain reaches mu only through
    # `_channel_media_input`, which the legacy fixed-alpha branch never calls.
    # On the legacy path the graph convolves the raw reach column — exactly what
    # `_legacy_adstock` reproduces — so refusing there would remove a capability
    # that worked. (That the configured gain is inert on the legacy path is a
    # separate pre-existing model bug, not this forward pass's to convert into a
    # forecast refusal.)
    if getattr(model, "_reach_freq", None) and getattr(
        model, "use_parametric_adstock", False
    ):
        out.append(
            (
                "Reach/frequency channels",
                "the frequency gain multiplies the media column before adstock "
                "and carries its own shape parameters; the forward pass "
                "convolves the RAW reach series and would silently omit the gain",
            )
        )

    if getattr(model, "_price_lever", None) is not None or getattr(
        model, "_promo_levers", None
    ):
        out.append(
            (
                "Price / promotion levers",
                "`lever_component` is part of the fitted mean and the forward "
                "pass has no branch for it, so every forecast would omit the "
                "price and promotion effects the model estimated",
            )
        )

    events = getattr(model, "event_features", None)
    if events is not None and getattr(events, "shape", (0, 0))[1] > 0:
        out.append(
            (
                "Event / holiday effects",
                "`event_component` is part of the fitted mean, and replaying it "
                "needs the original EventsConfig plus a rule that an event with "
                "no future occurrence yields a zero column rather than being "
                "dropped — subtle enough to refuse rather than guess",
            )
        )

    if getattr(getattr(model, "model_config", None), "channel_interactions", None):
        out.append(
            (
                "Cross-channel interactions",
                "`interaction_component` is part of the fitted mean and the "
                "forward pass has no branch for it",
            )
        )

    return out


def audit_refit(model: Any) -> list[tuple[str, str]]:
    """What a rolling-origin *refit* would change about the model under test.

    Distinct from :func:`audit_forward_pass`: these terms do not break the
    forward pass, they make the backtest grade a **different model** than the
    one the user holds. Reported by :func:`run_backtest`, not by the forecaster.
    """
    out: list[tuple[str, str]] = []
    if getattr(model, "experiments", None):
        out.append(
            (
                "Experiment calibration",
                "the calibration likelihoods reference full-period spend, so the "
                "refit on a training prefix drops them — the reported accuracy "
                "would be an UNCALIBRATED model's, carrying the calibrated "
                "model's name. Backtest the uncalibrated specification "
                "explicitly if that is what you want to measure",
            )
        )
    return out


def _raise_first(problems: list[tuple[str, str]]) -> None:
    if problems:
        feature, reason = problems[0]
        raise ForecastUnsupportedError(feature, reason, all_unsupported=problems)


class PosteriorForecaster:
    """Out-of-time forecasts from a fitted BayesianMMM's posterior draws.

    Replays the model's structural forward pass (intercept, trend, seasonality,
    geo and product level offsets, adstock, saturation, controls, observation
    noise) in NumPy at arbitrary future period positions, using the full spend
    history for adstock carryover.

    The replay is *exact*: over training positions, ``forecast(include_noise=
    False)`` equals the sum of the model's registered component Deterministics
    to floating-point tolerance. Terms it cannot replay are refused here, in the
    constructor, rather than at forecast time — so a caller cannot hold a
    forecaster whose output it is not allowed to trust.

    Parameters
    ----------
    model : BayesianMMM
        A *fitted* model, possibly trained on a prefix of the full series.
    strict : bool, default True
        When ``True``, an unreplayable term raises
        :class:`ForecastUnsupportedError`. ``False`` downgrades the refusal to a
        warning and records it on ``self.unsupported`` — for diagnostic use on a
        model you already know is incomplete. **Never use it for planning**: the
        resulting forecast omits a term the model estimated.

    Attributes
    ----------
    trend_extrapolation : TrendExtrapolation
        The stated policy for continuing the trend past the training window.
    unsupported : list[tuple[str, str]]
        Empty unless ``strict=False`` suppressed a refusal.

    Raises
    ------
    ForecastUnsupportedError
        The model carries a term the forward pass cannot replay.
    """

    def __init__(self, model: Any, *, strict: bool = True):
        # The audit reads CONFIGURATION, not the trace, so it runs first: a user
        # who configured price levers should be told the forecast is unavailable
        # before paying for the fit, not after.
        self.unsupported = audit_forward_pass(model)
        if self.unsupported:
            if strict:
                _raise_first(self.unsupported)
            for feature, reason in self.unsupported:
                logger.warning(
                    f"PosteriorForecaster(strict=False): {feature} is part of "
                    f"the fitted mean and will be OMITTED from the forecast — "
                    f"{reason}."
                )

        if model._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self._trend_type = str(
            getattr(model.trend_config.type, "value", model.trend_config.type)
        ).lower()

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
        # Student-t degrees of freedom, when that is the fitted family — read
        # once so _draw_noise stays a pure dispatch.
        _lik = getattr(model.model_config, "likelihood", None)
        _fam = getattr(getattr(_lik, "family", None), "value", None)
        self._nu = (
            float((getattr(_lik, "params", None) or {}).get("nu", 4.0))
            if _fam == "student_t"
            else None
        )
        self._trend_slope = get("trend_slope")
        # Flexible trends (spline / GP / piecewise) have no closed-form out-of-time
        # extrapolation, so we replay the fitted per-period trend component and hold
        # its LAST value beyond the training window (see _trend_at).
        # `trend_component` is registered per-OBS (model/base.py: every flexible
        # branch ends in `trend_unique[time_idx]`), NOT per-period. Obs are
        # period-major / cell-minor and the trend is constant across cells within
        # a period, so collapse to one value per period. Without this, a geo panel
        # indexed `tc[:, period]` reads obs `period` — i.e. period `period //
        # n_cells` — stretching the trend by a factor of n_cells (measured: 12.6%
        # of KPI level on a 6-cell spline panel) AND making the documented
        # hold-last-flat clamp unreachable, since positions never reach n_obs - 1.
        tc = get("trend_component")
        n_cells_ = max(int(getattr(model, "n_cells", 1) or 1), 1)
        if tc is not None and n_cells_ > 1 and tc.shape[-1] == n_cells_ * int(
            getattr(model, "n_periods", 0) or 0
        ):
            tc = tc[:, ::n_cells_]
        self._trend_component = tc
        self._beta_controls = get("beta_controls")
        # Per-obs geo and product offsets (each constant within its cell). The obs
        # layout is period-major / cell-minor (obs = period*n_cells + cell), so the
        # first n_cells obs carry one offset per cell — see _cell_offsets.
        # ``product_component`` was previously never read, so on a geo x product
        # panel with product pooling the forecast omitted the product level offset.
        self._geo_component = get("geo_component")
        self._product_component = get("product_component")
        self._season = {
            name: get(f"season_{name}") for name in model.seasonality_features
        }

        # Policy mirrors _trend_at's three branches exactly. A 'linear' trend
        # whose slope is absent from the trace degrades to 'none' there, so it
        # must degrade here too or the reported policy would be a fiction.
        if self._trend_type == "none":
            policy = "none"
        elif self._trend_type == "linear":
            policy = "none" if self._trend_slope is None else "linear"
        else:
            policy = "held_flat"
        self.trend_extrapolation = TrendExtrapolation(
            policy=policy,
            trend_type=self._trend_type,
            n_train_periods=int(getattr(model, "n_periods", 0) or 0),
        )
        if (
            self.trend_extrapolation.policy == "held_flat"
            and self._trend_component is None
        ):
            raise ForecastUnsupportedError(
                f"{self._trend_type} trend",
                "the trace carries no 'trend_component', so there is no fitted "
                "per-period level to carry forward",
            )

    @property
    def n_samples(self) -> int:
        return self._n_samples

    # -- components ---------------------------------------------------------

    def _trend_at(self, positions: np.ndarray, train_offset: int = 0) -> np.ndarray:
        """Trend evaluated at absolute ``positions``.

        ``train_offset`` is the absolute position of the trained model's first
        period — nonzero for rolling-window training (e.g. validator CV), where
        the clone's ``t_scaled = 0`` corresponds to that offset, not position 0.

        * ``none``   -> zero.
        * ``linear`` -> slope extrapolated on the training time scale (closed form).
        * spline/GP/piecewise -> replay the fitted ``trend_component`` and HOLD ITS
          LAST value beyond the training window. A flexible basis has no
          model-defined out-of-time forecast, so this is a documented heuristic
          (no further trend growth assumed); interpret long-horizon backtests of
          flexible-trend models with that caveat.
        """
        if self._trend_type == "none":
            return np.zeros((len(positions), self._n_samples))

        if self._trend_type == "linear":
            if self._trend_slope is None:
                return np.zeros((len(positions), self._n_samples))
            # Training used t_scaled = linspace(0,1,n_train) = pos/(n_train-1);
            # future positions extrapolate past 1 on the same scale.
            denom = max(self.model.n_periods - 1, 1)
            t_scaled = (positions - train_offset) / denom
            return t_scaled[:, None] * self._trend_slope[None, :]

        # Flexible trend: index the fitted component, clamping to [0, n_train-1]
        # so positions beyond the training window hold the last fitted level.
        tc = self._trend_component  # (n_samples, n_train)
        if tc is None:
            return np.zeros((len(positions), self._n_samples))
        n_train = tc.shape[-1]
        idx = np.clip(positions - train_offset, 0, n_train - 1)
        return tc[:, idx].T

    def _seasonality_at(
        self, positions: np.ndarray, train_offset: int = 0
    ) -> np.ndarray:
        """Fourier seasonality evaluated at absolute period positions.

        ``train_offset`` mirrors :meth:`_trend_at`: the model built its features
        on ``t = np.arange(n_periods)`` starting at 0, so for a clone whose
        window starts at absolute period ``s`` the basis must be evaluated at
        ``position - s``. It is 0 on the backtest path (every prefix starts at
        period 0) but nonzero for the validator's ROLLING-window cross
        validation, which previously evaluated the basis ``s`` periods out of
        phase.
        """
        positions = np.asarray(positions) - train_offset
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
        self,
        X_media_full_raw: np.ndarray,
        positions: np.ndarray,
        cell: int | None = None,
    ) -> np.ndarray:
        """Sum of saturated channel contributions, (n_pos, n_samples).

        ``cell`` selects the per-geo coefficient when the model was fit with
        per-geo effectiveness (V3): ``beta_{ch}`` is then (n_samples, n_geos) and
        we index column ``cell``. A scalar (n_samples,) beta is geo-shared and
        used for every cell.

        Delegates to :meth:`media_by_channel_at` and sums, so a per-channel
        decomposition can never drift from the pooled path it is supposed to
        decompose — the same "summed rather than assumed" discipline the geo
        level-offsets use above.
        """
        per_channel = self.media_by_channel_at(X_media_full_raw, positions, cell)
        out = np.zeros((len(positions), self._n_samples))
        for contrib in per_channel.values():
            out += contrib
        return out

    def media_by_channel_at(
        self,
        X_media_full_raw: np.ndarray,
        positions: np.ndarray,
        cell: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Saturated contribution PER CHANNEL, each ``(n_pos, n_samples)``.

        The decomposition behind :meth:`_media_at`. A channel whose ``beta_`` is
        absent from the trace is omitted rather than zero-filled, so a caller
        cannot mistake "not estimated" for "estimated at zero".
        """
        model = self.model
        out: dict[str, np.ndarray] = {}
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
            if beta.ndim == 2:  # per-geo (n_samples, n_geos)
                if cell is not None:
                    # cells are period-major geo×product; map cell -> geo column.
                    n_products = getattr(model, "n_products", 1) or 1
                    beta = beta[:, cell // n_products]
                else:
                    beta = beta.mean(axis=1)
            out[ch] = x_sat * beta[None, :]
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
        """Apply the channel's saturation, per posterior draw.

        Delegates to :func:`mmm_framework.frequentist._transforms.saturate`, the
        single numpy mirror of
        :func:`~mmm_framework.model.base._apply_saturation_pt`, so the forecaster
        and the frequentist design matrix (#180) cannot drift from the graph or
        from each other.

        This method previously carried its own copy covering four families and
        returning ``x_ad`` unchanged for anything else — so a channel configured
        with ``SaturationType.ROOT`` was forecast **unsaturated**, and every
        backtest metric on such a model was wrong. Dispatching through the shared
        table makes an unhandled family a loud error rather than a silent
        identity.

        Posterior parameters arrive shaped ``(n_draws,)`` and are reshaped to
        ``(1, n_draws)`` to broadcast against ``x_ad``'s ``(n_obs, n_draws)``.
        """
        kind = self.model._get_saturation_config(channel).type
        params: dict[str, np.ndarray] = {}
        for name in SATURATION_PARAMS[kind]:
            values = self._get(f"{name}_{channel}")
            if values is None:
                raise KeyError(
                    f"Saturation parameter '{name}_{channel}' is not in the "
                    f"trace, so channel '{channel}' ({kind.value} saturation) "
                    "cannot be forecast. Was the model fitted with a different "
                    "saturation configuration than the one it now carries?"
                )
            params[name] = np.asarray(values)[None, :]
        return saturate(x_ad, kind, **params)

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

        if model.n_cells > 1:
            return self._forecast_geo(
                X_media_full_raw,
                X_controls_full_raw,
                positions,
                include_noise=include_noise,
                random_seed=random_seed,
                train_offset=train_offset,
            )

        mu = np.zeros((len(positions), self._n_samples))
        if self._intercept is not None:
            mu += self._intercept[None, :]
        mu += self._trend_at(positions, train_offset)
        mu += self._seasonality_at(positions, train_offset)
        # Exactly zero on a true national panel (one cell => no geo/product
        # hierarchy), but summed rather than assumed so the component-sum
        # identity holds structurally rather than by coincidence.
        mu += self._level_offsets()[:, 0][None, :]  # (1, n_samples)
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
            mu = mu + self._draw_noise(mu.shape, random_seed) * self._sigma[None, :]

        y = mu * model.y_std + model.y_mean
        return y.T  # (n_samples, n_pos)

    def _draw_noise(self, shape: tuple[int, ...], random_seed: int | None) -> np.ndarray:
        """Standardized observation noise matching the fitted likelihood family.

        ``include_noise=True`` exists so ``BacktestResult.coverage()`` grades the
        *predictive* distribution. Drawing Gaussian noise for a Student-t fit
        would understate the tails and report interval coverage that is simply
        not the model's — so the family is honoured here rather than assumed.
        """
        rng = np.random.default_rng(random_seed)
        if self._nu is not None:
            # StudentT(nu, mu, sigma) — sigma is the SCALE, not the sd.
            return rng.standard_t(self._nu, size=shape)
        return rng.normal(0.0, 1.0, size=shape)

    def _cell_offsets(self, component: np.ndarray | None) -> np.ndarray:
        """Per-cell level offset, ``(n_samples, n_cells)``.

        ``geo_component`` and ``product_component`` are each constant within a
        cell, and obs are period-major / cell-minor (obs = period*n_cells +
        cell), so the first ``n_cells`` obs are period 0's cells — one offset
        per cell.
        """
        n_cells = self.model.n_cells
        if component is None or component.shape[-1] < n_cells:
            return np.zeros((self._n_samples, n_cells))
        return component[:, :n_cells]

    def _geo_offsets(self) -> np.ndarray:
        """Per-cell geo offset, ``(n_samples, n_cells)``. Back-compat alias."""
        return self._cell_offsets(self._geo_component)

    def _level_offsets(self) -> np.ndarray:
        """Combined geo + product per-cell level offset, ``(n_samples, n_cells)``.

        Both enter the fitted mean as ``effect[idx_data]`` level terms, so the
        forecast needs their sum. ``product_component`` was omitted entirely
        before v1.3.1.
        """
        return self._cell_offsets(self._geo_component) + self._cell_offsets(
            self._product_component
        )

    def _forecast_geo(
        self,
        X_media_full_raw: np.ndarray,
        X_controls_full_raw: np.ndarray | None,
        obs_positions: np.ndarray,
        *,
        include_noise: bool,
        random_seed: int | None,
        train_offset: int,
    ) -> np.ndarray:
        """Geo-panel forward pass. Reuses the single-cell components PER CELL.

        The geo mean is the national mean plus a per-cell offset, with each cell's
        OWN media series (so adstock carryover stays within a cell — no cross-geo
        bleed). Obs are period-major / cell-minor; we build the full (period, cell)
        grid and select the requested obs.
        """
        model = self.model
        n_cells = model.n_cells
        n_obs = X_media_full_raw.shape[0]
        n_full = n_obs // n_cells
        all_periods = np.arange(n_full)

        Xm = np.asarray(X_media_full_raw, dtype=float).reshape(n_full, n_cells, -1)
        Xc = (
            np.asarray(X_controls_full_raw, dtype=float).reshape(n_full, n_cells, -1)
            if X_controls_full_raw is not None
            else None
        )

        # Shared (across cells) components, evaluated on the period axis.
        shared = np.zeros((n_full, self._n_samples))
        if self._intercept is not None:
            shared += self._intercept[None, :]
        shared += self._trend_at(all_periods, train_offset)
        shared += self._seasonality_at(all_periods, train_offset)

        geo_off = self._level_offsets()  # (n_samples, n_cells): geo + product

        mu_grid = np.empty((n_full, n_cells, self._n_samples))
        for j in range(n_cells):
            mu_j = shared + geo_off[None, :, j]
            # per-cell adstock; cell=j selects this geo's coefficient under V3
            mu_j = mu_j + self._media_at(Xm[:, j, :], all_periods, cell=j)
            if model.n_controls > 0:
                if Xc is None:
                    raise ValueError(
                        "Model was fitted with controls; pass X_controls_full_raw."
                    )
                x_ctrl = (Xc[:, j, :] - model.control_mean) / model.control_std
                if self._beta_controls is not None:
                    mu_j = mu_j + x_ctrl @ self._beta_controls.T
            mu_grid[:, j, :] = mu_j

        mu_obs = mu_grid.reshape(n_full * n_cells, self._n_samples)[obs_positions]

        if include_noise and self._sigma is not None:
            mu_obs = (
                mu_obs
                + self._draw_noise(mu_obs.shape, random_seed) * self._sigma[None, :]
            )

        return (mu_obs * model.y_std + model.y_mean).T  # (n_samples, n_pos)


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

    @property
    def mape(self) -> float:
        """Headline out-of-time MAPE of the MMM over all records (the number
        the docs quote). Convenience for ``summary().loc["mmm", "mape"]``."""
        m = _point_metrics(
            self.records["y_true"].to_numpy(), self.records["y_pred"].to_numpy()
        )
        return float(m["mape"])

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
    """First ``n_train`` periods of the panel (national OR geo/product).

    Obs are period-major / cell-minor, so the first ``n_train`` periods are the
    first ``n_train * n_cells`` obs.
    """
    from ..data_loader import PanelCoordinates, PanelDataset

    n_periods = panel.coords.n_periods
    n_cells = max(len(panel.y) // n_periods, 1)
    idx = np.arange(n_train * n_cells)
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


def rebuild_like(model: Any, panel: Any, **overrides: Any) -> Any:
    """A fresh, unfitted model of ``type(model)`` on ``panel``.

    The single place that reconstructs a model from another one. Preserving the
    CLASS and ``model_params`` is load-bearing: hard-constructing ``BayesianMMM``
    means a garden or custom model is fit and graded as a plain additive MMM and
    its results reported under the custom model's name.

    Raises
    ------
    ForecastUnsupportedError
        The class cannot be rebuilt from ``(panel, model_config, trend_config,
        adstock_alphas, model_params)`` — e.g. a model declaring
        ``REQUIRED_DATASET_CAPABILITIES`` the sliced panel no longer satisfies.
        Refusing beats falling back to a different model class.
    """
    cls = type(model)
    kwargs: dict[str, Any] = {
        "panel": panel,
        "model_config": model.model_config,
        "trend_config": model.trend_config,
        "adstock_alphas": model.adstock_alphas,
    }
    if getattr(model, "model_params", None) is not None:
        kwargs["model_params"] = model.model_params
    kwargs.update(overrides)
    try:
        return cls(**kwargs)
    except Exception as exc:
        raise ForecastUnsupportedError(
            f"Model class {cls.__name__}",
            "it cannot be reconstructed on a modified panel from "
            "(panel, model_config, trend_config, adstock_alphas, model_params) "
            f"— {type(exc).__name__}: {exc}. Refusing rather than falling back "
            "to a plain BayesianMMM, which would report a different model's "
            "results under this model's name",
        ) from exc


def _clone_for_prefix(model: Any, n_train: int) -> Any:
    """A fresh, unfitted model **of the same class** on the first ``n_train`` periods.

    Preserving ``type(model)`` is load-bearing. This used to hard-construct
    ``BayesianMMM(...)``, so backtesting a garden or custom model
    (:class:`~mmm_framework.garden.base.CustomMMM` subclasses such as
    ``LatentFactorMMM`` or the awareness model) silently fit and graded a plain
    additive MMM and reported its MAPE under the custom model's name.

    ``model_params`` (a garden model's ``CONFIG_SCHEMA`` payload) is forwarded
    for the same reason: without it the clone falls back to schema defaults and
    grades a differently-configured model.

    Experiment-calibration likelihoods are still dropped — their estimands are
    defined on the full-period spend — which is why :func:`run_backtest` refuses
    a calibrated model up front via :func:`audit_refit` rather than quietly
    grading the uncalibrated one.
    """
    return rebuild_like(model, _slice_panel_prefix(model.panel, n_train))


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

    # Audit the ORIGINAL model, before the refit loop: _clone_for_prefix drops
    # experiment calibration, so auditing only the clone would never see it, and
    # a refusal is worth far more before N expensive refits than after them.
    _raise_first(audit_forward_pass(model) + audit_refit(model))

    # Obs are period-major / cell-minor: periods [a, b) over all cells map to the
    # contiguous obs block [a*cells, b*cells). cells == 1 reduces to national.
    cells = max(model.n_cells, 1)

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
        end = min(origin + config.horizon, n_periods)
        n_fc_periods = end - origin
        # Obs for periods [origin, end) over all cells (contiguous, period-major).
        positions = np.arange(origin * cells, end * cells)
        logger.info(
            f"Backtest origin {k + 1}/{len(origins)}: "
            f"train=[0, {origin}), forecast={n_fc_periods} periods x {cells} cell(s)"
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
        # Last-value baseline per cell (the last training period's value for each
        # cell), tiled across the forecast periods.
        last_block = y_full[(origin - 1) * cells : origin * cells]
        pred_naive = np.tile(last_block, n_fc_periods)
        # Seasonal-naive + MASE scale in OBS space with an obs stride of season*cells
        # (period-major layout => same cell, `season` periods back).
        pred_snaive = _seasonal_naive_pred(
            y_full, origin * cells, positions, season * cells
        )
        mase_scales[origin] = _seasonal_naive_scale(
            y_full[: origin * cells], season * cells
        )

        bounds = {}
        for level in config.coverage_levels:
            pct = int(round(level * 100))
            alpha = (1 - level) / 2
            bounds[pct] = (
                np.percentile(samples, alpha * 100, axis=0),
                np.percentile(samples, (1 - alpha) * 100, axis=0),
            )

        for i, p in enumerate(positions):
            pp, cell = divmod(int(p), cells)  # obs -> (period, cell)
            row: dict[str, Any] = {
                "origin": origin,
                "position": int(p),
                "cell": cell,
                "date": periods[pp],
                "horizon": pp - origin + 1,
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
