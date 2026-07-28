"""Forward forecast under a spend plan, with the caveats that make it readable.

The framework could replay a fitted model forward — ``PosteriorForecaster`` does
exactly that — but only ever backwards, to grade itself. Nothing turned a
*future* plan into a forecast, even though ``planning/flighting.py`` already
produces the per-period forward spend matrix such a call needs.

The reason this module is small and its caveat block is large: **a single mean
line with a tight band reads like a measurement, and it is a counterfactual
under a plan the model has never observed.** Three specific ways the band lies,
each answered here by a computed field rather than a sentence:

1. **A flexible trend does not extrapolate.** Spline / GP / piecewise bases have
   no out-of-time continuation, so the forecaster holds the last fitted level
   flat. The consequence is not that the level is wrong — it is that the
   interval *does not widen with horizon at all*. A 26-week-out band the same
   width as a 1-week-out band is not a forecast of week 26.
2. **The observation noise is iid.** ``residual_diagnostics`` routinely finds
   autocorrelation on the same residuals, and #186 measured 79.6% vs 90.4%
   coverage at ρ=0.6 for exactly this. The Ljung-Box p is measured and reported;
   it is deliberately NOT corrected by a fudge factor, because an AR(1)
   predictive variance is a modelling change (v1.5), and because a widened band
   with no stated reason is worse than a narrow one with a stated reason.
3. **A plan above the observed spend extrapolates the saturation FORM.** Future
   media is normalized by the *training* max, so a channel funded past what the
   model ever saw is being asked about a curve region that has no data. Flagged
   per channel, reusing the optimizer's own extrapolation machinery.

Hence the governing rule, enforced structurally: :meth:`ForecastResult.headline`
**raises** when the caveats were not computed. A caller cannot obtain the number
without the block that qualifies it.

Whether a forecast may be *committed* is a separate question, and deliberately
not answered here — that gate belongs to the plan-of-record work. This module's
job is to make the forecast computable and honestly described.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ForecastCaveats",
    "ForecastResult",
    "forecast_under_plan",
]

#: Per-period draws are stored base64 float32, thinned — the encoding
#: ``reporting/interactive/facts.py`` uses. A committed plan must keep per-period
#: DRAWS (a window-total interval cannot be recovered by summing per-period
#: bounds), and 4000x52 float64 per plan version is how the sessions.db bloat
#: incident started.
DEFAULT_MAX_DRAWS = 200

#: Below this many posterior draws there is nothing to take quantiles of, and a
#: collapsed interval misreads as precision (#249). A MAP fit has exactly one.
MIN_DRAWS_FOR_INTERVAL = 10


def _encode_draws(draws: np.ndarray) -> str:
    """(n_draws, n_periods) float32 little-endian, base64."""
    arr = np.ascontiguousarray(np.asarray(draws, dtype="<f4"))
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _thin_index(n: int, max_draws: int) -> np.ndarray:
    """Which draws to keep. Returned as an index so the SAME subset can be
    applied to the per-channel decomposition — a summary computed over a
    different subset than the one stored would leave the artifact and the
    headline disagreeing about the same forecast."""
    if max_draws <= 0 or n <= max_draws:
        return np.arange(n)
    return np.linspace(0, n - 1, max_draws).astype(int)


@dataclass
class ForecastCaveats:
    """Everything that qualifies the number, computed rather than written.

    Every field here is derived from the fitted model or the plan. None of it is
    a template sentence, because a template sentence is equally true of a good
    forecast and a bad one.
    """

    #: ``{policy, trend_type, n_train_periods}`` — the forecaster's stated policy.
    trend_extrapolation: dict[str, Any]
    #: True when the trend policy produces a band that cannot widen with horizon.
    interval_widens_with_horizon: bool
    #: Channels planned above the spend the model observed, with the multiple.
    extrapolated_channels: list[dict[str, Any]]
    #: ``{ljung_box_p, lag, autocorrelated}`` — measured on training residuals.
    residual_autocorrelation: dict[str, Any]
    #: Interval vocabulary from ``diagnostics.provenance`` — a bootstrap trace
    #: carries the same variable names and would otherwise render a CONFIDENCE
    #: interval as a credible one.
    interval_noun: str
    interval_phrase: str
    inference_family: str
    #: Fit provenance, folded the way ``extractors/base._merge_fit_provenance`` does.
    approximate: bool
    fit_method: str | None
    #: Terms the forward pass could not replay (only when ``strict=False``).
    unsupported: list[Any] = field(default_factory=list)
    #: False when the posterior has too few draws to estimate an interval at
    #: all — a MAP fit has ONE. Following #249: a collapsed interval is the
    #: visual language of an extremely precise estimate, which is the opposite
    #: of what a single-draw posterior means, so it is reported as absent.
    interval_available: bool = True
    n_posterior_draws: int | None = None

    def statements(self) -> list[str]:
        """The caveats as sentences, each derived from a measured field."""
        out: list[str] = []
        if not self.interval_available:
            out.append(
                f"No interval: this posterior has {self.n_posterior_draws} draw(s), "
                "so there is nothing to take quantiles of. The bounds would "
                "collapse onto the point estimate and read as extreme precision "
                "— the opposite of what this fit means."
            )
        pol = self.trend_extrapolation.get("policy")
        if not self.interval_widens_with_horizon:
            out.append(
                f"The {self.trend_extrapolation.get('trend_type')} trend has no "
                f"out-of-time continuation, so it is {pol}: the interval does "
                "not widen with horizon and a far-out period is no less certain "
                "than a near one, which cannot be true."
            )
        if self.residual_autocorrelation.get("autocorrelated"):
            p = self.residual_autocorrelation.get("ljung_box_p")
            out.append(
                f"Training residuals are autocorrelated (Ljung-Box p={p:.3g}), "
                "and the predictive noise is iid, so this interval is TOO "
                "NARROW. It is reported uncorrected rather than widened by an "
                "unstated factor."
            )
        if self.extrapolated_channels:
            names = ", ".join(
                f"{c['channel']} ({c['multiple']:.2f}x observed max)"
                for c in self.extrapolated_channels
            )
            out.append(
                f"Planned above observed spend: {names}. The saturation curve "
                "has no data in that region, so the shape is extrapolated."
            )
        if self.approximate:
            out.append(
                f"Fitted with {self.fit_method}, an approximate method — the "
                "uncertainty is not calibrated. Re-fit with NUTS before "
                "committing to this interval."
            )
        if self.unsupported:
            out.append(
                "The forward pass could not replay: "
                + ", ".join(str(u) for u in self.unsupported)
                + ". The level is shifted by the omitted term(s)."
            )
        return out


@dataclass
class ForecastResult:
    """A forecast, its interval, its decomposition, and its caveats."""

    periods: list[str]
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    interval: float
    #: Per-channel media contribution (posterior mean), original KPI scale.
    by_channel: dict[str, np.ndarray]
    #: Everything that is not media: intercept + trend + seasonality + controls.
    baseline: np.ndarray
    n_draws: int
    #: base64 float32 ``(n_draws, n_periods)`` — see DEFAULT_MAX_DRAWS.
    draws_b64: str
    caveats: ForecastCaveats | None = None
    calendar: Any = None

    def headline(self) -> dict[str, Any]:
        """The number a reader would quote — and its caveats, inseparably.

        Raises when caveats were not computed. This is the module's governing
        rule made structural: there is no code path that yields the headline
        without the block that qualifies it.
        """
        if self.caveats is None:
            raise ValueError(
                "This forecast has no caveat block, so its headline cannot be "
                "rendered. A forecast is a counterfactual under a plan the "
                "model never observed; the interval's known failure modes "
                "(trend extrapolation policy, residual autocorrelation, spend "
                "beyond observed support) travel with the number or the number "
                "does not travel. Build it with forecast_under_plan(), which "
                "always computes them."
            )
        has_interval = self.caveats.interval_available
        lo, hi = self.window_total_interval() if has_interval else (None, None)
        return {
            "total": float(np.sum(self.mean)),
            # The window total's interval comes from summing DRAWS, not bounds:
            # the periods are correlated under the posterior and their errors
            # partly cancel, so summing per-period bounds gives a wider,
            # wrong number.
            "total_lower": lo,
            "total_upper": hi,
            "interval": self.interval if has_interval else None,
            "interval_available": has_interval,
            "interval_noun": self.caveats.interval_noun,
            "n_periods": len(self.periods),
            "periods": list(self.periods),
            "caveats": self.caveats.statements(),
        }

    def draws(self) -> np.ndarray:
        """Decode the stored per-period draws, ``(n_draws, n_periods)``."""
        raw = base64.b64decode(self.draws_b64)
        return np.frombuffer(raw, dtype="<f4").reshape(self.n_draws, len(self.periods))

    def window_total_interval(self, interval: float | None = None) -> tuple[float, float]:
        """Interval on the WINDOW TOTAL — the reason draws are stored.

        Summing the per-period bounds would give a wider, wrong interval: the
        periods are not independent under the posterior, and their errors partly
        cancel. This sums per draw and then takes quantiles.
        """
        p = self.interval if interval is None else interval
        totals = self.draws().sum(axis=1)
        lo, hi = (1 - p) / 2 * 100, (1 + p) / 2 * 100
        return float(np.percentile(totals, lo)), float(np.percentile(totals, hi))


# ---------------------------------------------------------------------------
# plan normalization
# ---------------------------------------------------------------------------


def _normalize_plan(
    future_media: Any, channel_names: list[str]
) -> tuple[np.ndarray, int]:
    """``(n_future, n_channels)`` raw spend from any of the accepted shapes.

    Accepts ``{channel: [per-period]}``, a flighting result (``{'by_channel':
    ...}`` or ``{'schedule': [...]}``), or an array already in channel order.
    A channel the model does not know is refused by name rather than dropped —
    a silently ignored plan line is a forecast of a different plan.
    """
    if isinstance(future_media, np.ndarray):
        arr = np.asarray(future_media, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(channel_names):
            raise ValueError(
                f"future_media array must be (n_periods, {len(channel_names)}) "
                f"in channel order {channel_names}; got {arr.shape}."
            )
        return arr, arr.shape[0]

    by_channel: dict[str, Any]
    if isinstance(future_media, dict) and "by_channel" in future_media:
        by_channel = dict(future_media["by_channel"])
    elif isinstance(future_media, dict) and "schedule" in future_media:
        rows = list(future_media["schedule"])
        by_channel = {
            ch: [float(r.get(ch, 0.0)) for r in rows]
            for ch in future_media.get("channels", channel_names)
        }
    elif isinstance(future_media, dict):
        by_channel = dict(future_media)
    else:
        raise TypeError(
            "future_media must be {channel: [per-period]}, a flighting schedule "
            "dict, or an (n_periods, n_channels) array."
        )

    unknown = [c for c in by_channel if c not in channel_names]
    if unknown:
        raise ValueError(
            f"Plan names channels the model does not have: {unknown}. "
            f"Modeled channels: {channel_names}. A dropped plan line would make "
            "this a forecast of a different plan."
        )
    lengths = {len(list(v)) for v in by_channel.values()}
    if not lengths:
        raise ValueError("future_media is empty — nothing to forecast.")
    if len(lengths) > 1:
        raise ValueError(
            f"All channels must plan the same number of periods; got {sorted(lengths)}."
        )
    n_future = lengths.pop()
    if n_future < 1:
        raise ValueError("future_media must cover at least one period.")

    out = np.zeros((n_future, len(channel_names)), dtype=float)
    for c, ch in enumerate(channel_names):
        if ch in by_channel:
            out[:, c] = np.asarray(list(by_channel[ch]), dtype=float)
    return out, n_future


# ---------------------------------------------------------------------------
# caveat computation
# ---------------------------------------------------------------------------


def _ljung_box(residuals: np.ndarray, lag: int = 12) -> tuple[float | None, int]:
    """Ljung-Box p at ``lag`` (numpy only). ``None`` when too short to test."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    lag = int(min(lag, max(1, n // 5)))
    if n < 3 * lag or lag < 1:
        return None, lag
    r = r - r.mean()
    denom = float(np.sum(r * r))
    if denom <= 0:
        return None, lag
    q = 0.0
    for k in range(1, lag + 1):
        rho = float(np.sum(r[k:] * r[:-k]) / denom)
        q += rho * rho / (n - k)
    q *= n * (n + 2)
    # chi2 survival with `lag` dof, via the regularized upper incomplete gamma.
    from math import erfc, exp, lgamma, sqrt

    def _chi2_sf(x: float, k: int) -> float:
        if x <= 0:
            return 1.0
        if k == 1:
            return erfc(sqrt(x / 2.0))
        if k == 2:
            return exp(-x / 2.0)
        # series for the lower regularized gamma P(k/2, x/2)
        a, z = k / 2.0, x / 2.0
        if z < a + 1.0:
            term = 1.0 / a
            total = term
            for i in range(1, 500):
                term *= z / (a + i)
                total += term
                if abs(term) < abs(total) * 1e-14:
                    break
            p = total * exp(-z + a * np.log(z) - lgamma(a))
            return float(max(0.0, min(1.0, 1.0 - p)))
        # continued fraction for Q(a, z)
        tiny = 1e-300
        b, c, d = z + 1.0 - a, 1.0 / tiny, 1.0 / (z + 1.0 - a)
        h = d
        for i in range(1, 500):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-14:
                break
        q_val = exp(-z + a * np.log(z) - lgamma(a)) * h
        return float(max(0.0, min(1.0, q_val)))

    return _chi2_sf(q, lag), lag


def _residual_autocorrelation(model: Any) -> dict[str, Any]:
    """Ljung-Box on the fitted model's training residuals."""
    out: dict[str, Any] = {"ljung_box_p": None, "lag": None, "autocorrelated": None}
    try:
        trace = getattr(model, "_trace", None)
        if trace is None:
            return out
        post = trace.posterior
        if "mu" in post:
            fitted = np.asarray(post["mu"].mean(dim=["chain", "draw"]).values)
        else:
            comp = model.compute_component_decomposition()
            fitted = np.asarray(comp.fitted_mean())
        obs = np.asarray(model.y_raw, dtype=float)
        if fitted.shape != obs.shape:
            fitted = np.asarray(fitted).reshape(-1)[: obs.size]
        resid = obs - (fitted * model.y_std + model.y_mean if fitted.max() < 50 else fitted)
        p, lag = _ljung_box(resid)
        out.update(
            ljung_box_p=p,
            lag=lag,
            autocorrelated=(None if p is None else bool(p < 0.05)),
        )
    except Exception:  # noqa: BLE001 — a caveat must never fail the forecast
        pass
    return out


def _extrapolated_channels(
    model: Any, future_media: np.ndarray, channel_names: list[str]
) -> list[dict[str, Any]]:
    """Channels planned above the max spend the model observed.

    Future media is normalized by the TRAINING max inside the forward pass, so a
    plan above it asks the saturation curve about a region with no data —
    the same test the budget optimizer applies to its own recommendations.
    """
    out: list[dict[str, Any]] = []
    raw_max = getattr(model, "_media_raw_max", None) or {}
    for c, ch in enumerate(channel_names):
        observed = float(raw_max.get(ch, 0.0) or 0.0)
        planned = float(np.max(future_media[:, c])) if future_media.size else 0.0
        if observed > 0 and planned > observed:
            out.append(
                {
                    "channel": ch,
                    "planned_max": planned,
                    "observed_max": observed,
                    "multiple": planned / observed,
                }
            )
    return out


def _fit_provenance(model: Any) -> dict[str, Any]:
    """``approximate`` / ``fit_method`` / interval vocabulary for this fit."""
    from ..diagnostics import provenance

    diagnostics: dict[str, Any] = {}
    res = getattr(model, "_results", None)
    if res is not None and isinstance(getattr(res, "diagnostics", None), dict):
        diagnostics = res.diagnostics
    fit_diag = getattr(model, "_fit_diagnostics", None) or {}
    if not diagnostics and isinstance(fit_diag, dict):
        diagnostics = fit_diag

    family = provenance.family_of(diagnostics)
    fm = getattr(getattr(model, "model_config", None), "fit_method", None)
    fit_method = getattr(fm, "value", fm)
    approximate = bool(diagnostics.get("approximate", False))
    if not approximate and fit_method is not None and family == provenance.BAYESIAN:
        try:
            from ..config.enums import FitMethod

            approximate = FitMethod(str(fit_method).lower()).is_approximate
        except ValueError:
            approximate = False
    return {
        "inference_family": family,
        "approximate": approximate,
        "fit_method": None if fit_method is None else str(fit_method),
        "interval_noun": provenance.interval_noun(family),
        "family": family,
    }


# ---------------------------------------------------------------------------
# the forecast
# ---------------------------------------------------------------------------


def forecast_under_plan(
    model: Any,
    future_media: Any,
    *,
    calendar: Any = None,
    future_controls: Any = None,
    interval: float = 0.9,
    include_noise: bool = True,
    max_draws: int = DEFAULT_MAX_DRAWS,
    random_seed: int | None = None,
    strict: bool = True,
) -> ForecastResult:
    """Forecast the KPI forward under a spend plan.

    One signature for national and geo panels. The forward pass addresses
    national models on the PERIOD axis and geo models on the OBS axis
    (period-major, cell-minor); that asymmetry is resolved here so a caller
    never sees it — a geo forecast is returned summed to the national total,
    which is the number a plan is judged on.

    Parameters
    ----------
    model
        A fitted ``BayesianMMM``.
    future_media
        ``{channel: [per-period]}``, a flighting schedule dict, or an
        ``(n_periods, n_channels)`` array in the model's channel order.
    calendar
        Optional :class:`~mmm_framework.planning.calendar.PlanningCalendar`
        supplying dated period labels. Without it, periods are labelled by
        their forward index.
    future_controls
        Future values of each control, same shapes as ``future_media``.
        Required when the model has controls: their future values are a
        planning assumption and there is no defensible default.
    interval
        Central interval width; equal-tailed, matching ``compute_hdi_bounds``.
    include_noise
        Include observation noise, making this a PREDICTIVE interval (the
        interval a plan is judged against) rather than an interval on the mean.

    Returns
    -------
    ForecastResult
        With :attr:`ForecastResult.caveats` always populated — see the module
        docstring for why that is not optional.
    """
    from ..validation.backtest import PosteriorForecaster

    if getattr(model, "_trace", None) is None:
        raise ValueError("Model is not fitted. Call fit() before forecasting.")
    if not 0 < interval < 1:
        raise ValueError(f"interval must be in (0, 1); got {interval}.")

    channel_names = list(model.channel_names)
    plan, n_future = _normalize_plan(future_media, channel_names)

    forecaster = PosteriorForecaster(model, strict=strict)

    n_cells = int(getattr(model, "n_cells", 1) or 1)
    n_train = int(model.n_obs // n_cells)

    # Full history + plan, so adstock carryover into the forecast is correct.
    hist = np.asarray(model.X_media_raw, dtype=float)
    if n_cells > 1:
        hist_periods = hist.reshape(n_train, n_cells, -1)
        # The plan is national; split it evenly across cells in proportion to
        # each cell's share of TRAINING spend, so a geo forecast answers the
        # same plan a national one would.
        share = hist_periods.sum(axis=0)  # (n_cells, n_channels)
        tot = share.sum(axis=0, keepdims=True)
        weights = np.divide(share, tot, out=np.full_like(share, 1.0 / n_cells), where=tot > 0)
        future_cells = plan[:, None, :] * weights[None, :, :]
        full = np.concatenate([hist_periods, future_cells], axis=0)
        X_media_full = full.reshape((n_train + n_future) * n_cells, -1)
    else:
        X_media_full = np.concatenate([hist, plan], axis=0)

    X_controls_full = None
    if getattr(model, "n_controls", 0) > 0:
        if future_controls is None:
            raise ValueError(
                f"This model has controls {list(model.control_names)}, whose "
                "future values are a planning assumption with no defensible "
                "default. Pass future_controls."
            )
        ctrl_plan, n_ctrl = _normalize_plan(future_controls, list(model.control_names))
        if n_ctrl != n_future:
            raise ValueError(
                f"future_controls covers {n_ctrl} periods but future_media "
                f"covers {n_future}."
            )
        ctrl_hist = np.asarray(model.X_controls_raw, dtype=float)
        if n_cells > 1:
            ch_periods = ctrl_hist.reshape(n_train, n_cells, -1)
            fut = np.repeat(ctrl_plan[:, None, :], n_cells, axis=1)
            X_controls_full = np.concatenate([ch_periods, fut], axis=0).reshape(
                (n_train + n_future) * n_cells, -1
            )
        else:
            X_controls_full = np.concatenate([ctrl_hist, ctrl_plan], axis=0)

    periods = np.arange(n_train, n_train + n_future)
    if n_cells > 1:
        # obs axis is period-major, cell-minor
        positions = np.concatenate(
            [np.arange(p * n_cells, (p + 1) * n_cells) for p in periods]
        )
    else:
        positions = periods

    draws = forecaster.forecast(
        X_media_full,
        X_controls_full,
        positions,
        include_noise=include_noise,
        random_seed=random_seed,
    )  # (n_samples, n_pos)

    if n_cells > 1:
        # Sum cells back to the national total the plan is judged on.
        draws = draws.reshape(draws.shape[0], n_future, n_cells).sum(axis=2)

    # Thin FIRST, then summarize, so recomputing from the stored draws
    # reproduces the reported mean and bounds exactly. Summarizing over the full
    # set and storing a subset leaves the artifact and the headline disagreeing
    # about the same forecast — and #225 will compare actuals against the stored
    # draws, not against the summary.
    keep = _thin_index(draws.shape[0], max_draws)
    draws = draws[keep]

    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100
    mean = draws.mean(axis=0)
    # A single-draw posterior (MAP) has no interval to estimate. Reporting NaN
    # rather than a zero-width band, per #249: bounds collapsed onto the point
    # estimate are the visual language of extreme precision, and travel out of
    # context far more readily than the "approximate fit" caveat does.
    interval_available = draws.shape[0] >= MIN_DRAWS_FOR_INTERVAL
    if interval_available:
        lower = np.percentile(draws, lo_q, axis=0)
        upper = np.percentile(draws, hi_q, axis=0)
    else:
        lower = np.full(mean.shape, np.nan)
        upper = np.full(mean.shape, np.nan)

    # Per-channel decomposition, from the same call `_media_at` sums, so the
    # parts cannot drift from the whole.
    y_std = float(getattr(model, "y_std", 1.0))
    by_channel: dict[str, np.ndarray] = {}
    if n_cells > 1:
        per_cell = np.zeros((n_future, len(channel_names)))
        Xm = X_media_full.reshape(n_train + n_future, n_cells, -1)
        acc: dict[str, np.ndarray] = {}
        for j in range(n_cells):
            contrib = forecaster.media_by_channel_at(Xm[:, j, :], periods, cell=j)
            for ch, v in contrib.items():
                acc[ch] = acc.get(ch, 0.0) + v[:, keep].mean(axis=1)
        by_channel = {ch: v * y_std for ch, v in acc.items()}
        del per_cell
    else:
        contrib = forecaster.media_by_channel_at(X_media_full, periods)
        by_channel = {ch: v[:, keep].mean(axis=1) * y_std for ch, v in contrib.items()}

    media_total = (
        np.sum(list(by_channel.values()), axis=0)
        if by_channel
        else np.zeros(n_future)
    )
    baseline = mean - media_total

    prov = _fit_provenance(model)
    policy = forecaster.trend_extrapolation
    trend_block = {
        "policy": getattr(policy, "policy", None),
        "trend_type": getattr(policy, "trend_type", None),
        "n_train_periods": getattr(policy, "n_train_periods", n_train),
    }
    caveats = ForecastCaveats(
        trend_extrapolation=trend_block,
        # `none` and `held_flat` both produce a band whose width is driven only
        # by parameter + noise uncertainty, not by distance from the data.
        interval_widens_with_horizon=(trend_block["policy"] == "linear"),
        extrapolated_channels=_extrapolated_channels(model, plan, channel_names),
        residual_autocorrelation=_residual_autocorrelation(model),
        interval_noun=prov["interval_noun"],
        interval_phrase=f"{int(round(interval * 100))}% {prov['interval_noun']}",
        inference_family=prov["inference_family"],
        approximate=prov["approximate"],
        fit_method=prov["fit_method"],
        unsupported=list(getattr(forecaster, "unsupported", []) or []),
        interval_available=interval_available,
        n_posterior_draws=int(draws.shape[0]),
    )

    if calendar is not None and getattr(calendar, "n_periods", 0) >= n_future:
        labels = list(calendar.periods())[:n_future]
    else:
        labels = [f"t+{i + 1}" for i in range(n_future)]

    return ForecastResult(
        periods=labels,
        mean=mean,
        lower=lower,
        upper=upper,
        interval=interval,
        by_channel=by_channel,
        baseline=baseline,
        n_draws=int(draws.shape[0]),
        draws_b64=_encode_draws(draws),
        caveats=caveats,
        calendar=calendar,
    )
