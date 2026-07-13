"""Holiday / event regressor construction (#143).

Turn an :class:`~mmm_framework.config.events.EventsConfig` (named country
holidays + user-defined events) into a matrix of regressor columns aligned to
the model's period index. Each column peaks at the event and optionally decays
over shoulder weeks, so a sharp date-specific spike (Black Friday, a launch) is
representable where the smooth Fourier seasonality is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..config.events import EventsConfig, EventSpec


def _period_spacing_days(periods: pd.DatetimeIndex) -> float:
    if len(periods) < 2:
        return 7.0
    return float(
        np.median(np.diff(periods.values).astype("timedelta64[D]").astype(int))
    )


def _window_column(
    n: int, peaks: list[int], pre: int, post: int, decay: float
) -> np.ndarray:
    """A 0..1 column: 1.0 at each peak, geometric decay over the shoulders."""
    col = np.zeros(n, dtype=float)
    for p in peaks:
        col[p] = max(col[p], 1.0)
        for k in range(1, pre + 1):
            if p - k >= 0:
                col[p - k] = max(col[p - k], (1.0 - decay) ** k)
        for k in range(1, post + 1):
            if p + k < n:
                col[p + k] = max(col[p + k], (1.0 - decay) ** k)
    return col


def _peaks_for_dates(
    periods: pd.DatetimeIndex, dates: list, tol_days: float
) -> list[int]:
    """Map each date to the nearest period; drop dates outside the data window."""
    peaks: list[int] = []
    p_int = periods.asi8  # ns since epoch
    for d in dates:
        ts = pd.Timestamp(d)
        dist = np.abs((p_int - ts.value)) / 86_400_000_000_000.0  # ns -> days
        j = int(np.argmin(dist))
        if dist[j] <= tol_days:
            peaks.append(j)
    return sorted(set(peaks))


def _holiday_dates(config: "EventsConfig", years: list[int]) -> dict[str, list]:
    """``{holiday name: [dates]}`` from the country calendar (needs ``holidays``)."""
    if not config.country:
        return {}
    try:
        import holidays as _holidays
    except ImportError as e:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "Named country holidays require the optional 'holidays' package "
            "(`pip install holidays`). Custom events work without it."
        ) from e

    cal = _holidays.country_holidays(config.country, years=years)
    wanted = [h.lower() for h in config.holidays]
    out: dict[str, list] = {}
    for date, name in cal.items():
        if wanted and not any(w in name.lower() for w in wanted):
            continue
        out.setdefault(name, []).append(date)
    return out


def build_event_regressors(
    periods: pd.DatetimeIndex, config: "EventsConfig"
) -> pd.DataFrame:
    """Build the (n_periods x n_events) event-regressor matrix.

    Columns are named by event; values are in ``[0, 1]`` (1 at the peak period).
    A holiday/event outside the data window contributes no column. Returns an
    empty frame (correct n_periods index) when nothing matches.
    """
    periods = pd.DatetimeIndex(periods)
    n = len(periods)
    tol = _period_spacing_days(periods)
    cols: dict[str, np.ndarray] = {}

    # Named country holidays (recur across the data's years).
    years = sorted({int(y) for y in periods.year})
    for name, dates in _holiday_dates(config, years).items():
        peaks = _peaks_for_dates(periods, dates, tol)
        if peaks:
            cols[name] = _window_column(
                n,
                peaks,
                config.holiday_pre_weeks,
                config.holiday_post_weeks,
                config.holiday_decay,
            )

    # User-defined events.
    ev: EventSpec
    for ev in config.custom_events:
        dates = list(ev.dates or [])
        if ev.holiday:  # a custom window over a named holiday
            years = sorted({int(y) for y in periods.year})
            for _n, ds in _holiday_dates(
                type(config)(country=config.country, holidays=[ev.holiday]), years
            ).items():
                dates.extend(ds)
        peaks = _peaks_for_dates(periods, dates, tol)
        if peaks:
            cols[ev.name] = _window_column(
                n, peaks, ev.pre_weeks, ev.post_weeks, ev.decay
            )

    return pd.DataFrame(cols, index=periods)
