"""In-flight pacing — actual-vs-plan loop (issue #107).

MMM is retrospective and the measurement loop re-tests on a decay cadence, but
nothing connects a *recommended plan* to what's *actually being spent* in-flight.
Pacing — planned vs actual delivery — is where media planners live day-to-day,
and between fits it is a blind spot: the recommendation is delivered and then
goes dark until the next model run.

This module closes that loop with pure, testable numpy:

* :func:`compute_pacing` — aligns a plan and actual delivery per channel/period,
  computes divergence, and flags channels pacing outside a threshold.
* :func:`expected_outcome_delta` — the expected KPI impact of the divergence,
  read off the fitted response curves *with uncertainty* (contribution at actual
  spend vs at planned spend, per posterior draw).
* :func:`pacing_report` — combines both into the report/agent payload.

The plan and the actuals can be given as a flighting schedule
(``{"schedule": [{"period", "<channel>": spend, ...}]}``), a list of period rows,
a ``{channel: [per-period]}`` map, or a ``{channel: total}`` map — all normalize
to per-channel per-period arrays aligned by period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "PacingChannel",
    "PacingResult",
    "compute_pacing",
    "expected_outcome_delta",
    "pacing_report",
    "DEFAULT_PACING_THRESHOLD",
]

#: Default divergence threshold (fraction of planned spend) above which a channel
#: is flagged as off-pace.
DEFAULT_PACING_THRESHOLD = 0.10

_META_KEYS = {"period", "total", "label", "date"}


def _rows_to_channel_series(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Pivot a list of ``{period, <channel>: spend}`` rows to per-channel arrays."""
    periods: list[str] = []
    channels: list[str] = []
    for r in rows:
        for k in r:
            if k not in _META_KEYS and k not in channels:
                channels.append(k)
    series = {c: np.zeros(len(rows)) for c in channels}
    for i, r in enumerate(rows):
        periods.append(str(r.get("period", r.get("label", i))))
        for c in channels:
            try:
                series[c][i] = float(r.get(c, 0.0) or 0.0)
            except (TypeError, ValueError):
                series[c][i] = 0.0
    return series, periods


def _normalize(data: Any) -> tuple[dict[str, np.ndarray], list[str]]:
    """Normalize a plan / actuals input to ``({channel: per-period array}, periods)``."""
    if data is None:
        return {}, []
    # Flighting schedule dict.
    if isinstance(data, Mapping) and "schedule" in data:
        return _rows_to_channel_series(list(data["schedule"] or []))
    # List of period rows.
    if isinstance(data, (list, tuple)):
        return _rows_to_channel_series(list(data))
    # {channel: array} or {channel: scalar}.
    if isinstance(data, Mapping):
        series: dict[str, np.ndarray] = {}
        n = 1
        scalar = True
        for c, v in data.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v, dtype=float).ravel()
                scalar = False
            else:
                try:
                    arr = np.array([float(v)])
                except (TypeError, ValueError):
                    arr = np.array([0.0])
            series[str(c)] = arr
            n = max(n, arr.size)
        # Pad ragged arrays to a common length.
        for c in series:
            if series[c].size < n:
                series[c] = np.pad(series[c], (0, n - series[c].size))
        periods = ["total"] if scalar else [str(i) for i in range(n)]
        return series, periods
    return {}, []


@dataclass
class PacingChannel:
    """One channel's pacing status."""

    channel: str
    planned: float
    actual: float
    divergence_pct: float  # (actual - planned) / planned
    status: str  # "on-track" | "over-pacing" | "under-pacing" | "not-started"
    planned_series: list[float] = field(default_factory=list)
    actual_series: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "planned": self.planned,
            "actual": self.actual,
            "divergence_pct": self.divergence_pct,
            "status": self.status,
            "planned_series": list(self.planned_series),
            "actual_series": list(self.actual_series),
        }


@dataclass
class PacingResult:
    """Portfolio + per-channel pacing, with the flagged (off-pace) channels."""

    channels: list[PacingChannel]
    periods: list[str]
    threshold: float
    planned_total: float
    actual_total: float
    divergence_pct: float
    flagged: list[str]
    outcome_delta: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": [c.to_dict() for c in self.channels],
            "periods": list(self.periods),
            "threshold": self.threshold,
            "planned_total": self.planned_total,
            "actual_total": self.actual_total,
            "divergence_pct": self.divergence_pct,
            "flagged": list(self.flagged),
            "outcome_delta": self.outcome_delta,
        }


def _status(divergence: float, planned: float, actual: float, threshold: float) -> str:
    if planned <= 0 and actual <= 0:
        return "not-started"
    if abs(divergence) <= threshold:
        return "on-track"
    return "over-pacing" if divergence > 0 else "under-pacing"


def compute_pacing(
    planned: Any,
    actual: Any,
    *,
    threshold: float = DEFAULT_PACING_THRESHOLD,
) -> PacingResult:
    """Compare a plan against actual delivery, per channel and overall.

    Both inputs normalize to per-channel per-period arrays; comparison is over
    the periods present in ``actual`` (the elapsed window) — the planned series is
    truncated to the same length so mid-flight pacing compares like with like.
    """
    plan_series, plan_periods = _normalize(planned)
    act_series, act_periods = _normalize(actual)
    channels = list(dict.fromkeys([*plan_series.keys(), *act_series.keys()]))

    # Elapsed window = the number of actual periods (truncate the plan to match).
    n_elapsed = max((v.size for v in act_series.values()), default=0)
    periods = act_periods[:n_elapsed] if act_periods else plan_periods

    rows: list[PacingChannel] = []
    flagged: list[str] = []
    p_total = a_total = 0.0
    for ch in channels:
        p = plan_series.get(ch, np.zeros(0))
        a = act_series.get(ch, np.zeros(0))
        p_elapsed = p[:n_elapsed] if n_elapsed else p
        planned_c = float(np.nansum(p_elapsed))
        actual_c = float(np.nansum(a))
        div = (
            (actual_c - planned_c) / planned_c
            if planned_c > 1e-9
            else (0.0 if actual_c <= 1e-9 else float("inf"))
        )
        status = _status(div, planned_c, actual_c, threshold)
        if status in ("over-pacing", "under-pacing"):
            flagged.append(ch)
        rows.append(
            PacingChannel(
                channel=ch,
                planned=planned_c,
                actual=actual_c,
                divergence_pct=div,
                status=status,
                planned_series=[float(x) for x in p_elapsed],
                actual_series=[float(x) for x in a],
            )
        )
        p_total += planned_c
        a_total += actual_c

    port_div = (a_total - p_total) / p_total if p_total > 1e-9 else 0.0
    return PacingResult(
        channels=rows,
        periods=periods,
        threshold=threshold,
        planned_total=p_total,
        actual_total=a_total,
        divergence_pct=port_div,
        flagged=flagged,
    )


def expected_outcome_delta(
    curves: Any,
    planned_totals: Mapping[str, float],
    actual_totals: Mapping[str, float],
    *,
    hdi_prob: float = 0.9,
) -> dict[str, float] | None:
    """Expected KPI delta from the divergence, off the fitted response curves.

    ``curves`` is a :class:`~mmm_framework.planning.budget.ResponseCurves` (or any
    object exposing ``channel_names``, ``spend_grid`` ``(C,G)`` and
    ``contributions`` ``(D,C,G)``). For each posterior draw we read the channel's
    contribution at the ACTUAL total vs the PLANNED total and sum the difference —
    turning parameter uncertainty into a credible interval on the outcome impact.
    Returns ``{"mean","lower","upper","planned_kpi","actual_kpi"}`` or ``None``.
    """
    names = list(getattr(curves, "channel_names", []) or [])
    grid = getattr(curves, "spend_grid", None)
    contrib = getattr(curves, "contributions", None)
    if not names or grid is None or contrib is None:
        return None
    grid = np.asarray(grid, dtype=float)
    contrib = np.asarray(contrib, dtype=float)
    D = contrib.shape[0]
    planned_kpi = np.zeros(D)
    actual_kpi = np.zeros(D)
    for c, ch in enumerate(names):
        p = float(planned_totals.get(ch, 0.0) or 0.0)
        a = float(actual_totals.get(ch, 0.0) or 0.0)
        gc = grid[c]
        for d in range(D):
            planned_kpi[d] += float(np.interp(p, gc, contrib[d, c]))
            actual_kpi[d] += float(np.interp(a, gc, contrib[d, c]))
    delta = actual_kpi - planned_kpi
    lo_q = (1 - hdi_prob) / 2 * 100
    return {
        "mean": float(delta.mean()),
        "lower": float(np.percentile(delta, lo_q)),
        "upper": float(np.percentile(delta, 100 - lo_q)),
        "planned_kpi": float(planned_kpi.mean()),
        "actual_kpi": float(actual_kpi.mean()),
    }


def pacing_report(
    planned: Any,
    actual: Any,
    *,
    curves: Any = None,
    threshold: float = DEFAULT_PACING_THRESHOLD,
    hdi_prob: float = 0.9,
) -> PacingResult:
    """Full pacing assessment: divergence + (when ``curves`` given) the expected
    KPI impact of that divergence."""
    result = compute_pacing(planned, actual, threshold=threshold)
    if curves is not None:
        planned_totals = {c.channel: c.planned for c in result.channels}
        actual_totals = {c.channel: c.actual for c in result.channels}
        result.outcome_delta = expected_outcome_delta(
            curves, planned_totals, actual_totals, hdi_prob=hdi_prob
        )
    return result
