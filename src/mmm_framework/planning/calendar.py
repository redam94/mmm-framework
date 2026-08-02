"""Period labels, and the fiscal calendar a finance audience plans on.

The framework's panel indexes time as ``0..n_periods-1``. A CFO's plan is on a
fiscal calendar. Before v1.4 nothing bridged the two: ``grep -rni fiscal`` over
``src/`` returned only prose.

The bug this module exists to close is not the missing feature — it is the join.
``platform/pacing.py`` emitted actual-delivery rows in ``sorted()`` order, which
is **lexicographic** (``P1, P10, P11, P12, P13, P2, ...``), while
``planning/pacing.py`` aligned **positionally**. Per-channel totals survived
because sums are order-invariant, which is why it went unnoticed; the per-period
series a user reads were shuffled once there were more than nine periods, and a
mid-flight upload covering weeks 5-8 was compared against the plan's weeks 1-4.
Measured on a non-flat plan, that reports **+296% over-pacing** where the truth
is +52%. No interval can express "right arithmetic, wrong rows".

Two deliberate limits:

* **Gregorian plus ``fy_start_month``, or an explicit user-supplied period map.**
  No 4-5-4 retail calendar: the 52-vs-53-week year-end rule is per-company
  policy the framework cannot verify from anything the panel carries, and
  shipping one would imply knowledge we do not have.
* **The calendar is derived from the model's own period index, never invented.**
  A plan window the calendar does not fully cover is a hard refusal, not a
  silent truncation — a dropped week is a dropped dollar.

The Fourier seasonality basis is untouched by any of this; it runs on
``t = np.arange(n_periods)`` and is unaffected by labelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CalendarCoverageError",
    "PlanningCalendar",
    "label_sort_key",
]

#: Cadence -> offset for period boundaries. Deliberately NOT the "W"/"MS"
#: aliases: "W" anchors to Sunday and "MS" to the first of the month, so a
#: calendar starting on any other day would silently shift every boundary and
#: the first period would not begin on ``start``. These offsets are anchored on
#: the start date itself, matching the panel's weekday-anchored DatetimeIndex.
_CADENCE_FREQ = {
    "weekly": pd.Timedelta(days=7),
    "daily": pd.Timedelta(days=1),
    "monthly": pd.DateOffset(months=1),
}


class CalendarCoverageError(ValueError):
    """A plan window is not fully covered by the calendar.

    Raised rather than truncating: a period silently dropped from a plan is a
    dollar silently dropped from a commitment.
    """


def label_sort_key(label: str) -> tuple[int, Any]:
    """Sort key that orders ``P2`` before ``P10``.

    The default labels are ``P1..Pn`` and ``sorted()`` on them is lexicographic,
    which is the ordering defect this module was written to fix. Numeric-suffixed
    labels sort numerically; anything else falls back to string order, after the
    numeric ones so a mixed vocabulary is still deterministic.
    """
    text = str(label)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits and text.rstrip("0123456789") + digits == text:
        return (0, int(digits))
    return (1, text)


@dataclass(frozen=True)
class PlanningCalendar:
    """Maps plan periods to dates, labels and fiscal groups.

    Parameters
    ----------
    start : date-like
        First period's start date.
    n_periods : int
        Number of periods in the plan window.
    cadence : {'weekly', 'daily', 'monthly'}
        Period length. Must match the panel's cadence.
    fy_start_month : int
        Month (1-12) the fiscal year begins. ``1`` is the calendar year.
    labels : sequence of str, optional
        Explicit labels. Defaults to ISO dates for the period start, which sort
        correctly as strings and are unambiguous to a finance reader — unlike
        ``P1..Pn``, whose lexicographic order is wrong.
    """

    start: pd.Timestamp
    n_periods: int
    cadence: str = "weekly"
    fy_start_month: int = 1
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_periods < 1:
            raise ValueError("n_periods must be >= 1")
        if self.cadence not in _CADENCE_FREQ:
            raise ValueError(
                f"cadence must be one of {sorted(_CADENCE_FREQ)}, got {self.cadence!r}"
            )
        if not 1 <= int(self.fy_start_month) <= 12:
            raise ValueError("fy_start_month must be in 1..12")
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        if not self.labels:
            object.__setattr__(
                self, "labels", tuple(d.date().isoformat() for d in self.starts())
            )
        elif len(self.labels) != self.n_periods:
            raise ValueError(
                f"labels has {len(self.labels)} entries for {self.n_periods} periods"
            )
        elif len(set(self.labels)) != len(self.labels):
            raise ValueError(
                "labels must be unique — a duplicate label is a double count"
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: Any,
        n_periods: int,
        *,
        fy_start_month: int = 1,
        labels: Sequence[str] | None = None,
    ) -> "PlanningCalendar":
        """A forward calendar starting the period AFTER the model's last.

        Derived from the panel's own ``DatetimeIndex`` rather than invented, so
        a plan cannot silently sit on a different cadence than the fit.
        """
        periods = pd.DatetimeIndex(pd.Index(model.panel.coords.periods))
        if len(periods) < 2:
            raise ValueError("model panel needs >= 2 periods to infer a cadence")
        step = periods[-1] - periods[-2]
        cadence = (
            "weekly"
            if pd.Timedelta("6D") <= step <= pd.Timedelta("8D")
            else "daily" if step <= pd.Timedelta("1D") else "monthly"
        )
        return cls(
            start=periods[-1] + step,
            n_periods=int(n_periods),
            cadence=cadence,
            fy_start_month=fy_start_month,
            labels=tuple(labels) if labels else (),
        )

    # -- periods -----------------------------------------------------------

    def starts(self) -> pd.DatetimeIndex:
        """Start date of each period."""
        return pd.date_range(
            self.start, periods=self.n_periods, freq=_CADENCE_FREQ[self.cadence]
        )

    def periods(self) -> list[str]:
        """Ordered period labels."""
        return list(self.labels)

    def period_bounds(self, label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """``(start, end_exclusive)`` for one label."""
        try:
            i = self.labels.index(str(label))
        except ValueError as exc:
            raise KeyError(f"{label!r} is not a period of this calendar") from exc
        starts = self.starts()
        end = (
            starts[i + 1]
            if i + 1 < self.n_periods
            else starts[i]
            + (starts[1] - starts[0] if self.n_periods > 1 else pd.Timedelta("7D"))
        )
        return starts[i], end

    def period_of(self, dates: Iterable[Any]) -> list[str | None]:
        """Label containing each date, or ``None`` when outside the window.

        Exhaustive and disjoint over the window by construction: bins are
        half-open ``[start, next_start)`` from a single monotonic edge array, so
        no date can land in two periods and none inside the window can land in
        none.
        """
        idx = pd.DatetimeIndex(pd.Index(list(dates)))
        starts = self.starts()
        step = starts[1] - starts[0] if self.n_periods > 1 else pd.Timedelta("7D")
        edges = starts.append(pd.DatetimeIndex([starts[-1] + step]))
        pos = np.searchsorted(edges.values, idx.values, side="right") - 1
        out: list[str | None] = []
        for p, d in zip(pos, idx):
            out.append(
                self.labels[p]
                if 0 <= p < self.n_periods and edges[0] <= d < edges[-1]
                else None
            )
        return out

    # -- fiscal ------------------------------------------------------------

    def fiscal_year_of(self, date: Any) -> int:
        """Fiscal year label for a date. December can belong to the next FY."""
        ts = pd.Timestamp(date)
        return int(ts.year + (1 if ts.month >= self.fy_start_month > 1 else 0))

    def fiscal_groups(self) -> dict[int, list[str]]:
        """``{fiscal_year: [labels]}``, preserving period order."""
        groups: dict[int, list[str]] = {}
        for label, start in zip(self.labels, self.starts()):
            groups.setdefault(self.fiscal_year_of(start), []).append(label)
        return groups

    # -- coverage ----------------------------------------------------------

    def require_covers(self, dates: Iterable[Any]) -> None:
        """Raise unless every date falls inside the window.

        A partially-covered plan is refused rather than truncated.
        """
        idx = pd.DatetimeIndex(pd.Index(list(dates)))
        missing = [d for d, lab in zip(idx, self.period_of(idx)) if lab is None]
        if missing:
            shown = ", ".join(str(d.date()) for d in missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise CalendarCoverageError(
                f"{len(missing)} date(s) fall outside the plan window "
                f"{self.labels[0]}..{self.labels[-1]}: {shown}{more}. Refusing "
                "rather than dropping them — an uncovered period is an "
                "uncounted dollar."
            )

    def to_dict(self) -> dict[str, Any]:
        """Payload carried by plan rows and REST responses."""
        return {
            "start": self.start.date().isoformat(),
            "n_periods": int(self.n_periods),
            "cadence": self.cadence,
            "fy_start_month": int(self.fy_start_month),
            "labels": list(self.labels),
        }
