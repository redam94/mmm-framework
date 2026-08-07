# Planning calendar — the forward date vocabulary plans, forecasts and delivery share

The model indexes time as `0..n_periods-1`; a CFO plans on a fiscal calendar; a delivery feed
carries dates. Before v1.4 nothing bridged the three, and the bridge that existed by accident was
wrong: `platform/pacing.py` emitted actual-delivery rows in `sorted()` order — **lexicographic**
(`P1, P10, P11, P12, P13, P2, ...`) — while `planning/pacing.py` aligned **positionally**.
Per-channel totals survived (sums are order-invariant), which is why it shipped unnoticed; the
per-period series a user reads were shuffled once a plan passed nine periods, and a mid-flight
upload for weeks 5-8 was graded against the plan's weeks 1-4. Measured on a ramped plan that is
**+296% over-pacing reported where the truth is +52%** (#216). No interval can express "right
arithmetic, wrong rows". `PlanningCalendar` exists so that plans, forecasts and delivery all speak
one dated period vocabulary and join **by label**, never by position.

Module: `src/mmm_framework/planning/calendar.py`. Not re-exported from `mmm_framework.planning` —
import it from the submodule.

## Design

`PlanningCalendar` is a small frozen dataclass: `start` (first period's start date), `n_periods`,
`cadence` (`weekly` / `daily` / `monthly`), `fy_start_month`, and optional explicit `labels`.
Default labels are **ISO dates of each period start**, chosen because they sort correctly as
strings and are unambiguous to a finance reader — the `P1..Pn` vocabulary they replace is exactly
the one whose string sort is wrong.

```python
from mmm_framework.planning.calendar import PlanningCalendar, label_sort_key
from mmm_framework.planning import build_flighting_schedule

cal = PlanningCalendar(start="2026-01-05", n_periods=13, cadence="weekly", fy_start_month=7)
cal.periods()                      # ordered ISO-date labels
cal.period_of(["2026-01-19"])      # -> label containing each date, or None
cal.fiscal_groups()                # {fiscal_year: [labels]} in period order
schedule = build_flighting_schedule({"tv": 130_000.0}, 13, pattern="even", calendar=cal)
sorted(["P2", "P10"], key=label_sort_key)   # ["P2", "P10"] — numeric, not lexicographic
```

Key surface:

- `starts()` — a `pd.date_range` anchored on `start` itself. The cadence offsets are deliberately
  NOT the pandas `"W"`/`"MS"` aliases: `"W"` snaps to Sunday and `"MS"` to the 1st, which would
  silently shift every boundary off the panel's weekday-anchored index.
- `period_of(dates)` — half-open `[start, next_start)` bins from a single monotonic edge array, so
  the map is **exhaustive and disjoint** over the window by construction: no date lands in two
  periods, none inside the window lands in none. Dates outside the window map to `None`.
- `period_bounds(label)` — `(start, end_exclusive)`; unknown label is a `KeyError`.
- `fiscal_year_of(date)` / `fiscal_groups()` — Gregorian year shifted by `fy_start_month`
  (December can belong to the *next* FY).
- `require_covers(dates)` — raises `CalendarCoverageError` naming the uncovered dates.
- `to_dict()` — the JSON payload plan rows and REST responses carry.
- `from_model(model, n_periods, ...)` — a forward calendar starting one step **after** the panel's
  last period, cadence inferred from the panel's own `DatetimeIndex` step.
- `label_sort_key` (module function) — orders `P2` before `P10`; numeric-suffixed labels sort
  numerically, everything else falls back to string order after them. `platform/pacing.py` sorts
  its per-period rows with it.

Two deliberate scope limits, stated in the module docstring: **Gregorian plus `fy_start_month`
only** (no 4-5-4 retail calendar — the 52-vs-53-week year-end rule is per-company policy the panel
cannot reveal), and **the calendar is derived from the model's own period index, never invented**.

## How the forward calendar is derived (`agents/model_ops.py::_forward_calendar`)

A plan is for the periods after the model's history, and the model knows when that history ends
and at what cadence — so the caller should not have to supply dates for the plan to be joinable to
delivery. `_forward_calendar(mmm, n_periods, flighting)`:

1. Reads `mmm.panel.index`; needs a `DatetimeIndex` with ≥ 2 entries. An undated or irregular
   panel yields `None` — the schedule keeps positional labels rather than inventing dates.
2. Infers cadence from the last step (1d → daily, 7d → weekly, 28–31d → monthly); an explicit
   `flighting["cadence"]` wins.
3. Advances by the **cadence, not the raw gap**. A month is not a fixed day count: `idx[-1] + step`
   lands a monthly plan on 2025-07-31 where the next period is 2025-08-01, and a calendar whose
   labels miss the delivery dates falls back to the positional join this derivation exists to
   avoid (measured: +4.3% "on-track" reported where the truth is +41.2% over-pacing). Month-END
   panels are detected and followed (`06-30 + 1mo` → `07-31`, not `07-30`), because
   `PlanningCalendar` is start-anchored by design.
4. An explicit `flighting["start_date"]` overrides the derived start.

Note the two derivation paths differ: `PlanningCalendar.from_model` reads
`model.panel.coords.periods` and advances by the raw step; `_forward_calendar` reads
`mmm.panel.index` and carries the month-end handling. The agent ops use `_forward_calendar`.

## Flighting on the calendar (`planning/flighting.py`)

`build_flighting_schedule(channel_budgets, n_periods, pattern=..., calendar=...)` spreads each
channel's total across the horizon (`even`, `front_loaded`/`back_loaded`, `pulsed`, `seasonal`,
`custom`, with `per_channel_pattern` overrides). Label precedence: explicit `period_labels` win;
else the calendar's dated labels; else the `P1..Pn` fallback (kept for undated panels, with the
lexicographic caveat above). The returned dict's `"periods"` / `"schedule"[i]["period"]` carry the
calendar labels, which is what makes the plan joinable to delivery by date.

## Forecast and committed plans

`forecast_plan` (agents/model_ops.py) derives the calendar once, hands it to both
`build_flighting_schedule` (when expanding `channel_budgets`) and
`forecast_under_plan(..., calendar=cal)` (planning/forecast.py) — so the forecast covers exactly
the periods a saved plan would, under the same labels (`t+1..t+n` only when there is no calendar).
The forecast payload and `plan_budget`'s committed plan both stamp
`{"start", "n_periods", "cadence", "fy_start_month"}` into a `"calendar"` key, so a saved plan
carries its own date vocabulary rather than depending on whoever reads it back to reconstruct
one. `platform/plan_of_record.py` persists that dict with the commitment.

## Refusals

- **Partial coverage** — `require_covers` raises `CalendarCoverageError` (naming up to five
  offending dates) instead of truncating. A period silently dropped from a plan is a dollar
  silently dropped from a commitment.
- **Short calendar** — `build_flighting_schedule` raises when the calendar covers fewer periods
  than the schedule needs, rather than labelling the tail `P{n}` (a mixed vocabulary would
  re-break the sort).
- **Duplicate labels** — rejected at construction: a duplicate label is a double count.
- **Label count mismatch**, **cadence outside the vocabulary**, **`fy_start_month` outside 1..12**,
  **`n_periods < 1** — all `ValueError` at construction.
- **`from_model` on a < 2-period panel** — no step to infer a cadence from; `ValueError`.
- **Unknown label in `period_bounds`** — `KeyError`, not a nearest-match guess.
- The one place the module *degrades* instead of refusing: `_forward_calendar` returns `None` for
  an undated panel, and downstream keeps positional labels. Undated data genuinely has no
  calendar; inventing one would be worse than the fallback.

## Interactions

- `planning/pacing.py::compute_pacing` — joins plan vs actual **by label** when both sides carry a
  real (non-placeholder, unique, non-all-numeric) vocabulary; falls back to positional truncation
  otherwise and records which join it used (`"label"` / `"positional"`).
- `platform/pacing.py` — sorts delivery periods with `label_sort_key`.
- `platform/plan_of_record.py` — stores the plan's `calendar` dict.
- `planning/forecast.py` — labels forecast periods from the calendar.
- Seasonality is **untouched**: the Fourier basis runs on `t = np.arange(n_periods)` and is
  unaffected by labelling.

## Test anchors

- `tests/planning/test_calendar.py` — the shipped misalignment (mid-flight join, ramp-based
  regressions), exhaustive-and-disjoint sweep, fiscal grouping, coverage refusal, `from_model`
  cadence inference, flighting-label integration, `label_sort_key`.
- `tests/test_planner_budget.py` — flighting patterns, `plan_budget` geo + flighting.
- `tests/test_forecast.py` — forecast-under-plan with the derived calendar.
- `tests/test_delivery_pacing.py` — the platform-side ordering.

## Gotchas

- **A flat plan hides the join bug entirely** — positional and label joins agree when every period
  plans the same spend. Regression tests must use a ramp; the calendar test file says so in its
  docstring and every join test there does.
- The `P1..Pn` fallback still exists (undated panels) and still sorts lexicographically as plain
  strings — anything ordering those labels must go through `label_sort_key`.
- `start_date` in a flighting spec silently overrides the panel-derived start; the cadence is still
  inferred from the panel unless also given, so an explicit start with an implicit cadence can
  produce a calendar the caller did not expect.
- `_forward_calendar` swallows all derivation errors to `None` by design; if a plan you expected to
  be dated comes back with positional labels, check the panel index first, not the calendar code.
