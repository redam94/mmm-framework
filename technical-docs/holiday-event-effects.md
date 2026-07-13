# Holiday / event effects — #143

Sharp, date-specific effects — Black Friday, Prime Day, a product launch, a
one-off PR moment — are **not** smooth-seasonal, so the yearly Fourier basis
cannot represent them and they bleed into media or residual noise around the
highest-revenue weeks. This adds a first-class event block, distinct from
seasonality, so the user declares events instead of hand-authoring dummy columns.

## Usage

```python
from mmm_framework.config import EventsConfig, EventSpec

events = EventsConfig(
    country="US",                                   # named-holiday calendar
    holidays=["Thanksgiving", "Christmas"],         # (empty ⇒ all holidays)
    holiday_post_weeks=1, holiday_decay=0.5,        # shoulder + decay for holidays
    custom_events=[
        EventSpec(name="Launch", dates=["2024-06-15"], post_weeks=3, decay=0.5),
        EventSpec(name="BlackFriday", holiday="Black Friday", pre_weeks=1),
    ],
    prior_sigma=0.5,
)
model_config = ModelConfigBuilder().with_events(events).build()
```

`build_event_regressors(periods, events)` turns this into a matrix of columns
(one per holiday/event), aligned to the model's period index: 1.0 at the event
period, geometric decay `(1 − decay)^k` over `pre_weeks` / `post_weeks`
shoulders, `max`-combined across recurrences. An event outside the data window
contributes no column (a warning fires if nothing matches).

Named country holidays use the optional [`holidays`](https://pypi.org/project/holidays/)
package (a declared dependency); custom events work regardless.

## In the model

Off by default (`ModelConfig.events is None`) — no RVs, graph byte-identical
(R0.1). When set, an additive block:

```
event_component_t = X_events_t · beta_events      # beta_events ~ Normal(0, prior_sigma)
mu = ... + seasonality + event_component + ...
```

`event_component` is a separate Deterministic from `seasonality_component`, and
the decomposition carries an **`events` / `total_events`** term, so the waterfall
still closes and event contributions are reported as their own line ("Events /
Holidays" in `ComponentDecomposition.summary()`, `"Events"` in the report's
component totals) — never folded into seasonality.

## Defaults & double-counting

- Holidays default to a 1-week post shoulder with 0.5 decay; custom events
  default to a single-period spike. Widen with `pre_weeks` / `post_weeks`.
- Events and Fourier seasonality are **separate additive terms**. Keep the
  yearly Fourier order modest so a recurring holiday is captured by its event
  regressor, not absorbed into a high-order harmonic — the regressor is the
  sharper, more interpretable representation of a date-specific spike.
- The event coefficient is Normal(0, σ), so an event can lift **or** depress
  sales (a promotion pull-forward, a stock-out).

## Deferred

Pooling across recurring instances of the same holiday (a hierarchical prior so
"Christmas 2023" and "Christmas 2024" borrow strength) — a follow-up; today each
named holiday is one regressor whose column already spans all its occurrences.

## Tests

`tests/test_holiday_events.py` — regressor helper (windows/decay, named
holidays, out-of-window), off-has-no-block, on-structure, spike attribution +
waterfall closure, reporting separation from seasonality.
