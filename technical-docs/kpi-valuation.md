# KPI Valuation — what one KPI unit is worth, and where that number came from

`src/mmm_framework/finance/valuation.py` is the single place the framework answers the question
"how many dollars is one unit of the KPI worth?". It exists because the answer used to be a
silent `1.0`. A missing valuation is an assertion that one KPI unit equals one dollar — true only
by coincidence, and wrong by ~1000x on a KPI column denominated in thousands. Before v1.4 the
Planner's "Fund to breakeven" control turned that default into a budget recommendation roughly
1000x too large, and rendered it with credible intervals. The core design decision, stated once
and enforced everywhere: **"unresolved" is a state, not a number.** Consumers that need money
must refuse loudly rather than guess.

A second decision rides along: **margin is exogenous and never estimated.** Nothing in the
framework infers a gross margin from data. It is always supplied by a human or a saved
preference, and the return type carries `source` so every rendering surface can say which.

## Design

### Types

- **`KpiKind`** — enum: `REVENUE` (KPI is already money; value per unit = gross margin),
  `UNITS` (a conversion count; value per unit = gross margin x price), `OTHER` (sessions,
  awareness points — genuinely not convertible to dollars).
- **`KpiValuation`** — a Pydantic model (`extra="forbid"`) declaring the conversion:
  `kind`, `gross_margin`, `price`, `currency`, `scale`. `value_per_kpi()` returns the dollars
  per KPI unit or `None` when not convertible. A model validator rejects `kind='units'` with a
  margin but no price at construction time.
- **`ResolvedValue`** — frozen dataclass: `value_per_kpi`, `source`, `kind`, `currency`,
  `warnings`. `is_dollar` (property: `value_per_kpi is not None`) is the load-bearing field.
  `require(what)` returns the value or raises; `describe()` renders provenance for humans;
  `to_dict()` is the payload shape carried by plan rows, run metrics and REST responses.
- **`UnresolvedValueError`** — `ValueError` subclass raised instead of defaulting to 1.0.
  Takes `what` (the specific decision that needed the number) so the message names it, and the
  message enumerates the fixes: supply a valuation, set the project `economics` preference, or
  use a fixed-budget allocation, which needs no valuation.
- **`kpi_to_dollars(...)`** — the resolver. Keyword-only inputs, one `ResolvedValue` out.

```python
from mmm_framework.finance import KpiKind, KpiValuation, kpi_to_dollars

resolved = kpi_to_dollars(
    override=KpiValuation(kind=KpiKind.UNITS, gross_margin=0.6, price=10.0)
)
assert resolved.value_per_kpi == 6.0 and resolved.source == "param"
value = resolved.require("Fund-to-breakeven allocation")  # raises if unresolved

nothing = kpi_to_dollars()          # no inputs at all
assert nothing.is_dollar is False   # unresolved stays unresolved — never 1.0
```

### Precedence (highest first)

1. `override` — an explicit valuation passed by the caller (`source="param"`);
2. `spec["valuation"]` — the model spec's valuation block (`source="spec"`);
3. `preferences["economics"]` — the project's saved economics preference;
4. `branding["economics"]` — the branding blob's economics block;
5. **unresolved** — `ResolvedValue(value_per_kpi=None, source="none")`.

Two subtleties in the chain:

- **`kind='other'` is a resolved ANSWER, not a miss.** When a candidate declares the KPI is
  neither money nor units, the chain STOPS and returns unresolved from that source. Falling
  through to a lower-precedence blob would let a stale branding margin contradict the explicit
  statement that this KPI cannot be dollar-denominated.
- **Invalid stored blobs are skipped, with a warning, not raised.** A saved preference is
  untrusted input — it may predate the bounds or have been written by an agent. `_coerce`
  appends `"ignored invalid <source> valuation: ..."` to `warnings` and the chain falls
  through, so one bad saved preference cannot break every plan. If the bad value looks like a
  percentage (`gross_margin > 1`), the warning includes the fraction hint.

### The two fields that encode past incidents

- **`gross_margin` is bounded `(0, 1]`.** The pre-1.4 resolver guarded only `m <= 0`, so
  `gross_margin=40` (a user meaning 40%) was accepted and multiplied every profit number by 40.
  Since `save_preference` validated nothing but branding, an agent could persist it. The bound
  lives on the Pydantic field, so it holds at construction, at `save_preference` (which
  validates through the same model), and at resolve time.
- **`scale` is dollars per one unit of the KPI column *as modelled*.** A KPI in thousands takes
  `scale=1000`; this is the exact case that made the old 1.0 default ~1000x wrong. `scale`
  multiplies into both the revenue and units formulas.

## Refusals

- **`ResolvedValue.require(what)` raises `UnresolvedValueError`** when nothing resolved.
  Consumers call `require`, never read `value_per_kpi` directly, so the error names the
  decision ("Fund-to-breakeven allocation (mode='free')") rather than surfacing as an
  unexplained `TypeError` three frames deeper.
- **`planning/budget.py` `mode='free'` refuses without a valuation.** Free-mode allocation
  trades KPI against SPEND, so the exchange rate is load-bearing: a silent 1.0 funds every
  channel to a fabricated breakeven line. `mode='fixed'` is a scale-free argmax over a given
  total and runs fine with no valuation — the refusal is scoped to exactly the mode that needs
  the number. The agent tool (`run_budget_optimizer` in `agents/tools.py`) and the kernel op
  (`agents/model_ops.py`) both default `value_per_kpi` to `None`, NOT `1.0`, for the same
  reason (#226); the server surfaces the refusal as a 400.
- **`KpiValuation(kind='units', gross_margin=..., price=None)` is rejected at construction** —
  units convert as margin x price, so a price-less units valuation has no meaning. The error
  points revenue-denominated KPIs at `kind='revenue'`.
- **`save_preference` with `key="economics"` rejects invalid payloads** (`agents/tools.py`),
  validating through `KpiValuation` itself and hinting `gross_margin is a FRACTION` when the
  value looks like a percentage. This closes the persistence hole that let bad margins reach
  every downstream profit number.
- **`is_dollar=False` suppresses every dollar figure downstream.** Rendering surfaces gate
  dollar columns on it rather than formatting `None` or inventing a rate: the variance-to-plan
  tables (`reporting/sections.py`, `reporting/augur_sections.py`) only emit dollar cells when
  `value_per_kpi` is present, and the two-bucket bridge's SUPPLIED lines
  (`platform/variance.py`) need a dollar-denominated valuation to map at all.

## Interactions with neighbouring subsystems

- **`agents/tools.py` — commit path.** `commit_plan_of_record` resolves
  `kpi_to_dollars(preferences=..., branding=...)` from the project's saved preferences and
  passes `resolved.to_dict()` into `assess_committability`
  (`platform/plan_of_record.py`) as one of the commitment gates; the gate verdict names its
  refusal and says whether it is overridable.
- **`agents/tools.py` — CFO path.** `generate_cfo_onepager` takes an explicit `margin`
  parameter (a human-supplied 0–1 fraction) for the profit-at-risk lines; the one-pager's
  incremental-contribution numbers stay in KPI units when no margin is given, consistent with
  the margin-is-exogenous rule. `reporting/helpers/cfo.py` builds the section.
- **`platform/variance.py`.** `collect_variance_inputs` resolves the valuation once via
  `kpi_to_dollars` and ships `to_dict()` alongside the committed plan, delivery and actuals so
  the variance bridge and its report sections all read the same provenance-carrying payload.
- **`planning/experiment_value.py`** is the in-repo precedent this module generalizes: it
  already set `dollar = margin_per_kpi is not None` and labelled non-dollar output.
  `planning/payback.py`, `planning/decision_arms.py` and `finance/evidence.py` consume the
  same resolved shape.
- **Lean core.** `finance/` is core (no web/LLM deps); `tests/finance/test_valuation.py::
  test_module_is_lean_core` pins it.

## Test anchors

- `tests/finance/test_valuation.py` — the resolver's contract: conversion arithmetic per kind,
  the `(0, 1]` margin bound (parametrized over `40.0`), `scale` carrying the denomination,
  precedence order, `other` stopping the chain, invalid-blob skip-with-warning,
  `require` raising with the decision name, `to_dict` provenance, lean-core import.
- `tests/finance/test_valuation_wiring.py` — the consumers: free-mode allocation is a 400
  without a valuation and resolves from the project economics preference; valuation-free paths
  (fixed mode) are not refused; `save_preference` rejects percentage margins with the pointed
  hint; the server resolver delegates to this single chain and returns the source.

## Gotchas

- **Read `is_dollar` / call `require`, never truth-test `value_per_kpi`.** A resolved value can
  legitimately be small; the only unresolved signal is `None`.
- **`to_dict()` is a wire shape consumed by name** (plan rows, run metrics, REST, report
  sections read `value_per_kpi` / `is_dollar` / `source` out of plain dicts). Renaming a key is
  a cross-surface breaking change, not a refactor.
- **`_coerce` filters to `KpiValuation.model_fields` before validating**, so a stored blob with
  extra bookkeeping keys still resolves — but a *direct* `KpiValuation(...)` construction
  forbids extras. Passing `margin=` instead of `gross_margin=` raises; a stored blob with the
  same typo silently resolves to nothing. The warning list is how you find out.
- The resolver never raises. All refusal behaviour lives in consumers via `require` or explicit
  `is_dollar` checks; if you add a consumer that turns KPI into money, route it through
  `require(what)` with a `what` string a user can act on.
