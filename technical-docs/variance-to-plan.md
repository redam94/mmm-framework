# Variance to plan: the two-bucket bridge

`src/mmm_framework/planning/variance.py` answers "we committed to a forecast, the quarter is
over, why did we miss?" without manufacturing causal claims. The finance-native instinct is a
three-bucket waterfall — delivery, effectiveness, other — but the "effectiveness" bucket is a
subtraction between two posteriors (the committed fit and a refit) that differ in data, window,
spec and MC noise all at once. Nothing in that subtraction isolates effectiveness. So this
subsystem ships exactly the two buckets that ARE identifiable without a refit, refuses the
third loudly, and leads every rendering with the only verdict that matters first: did the
realized total land inside the interval the team committed to?

## Design

Two halves, split so the compute stays lean-core testable:

- **Engine** — `planning/variance.py`: `variance_to_plan(model, committed_payload,
  actual_media, actuals, *, supplied=, value_per_kpi=, refit_run_id=, channel_meta=,
  fit_diagnostics=)` → frozen `VarianceBridge`. numpy + the finance vocabulary only.
- **Assembler** — `platform/variance.py`: `collect_variance_inputs(project_id, *, supplied=)`
  reads the stores (committed plan of record, delivery ledger, actuals, valuation
  preferences), returns a JSON-safe dict that crosses the REST-job and agent-kernel
  boundaries unchanged, and raises `VarianceInputError` for the conditions under which a job
  should never start. The model-op wrapper is `agents/model_ops.py::variance_to_plan`.

The actuals themselves come from `platform/actuals.py`: `parse_actuals_records` (CSV/TSV/JSON
sniffing) and `reconcile_against_panel`, backed by an **as-of-dated, append-preserving** store
(`sessions.record_actuals` / `latest_actuals_for_project`) — restating a period is a new row
under a new `as_of`, never an overwrite. Actuals are an independent record, not a derivative
of the panel: when they disagree with the panel's own KPI, `reconcile_against_panel` reports
the signed gap per period and never silently prefers one source.

### The two buckets

1. **Delivery variance** — `g_plan(S_actual) − g_plan(S_plan)`: two paired runs of
   `forecast_under_plan` on the **committed** posterior with the **recorded seed** and
   thinning, one on the plan as committed, one on the spend actually delivered. Per-channel
   rows come from the forecast's own decomposition (`MODELLED`, basis "committed posterior,
   paired counterfactual"); the interval lives on the total, from the paired per-draw deltas
   (`fc_actual.draws() − fc_plan.draws()`, summed over the window). Point values carry no
   per-channel band because only the total's paired interval is honest here.
2. **Unexplained** — `actual KPI − g_plan(S_actual) − Σ supplied`: a `RESIDUAL` row LABELLED
   for what it mixes (baseline movement, competitor action, data error, model error, noise),
   never attributed.

### Closure is algebraic, not "usually within tolerance"

The forecast's own decomposition closes (mean = Σ by_channel + baseline), so per-channel
deltas plus the baseline-interaction delta reproduce the forecast-to-forecast gap
bit-for-bit, and unexplained is defined as the remainder to actuals. `VarianceBridge.closes`
checks Σ rows == `actual − committed` at 1e-9. The one thing that can break the identity is
the re-run of the committed plan drifting from the committed snapshot: sub-tolerance drift
(≤ 1e-6 relative) is carried as its own `RESIDUAL` row named "Reproduction drift" so the
bridge still closes exactly; larger drift refuses outright, because a model that does not
reproduce the committed forecast would make the delivery bucket a different posterior's
opinion wearing the committed label — the refused refit comparison in disguise.

### The committed-interval verdict leads

`within_committed_interval` is computed from the committed **per-period draws** (the #225
snapshot stores them whole: base64 little-endian float32, `(n_draws, n_periods)`), summed to
window totals and quantiled at the committed mass. A window-total interval cannot be
recovered from per-period bounds. A miss inside the band is "within the committed
uncertainty" — the bridge then explains composition, not a surprise. Undecodable or too-few
draws make the verdict *unavailable, not passed* (a caveat says so). Fewer than 10 draws is
treated as too few.

### Supplied lines

`supplied_line(name, value, *, source_note, channel=None, kpi_kind_is_dollar=True)` builds a
`SUPPLIED` `BridgeLine` (finance/lines.py) for human adjustments — gross-to-net, returns,
trade spend. The provenance vocabulary enforces the shape (required non-blank `source_note`,
no interval fields: a supplied number has no sampling distribution, and a band would dress an
assertion as an estimate). Supplied lines subtract from the **unexplained remainder only**,
never from a channel, so the bridge still closes and no per-channel ROI gets net-rescaled.

## Refusals

Engine (`ValueError`, enforced for every caller including store-bypassing tests):

- **Refit "effectiveness" split** — any `refit_run_id` appends `REFIT_REFUSAL` to
  `bridge.refusals`; the platform wrapper attaches the `compare_runs` diff (what CHANGED
  between the runs) as the sayable alternative. The word "effectiveness" appears on no row,
  header or markdown surface — pinned by `test_the_word_effectiveness_appears_nowhere` and
  `test_op_markdown_never_says_effectiveness`.
- **Non-reproducing model** — re-run of the committed plan differs from the snapshot beyond
  1e-6 relative: refuse, load the committed run.
- **Partial actuals coverage** — any committed period without realized KPI: a bridge over a
  partially-realized window would compare a full-window commitment against a part-window
  actual.
- **Per-channel supplied line** (`channel` not None) — would produce a net-scaled per-channel
  number the model never estimated.
- **Non-dollar KPI supplied line** — a monetary netting adjustment on a units/index KPI
  silently changes what the bridge is a bridge OF; declare a valuation first.
- **Non-SUPPLIED provenance in `supplied=`** — a "modelled" adjustment would be a model
  claim nobody modelled.
- **No forecast periods / no `plan_media`** in the committed payload.

Assembler (`VarianceInputError`, before any job starts):

- **No committed plan of record** — drafts do not qualify (editable after the fact).
- **No `plan_media`** on the commitment (pre-v1.4 commitments; re-commit).
- **No realized KPI uploaded** for the project.
- **Delivery gaps over the committed window** — assuming plan-as-delivered would fabricate a
  zero delivery variance as if it were measured; the refusal names the missing cells.
- **Changed dataset fingerprint** since the commitment (md5 via `runs.data_fingerprint`).
- **No recorded model path** — the committed model cannot be reloaded.
- **Supplied line without name or source_note, or non-numeric value** — fails the POST.

Suppressions that are gates, not refusals: any efficiency-measured channel
(`channel_meta` `is_monetary=False`) suppresses the blended dollar headline
(`dollar_headline_suppressed`, `rows_dollars=None`); `value_per_kpi=None` suppresses every
dollar figure rather than inventing one; `fit_diagnostics["at_boundary"]` (a `sum_equals`-
constrained frequentist fit) suppresses the "independent reconciliation" framing, since parts
of that closure hold by construction.

```python
from mmm_framework.finance.lines import SUPPLIED, BridgeLine
from mmm_framework.planning.variance import REFIT_REFUSAL, supplied_line, variance_to_plan
from mmm_framework.platform.variance import VarianceInputError, collect_variance_inputs

returns = supplied_line(
    "Returns and allowances",
    -125_000.0,
    source_note="FY26 Q2 returns ledger, finance close 2026-07-14",
)
assert returns.provenance is SUPPLIED and isinstance(returns, BridgeLine)
```

## Interactions

- **`planning/forecast.py`** — `forecast_under_plan` is the counterfactual engine; the paired
  seed/thinning is the #225 reproduction guarantee the drift check leans on.
- **`platform/plan_of_record.py`** — `reproduce_committed_plan` is the commit-time twin: a
  commitment must reproduce from provenance to 1e-9 before it is committable.
- **`platform/runs.py`** — `compare_runs` supplies the refit diff; `data_fingerprint` gates
  the dataset.
- **`finance/lines.py` + `finance/valuation.py`** — the provenance vocabulary and the
  KPI→dollars resolution (`kpi_to_dollars`) behind `value_per_kpi`/`value_source`.
- **`agents/model_ops.py::variance_to_plan`** — kernel/REST wrapper: reloads the committed
  run, calls the engine, renders markdown, persists the artifact.
- **`synth/`** — the exact-truth grading path: actuals generated from the synth world's own
  `response_fn` let tests assert the delivery interval covers the true delivery delta and
  that unexplained is small in a clean world.

## Test anchors

- `tests/test_variance_bridge.py` — closure at 1e-9, per-channel rows and provenances, the
  draws-not-bounds verdict, both refusal families, the effectiveness-word bans, drift row vs
  drift refusal, dollar suppression, response_fn truth coverage, model-op and endpoint
  layers, commit reproduction.
- `tests/test_actuals_store.py` — as-of restatement semantics, parser formats, panel
  reconciliation (signed gaps, unmatched periods shift nothing), lean-import guard, REST
  round-trip.
- Notebook walkthrough: `nbs/demos/variance_to_plan.ipynb`. Narrative background:
  *Variance to plan* in `technical-docs/engineering-notes.md`.

## Gotchas

- Pass the **committed** model, reloaded from the version's provenance. A newer fit will
  usually trip the drift refusal — but a refit that happens to reproduce the plan total
  would silently pass, which is why the platform wrapper always reloads
  `committed_run_name` rather than "the latest model".
- The engine's seed fallback chain ends at 42; a payload that never recorded
  `random_seed` still pairs its two forecasts (same seed both sides), so the delivery
  interval stays paired even when the snapshot predates seed recording.
- `reconcile_against_panel` is diagnostic, not a gate: the bridge runs on uploaded actuals
  even when they disagree with the panel. Surfaces that show the bridge should show the
  reconciliation beside it.
- Engine refusals are plain `ValueError`; only the assembler raises `VarianceInputError`
  (a `ValueError` subclass). Catch the subclass at the API boundary, not in the engine.
