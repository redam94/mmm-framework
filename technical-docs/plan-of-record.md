# Plan of Record — committed, hash-chained, reproducible plan versions

A committed plan is the number every later variance is computed against. Two failure modes
motivate this subsystem (issue #225, with the reproduction inputs fixed in #227): a commitment
nobody can regenerate months later ("a screenshot wearing a commitment's clothes"), and a
commitment quietly edited after variances were already graded against it. The design answer is a
gate that refuses to commit an indefensible number, an append-only hash-chained store that makes
after-the-fact edits detectable, and a reproduction path that re-derives the committed number
from its own recorded provenance.

Two modules split the work:

- `src/mmm_framework/platform/plan_of_record.py` — the **gate** (`assess_committability`), the
  **frozen payload** (`build_commit_payload`), and **reproduction**
  (`reproduce_committed_plan`). Lean-core: stdlib at module level; PyMC-adjacent imports are
  function-local inside reproduction only, so the server and agent import the gate for free.
- `src/mmm_framework/platform/sessions.py` — the `plan_versions` SQLite store:
  `commit_plan_version`, `list_plan_versions`, `get_plan_version`, `latest_committed_plan`,
  `update_plan_version` (which only refuses), `verify_plan_chain`, `ImmutablePlanVersionError`.

## The store: append-only, hash-chained, per (org, plan_family)

`commit_plan_version(plan_family=..., org_id=..., payload=...)` appends a row with
`version = MAX(version) + 1` within the `(org_id, plan_family)` lineage, allocated inside the
same transaction as the insert (the unique index on `(org_id, plan_family, version)` is the
concurrency backstop). The prior `committed` row in the family is flipped to `superseded` —
**status is the only mutable column** on a committed row, and it is a lifecycle marker, not
content: the payload and the chain hash are untouched, so verification is unaffected.

Each row carries `prev_hash` and `hash`, where `hash = sha256(prev_hash | org_id | plan_family |
version | payload_json | created_at)`. The **payload itself** is hashed, not a digest of it: a
chain covering only metadata would leave the committed numbers editable while the chain still
verified. `verify_plan_chain(plan_family, org_id)` re-derives the chain oldest→newest and
returns `{intact, n}` or `{intact: False, broken_at, broken_version}` naming the first bad row.
The chain is per tenant — the hash covers `org_id`, so one org's tampering never breaks
another's chain, and two orgs can reuse a family name.

`latest_committed_plan(project_id)` is what pacing and variance read: the current committed
version, never "whichever draft was edited most recently"
(`platform/variance.py` and the pacing retarget both go through it).

## The gate: `assess_committability`

Input is the payload of the `forecast_plan` op (#223) plus provenance, valuation, and any
overrides. Every gate is derived from a field the forecast already computed
(`caveat_fields`), never from a fresh judgement. The refusals, from the code:

| Gate id | Fires when | Overridable |
|---|---|---|
| `forecast` | No forecast at all — an allocation alone states no KPI level to vary against | no |
| `trend_horizon` | `interval_widens_with_horizon is False` (held-flat spline/GP/piecewise trend) and the window exceeds `DEFAULT_FLEXIBLE_TREND_HORIZON_CAP` (13 periods) — the band is decorative past that | yes |
| `residual_autocorrelation` | Ljung-Box rejected: the interval is knowingly too narrow, so variances flag too often | yes |
| `spend_support` | The plan funds channels past observed spend (`extrapolated_channels`) — the committed number is curve fiction | yes |
| `interval_available` | `interval_available is False` (MAP/ADVI single-draw posterior) — a point estimate states a precision the fit lacks | yes |
| `provenance` | `provenance_gaps()` finds any of `run_id`, `spec_hash`, `data_fingerprint`, `model_path` unresolved | **no** |
| `valuation` | A dollar-denominated plan with no `value_per_kpi` — one KPI unit would silently be worth $1 | **no** |

`CommitRefusal.overridable` is a property: `True` iff the gate is in `ALL_GATES` (the four
forecast-quality gates). Provenance and valuation refusals are structural — waiving them would
produce exactly the unreproducible or un-denominable commitment the store exists to prevent.
(The property once did not exist; any assess with a refusal crashed the agent tool with
`AttributeError` — it is load-bearing for the FE and the agent tool.) Every refusal carries a
`remedy` naming what to *change*, distinct from what to *waive*.

Overrides are gate-specific (`{gate_id: acknowledgement_text}`); unknown or empty entries waive
nothing. An overridden gate stops blocking but is **recorded into the committed payload** via
`Committability.overrides` — an override that is not written down is indistinguishable from a
gate that never fired.

## The payload: `build_commit_payload`

Everything a later reader needs to (a) know what was promised and (b) regenerate it:
`plan_payload` (the working draft's own payload frozen verbatim — pacing joins delivery against
this shape), the forecast **stored whole including its base64 per-period draws** (a window-total
interval cannot be recovered from per-period bounds; variance grades against draws), allocation,
flighting, calendar, provenance, valuation, objective, and the `Committability.to_dict()` with
its overrides.

```python
from mmm_framework.platform.plan_of_record import (
    assess_committability,
    build_commit_payload,
)
from mmm_framework.platform.sessions import commit_plan_version, verify_plan_chain

verdict = assess_committability(forecast, provenance=provenance, valuation=valuation)
if verdict.committable:
    payload = build_commit_payload(
        forecast=forecast, provenance=provenance, committability=verdict
    )
    version = commit_plan_version(
        plan_family="fy26-us", org_id="org_1", payload=payload
    )
    assert verify_plan_chain("fy26-us", "org_1")["intact"]
```

## Reproduction: "here is a number anyone can regenerate"

`reproduce_committed_plan(version)` reloads the model from the run's saved directory, rebuilds
the panel from the **saved** run's spec (never a current session spec, which may have been
edited since the fit), and re-runs `forecast_under_plan` with the recorded `plan_media`,
`plan_controls`, and `random_seed`. Recomputed mean/lower/upper must match the stored snapshot
to `tolerance` (default 1e-9); `ReproductionResult.diffs` names what moved.

The #227 fix: the `forecast_plan` op now stamps `plan_media` (the *normalized* per-period plan
the forecast actually ran on, after channel_budgets→flighting expansion), `plan_controls`, and
`random_seed` into the forecast snapshot (`agents/model_ops.py`). Before that, the reproduction
path read only payload-top-level fields, so every real commitment refused with "records no
per-period spend plan" while the acceptance criterion read as met. The reader still accepts
top-level fields for older payloads, and the seed also rides provenance.

**Refusing is not reporting a mismatch.** `ReproductionResult.refused=True` means the check
could not even be *attempted* on the committed inputs; `reproduced=False, refused=False` means
the check ran and the numbers differ. Conflating them would blame the model for a moved file.
Refusals: missing provenance ("should not have been committable"), a gone model directory, no
recorded dataset path, an unreadable dataset, a **changed dataset fingerprint** (recorded md5 vs
current — "the committed number was correct for the data it was made on; recomputing it against
different data would not verify it"), and a saved run carrying no spec. A tampered snapshot, by
contrast, reproduces `False` without refusing — the tests pin that distinction.

## Refusals summary (the loud-failure inventory)

1. `assess_committability` — the seven gates above; only the four `ALL_GATES` are waivable.
2. `update_plan_version` — unconditionally raises `ImmutablePlanVersionError` (409 at the API).
   It exists so the refusal has one home and one error type; editing a committed row would make
   every variance already computed against it retroactively wrong while the audit trail looked
   intact.
3. `list_plan_versions(plan_family=...)` without `org_id` — raises `ValueError`: family names
   are caller-supplied, so a family-only query would leak another tenant's versions.
4. `reproduce_committed_plan` — the six refuse-before-checking conditions above.

## Neighbours

- **Forecast (#223)**: `planning/forecast.py` computes the caveat fields the gates read;
  `agents/model_ops.py::forecast_plan` builds the snapshot (draws + #227 reproduction inputs).
- **Provenance**: `platform/runs.py::data_fingerprint`; provenance stamping is genuinely
  best-effort at fit time (try/except in the agent host path), which is *why* the gate is hard.
- **Variance & pacing**: `platform/variance.py` and the pacing retarget compare delivery
  against `latest_committed_plan`, and grade against the stored draws.
- **Surfaces**: server routes in `server/src/mmm_framework_server/main.py` (commit / plan-of-
  record / verify endpoints) and the agent tool in `agents/tools.py` — both call the same gate.

## Test anchors

- `tests/test_plan_of_record.py` — versioning, byte-identical earlier versions, chain
  verification and tamper naming, every gate and remedy, override specificity,
  provenance-not-overridable, payload roundtrip and size, all reproduction refusals, the
  end-to-end regenerate-to-tolerance and tampered-snapshot-is-False cases, tenant isolation,
  frequentist commitments.
- `tests/test_budget_plan_endpoints.py` — the HTTP surface, including the 409 mapping.
- `tests/test_variance_bridge.py` — the roundtrip: committed version → variance against it.

## Gotchas

- `commit_plan_version` serializes the payload with `sort_keys=True, default=str`; anything not
  JSON-native is stringified into the hashed bytes, so a payload must be JSON-safe *before*
  commit if you ever want to compare it structurally.
- The seed defaulting chain in reproduction ends at `42` — the same literal `forecast_plan`
  stamps. Change one without the other and every new commitment stops reproducing.
- `reproduce_committed_plan(models_dir=...)` rebases only the model path's basename; the
  dataset path is used as recorded. Relocated deployments must keep dataset paths stable or the
  fingerprint check refuses.
- Supersede-then-insert happens in one transaction, but `latest_committed_plan` orders by
  `created_at DESC, version DESC` — same-timestamp commits resolve by version, which is why the
  version allocation lives inside the transaction.
