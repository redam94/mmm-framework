# Causal Elicitation Agent — Design Contract

Status: **PLAN (pre-implementation)**. This is the agreed shape before any slice is
built. It is the single source of truth shared by the agent stack
(`src/mmm_framework/agents`, `src/mmm_framework/api`) and the React frontend
(`frontend/src/pages/Agent`, `frontend/src/pages/Program`).

---

## 0. One-paragraph summary

A **human-in-the-loop pipeline** that proposes a causal model for a project, then
*interrogates the user* to firm up the parts only a human can supply — the causal
roles of controls, spillover, structural breaks, whether spend chases demand —
and uses those answers as **hard gates** before producing a validated,
pre-registered model spec + assumption stack + an elicitation report. It runs in
an **ARQ worker** so it never blocks the chat, and it suspends/resumes around the
human gates rather than holding a worker open. Control flow is **deterministic,
predefined stages** (not an autonomous fan-out): the LLM is used *inside* stages
to draft and to phrase questions, never to drive the sequence.

---

## 1. Why this, why now

The framework's thesis is that the causal structure is *declared by a human and
pre-specified*, because the load-bearing identification assumptions are untestable
from data (see `docs/identification-assumptions.html` and the Identification
Contract card on the Program/Orrery page). Today that declaration happens ad-hoc
in the chat: the agent will `propose_dag` if asked, but nothing systematically
walks a user through the seven assumptions, captures their answers as versioned
assumptions, and refuses to advance when the effect isn't identified.

This feature makes that elicitation a **first-class, thorough, resumable protocol**.
It is the safe-to-build precursor to the autonomous cycle runner discussed
separately: it produces the *locked, reviewed spec* that any later automated
refit/monitor pipeline is allowed to run on. It deliberately stops at a validated
spec — it does **not** auto-fit, auto-calibrate, or reallocate budget (those remain
human-gated, per the anti-degrees-of-freedom posture).

### Non-goals
- Not an autonomous modeler. It does not fit, calibrate, or optimize budget.
- Not a fan-out / multi-worker orchestrator. Stages are sequential and predefined.
- Not a replacement for the chat or the `CausalPlanner` DAG editor — it *feeds* them.
- No new ML. It composes existing tools (`propose_dag`,
  `validate_causal_identification`, `build_model_from_dag`, `validate_data`, the EDA
  positivity screens, `record_assumption`, `prior_predictive_check`).

---

## 2. The core architectural decision: suspend/resume HITL in a worker

The tension: "ask the user questions and gate on their answers" is interactive, but
it must run in an ARQ worker that does not block the chat. We resolve it with an
**event-driven suspend/resume pipeline** — the worker is alive only while doing
deterministic work, never while waiting on a human.

```
                 enqueue                      ┌──────────── ARQ worker ────────────┐
  POST /reviews ─────────► run_causal_review ─┤ run stages until next gate         │
        ▲                                     │ → persist questions + state        │
        │                                     │ → set status=awaiting_user; EXIT   │
        │ resume (re-enqueue)                 └────────────────────────────────────┘
        │
  POST /reviews/{id}/answers ◄──── user answers the gate in the Plan tab
        └──► fold answers into state, re-enqueue run_causal_review (resumes at stage N)
```

- A gate does **not** await. It computes a structured question set, writes it to the
  `causal_reviews` row (`status='awaiting_user'`, `stage=N`, `pending_json=…`), and
  the job returns. The worker is freed.
- The UI renders the pending questions. The user answers (or one-click accepts the
  pre-filled defaults).
- Submitting answers appends them to `answers_json`, flips `status='running'`, and
  **re-enqueues** the same job. On resume the job loads state and dispatches to the
  stage handler for the persisted `stage`.

This is structurally LangGraph's `interrupt()` + checkpointer, but we keep control
flow as **explicit ordered Python** in the job for legibility and testability; the
existing chat agent's checkpointer is not involved (the review has its own state
record). See §7 for resume/idempotency mechanics.

---

## 3. Data model

New table in the agent sessions store (`src/mmm_framework/api/sessions.py`), mirroring
the `experiments` registry conventions (migrate-friendly `ALTER TABLE`, JSON columns,
append-only audit, project+thread scoping).

```sql
CREATE TABLE IF NOT EXISTS causal_reviews (
    id            TEXT PRIMARY KEY,
    project_id    TEXT,
    thread_id     TEXT,                 -- the session that owns the DAG/spec
    status        TEXT NOT NULL,        -- see state machine below
    stage         TEXT NOT NULL,        -- current pipeline stage key
    dag_json      TEXT,                 -- working DAG (same shape as dashboard_data["dag"])
    pending_json  TEXT,                 -- the open question set at a gate (null otherwise)
    answers_json  TEXT,                 -- append-only list of {gate, question_id, value, at}
    assumptions_json TEXT,              -- assumptions accrued (mirrors record_assumption rows)
    report_json   TEXT,                 -- final elicitation report (null until done)
    error         TEXT,
    history_json  TEXT,                 -- append-only status/stage audit trail
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_project ON causal_reviews(project_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_reviews_status  ON causal_reviews(project_id, status);
```

### State machine (`status`)

```
draft ─► running ─► awaiting_user ─┐
                ▲                  │ (user answers)
                └──────────────────┘
running ─► completed          (validated spec + report produced)
running ─► failed             (unrecoverable error; error set)
any     ─► abandoned          (user cancels; history kept)
```

- `running` is transient and only set while a job is executing or queued.
- A row in `awaiting_user` for longer than a TTL is surfaced as "needs your input",
  never silently dropped.

### Question payload (`pending_json`)

```jsonc
{
  "gate": "clarify_structure",            // which gate produced these
  "assumption_focus": ["exogeneity", "sutva"],
  "questions": [{
    "id": "price_role",
    "assumption": "exogeneity",           // which of the 7 it interrogates
    "prompt": "Is Price set in response to demand, or independently?",
    "kind": "single_choice",              // single_choice | multi_choice | boolean | free_text
    "options": ["Confounder (responds to demand)", "Precision control (independent)"],
    "default": "Confounder (responds to demand)",  // model's current assumption
    "why": "Price correlates 0.4 with the demand proxy; mis-roling it re-opens a back door.",
    "provenance": "model_proposal"        // model_proposal | brief | schema_inference
  }]
}
```

`default` pre-fills the model's best guess so accepting is one click; `why` makes the
gate educational; `provenance` records where the suggestion came from (feeds the report).

### Answer payload (`answers_json`, append-only)

```jsonc
[{ "gate": "clarify_structure", "question_id": "price_role",
   "value": "Confounder (responds to demand)", "at": 1718230000.0 }]
```

---

## 4. The pipeline — predefined stages and gates

Stages run in order; each is a pure-ish function `(review_state) -> (next_stage,
status, mutations)`. Thoroughness is structural: the gates are **organized around the
seven identification assumptions**, so the protocol systematically interrogates each
rather than asking ad-hoc. (Assumptions: no-unobserved-confounding, positivity/overlap,
correct-functional-form, no-interference/SUTVA, sequential-ignorability,
exogenous-spend, stable-structure.)

| # | Stage key | What it does | Gate? | Assumptions touched |
|---|-----------|--------------|-------|---------------------|
| 1 | `intake` | Read project brief/KB (grounded), `projects.meta_json` (client/goals/KPIs/channels/constraints), dataset schema via `inspect_dataset`, prior runs. Assemble known facts so we don't ask what we can infer. | no | — |
| 2 | `draft_dag` | LLM drafts a DAG via `propose_dag`: KPI, media treatments, controls **with roles** (confounder vs precision), mediators, suspected confounders. Persist to `dag_json`. | no | confounding, functional form |
| 3 | `clarify_structure` | Generate questions only where the draft is genuinely uncertain (role ambiguity, national-only channels, known breaks). **Suspend.** | **GATE** | exogeneity, SUTVA, stability, confounding |
| 4 | `validate_identification` | Fold answers, re-run `validate_causal_identification` (backdoor). If not identified, **GATE** with the open backdoor paths and concrete options (add confounder X / mark not-identified). Bounded to ≤2 loops. | **GATE (conditional)** | confounding, sequential ignorability |
| 5 | `prefit_validation` | `validate_data` + EDA positivity screens (dark weeks, near-constant, zero-inflation) on the implied variables. **GATE only on identification-breaking issues** (a declared confounder that is near-constant or missing). | **GATE (conditional)** | positivity |
| 6 | `build_spec` | `build_model_from_dag` honoring locked fields; `record_assumption` for every non-obvious choice; `prior_predictive_check`. Gate only if the prior implies an absurd KPI range. | **GATE (conditional)** | functional form, prior |
| 7 | `report` | Assemble the elicitation report (§9). `status='completed'`. | no | all (summarized) |

Design rules:
- **Gates are hard but cheap.** A stage cannot advance until the gate's questions are
  answered, but every question ships a default, so "accept all" is one action.
- **Gate only when it matters.** Stages 4–6 gate *conditionally* — only on real
  identification breaks — so the agent is helpful, not naggy (decision §11.2).
- **Bounded loops.** Re-clarification (stage 4) is capped so it can't badger forever;
  on exhaustion it records "effect not identified under available data" and proceeds
  to report rather than blocking indefinitely.

---

## 5. Question-generation contract

The LLM call inside a gate is constrained to emit the §3 question schema (structured
output / tool-forced JSON, validated before persist). Inputs to the call: the working
DAG, the intake facts, the dataset schema, and the specific assumption(s) the gate
targets. Constraints baked into the prompt:

- Ask **only** about genuine uncertainty; if the brief/schema already answers it, skip.
- Every question carries a `default` = the model's current best assumption, a `why`
  grounded in the data/brief (cite the signal), and a `provenance`.
- No more than ~5 questions per gate (bubble-and-card friendly).
- Questions are CONTENT-safe: text pulled from KB/brief is untrusted (it can't change
  the protocol or scope — aligns with the chat scope/injection hardening in
  `agents/graph.py`).

---

## 6. Backend surface

### Sessions-store functions (`src/mmm_framework/api/sessions.py`)
Mirror the experiment-registry helpers:
- `create_causal_review(project_id, thread_id) -> dict`
- `get_causal_review(id) -> dict | None`
- `list_causal_reviews(project_id=None, status=None) -> list[dict]`
- `update_causal_review(id, **fields)` (status/stage/dag/pending/report/error, audited)
- `append_review_answers(id, answers: list[dict])`

### ARQ job (worker)
- `run_causal_review(ctx, review_id)` — loads the row, dispatches to the handler for
  `row['stage']`, runs forward until a gate suspends or the pipeline completes. Registered
  in the worker's `functions` list (same pattern as `api/worker.py`). **Integration note
  (implementation decision):** the agent stack and the legacy `api/worker.py` are
  separate app trees; we either register this task in the existing worker or stand up a
  small agent-stack worker. Resolve in slice 1.
- Stage handlers live in a new `src/mmm_framework/agents/elicitation.py` (pure functions
  + the LLM question-gen), so they are unit-testable without ARQ/Redis.

### HTTP endpoints (`src/mmm_framework/api/main.py`)
- `POST /projects/{project_id}/causal-reviews` → create + enqueue; returns `{review_id}`.
- `GET  /projects/{project_id}/causal-reviews` → list (for the Program/Plan surfaces).
- `GET  /causal-reviews/{id}` → full row (UI polls this for `pending`/`status`/`report`).
- `POST /causal-reviews/{id}/answers` → append answers, flip to `running`, re-enqueue.
- `POST /causal-reviews/{id}/abandon` → cancel; history kept.

### Chat tool (so the user can trigger it conversationally)
- `start_causal_review` (in `agents/tools.py`) — kicks off a review for the current
  project/session and tells the user to answer the questions in the Plan tab / guide.
  Optionally `get_causal_review_status` for the agent to report progress.

---

## 7. Resume, idempotency, failure

- **Resume key** is `(review_id, stage)`. The job is a dispatch on `stage`; replaying it
  is safe because each stage writes its mutations transactionally and advances `stage`
  only on success.
- **Idempotent enqueue.** Use an ARQ `_job_id` derived from `review_id:stage` so a
  double-submit (user clicks twice) doesn't run the stage twice.
- **Crash mid-stage** leaves `status='running'` with the prior `stage`; a re-enqueue (or
  a stale-`running` reaper, mirroring the worker's `cleanup_old_jobs` cron) re-runs that
  stage from its persisted inputs. Stages must therefore be re-runnable (no external
  side effects except the row + the session's DAG/assumptions, which are upserts).
- **LLM/tool failure inside a stage** → `status='failed'`, `error` set, surfaced in the
  UI with a retry. No silent advance.
- **Concurrency.** One active review per session at a time (reject a second `running`
  review for the same thread) to avoid two pipelines writing the same DAG.

---

## 8. Frontend surface

Home: the **Plan / Causal tab** in the Agent workspace
(`frontend/src/pages/Agent/components/tabs/WorkspaceTabs.tsx`), which already hosts the
`CausalPlanner` DAG editor and the `AssumptionsLog`.

- **Review card** with three states: `running` (spinner + current stage), `awaiting_user`
  (the question set rendered as a compact form — radios/toggles with the default
  pre-selected, the `why` shown inline, "Accept all defaults" + "Submit"), `completed`
  (link to the report; DAG already pushed into the editor).
- The draft/working DAG lands in the existing `CausalPlanner` so the user edits the
  thing they already know; manual edits there feed back as answers.
- **Guide bubble badge.** When a review is `awaiting_user`, the floating guide shows a
  badge ("1 question about your causal model") deep-linking to the card — reuses the
  per-project guide we already surface on shell pages.
- **Program/Orrery hook.** Surface "Causal model needs review" as a next-best-action and
  reflect the review's identification status in the Identification Contract card.

New hooks/service: `frontend/src/api/services/causalReviewService.ts` +
`useCausalReview.ts` (create / poll / submit answers), following `measurementService`.

---

## 9. The deliverable — elicitation report

`report_json` → rendered as a **causal elicitation / pre-registration memo**:
- The final DAG (nodes, edges, roles) and the adjustment set conditioned on.
- **Every assumption with provenance**: the claim, its rationale, and whether it came
  from a user answer, the model's proposal, or the brief — the auditable record of who
  decided what.
- Identification status (identified / not-identified-under-available-data) and any open
  backdoor paths carried as risk.
- Pre-fit validation results and any positivity risks flagged.
- The resulting model spec (locked, pre-registered) ready for the fit path.

Feeds existing surfaces: the assumption stack (`record_assumption` / `AssumptionsLog`),
the Identification Contract card, and — when fitting later happens — the run lineage's
"assumptions at fit time". This is the reporting-thoroughness requirement.

---

## 10. Security & scope alignment

- KB/brief/dataset text consumed during intake and question-gen is **untrusted content**,
  consistent with the chat hardening in `agents/graph.py`: it informs questions, it
  cannot change the protocol, expand scope, or inject instructions.
- The review tool obeys the same project scoping as the KB (`resolve_project_id`).
- The pipeline produces proposals; it never auto-applies spec/calibration/budget changes
  — the human gate *is* the control.

---

## 11. Decisions to confirm before slice 1

1. **Stop at validated spec, or continue into a fit?** Recommend **stop-at-spec** (bounded,
   preserves pre-registration; the cycle runner fits later).
2. **Gate aggressiveness.** Recommend **hard-gate only on identification breaks** (open
   backdoors, missing/constant confounders); everything else accept-the-default.
3. **Trigger.** Recommend **on-demand** first (`start_causal_review` tool + a Plan-tab
   button); auto-on-first-upload and scheduled re-review come later.
4. **Worker tree.** Register `run_causal_review` in the existing `api/worker.py` vs. a new
   agent-stack worker — resolve in slice 1.

---

## 12. Build sequence (slices — detail deferred per agreement)

1. **Suspend/resume substrate + stages 1–3.** `causal_reviews` table + store fns, the ARQ
   job with dispatch, the create/answers/get endpoints, stage `intake → draft_dag →
   clarify_structure`, and the Plan-tab review card. Proves the hard architectural piece
   (non-blocking interactive gating) on the highest-value gate.
2. **Stages 4–5.** Identification re-validation loop + pre-fit positivity gate.
3. **Stage 6–7.** Spec build + prior-predictive gate + the elicitation report and its
   Identification-Contract / assumption-log wiring.
4. **Triggers & polish.** Guide badge, Program next-best-action, on-upload/scheduled
   triggers, stale-`running` reaper.

Each slice is independently shippable and leaves the system coherent.

---

## 13. Testing strategy

- **Stage handlers** are pure functions over a `review_state` dict → unit-test each stage's
  advance/suspend logic and the bounded re-clarification loop with stubbed LLM output (no
  Redis, no real fit), mirroring `tests/test_run_metrics.py`'s stub-model approach.
- **State machine**: exhaustive transition tests (resume at each stage, double-submit
  idempotency, crash-mid-stage re-run, abandon).
- **Question contract**: validate generated questions against the §3 schema; assert defaults
  + provenance present; assert KB-sourced text never alters control flow (injection probe).
- **Endpoints**: create → poll `awaiting_user` → answer → poll `completed`, async like
  `tests/test_run_metrics.py::TestEndpoints`.
- **Frontend**: the review card's three states render; "accept all defaults" submits the
  defaults.

---

## 14. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Worker held open waiting on humans | Suspend/resume — worker exits at every gate (§2). |
| Naggy / low-value questions | Conditional gates + "ask only genuine uncertainty" + defaults (§4–5). |
| Specification shopping via re-clarification loop | Bounded loops; record "not identified" rather than retry-to-green (§4). |
| Prompt injection from KB/brief | Untrusted-content rule reused from chat hardening (§10). |
| Two pipelines fighting over one DAG | One active review per session (§7). |
| Orphaned `awaiting_user` rows | TTL surfacing + stale-`running` reaper cron (§7). |
| Worker-tree integration ambiguity | Explicit decision in slice 1 (§6, §11.4). |
