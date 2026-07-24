"""Per-turn workflow-step guard for the oracle agent.

The system prompt tells the agent to advance the canonical modeling workflow by
**at most one step per user turn** and then hand control back to the user. Weak
orchestrator models (e.g. ``gemini-flash-lite``) ignore that instruction and
chain the whole pipeline — define question → propose DAG → validate → build →
prior-predictive → fit → diagnostics → … — off a single high-level ask like
*"I want to understand the total impact of media on my sales"* (whose real
intent was only to register the research question).

The ``/chat`` graph is a plain ``agent → tools → agent`` loop with no per-turn
tool-call budget, so nothing structurally stops that runaway; the prompt is the
only brake, and a weak model slips it. This module is the structural backstop:
it counts how many *workflow-advancing* ("milestone") tool calls the agent has
already run in the CURRENT user turn and, once the budget is spent, tells the
graph to pause — return control to the user — instead of running the next one.

Two escape hatches keep it from being annoying:

* If the user's message explicitly opts into an end-to-end run ("run the whole
  pipeline", "do everything", "build and fit end to end"), the guard stands down
  for that turn (see :func:`wants_full_run`).
* ``MMM_AGENT_WORKFLOW_STEP_BUDGET`` (int, default ``1``) tunes how many
  milestone advances are allowed per turn before pausing; ``0`` (or negative)
  disables the guard entirely.

The guard is applied only to the orchestrator / single-agent graph — never to
the expert sub-agent, which is handed ONE task and must run it to completion.
Everything here is pure (no I/O, no LLM) so it is cheap and deterministic to
test.
"""

from __future__ import annotations

import os
from typing import Collection, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

# ---------------------------------------------------------------------------
# What counts as advancing the workflow
# ---------------------------------------------------------------------------

#: Tools that COMMIT to a step of the canonical modeling pipeline — the ones that
#: auto-ran end-to-end in the reported bug. Read-only / supporting tools
#: (``inspect_dataset``, ``validate_data``, ``run_eda``, ``get_model_diagnostics``,
#: ``get_roi_metrics``, ``record_assumption``, ``list_assumptions``,
#: ``search_knowledge_base``, ``library_reference`` …) are deliberately NOT here:
#: reading/inspecting as part of doing one step well should never trip the guard.
#:
#: In the deployed two-tier path the heavy steps (``prior_predictive_check`` /
#: ``fit_mmm_model`` / the analysis tools) reach the model as a
#: ``delegate_to_expert`` hand-off rather than the underlying tool name, so
#: ``delegate_to_expert`` is counted as one advance too — one expert hand-off is
#: one workflow step. Both the direct names and ``delegate_to_expert`` are listed
#: so the guard works in the two-tier path, the full-orchestrator path
#: (``MMM_AGENT_ORCHESTRATOR_FULL_TOOLS=1``) and any single-agent deployment.
MILESTONE_TOOLS: frozenset[str] = frozenset(
    {
        # Pipeline setup + fit — the steps shown auto-running in the screenshot.
        "define_research_question",
        "define_analysis_plan",
        "propose_dag",
        "validate_causal_identification",
        "build_model_from_dag",
        "configure_model",
        "prior_predictive_check",
        "fit_mmm_model",
        # Forward decisions (a NEW analysis / decision, not merely reading a fit).
        "run_marginal_analysis",
        "run_budget_optimizer",
        "run_budget_scenario",
        "recommend_lift_experiments",
        "compute_experiment_priorities",
        # Two-tier: heavy steps above arrive as an expert hand-off — count each.
        "delegate_to_expert",
    }
)

#: Human-readable label for the step the guard is about to defer, used in the
#: pause message ("I paused before <label>."). Missing names fall back to a
#: generic phrase.
STEP_LABELS: dict[str, str] = {
    "define_research_question": "defining the research question",
    "define_analysis_plan": "locking the analysis plan",
    "propose_dag": "proposing the causal DAG",
    "validate_causal_identification": "validating causal identification",
    "build_model_from_dag": "building the model from the DAG",
    "configure_model": "configuring the model",
    "prior_predictive_check": "the prior-predictive check",
    "fit_mmm_model": "fitting the model",
    "run_marginal_analysis": "the marginal analysis",
    "run_budget_optimizer": "budget optimization",
    "run_budget_scenario": "the budget scenario",
    "recommend_lift_experiments": "recommending lift experiments",
    "compute_experiment_priorities": "computing experiment priorities",
    "delegate_to_expert": "the next analysis step",
}


def _step_budget() -> int:
    """Milestone advances allowed per turn before pausing (env-tunable).

    Read on every call (not import-time) so tests and operators can flip it via
    ``MMM_AGENT_WORKFLOW_STEP_BUDGET`` without reimporting. Explicit ``0`` (or a
    negative value) disables the guard; a missing or unparseable value falls back
    to the default of ``1`` (a typo must never silently disable a safety guard).
    """
    raw = os.environ.get("MMM_AGENT_WORKFLOW_STEP_BUDGET")
    if raw is None or raw == "":
        return 1
    try:
        return int(raw)
    except (TypeError, ValueError):
        # A typo (e.g. "off", "1O") must NOT silently disable a safety guard —
        # fall back to the default. The documented disable path is an explicit 0
        # (or negative). See the module docstring.
        return 1


# ---------------------------------------------------------------------------
# Opt-in detection — "just do the whole thing"
# ---------------------------------------------------------------------------

#: Phrases in the user's own message that mean "don't stop between steps". Kept in
#: sync with the examples the system prompt teaches ("run the whole pipeline",
#: "do everything", "build and fit a model end to end"). Deliberately UNAMBIGUOUS
#: multi-word phrases only — bare fragments like "build and fit", "all steps" or
#: "every step" are excluded because they match ordinary non-opt-in requests
#: ("Can you build and fit an MMM?", "walk me through every step").
_OPT_IN_PHRASES: tuple[str, ...] = (
    "run the whole",
    "whole pipeline",
    "whole workflow",
    "whole process",
    "the whole thing",
    "entire workflow",
    "entire pipeline",
    "entire process",
    "full pipeline",
    "full workflow",
    "do everything",
    "run everything",
    "do it all",
    "run it all",
    "all at once",
    "end to end",
    "end-to-end",
    "start to finish",
    "without stopping",
    "without pausing",
    "don't stop",
    "do not stop",
    "dont stop",
    "don't pause",
    "do not pause",
    "dont pause",
    "go all the way",
    "autonomously",
    "complete the workflow",
    "complete workflow",
)

#: Negation tokens that, appearing BEFORE an opt-in phrase, flip its meaning
#: ("don't run the whole pipeline", "no need to do everything"). Matching in the
#: safe direction: a false negative here just keeps the guard ON (an extra pause),
#: whereas a false POSITIVE would re-open the pipeline runaway the guard prevents.
_NEGATORS: tuple[str, ...] = (
    "don't",
    "dont",
    "do not",
    "never",
    "no need",
    "rather not",
)


def _message_text(message: BaseMessage) -> str:
    """Flatten a message's content to plain text (handles block-list content)."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def _latest_user_text(messages: Sequence[BaseMessage]) -> str:
    """The most recent human message's text (the turn that is currently running)."""
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


def wants_full_run(text: str) -> bool:
    """True iff the user explicitly asked for an end-to-end, no-pause run.

    Matches the strong opt-in phrases the system prompt teaches, but stands down
    when a negation precedes the earliest match ("don't run the whole pipeline",
    "no need to do everything") so a negated request cannot silently disable the
    guard.
    """
    lowered = (text or "").lower()
    earliest: int | None = None
    for phrase in _OPT_IN_PHRASES:
        idx = lowered.find(phrase)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return False
    # A negation anywhere before the opt-in phrase turns it into an opt-OUT.
    prefix = lowered[:earliest]
    if any(neg in prefix for neg in _NEGATORS):
        return False
    return True


# ---------------------------------------------------------------------------
# Milestone counting + the decision
# ---------------------------------------------------------------------------


def _milestone_calls(
    message: BaseMessage, valid_tools: Collection[str] | None = None
) -> list[str]:
    """Milestone tool names requested by ``message`` that would actually EXECUTE.

    A call counts only when its name is a milestone AND — when ``valid_tools`` is
    given — is in the bound toolset. In the two-tier path a heavy tool the
    orchestrator can't run (e.g. a hallucinated ``fit_mmm_model``) is rejected and
    corrected without doing any work, so it must NOT burn the per-turn budget;
    passing the orchestrator's valid tool names filters it out. ``None`` (the
    default) counts every milestone name — used by the pure unit tests.
    """
    calls = getattr(message, "tool_calls", None) or []
    names = [c.get("name") for c in calls if c.get("name") in MILESTONE_TOOLS]
    if valid_tools is not None:
        names = [n for n in names if n in valid_tools]
    return names


def turn_milestone_count(
    messages: Sequence[BaseMessage], valid_tools: Collection[str] | None = None
) -> int:
    """How many milestone tool calls the agent already ran THIS turn.

    Counts executable milestone tool_calls in AI messages after the last human
    message, EXCLUDING the final (pending, not-yet-executed) message — those are
    the advances already committed this turn. The count resets every human turn.
    """
    msgs = list(messages)
    count = 0
    # Skip the last message (the pending AIMessage under consideration) and walk
    # back until the human message that opened this turn.
    for message in reversed(msgs[:-1]):
        if isinstance(message, HumanMessage):
            break
        count += len(_milestone_calls(message, valid_tools))
    return count


def next_paused_step(
    messages: Sequence[BaseMessage],
    *,
    budget: int | None = None,
    valid_tools: Collection[str] | None = None,
) -> str | None:
    """Decide whether the pending tool call should be paused.

    Returns the human label of the first milestone step being deferred when the
    guard trips, or ``None`` when the tools should just run. The guard trips iff
    running the last message's milestone calls would push this turn's TOTAL
    milestone advances over ``budget`` — the count includes the milestones
    batched INTO the pending message, so a single AIMessage carrying several
    milestone tools cannot slip the whole batch through in one super-step. It
    also stands down when the guard is disabled (budget <= 0) or the user opted
    into an end-to-end run this turn.
    """
    if budget is None:
        budget = _step_budget()
    msgs = list(messages)
    if budget <= 0 or not msgs:
        return None

    pending = _milestone_calls(msgs[-1], valid_tools)
    if not pending:
        return None  # non-milestone (or no) executable tool call → always allowed

    if wants_full_run(_latest_user_text(msgs)):
        return None  # user asked for the whole thing

    committed = turn_milestone_count(msgs, valid_tools)
    if committed + len(pending) <= budget:
        return None  # the whole pending batch still fits this turn's budget

    # Name the FIRST milestone that will be deferred (the first past the budget's
    # remaining room within the pending batch).
    remaining = max(0, budget - committed)
    first = pending[remaining] if remaining < len(pending) else pending[-1]
    return STEP_LABELS.get(first, "the next step")


def plan_pause(
    messages: Sequence[BaseMessage],
    *,
    budget: int | None = None,
    valid_tools: Collection[str] | None = None,
) -> tuple[list[dict], list[dict], str]:
    """Split the pending message's tool calls for the pause node.

    Returns ``(run_calls, defer_calls, label)`` as raw tool_call dicts:

    * ``run_calls`` — every non-(executable-milestone) call PLUS the first
      ``budget - already-run`` executable-milestone calls (so the turn still
      advances exactly one step even when the model batched several milestones).
    * ``defer_calls`` — the milestone calls beyond the budget, to be answered
      with "deferred" ToolMessages and re-issued on the next turn.
    * ``label`` — the human label of the first deferred milestone.

    Consistent with :func:`next_paused_step`: ``defer_calls`` is non-empty exactly
    when that function trips.
    """
    if budget is None:
        budget = _step_budget()
    msgs = list(messages)
    last = msgs[-1] if msgs else None
    calls = list(getattr(last, "tool_calls", None) or [])
    remaining = max(0, budget - turn_milestone_count(msgs, valid_tools))

    run_calls: list[dict] = []
    defer_calls: list[dict] = []
    used = 0
    for call in calls:
        name = call.get("name")
        is_milestone = name in MILESTONE_TOOLS and (
            valid_tools is None or name in valid_tools
        )
        if is_milestone and used >= remaining:
            defer_calls.append(call)
        else:
            if is_milestone:
                used += 1
            run_calls.append(call)

    label = next(
        (
            STEP_LABELS.get(c.get("name"), "the next step")
            for c in defer_calls
            if c.get("name") in MILESTONE_TOOLS
        ),
        "the next step",
    )
    return run_calls, defer_calls, label


__all__ = [
    "MILESTONE_TOOLS",
    "STEP_LABELS",
    "wants_full_run",
    "turn_milestone_count",
    "next_paused_step",
    "plan_pause",
]
