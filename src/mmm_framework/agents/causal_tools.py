"""Agent tools for the scientific causal MMM workflow.

These tools are added to the main TOOLS list in `agents/tools.py`. They share
the `_MODEL_CACHE` from that module via direct import.

Each tool needs the active `thread_id` to write to the session-scoped tables
in `api/sessions.py`. We get it through LangChain's `RunnableConfig`
injection — every LangGraph tool invocation receives the run config, which
includes `configurable.thread_id`.
"""
from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

# Injected at runtime by LangGraph; agent must NOT see this in the tool schema.
InjectedConfig = Annotated[RunnableConfig, InjectedToolArg]

from mmm_framework.api import sessions as sessions_store
from mmm_framework.dag_model_builder.dag_spec import (
    DAGEdge,
    DAGNode,
    DAGSpec,
    EdgeType,
    NodeType,
)
from mmm_framework.dag_model_builder.frontend_adapter import dag_spec_to_react_flow
from mmm_framework.dag_model_builder.identification import (
    identification_report,
    propose_adjustment_set,
    report_to_dict,
)
from mmm_framework.dag_model_builder.validation import validate_dag


def _thread_id_from(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


# ── 1. Define the research question (Step 1 of workflow) ─────────────────────

@tool
def define_research_question(
    question: str,
    business_decision: str,
    treatment_variable: str,
    outcome_variable: str,
    scope_notes: str = "",
    config: InjectedConfig = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Pre-register the causal question, the decision it supports, and the
    treatment/outcome variables, BEFORE looking at data. This is Step 1 of the
    canonical scientific workflow.

    Stored as a versioned assumption under key 'research_question'.

    Args:
        question: The causal/business question in plain English.
        business_decision: What decision will be made differently based on the answer.
        treatment_variable: Variable whose causal effect we want to estimate (e.g. "TV_spend").
        outcome_variable: The KPI/outcome variable (e.g. "Sales").
        scope_notes: Time window, geography, population scope.
    """
    tid = _thread_id_from(config)
    if not tid:
        return Command(update={"messages": [ToolMessage(
            content="No active thread id; cannot store research question.",
            tool_call_id=tool_call_id,
        )]})

    payload = {
        "question": question,
        "business_decision": business_decision,
        "treatment_variable": treatment_variable,
        "outcome_variable": outcome_variable,
        "scope_notes": scope_notes,
    }
    sessions_store.record_assumption(
        thread_id=tid,
        key="research_question",
        value=payload,
        rationale=f"Pre-registered before EDA. Decision: {business_decision}",
        category="research_question",
        change_note="initial registration",
    )

    dashboard = state.get("dashboard_data", {}) if state else {}
    dashboard["research_question"] = payload

    content = (
        f"**Research question registered**\n\n"
        f"- **Question:** {question}\n"
        f"- **Decision:** {business_decision}\n"
        f"- **Treatment:** `{treatment_variable}` → **Outcome:** `{outcome_variable}`\n"
        f"- **Scope:** {scope_notes or '_not specified_'}\n\n"
        f"This is now Assumption #1 (`research_question`). Edits will be versioned."
    )
    return Command(update={
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        "dashboard_data": dashboard,
    })


# ── 2. Propose a causal DAG (Step 2 of workflow) ─────────────────────────────

def _normalize_node_ids(names: list[str]) -> list[tuple[str, str]]:
    """Return [(node_id, variable_name), ...] with deterministic id slugs."""
    out = []
    seen = set()
    for n in names:
        slug = (n.lower().replace(" ", "_")
                .replace("-", "_").replace("/", "_"))
        base = slug or "node"
        candidate = base
        i = 2
        while candidate in seen:
            candidate = f"{base}_{i}"
            i += 1
        seen.add(candidate)
        out.append((candidate, n))
    return out


@tool
def propose_dag(
    kpi: str,
    media_channels: list[str],
    controls: list[str] = None,
    mediators: list[str] = None,
    mediator_inputs: dict[str, list[str]] = None,
    direct_media: list[str] = None,
    confounders: list[dict[str, Any]] = None,
    cross_effects: list[dict[str, str]] = None,
    narrative: str = "",
    config: InjectedConfig = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Build an explicit causal DAG for the MMM.

    IMPORTANT — mediator wiring:
    You must explicitly say which media variables route through each mediator
    via `mediator_inputs`. A bare list of `mediators` without `mediator_inputs`
    will leave them as floating nodes (no incoming edges); the agent must
    decide and pass this — defaulting to "all media → all mediators" produces
    causally wrong DAGs in almost every real case (e.g., TV drives awareness
    but search drives consideration, not both).

    Args:
        kpi: Name of the KPI / outcome variable (e.g. "Sales").
        media_channels: Media variables (treatments).
        controls: Control variables that affect KPI (default: empty).
        mediators: Mediator variable names. Each will get a MEDIATED edge to the KPI.
        mediator_inputs: {mediator_name: [media_names that drive it]}. The agent
            must specify this — bare `mediators` are not auto-wired to media.
        direct_media: Media that ALSO have a direct path to KPI in addition to
            (or instead of) going through any mediator. Default: all media that
            don't appear as an input to ANY mediator.
        confounders: List of dicts like {"name": "Economy", "affects": ["TV", "Sales"]}
            — each named confounder is added as a CONTROL node with edges to all listed
            variables. This is how you encode backdoor structure.
        cross_effects: List of {"source": "Display_KPI", "target": "Search_KPI"} for
            halo/cannibalization between outcomes.
        narrative: One-paragraph story of how the data was generated (the agent's
            justification). Stored as the rationale on the `dag_structure` assumption.
    """
    tid = _thread_id_from(config)
    controls = controls or []
    mediators = mediators or []
    mediator_inputs = mediator_inputs or {}
    confounders = confounders or []
    cross_effects = cross_effects or []

    nodes: list[DAGNode] = []
    edges: list[DAGEdge] = []

    # KPI
    kpi_pair = _normalize_node_ids([kpi])[0]
    nodes.append(DAGNode(id=kpi_pair[0], variable_name=kpi_pair[1], node_type=NodeType.KPI))

    # Media (build name → id lookup as we go)
    media_pairs = _normalize_node_ids(media_channels)
    media_name_to_id = {name: nid for nid, name in media_pairs}
    for mid, mname in media_pairs:
        nodes.append(DAGNode(id=mid, variable_name=mname, node_type=NodeType.MEDIA))

    # Controls
    control_pairs = _normalize_node_ids(controls)
    for cid, cname in control_pairs:
        nodes.append(DAGNode(id=cid, variable_name=cname, node_type=NodeType.CONTROL))
        edges.append(DAGEdge(source=cid, target=kpi_pair[0]))

    # Mediators with explicit upstream mapping
    mediator_pairs = _normalize_node_ids(mediators)
    mediator_name_to_id = {name: nid for nid, name in mediator_pairs}
    media_routed_through_mediator: set[str] = set()
    for med_id, med_name in mediator_pairs:
        nodes.append(DAGNode(id=med_id, variable_name=med_name, node_type=NodeType.MEDIATOR))
        edges.append(DAGEdge(source=med_id, target=kpi_pair[0], edge_type=EdgeType.MEDIATED))
        for upstream_media_name in mediator_inputs.get(med_name, []):
            src_id = media_name_to_id.get(upstream_media_name)
            if src_id:
                edges.append(DAGEdge(source=src_id, target=med_id, edge_type=EdgeType.MEDIATED))
                media_routed_through_mediator.add(src_id)

    # Direct media → KPI edges. Default: any media not already routed through a mediator.
    if direct_media is None:
        for media_id, _ in media_pairs:
            if media_id not in media_routed_through_mediator:
                edges.append(DAGEdge(source=media_id, target=kpi_pair[0]))
    else:
        for name in direct_media:
            mid = media_name_to_id.get(name)
            if mid:
                edges.append(DAGEdge(source=mid, target=kpi_pair[0]))

    # Confounders — each becomes a CONTROL node fan-out to its listed affects
    name_to_id = {n.variable_name: n.id for n in nodes}
    for conf in confounders:
        cname = conf.get("name", "Confounder")
        affects = conf.get("affects", [])
        cid_pair = _normalize_node_ids([cname])[0]
        if cid_pair[0] in {n.id for n in nodes}:
            continue
        nodes.append(DAGNode(id=cid_pair[0], variable_name=cid_pair[1], node_type=NodeType.CONTROL))
        name_to_id[cid_pair[1]] = cid_pair[0]
        for target_name in affects:
            tid_node = name_to_id.get(target_name)
            if tid_node is None:
                # The user named a target that isn't a node yet; add it as a control
                tid_node = _normalize_node_ids([target_name])[0][0]
                nodes.append(DAGNode(id=tid_node, variable_name=target_name, node_type=NodeType.CONTROL))
                name_to_id[target_name] = tid_node
            edges.append(DAGEdge(source=cid_pair[0], target=tid_node))

    # Cross effects between outcomes
    for ce in cross_effects:
        src = name_to_id.get(ce.get("source", ""))
        tgt = name_to_id.get(ce.get("target", ""))
        if src and tgt:
            edges.append(DAGEdge(source=src, target=tgt, edge_type=EdgeType.CROSS_EFFECT))

    spec = DAGSpec(nodes=nodes, edges=edges)
    validation = validate_dag(spec)
    react_flow = dag_spec_to_react_flow(spec)

    # Persist as assumption and to dashboard
    if tid:
        sessions_store.record_assumption(
            thread_id=tid,
            key="dag_structure",
            value=spec.model_dump(mode="json"),
            rationale=narrative or "DAG proposed by agent; not yet user-confirmed.",
            category="causal_structure",
            change_note="initial proposal",
        )

    dashboard = state.get("dashboard_data", {}) if state else {}
    dashboard["dag"] = {
        "spec": spec.model_dump(mode="json"),
        "react_flow": react_flow,
        "validation": {
            "valid": validation.valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
        },
    }

    lines = [
        f"**Proposed DAG** ({len(nodes)} nodes, {len(edges)} edges)",
        "",
        f"- KPI: `{kpi}`",
        f"- Media: {', '.join(f'`{m}`' for m in media_channels) or '_none_'}",
        f"- Controls: {', '.join(f'`{c}`' for c in controls) or '_none_'}",
        f"- Mediators: {', '.join(f'`{m}`' for m in mediators) or '_none_'}",
        f"- Confounders: {', '.join('`' + c.get('name', '?') + '`' for c in confounders) or '_none declared_'}",
    ]
    if not validation.valid:
        lines.append("")
        lines.append("⚠️ Validation errors:")
        for err in validation.errors:
            lines.append(f"  - {err}")
    if validation.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in validation.warnings:
            lines.append(f"  - {w}")
    lines.append("")
    lines.append("Use `validate_causal_identification` next to check whether the causal effect is identified.")

    return Command(update={
        "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        "dashboard_data": dashboard,
    })


# ── 3. Validate causal identification (backdoor) ─────────────────────────────

@tool
def validate_causal_identification(
    treatment: str,
    outcome: str,
    adjustment_set: list[str] = None,
    config: InjectedConfig = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Check whether the causal effect of `treatment` on `outcome` is identified
    in the current DAG (must have already called `propose_dag`). Uses Pearl's
    backdoor criterion. If `adjustment_set` is not supplied, a heuristic set
    is proposed and reported.

    NOTE: Backdoor-only. Frontdoor/IV identification is not checked here.

    Args:
        treatment: variable_name OR node_id of the treatment.
        outcome: variable_name OR node_id of the outcome.
        adjustment_set: variable_names or node_ids to condition on. None ⇒ propose one.
    """
    dashboard = state.get("dashboard_data", {}) if state else {}
    dag_payload = dashboard.get("dag")
    if not dag_payload:
        return Command(update={"messages": [ToolMessage(
            content="No DAG found in state. Call `propose_dag` first.",
            tool_call_id=tool_call_id,
        )]})

    spec = DAGSpec.model_validate(dag_payload["spec"])

    def _resolve(name: str) -> str | None:
        n = spec.get_node(name) or spec.get_node_by_variable(name)
        return n.id if n else None

    t_id = _resolve(treatment)
    y_id = _resolve(outcome)
    if not t_id or not y_id:
        return Command(update={"messages": [ToolMessage(
            content=(
                f"Could not resolve treatment='{treatment}' or outcome='{outcome}' "
                f"to a node in the DAG. Known variables: "
                f"{', '.join(spec.variable_names)}"
            ),
            tool_call_id=tool_call_id,
        )]})

    z_ids: list[str] | None = None
    if adjustment_set is not None:
        z_ids = []
        for z in adjustment_set:
            zid = _resolve(z)
            if zid:
                z_ids.append(zid)

    rep = identification_report(spec, t_id, y_id, adjustment_set=z_ids)
    rep_dict = report_to_dict(rep)
    dashboard["identification"] = rep_dict

    # Record as an assumption (so the analyst can argue with the claim later)
    tid = _thread_id_from(config)
    if tid:
        sessions_store.record_assumption(
            thread_id=tid,
            key=f"identification::{t_id}->{y_id}",
            value=rep_dict,
            rationale=(
                f"Backdoor-criterion check with adjustment set "
                f"{rep.adjustment_set or '∅'}; identifiable={rep.identifiable}."
            ),
            category="identification",
            change_note=f"adjustment_set={rep.adjustment_set}",
        )

    lines = [
        f"**Causal identification: `{treatment}` → `{outcome}`**",
        "",
        f"- Backdoor paths found: **{len(rep.backdoor_paths)}**",
        f"- Adjustment set: {rep.adjustment_set or '_∅_'}",
        f"- Identifiable (under this DAG): **{'✅ Yes' if rep.identifiable else '❌ No'}**",
        "",
    ]
    if rep.backdoor_paths:
        lines.append("Backdoor paths:")
        for p in rep.backdoor_paths:
            tag = "blocked" if p.is_blocked else "**OPEN**"
            blockers = (" — by " + ", ".join(p.blocked_by)) if p.blocked_by else ""
            lines.append(f"  - {p.render()}  [{tag}{blockers}]")
    for note in rep.notes:
        lines.append(f"\n_{note}_")

    return Command(update={
        "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        "dashboard_data": dashboard,
    })


# ── 4. Assumptions log ───────────────────────────────────────────────────────

VALID_CATEGORIES = sorted(sessions_store.ASSUMPTION_CATEGORIES)


@tool
def record_assumption(
    key: str,
    value: Any,
    rationale: str,
    category: str = "other",
    change_note: str = "",
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Add (or update) a modeling assumption to the session's versioned log.
    Use this when making a non-obvious modeling choice the analyst would want
    to see, argue with, or revise later.

    Args:
        key: stable identifier for the assumption (e.g. "weekly_aggregation",
            "no_unmeasured_confounders", "tv_carryover_long").
        value: the assumption itself (any JSON-able value).
        rationale: WHY this is being assumed. Plain prose.
        category: one of: research_question, causal_structure, data,
            functional_form, prior, identification, external_evidence, other.
        change_note: short note describing what changed since the previous version.
    """
    tid = _thread_id_from(config)
    if not tid:
        return Command(update={"messages": [ToolMessage(
            content="No active thread; cannot record assumption.",
            tool_call_id=tool_call_id,
        )]})
    rec = sessions_store.record_assumption(
        thread_id=tid, key=key, value=value, rationale=rationale,
        category=category, change_note=change_note or None,
    )
    return Command(update={"messages": [ToolMessage(
        content=(
            f"📌 Recorded assumption `{key}` v{rec['version']} "
            f"(category: {rec['category']})."
        ),
        tool_call_id=tool_call_id,
    )]})


@tool
def list_assumptions(
    include_history: bool = False,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """List modeling assumptions for the current session.

    Args:
        include_history: if True, returns every version of every assumption;
            otherwise only the current value per key.
    """
    tid = _thread_id_from(config)
    if not tid:
        return Command(update={"messages": [ToolMessage(
            content="No active thread; nothing to list.", tool_call_id=tool_call_id,
        )]})
    items = sessions_store.list_assumptions(tid, include_history=include_history)
    if not items:
        return Command(update={"messages": [ToolMessage(
            content="_No assumptions recorded yet._", tool_call_id=tool_call_id,
        )]})

    lines = [f"### Assumptions ({len(items)})", ""]
    by_cat: dict[str, list[dict]] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    for cat in sorted(by_cat):
        lines.append(f"**{cat}**")
        for a in by_cat[cat]:
            v_preview = json.dumps(a["value"])[:80]
            lines.append(
                f"- `{a['key']}` v{a['version']} — {v_preview}\n  _{a['rationale']}_"
            )
        lines.append("")
    return Command(update={"messages": [ToolMessage(
        content="\n".join(lines), tool_call_id=tool_call_id,
    )]})


# ── 5. Workflow step override ────────────────────────────────────────────────

@tool
def mark_workflow_step(
    step: int,
    status: str = "done",
    notes: str = "",
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Manually mark a workflow step's status.

    Most steps are inferred automatically from session state; only use this
    when you need to override the inferred status (e.g. mark Step 8 'skipped'
    or Step 4 'done' after the user accepts the prior predictive check).

    Args:
        step: 1..9, matching the canonical scientific workflow.
        status: one of: pending, in_progress, done, skipped.
        notes: short note for the UI tooltip.
    """
    tid = _thread_id_from(config)
    if not tid:
        return Command(update={"messages": [ToolMessage(
            content="No active thread; cannot mark step.", tool_call_id=tool_call_id,
        )]})
    if status not in {"pending", "in_progress", "done", "skipped"}:
        status = "in_progress"
    if step < 1 or step > 9:
        return Command(update={"messages": [ToolMessage(
            content="step must be in [1, 9].", tool_call_id=tool_call_id,
        )]})
    sessions_store.set_workflow_step(tid, step, status, notes or None)
    return Command(update={"messages": [ToolMessage(
        content=f"Step {step} marked **{status}**" + (f": {notes}" if notes else "."),
        tool_call_id=tool_call_id,
    )]})


# ── 6. Prior predictive check (Step 4) ───────────────────────────────────────

@tool
def prior_predictive_check(
    n_samples: int = 500,
    config: InjectedConfig = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Step 4 of the workflow. Sample from the prior predictive distribution
    and report whether implied outcomes are on a physically plausible scale.

    LIMITATION: with the current `fit_mmm_model` pipeline, the PyMC model
    object is only kept in cache AFTER fitting. So this tool effectively runs
    a *retrospective* prior predictive check (priors are fixed, model
    structure exists, but we're using the already-fit model). For a true
    pre-fit check, do a short throwaway fit (draws=10, tune=10) first — it's
    still faster than the real fit and gives us a model to draw from.
    """
    from mmm_framework.agents.tools import _MODEL_CACHE  # avoid circular import

    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(update={"messages": [ToolMessage(
            content=(
                "Prior predictive check needs a built PyMC model object. "
                "Run a quick throwaway fit (draws=10, tune=10) first, then "
                "call this tool — the result reflects the priors, not the "
                "(barely-touched) posterior."
            ),
            tool_call_id=tool_call_id,
        )]})
    try:
        idata = mmm.sample_prior_predictive(samples=int(n_samples))
    except Exception as e:
        return Command(update={"messages": [ToolMessage(
            content=f"Prior predictive sampling failed: {e}",
            tool_call_id=tool_call_id,
        )]})

    # Summary stats on the prior-predictive KPI
    try:
        import numpy as np
        pp = idata.prior_predictive
        var = list(pp.data_vars)[0]  # first observed-data variable
        arr = pp[var].values.reshape(-1)
        summary = {
            "samples": int(arr.size),
            "min": float(np.nanmin(arr)),
            "p05": float(np.nanpercentile(arr, 5)),
            "median": float(np.nanmedian(arr)),
            "p95": float(np.nanpercentile(arr, 95)),
            "max": float(np.nanmax(arr)),
            "frac_negative": float(np.mean(arr < 0)),
        }
    except Exception as e:
        summary = {"error": str(e)}

    dashboard = state.get("dashboard_data", {}) if state else {}
    dashboard["prior_predictive_summary"] = summary

    tid = _thread_id_from(config)
    flag_neg = summary.get("frac_negative", 0) > 0.05
    if tid:
        sessions_store.record_assumption(
            thread_id=tid,
            key="prior_predictive_check",
            value=summary,
            rationale=(
                "Prior predictive sanity check. "
                + ("⚠️ More than 5% of samples imply negative outcomes — consider tighter priors."
                   if flag_neg else "Implied outcome range looks plausible.")
            ),
            category="prior",
            change_note=f"n_samples={n_samples}",
        )

    lines = ["### Prior Predictive Check", ""]
    if "error" in summary:
        lines.append(f"Could not summarize: {summary['error']}")
    else:
        lines.append(f"- Samples: {summary['samples']:,}")
        lines.append(f"- Implied KPI range (5–95%): [{summary['p05']:,.0f}, {summary['p95']:,.0f}]")
        lines.append(f"- Median: {summary['median']:,.0f}")
        lines.append(f"- Fraction negative: {summary['frac_negative']:.1%}")
        if flag_neg:
            lines.append("\n⚠️ >5% of prior-predictive draws are negative. Tighten priors before fitting.")
    return Command(update={
        "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        "dashboard_data": dashboard,
    })


# ── 7. Leave-one-out decomposition (Step 8, sensitivity) ─────────────────────

@tool
def leave_one_out_decomposition(
    component_to_drop: str,
    config: InjectedConfig = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Sensitivity to a single component: recompute the KPI decomposition with
    one component (e.g. a media channel name) set to zero and report how
    much the remaining contributions absorb. This does NOT refit the model —
    it reweights existing posterior decomposition contributions.

    Use for quick "what if this channel weren't there" questions. For honest
    prior or specification sensitivity, refit via `fit_mmm_model` after
    perturbing the spec; this tool can't substitute for that.

    Args:
        component_to_drop: name of the component to zero out (case-insensitive).
            Must match a component in the existing decomposition.
    """
    from mmm_framework.agents.tools import _MODEL_CACHE
    from mmm_framework.reporting.helpers import compute_component_decomposition

    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(update={"messages": [ToolMessage(
            content="No fitted model; fit one first.", tool_call_id=tool_call_id,
        )]})
    try:
        decomp = compute_component_decomposition(mmm, include_time_series=False)
    except Exception as e:
        return Command(update={"messages": [ToolMessage(
            content=f"Could not compute decomposition: {e}",
            tool_call_id=tool_call_id,
        )]})

    target = component_to_drop.strip().lower()
    components = [(d.component, d.total_contribution) for d in decomp]
    match_idx = next((i for i, (c, _) in enumerate(components) if c.lower() == target), -1)
    if match_idx < 0:
        return Command(update={"messages": [ToolMessage(
            content=(
                f"Component `{component_to_drop}` not found. Known: "
                + ", ".join(f"`{c}`" for c, _ in components)
            ),
            tool_call_id=tool_call_id,
        )]})

    total = sum(v for _, v in components)
    dropped_name, dropped_val = components[match_idx]
    remaining = [(c, v) for c, v in components if c != dropped_name]
    remaining_total = sum(v for _, v in remaining)
    new_pct = [(c, v / remaining_total if remaining_total else 0.0) for c, v in remaining]
    pct_loss = dropped_val / total if total else 0.0

    dashboard = state.get("dashboard_data", {}) if state else {}
    dashboard["sensitivity_loo"] = {
        "dropped": dropped_name,
        "fraction_dropped": pct_loss,
        "remaining_decomposition": [
            {"component": c, "pct_of_remaining": p} for c, p in new_pct
        ],
    }

    tid = _thread_id_from(config)
    if tid:
        sessions_store.record_assumption(
            thread_id=tid,
            key=f"sensitivity::loo::{dropped_name}",
            value=dashboard["sensitivity_loo"],
            rationale=(
                f"Leave-one-out: dropping `{dropped_name}` removes {pct_loss:.1%} "
                "of total fitted KPI (post-hoc reweighting; not a refit)."
            ),
            category="other",
            change_note="leave-one-out decomposition",
        )

    lines = [
        "⚠️ **This is NOT a sensitivity refit.** It only reweights the *existing* "
        "posterior decomposition assuming the dropped channel contributed zero. "
        "Use this for quick what-if framing only; for honest sensitivity to "
        "the fit, re-run `fit_mmm_model` with the channel removed.",
        "",
        f"### Leave-one-out: drop `{dropped_name}`",
        "",
        f"- `{dropped_name}` accounts for **{pct_loss:.1%}** of fitted KPI.",
        f"- Remaining components renormalize to:",
    ]
    for c, p in new_pct:
        lines.append(f"  - `{c}`: {p:.1%}")
    return Command(update={
        "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        "dashboard_data": dashboard,
    })


# ── Define (lock) an analysis plan ───────────────────────────────────────────

@tool
def define_analysis_plan(
    name: str = "Analysis Plan",
    config: InjectedConfig = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Snapshot the current research question, causal DAG, and all recorded
    assumptions into a LOCKED analysis plan artifact.

    Call this after the research question, DAG, and key priors have been
    established — before fitting the model. The locked plan serves as a
    pre-registration document to prevent post-hoc rationalisation.

    Args:
        name: Human-readable name for this plan (e.g. "Q4 MMM pre-registration").
    """
    tid = _thread_id_from(config)
    if not tid:
        return Command(update={"messages": [ToolMessage(
            content="No active thread id; cannot lock analysis plan.",
            tool_call_id=tool_call_id,
        )]})

    # Pull latest assumptions from session store
    current_assumptions = sessions_store.list_assumptions(tid, include_history=False)

    # Extract research question if present
    rq = next((a["value"] for a in current_assumptions if a["key"] == "research_question"), None)

    # Pull DAG from dashboard_data if available
    dashboard = state.get("dashboard_data", {}) if state else {}
    dag_data = dashboard.get("dag")

    payload: dict[str, Any] = {
        "research_question": rq,
        "dag": dag_data,
        "assumptions": [
            {k: a[k] for k in ("key", "category", "value", "rationale", "version")}
            for a in current_assumptions
        ],
    }

    plan = sessions_store.lock_analysis_plan(thread_id=tid, name=name, payload=payload)
    plan_id = plan["id"]

    assumption_count = len(current_assumptions)
    lines = [
        f"**Analysis plan locked** — `{plan_id[:8]}…`",
        "",
        f"- **Name:** {name}",
        f"- **Research question:** {'✓' if rq else '⚠ not recorded yet'}",
        f"- **DAG:** {'✓' if dag_data else '⚠ not recorded yet'}",
        f"- **Recorded assumptions:** {assumption_count}",
        "",
        "This plan is now immutable. Any changes after this point are explicitly "
        "tracked as divergences from the pre-registered analysis.",
    ]

    return Command(update={
        "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        "dashboard_data": {**dashboard, "analysis_plan_id": plan_id, "analysis_plan_name": name},
    })


CAUSAL_TOOLS = [
    define_research_question,
    propose_dag,
    validate_causal_identification,
    record_assumption,
    list_assumptions,
    mark_workflow_step,
    prior_predictive_check,
    leave_one_out_decomposition,
    define_analysis_plan,
]
