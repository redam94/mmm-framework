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
from typing import Annotated, Any, Union

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
    classify_dag_roles,
    frontdoor_criterion,
    identification_report,
    iv_criterion,
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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No active thread id; cannot store research question.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

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
    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard,
        }
    )


# ── 2. Propose a causal DAG (Step 2 of workflow) ─────────────────────────────


def _normalize_node_ids(names: list[str]) -> list[tuple[str, str]]:
    """Return [(node_id, variable_name), ...] with deterministic id slugs."""
    out = []
    seen = set()
    for n in names:
        slug = n.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
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
    instruments: list[dict[str, str]] = None,
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
    instruments = instruments or []

    nodes: list[DAGNode] = []
    edges: list[DAGEdge] = []

    # KPI
    kpi_pair = _normalize_node_ids([kpi])[0]
    nodes.append(
        DAGNode(id=kpi_pair[0], variable_name=kpi_pair[1], node_type=NodeType.KPI)
    )

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
    media_routed_through_mediator: set[str] = set()
    for med_id, med_name in mediator_pairs:
        nodes.append(
            DAGNode(id=med_id, variable_name=med_name, node_type=NodeType.MEDIATOR)
        )
        edges.append(
            DAGEdge(source=med_id, target=kpi_pair[0], edge_type=EdgeType.MEDIATED)
        )
        for upstream_media_name in mediator_inputs.get(med_name, []):
            src_id = media_name_to_id.get(upstream_media_name)
            if src_id:
                edges.append(
                    DAGEdge(source=src_id, target=med_id, edge_type=EdgeType.MEDIATED)
                )
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
        # The confounder node may already exist (e.g. the same variable was also
        # listed in `controls`). In that case keep the existing node, but still
        # add its fan-out edges below — otherwise the confounder→target edges
        # (the whole point of declaring a confounder) silently disappear.
        if cid_pair[0] not in {n.id for n in nodes}:
            nodes.append(
                DAGNode(
                    id=cid_pair[0],
                    variable_name=cid_pair[1],
                    node_type=NodeType.CONTROL,
                )
            )
            name_to_id[cid_pair[1]] = cid_pair[0]
        for target_name in affects:
            tid_node = name_to_id.get(target_name)
            if tid_node is None:
                # The user named a target that isn't a node yet; add it as a control
                tid_node = _normalize_node_ids([target_name])[0][0]
                nodes.append(
                    DAGNode(
                        id=tid_node,
                        variable_name=target_name,
                        node_type=NodeType.CONTROL,
                    )
                )
                name_to_id[target_name] = tid_node
            edges.append(DAGEdge(source=cid_pair[0], target=tid_node))

    # Instruments — each becomes an INSTRUMENT node with a DIRECT edge into its
    # treatment (exogenous variation that reaches the KPI only through the
    # treatment). Enables the IV identification check.
    for inst in instruments:
        iname = inst.get("name", "Instrument")
        treatment_name = inst.get("treatment")
        iid_pair = _normalize_node_ids([iname])[0]
        if iid_pair[0] in {n.id for n in nodes}:
            continue
        nodes.append(
            DAGNode(
                id=iid_pair[0],
                variable_name=iid_pair[1],
                node_type=NodeType.INSTRUMENT,
            )
        )
        name_to_id[iid_pair[1]] = iid_pair[0]
        treat_id = name_to_id.get(treatment_name)
        if treat_id is not None:
            edges.append(DAGEdge(source=iid_pair[0], target=treat_id))

    # Cross effects between outcomes
    for ce in cross_effects:
        src = name_to_id.get(ce.get("source", ""))
        tgt = name_to_id.get(ce.get("target", ""))
        if src and tgt:
            edges.append(
                DAGEdge(source=src, target=tgt, edge_type=EdgeType.CROSS_EFFECT)
            )

    # Dedup edges by (source, target), keeping the first occurrence so the
    # original edge_type is preserved. A confounder whose `affects` includes the
    # KPI (or a node already wired up as a control) would otherwise produce a
    # duplicate edge, e.g. Distribution→Sales added by both the controls loop and
    # the confounder fan-out.
    seen_edges: set[tuple[str, str]] = set()
    deduped_edges: list[DAGEdge] = []
    for edge in edges:
        key = (edge.source, edge.target)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped_edges.append(edge)
    edges = deduped_edges

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
    lines.append(
        "Use `validate_causal_identification` next to check whether the causal effect is identified."
    )

    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ],
            "dashboard_data": dashboard,
        }
    )


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

    NOTE: Checks back-door adjustment, and additionally reports front-door
    identification when the DAG has mediators and IV identification when it has
    declared instruments. Full-ID (Tian-Pearl) is not checked.

    Args:
        treatment: variable_name OR node_id of the treatment.
        outcome: variable_name OR node_id of the outcome.
        adjustment_set: variable_names or node_ids to condition on. None ⇒ propose one.
    """
    dashboard = state.get("dashboard_data", {}) if state else {}
    dag_payload = dashboard.get("dag")
    if not dag_payload:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No DAG found in state. Call `propose_dag` first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    spec = DAGSpec.model_validate(dag_payload["spec"])

    def _resolve(name: str) -> str | None:
        n = spec.get_node(name) or spec.get_node_by_variable(name)
        return n.id if n else None

    t_id = _resolve(treatment)
    y_id = _resolve(outcome)
    if not t_id or not y_id:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Could not resolve treatment='{treatment}' or outcome='{outcome}' "
                            f"to a node in the DAG. Known variables: "
                            f"{', '.join(spec.variable_names)}"
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

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

    # Automatic control-role guidance. Non-experts do not need to know causal
    # inference to avoid bad controls: classify every control against ALL media
    # treatments and tell the user, in plain language, what each one is and what
    # to do. Confounders are kept (un-shrunk); mediators/colliders are flagged
    # for removal (the model refuses to fit with them as controls).
    control_nodes = spec.control_nodes
    if control_nodes:
        media_ids = [n.id for n in spec.media_nodes]
        control_ids = [n.id for n in control_nodes]
        roles = classify_dag_roles(spec, media_ids, y_id, control_ids)
        marker = {
            "confounder": "✅ keep",
            "precision_control": "• keep",
            "mediator": "⛔ REMOVE",
            "collider": "⛔ REMOVE",
        }
        lines.append("")
        lines.append("**Control variable roles (auto-detected):**")
        for node in control_nodes:
            role, reason = roles.role_for(node.id)
            label = role.replace("_", " ")
            lines.append(
                f"  - `{node.variable_name}` — **{label}** "
                f"[{marker.get(role, '?')}]: {reason}"
            )
        if any(
            roles.role_for(n.id)[0] in ("mediator", "collider") for n in control_nodes
        ):
            lines.append(
                "\n_Variables marked ⛔ REMOVE are 'bad controls': conditioning on "
                "them biases the effect estimate. The model will refuse to fit "
                "while they are listed as controls — drop them from the control "
                "set. If one is genuinely a common cause of media and the KPI, "
                "re-draw it as a confounder instead._"
            )

    # Front-door identification (when mediators are present): an alternative to
    # back-door adjustment that can identify the effect *through* the mediation
    # pathway even when treatment and outcome are confounded.
    if spec.has_mediators:
        med_ids = [m.id for m in spec.mediator_nodes]
        med_names = {m.id: m.variable_name for m in spec.mediator_nodes}
        fd = frontdoor_criterion(spec, t_id, med_ids, y_id)
        verdict = "✅ Yes" if fd.identifiable else "❌ No"
        lines.append("")
        lines.append(
            f"**Front-door check** (through {', '.join(f'`{med_names[m]}`' for m in med_ids)}): "
            f"**{verdict}**"
        )
        if fd.identifiable:
            lines.append(
                "  - The effect is identified via the mediation pathway by the "
                "front-door formula, even without measuring the treatment-outcome "
                "confounders."
            )
            lines.append(
                "  - A linear front-door **estimate is available** as a cross-check "
                "via `mmm_framework.estimators.frontdoor_estimate(y, T, M)`. Note the "
                "fitted MMM itself still uses the **back-door additive estimator**, "
                "so compare the two: a large gap signals treatment–outcome "
                "confounding the back-door model can't remove."
            )
        else:
            for note in fd.notes:
                lines.append(f"  - {note}")

    # IV identification (when instruments are declared): identifies the effect
    # using exogenous variation that reaches the KPI only through a treatment --
    # one of the few routes that survives unobserved demand confounding.
    if spec.has_instruments:
        lines.append("")
        lines.append("**Instrumental-variable check:**")
        lines.append(
            "  - A 2SLS **estimate is available** as a cross-check via "
            "`mmm_framework.estimators.two_stage_least_squares(y, T, Z)` (reports a "
            "first-stage F for weak instruments). The fitted MMM uses the back-door "
            "additive estimator; the IV estimate is the confounding-robust comparison."
        )
        for z in spec.instrument_nodes:
            iv = iv_criterion(spec, z.id, t_id, y_id)
            verdict = "✅ valid" if iv.identifiable else "❌ invalid"
            lines.append(
                f"  - `{z.variable_name}` → `{treatment}`: **{verdict}** "
                f"(relevant={iv.is_relevant}, exogenous={iv.is_exogenous}, "
                f"exclusion={iv.satisfies_exclusion})"
            )
            if iv.identifiable and iv.weak_instrument_warning:
                lines.append(f"    - ⚠️ {iv.weak_instrument_warning}")
            elif not iv.identifiable:
                for note in iv.notes:
                    lines.append(f"    - {note}")

    # Honest framing: identifiability here is *conditional on the DAG being
    # complete*. The dominant MMM confounder -- unobserved demand (spend rises
    # when demand is expected to rise) -- is by definition NOT in the graph, and
    # no adjustment set can remove an unobserved confounder.
    lines.append("")
    lines.append(
        "⚠️ **Identification rests on a NO-UNOBSERVED-CONFOUNDING assumption.** "
        "Even when identifiable *under this DAG*, the estimate is only causal if "
        "every common cause of spend and the KPI is measured and included. In MMM "
        "the key confounder is usually **unobserved demand**, which no adjustment "
        "set can fix. Anchor effects with a geo-lift / incrementality experiment "
        "(`mmm_framework.calibration`) and quantify exposure with the "
        "unobserved-confounding robustness value "
        "(`ValidationConfigBuilder().with_unobserved_confounding()`)."
    )

    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ],
            "dashboard_data": dashboard,
        }
    )


# ── 4. Assumptions log ───────────────────────────────────────────────────────

VALID_CATEGORIES = sorted(sessions_store.ASSUMPTION_CATEGORIES)


@tool
def record_assumption(
    key: str,
    # A bare ``Any`` yields an untyped JSON schema that Google Gemini's
    # function-declaration validator rejects (null property schema → tool-binding
    # fails for the whole agent). A bare ``list`` is also rejected server-side
    # (an ``array`` schema must declare ``items``). This union is Gemini-valid
    # (rendered as ``anyOf``) and still covers any JSON-able value: ``dict``
    # carries arbitrary nested structure (including lists) for the rare
    # structured assumption. Pydantic's smart-union keeps the native type.
    value: Union[str, int, float, bool, dict],
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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No active thread; cannot record assumption.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    rec = sessions_store.record_assumption(
        thread_id=tid,
        key=key,
        value=value,
        rationale=rationale,
        category=category,
        change_note=change_note or None,
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        f"📌 Recorded assumption `{key}` v{rec['version']} "
                        f"(category: {rec['category']})."
                    ),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No active thread; nothing to list.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    items = sessions_store.list_assumptions(tid, include_history=include_history)
    if not items:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="_No assumptions recorded yet._",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

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
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n".join(lines),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No active thread; cannot mark step.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    if status not in {"pending", "in_progress", "done", "skipped"}:
        status = "in_progress"
    if step < 1 or step > 9:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="step must be in [1, 9].",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    sessions_store.set_workflow_step(tid, step, status, notes or None)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Step {step} marked **{status}**"
                    + (f": {notes}" if notes else "."),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


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

    Runs PRE-FIT: the model graph is built from the active model_spec and
    dataset without fitting, so call this BEFORE `fit_mmm_model` (no throwaway
    fit needed). It always reflects the CURRENT spec/priors — re-run it after
    any prior change. Requires a configured model and a loaded dataset.
    """
    from mmm_framework.agents.tools import _KERNELS, _modelop_command, _normalized_spec
    from mmm_framework.agents.runtime import set_current_thread, get_current_thread

    set_current_thread(_thread_id_from(config))
    spec = _normalized_spec((state or {}).get("model_spec"))
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "prior_predictive_check",
        {
            "n_samples": int(n_samples),
            "spec": spec if spec.get("kpi") else None,
            "dataset_path": (state or {}).get("dataset_path"),
        },
    )
    _assumption = res.pop("assumption", None) if isinstance(res, dict) else None
    tid = _thread_id_from(config)
    if _assumption and tid and not res.get("error"):
        sessions_store.record_assumption(thread_id=tid, **_assumption)
    return _modelop_command(res, state or {}, tool_call_id)


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
    from mmm_framework.agents.tools import _KERNELS, _modelop_command
    from mmm_framework.agents.runtime import set_current_thread, get_current_thread

    set_current_thread(_thread_id_from(config))
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "leave_one_out", {"component_to_drop": component_to_drop}
    )
    _assumption = res.pop("assumption", None) if isinstance(res, dict) else None
    tid = _thread_id_from(config)
    if _assumption and tid and not res.get("error"):
        sessions_store.record_assumption(thread_id=tid, **_assumption)
    return _modelop_command(res, state or {}, tool_call_id)


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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No active thread id; cannot lock analysis plan.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Pull latest assumptions from session store
    current_assumptions = sessions_store.list_assumptions(tid, include_history=False)

    # Extract research question if present
    rq = next(
        (a["value"] for a in current_assumptions if a["key"] == "research_question"),
        None,
    )

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

    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ],
            "dashboard_data": {
                **dashboard,
                "analysis_plan_id": plan_id,
                "analysis_plan_name": name,
            },
        }
    )


@tool
def check_spec_divergence(
    config: InjectedConfig = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Compare the CURRENT causal DAG against the most recently LOCKED analysis plan
    and report any divergences (channels/controls/edges or hyperparameters added,
    removed, or changed since pre-registration).

    Use this before fitting or reporting to make pre-registration enforceable:
    it turns "the spec was logged" into "the spec was checked". List items are
    matched by identity, so reordering is not flagged.
    """
    from mmm_framework.config import diff_spec, summarize_spec_diff

    tid = _thread_id_from(config)
    plans = sessions_store.list_analysis_plans(tid) if tid else []
    if not plans:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "No locked analysis plan found. Call "
                            "`define_analysis_plan` first to pre-register the spec."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Most recent plan (plans are returned newest-first or we take max locked_at).
    latest = max(plans, key=lambda p: p.get("locked_at", 0))
    frozen_dag = (latest.get("payload") or {}).get("dag") or {}
    frozen_spec = frozen_dag.get("spec", frozen_dag)

    dashboard = state.get("dashboard_data", {}) if state else {}
    current_dag = dashboard.get("dag") or {}
    current_spec = current_dag.get("spec", current_dag)

    changes = diff_spec(frozen_spec, current_spec)
    plan_id = latest.get("id", "")[:8]
    if not changes:
        content = (
            f"✅ The current DAG matches the pre-registered plan `{plan_id}…`. "
            "No divergence."
        )
    else:
        content = (
            f"⚠️ The current DAG **diverges** from the pre-registered plan "
            f"`{plan_id}…`. Reported results should disclose this.\n\n"
            + summarize_spec_diff(changes)
        )

    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            "dashboard_data": {
                **dashboard,
                "spec_divergences": [
                    {"path": ch.path, "kind": ch.kind, "old": ch.old, "new": ch.new}
                    for ch in changes
                ],
            },
        }
    )


@tool
def build_model_from_dag(
    state: Annotated[dict, InjectedState],
    reason: str = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Derive the model specification from the session's causal DAG (built with
    propose_dag or edited in the Causal tab): KPI node -> kpi, MEDIA nodes ->
    media_channels, CONTROL nodes -> control_variables. Per-channel settings
    (adstock/saturation/priors) already configured for surviving channels are
    preserved; user-locked fields are honored (conflicts go to the pending
    confirmation list, never silently applied).

    Call this after the DAG is validated, instead of configure_model, so the
    fitted model matches the pre-registered causal structure.
    """
    from mmm_framework.agents.tools import (
        _activate_thread,
        _commit_spec,
        _dataset_variable_names,
        _normalized_spec,
        _partition_latent_controls,
    )
    from mmm_framework.dag_model_builder.model_type_resolver import (
        describe_model_type,
        resolve_model_type,
    )

    _activate_thread(config)

    def _fail(msg: str) -> Command:
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
        )

    dag_payload = (state.get("dashboard_data") or {}).get("dag") or {}
    spec_dict = dag_payload.get("spec")
    if not spec_dict:
        return _fail(
            "No causal DAG found for this session — build one first with "
            "`propose_dag` (or in the Causal tab)."
        )
    validation = dag_payload.get("validation") or {}
    if validation and validation.get("valid") is False:
        errs = "; ".join(validation.get("errors") or []) or "unknown error"
        return _fail(
            f"The current DAG fails validation ({errs}). Fix it before deriving "
            "a model spec from it."
        )

    try:
        dag = DAGSpec.model_validate(spec_dict)
    except Exception as exc:
        return _fail(f"Could not parse the stored DAG spec: {exc}")

    kpi_nodes = dag.get_nodes_by_type(NodeType.KPI) or dag.get_nodes_by_type(
        NodeType.OUTCOME
    )
    if not kpi_nodes:
        return _fail("The DAG has no KPI/outcome node — add one before building.")
    media_nodes = dag.get_nodes_by_type(NodeType.MEDIA)
    if not media_nodes:
        return _fail("The DAG has no media nodes — add the treatments first.")
    control_nodes = dag.get_nodes_by_type(NodeType.CONTROL)
    mediator_nodes = dag.get_nodes_by_type(NodeType.MEDIATOR)
    instrument_nodes = dag.get_nodes_by_type(NodeType.INSTRUMENT)

    model_type = resolve_model_type(dag)
    type_note = describe_model_type(dag)

    import copy as _copy

    current = _normalized_spec(state.get("model_spec"))
    by_name_media = {
        c.get("name"): c for c in (current.get("media_channels") or []) if c.get("name")
    }
    by_name_ctrl = {
        c.get("name"): c
        for c in (current.get("control_variables") or [])
        if c.get("name")
    }

    # DAG proxies for latent baseline demand ("Trend", "Seasonality") are not
    # dataset variables — divert them to the built-in trend/seasonality
    # components instead of control regressors (where load_mff would fail with
    # "Missing expected variables"). Anything else missing from the dataset is
    # a hard error: silently dropping an adjustment-set member would break the
    # identification the DAG was validated for.
    ds_vars = _dataset_variable_names(state.get("dataset_path"))
    real_controls, latent_controls, missing_controls = _partition_latent_controls(
        [n.variable_name for n in control_nodes], ds_vars
    )
    if missing_controls:
        return _fail(
            "DAG control node(s) are not variables in the loaded dataset: "
            f"{', '.join(missing_controls)}. They are part of the adjustment set, "
            "so they cannot be silently dropped — add them to the dataset, or "
            "revise the DAG (latent baseline proxies should be named "
            "Trend/Seasonality to map onto the built-in components)."
        )

    candidate = _copy.deepcopy(current)
    candidate["kpi"] = kpi_nodes[0].variable_name
    candidate.setdefault("kpi_level", current.get("kpi_level") or "national")
    # Preserve any per-channel config the user/LLM already set for channels that
    # survive; new channels start with just their name.
    candidate["media_channels"] = [
        _copy.deepcopy(by_name_media.get(n.variable_name, {"name": n.variable_name}))
        for n in media_nodes
    ]
    candidate["control_variables"] = [
        _copy.deepcopy(by_name_ctrl.get(name, {"name": name})) for name in real_controls
    ]
    latent_lines = []
    for name, comp in latent_controls:
        if comp == "trend":
            if not candidate.get("trend"):
                candidate["trend"] = {"type": "linear"}
            latent_lines.append(
                f"- `{name}`: latent baseline proxy — modeled via the built-in "
                f"trend component (type={candidate['trend'].get('type', 'linear')}), "
                "not as a regressor."
            )
        else:
            seas = candidate.get("seasonality") or {}
            if not any(seas.get(k) for k in ("yearly", "monthly", "weekly")):
                candidate["seasonality"] = {"yearly": 2}
            latent_lines.append(
                f"- `{name}`: latent baseline proxy — modeled via the built-in "
                "seasonality component, not as a regressor."
            )
    candidate["dag_roles"] = {
        "mediators": [n.variable_name for n in mediator_nodes],
        "instruments": [n.variable_name for n in instrument_nodes],
    }
    candidate["dag_model_type"] = model_type.value

    lines = [
        "### Model spec derived from the causal DAG",
        f"- KPI: `{candidate['kpi']}`",
        f"- Media: {', '.join(f'`{n.variable_name}`' for n in media_nodes)}",
        ("- Controls: " + (", ".join(f"`{n}`" for n in real_controls) or "(none)")),
    ]
    lines.extend(latent_lines)
    if mediator_nodes:
        lines.append(
            "- Mediators (informational): "
            + ", ".join(f"`{n.variable_name}`" for n in mediator_nodes)
        )
    if instrument_nodes:
        lines.append(
            "- Instruments (identification only, not regressors): "
            + ", ".join(f"`{n.variable_name}`" for n in instrument_nodes)
        )
    lines.append(f"\n**Resolved model type:** {model_type.value} — {type_note}")
    if model_type.value != "bayesian_mmm":
        lines.append(
            "\n⚠️ **Honest scope note:** `fit_mmm_model` fits the basic Bayesian "
            "MMM only — mediators/multiple outcomes in the DAG are NOT modeled "
            "by it. For a nested/multivariate/combined model, build it with "
            "`DAGModelBuilder` via `execute_python` (see `library_reference`)."
        )

    return _commit_spec(
        state,
        candidate,
        tool_call_id,
        success_msg="\n".join(lines),
        reason=reason or "derive model spec from validated DAG",
        set_status="configured",
    )


CAUSAL_TOOLS = [
    define_research_question,
    propose_dag,
    validate_causal_identification,
    build_model_from_dag,
    record_assumption,
    list_assumptions,
    mark_workflow_step,
    prior_predictive_check,
    leave_one_out_decomposition,
    define_analysis_plan,
    check_spec_divergence,
]
