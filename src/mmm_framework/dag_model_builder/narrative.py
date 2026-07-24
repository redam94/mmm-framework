"""Plain-language reading of a causal DAG.

:func:`dag_human_reading` turns a :class:`~mmm_framework.dag_model_builder.
dag_spec.DAGSpec` into the story a careful analyst would tell a stakeholder:
what each arrow claims, what the *absence* of an arrow assumes, which variables
are confounders vs mediators vs plain controls (and what that means for how
their coefficients may be read), and which declared variables still need a
measured data series.

Everything here is deterministic — the agent relays this text rather than
inventing its own causal interpretation, so the reading can never drift from
the graph actually stored in the session. Control roles are derived from the
same back-door machinery the identification tool uses
(:func:`~mmm_framework.dag_model_builder.identification.find_backdoor_paths`),
so the reading and the identification verdict can never disagree about which
variables open a back-door.
"""

from __future__ import annotations

import re

from .dag_spec import DAGSpec, EdgeType, NodeType
from .identification import find_backdoor_paths

__all__ = ["dag_human_reading"]


def _norm_col(name: str) -> str:
    """Loose column-name normalization for measured-vs-unmeasured matching."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _fmt(names: list[str]) -> str:
    return ", ".join(f"`{n}`" for n in names)


def _directed_reaches(spec: DAGSpec, src: str, dst: str) -> bool:
    """True when a directed path src → … → dst exists."""
    adj: dict[str, list[str]] = {}
    for e in spec.edges:
        adj.setdefault(e.source, []).append(e.target)
    stack, seen = [src], set()
    while stack:
        n = stack.pop()
        if n == dst:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, []))
    return False


def _control_roles(spec: DAGSpec, primary_id: str) -> dict[str, tuple[str, list[str]]]:
    """Classify every CONTROL node by the same back-door semantics the
    identification tool uses. Returns ``{control_id: (role, related_names)}``
    with roles:

    - ``confounder``       — sits on a back-door path between ≥1 media and the
                             primary outcome (related = those media names)
    - ``mediation_confounder`` — on a back-door between a mediator and the
                             outcome (confounds the mediation pathway)
    - ``spend_driver``     — drives ≥1 media but opens NO back-door (an
                             instrument-like driver; conditioning can amplify
                             bias, not remove it)
    - ``precision``        — reaches the outcome but no treatment
    - ``dangling``         — touches nothing relevant as drawn
    """
    media = spec.media_nodes
    mediators = spec.mediator_nodes
    on_media_bd: dict[str, set[str]] = {}
    for m in media:
        for p in find_backdoor_paths(spec, m.id, primary_id):
            for nid in p.nodes[1:-1]:
                on_media_bd.setdefault(nid, set()).add(m.variable_name)
    on_med_bd: dict[str, set[str]] = {}
    for md in mediators:
        for p in find_backdoor_paths(spec, md.id, primary_id):
            for nid in p.nodes[1:-1]:
                on_med_bd.setdefault(nid, set()).add(md.variable_name)

    roles: dict[str, tuple[str, list[str]]] = {}
    for c in spec.control_nodes:
        kids = {k.id for k in spec.get_children(c.id)}
        media_kids = [m.variable_name for m in media if m.id in kids]
        if c.id in on_media_bd:
            roles[c.id] = ("confounder", sorted(on_media_bd[c.id]))
        elif c.id in on_med_bd:
            roles[c.id] = ("mediation_confounder", sorted(on_med_bd[c.id]))
        elif media_kids:
            roles[c.id] = ("spend_driver", media_kids)
        elif _directed_reaches(spec, c.id, primary_id):
            roles[c.id] = ("precision", [])
        else:
            roles[c.id] = ("dangling", [])
    return roles


def dag_human_reading(
    spec: DAGSpec,
    *,
    known_columns: set[str] | None = None,
) -> str:
    """A markdown, plain-English reading of what the DAG implies.

    Parameters
    ----------
    spec:
        The causal DAG.
    known_columns:
        Variable names known to exist as measured series (dataset columns
        and/or the model spec's kpi/media/control lists). When given, the
        reading ends with a measured-vs-needs-data checklist; matching is
        case/punctuation-insensitive.
    """
    outcomes = list(spec.kpi_nodes) + [
        o for o in spec.outcome_nodes if o not in spec.kpi_nodes
    ]
    if not outcomes:
        return "_This DAG has no KPI/outcome node yet — add one to read it._"
    primary = outcomes[0]
    outcome_ids = {o.id for o in outcomes}
    media = spec.media_nodes
    mediators = spec.mediator_nodes
    controls = spec.control_nodes
    instruments = spec.instrument_nodes
    roles = _control_roles(spec, primary.id)

    children = {n.id: [c.id for c in spec.get_children(n.id)] for n in spec.nodes}

    lines: list[str] = ["**What this DAG says, in plain terms**", ""]

    # ── media routes ────────────────────────────────────────────────────────
    if media:
        lines.append("*How each channel is claimed to work:*")
        for m in media:
            direct = primary.id in children.get(m.id, [])
            via = [
                md.variable_name
                for md in mediators
                if md.id in children.get(m.id, [])
                and _directed_reaches(spec, md.id, primary.id)
            ]
            other = [
                o.variable_name
                for o in outcomes[1:]
                if _directed_reaches(spec, m.id, o.id)
            ]
            if direct and via:
                lines.append(
                    f"- `{m.variable_name}` reaches `{primary.variable_name}` two "
                    f"ways: directly, AND through {_fmt(via)}. Its total effect is "
                    f"the sum of both routes."
                )
            elif via:
                lines.append(
                    f"- `{m.variable_name}` reaches `{primary.variable_name}` only "
                    f"through {_fmt(via)} — the graph claims no direct route."
                )
            elif direct:
                lines.append(
                    f"- `{m.variable_name}` → `{primary.variable_name}` directly."
                )
            elif other:
                lines.append(
                    f"- `{m.variable_name}` moves {_fmt(other)} (a secondary "
                    f"outcome); it touches `{primary.variable_name}` only through "
                    f"whatever cross-outcome links are drawn."
                )
            elif _directed_reaches(spec, m.id, primary.id):
                lines.append(
                    f"- `{m.variable_name}` reaches `{primary.variable_name}` "
                    f"indirectly (through the drawn intermediate nodes)."
                )
            else:
                lines.append(
                    f"- ⚠️ `{m.variable_name}` has NO route to any outcome in this "
                    f"graph — as drawn, it cannot move the KPI at all. Wire its "
                    f"edge (or remove the node)."
                )
        lines.append("")

    # ── mediators ───────────────────────────────────────────────────────────
    if mediators:
        lines.append("*Mediators (ON the causal path):*")
        mediator_ids = {md.id for md in mediators}
        for md in mediators:
            drivers = [
                p.variable_name
                for p in spec.get_parents(md.id)
                if p.node_type == NodeType.MEDIA or p.id in mediator_ids
            ]
            reaches_out = any(_directed_reaches(spec, md.id, o) for o in outcome_ids)
            if not reaches_out:
                lines.append(
                    f"- ⚠️ `{md.variable_name}` does not reach any outcome as "
                    f"drawn — its outgoing edge is missing, so nothing it carries "
                    f"arrives anywhere."
                )
                continue
            drv = _fmt(drivers) if drivers else "_no driver wired in yet_"
            lines.append(
                f"- `{md.variable_name}` carries part of {drv}'s effect to "
                f"`{primary.variable_name}`. Because it sits ON the causal path, "
                f"it must NEVER be listed as a control — conditioning on it would "
                f"hide exactly the effect being measured. The channels' total "
                f"effect already includes what flows through it."
            )
            if not drivers:
                lines.append(
                    f"  - ⚠️ `{md.variable_name}` has no incoming driver edge "
                    f"(media or upstream mediator) — say which channels drive it "
                    f"(`mediator_inputs`), or it floats disconnected."
                )
        lines.append("")

    # ── controls, by their actual causal role ───────────────────────────────
    if controls:
        conf_lines: list[str] = []
        medconf_lines: list[str] = []
        driver_lines: list[str] = []
        ctrl_lines: list[str] = []
        for c in controls:
            role, related = roles[c.id]
            if role == "confounder":
                conf_lines.append(
                    f"- `{c.variable_name}` drives BOTH {_fmt(related)} and "
                    f"`{primary.variable_name}` — a back-door. Including it lets "
                    f"those channels shed `{c.variable_name}`'s credit; leaving it "
                    f"out hands its lift to whichever channel tracks it. Its own "
                    f"coefficient is associational, not a causal effect."
                )
            elif role == "mediation_confounder":
                medconf_lines.append(
                    f"- `{c.variable_name}` is a common cause of {_fmt(related)} "
                    f"and `{primary.variable_name}` — it confounds the mediation "
                    f"pathway. Keep it in the graph and controlled when splitting "
                    f"direct vs routed effects; its own coefficient is still "
                    f"associational."
                )
            elif role == "spend_driver":
                driver_lines.append(
                    f"- `{c.variable_name}` steers {_fmt(related)} but, as drawn, "
                    f"never touches `{primary.variable_name}` on its own — an "
                    f"instrument-like spend driver, NOT a confounder: it opens no "
                    f"back-door, and conditioning on it can amplify any remaining "
                    f"unobserved confounding rather than remove it. If you believe "
                    f"`{c.variable_name}` also moves `{primary.variable_name}`, "
                    f"add that edge."
                )
            elif role == "precision":
                ctrl_lines.append(
                    f"- `{c.variable_name}` affects only `{primary.variable_name}` "
                    f"in this graph: a precision control. It cleans the media "
                    f"read; its coefficient carries no causal claim and it is not "
                    f"a lever ('change {c.variable_name} → gain "
                    f"{primary.variable_name}' does not follow)."
                )
            else:
                ctrl_lines.append(
                    f"- ⚠️ `{c.variable_name}` touches nothing relevant as drawn — "
                    f"wire its edges or drop it."
                )
        if conf_lines:
            lines.append("*Confounders (common causes — the reason controls exist):*")
            lines.extend(conf_lines)
            lines.append("")
        if medconf_lines:
            lines.append("*Mediation-pathway confounders:*")
            lines.extend(medconf_lines)
            lines.append("")
        if driver_lines:
            lines.append("*Spend drivers (no back-door as drawn):*")
            lines.extend(driver_lines)
            lines.append("")
        if ctrl_lines:
            lines.append("*Plain controls (KPI-only):*")
            lines.extend(ctrl_lines)
            lines.append("")

    # ── instruments ─────────────────────────────────────────────────────────
    if instruments:
        lines.append("*Instruments (exogenous nudges):*")
        for z in instruments:
            treats = [c.variable_name for c in spec.get_children(z.id)]
            lines.append(
                f"- `{z.variable_name}` moves {_fmt(treats) or 'a treatment'} for "
                f"reasons unrelated to demand, and touches "
                f"`{primary.variable_name}` only through it — variation that "
                f"survives unobserved-demand confounding. Its value rests on the "
                f"exclusion restriction (no other route to the KPI), which data "
                f"cannot verify."
            )
        lines.append("")

    # ── cross effects ───────────────────────────────────────────────────────
    ce = [e for e in spec.edges if e.edge_type == EdgeType.CROSS_EFFECT]
    if ce:
        lines.append("*Cross-effects (halo / cannibalization):*")
        for e in ce:
            s = spec.get_node(e.source)
            t = spec.get_node(e.target)
            if s and t:
                lines.append(
                    f"- `{s.variable_name}` spills over into `{t.variable_name}` — "
                    f"one outcome feeding another, beyond each channel's own effect."
                )
        lines.append("")

    # ── what the missing arrows assume ──────────────────────────────────────
    media_names = _fmt([m.variable_name for m in media]) or "any spend"
    lines.append("*What the missing arrows assume (data cannot test these):*")
    lines.append(
        f"- No arrow runs from `{primary.variable_name}` back into spend: budgets "
        f"do NOT react to the KPI within the model's time step. If spend is "
        f"actually set by chasing results, that missing arrow is real confounding "
        f"this graph hides."
    )
    lines.append(
        f"- No unmeasured common cause of {media_names} and "
        f"`{primary.variable_name}` exists beyond what is drawn — the "
        f"no-unobserved-confounding assumption. Anything known to drive both "
        f"(demand, pricing power, distribution wins) belongs IN the graph, "
        f"measured or not."
    )
    lines.append("")

    # ── data checklist ──────────────────────────────────────────────────────
    if known_columns is not None:
        known = {_norm_col(c) for c in known_columns}
        measured: list[str] = []
        missing_backdoor: list[str] = []
        missing_med: list[str] = []
        missing_other: list[str] = []
        for n in spec.nodes:
            if _norm_col(n.variable_name) in known:
                measured.append(n.variable_name)
            elif n.node_type == NodeType.CONTROL and roles.get(n.id, ("", []))[0] in (
                "confounder",
                "mediation_confounder",
            ):
                missing_backdoor.append(n.variable_name)
            elif n.node_type == NodeType.MEDIATOR:
                missing_med.append(n.variable_name)
            else:
                missing_other.append(n.variable_name)
        lines.append("*Data this DAG needs:*")
        if measured:
            lines.append(f"- Measured ✓: {_fmt(sorted(measured))}")
        if missing_backdoor:
            lines.append(
                f"- ⚠️ Needs a series: {_fmt(sorted(missing_backdoor))} — each "
                f"needs a measured series covering the modeling window (a proxy "
                f"works: category index, promo calendar, distribution/ACV…). An "
                f"unmeasured confounder leaves its back-door OPEN no matter what "
                f"the model controls for."
            )
        if missing_med:
            lines.append(
                f"- ⚠️ Needs a series: {_fmt(sorted(missing_med))} — a mediator "
                f"must be observed to split routed vs direct effect (survey "
                f"tracker, branded-search index…). Without it the TOTAL media "
                f"effect is still estimable — just keep it out of the controls."
            )
        if missing_other:
            lines.append(
                f"- ⚠️ Not seen in the data: {_fmt(sorted(missing_other))} — each "
                f"needs a measured series to play its drawn role."
            )
    return "\n".join(lines).rstrip()
