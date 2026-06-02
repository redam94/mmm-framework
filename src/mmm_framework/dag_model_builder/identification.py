"""Causal identification for MMM DAGs.

Implements three standard graphical identification strategies:

- **Back-door** adjustment (:func:`identification_report`, :func:`classify_dag_roles`);
- **Front-door** identification through a declared mediator set
  (:func:`frontdoor_criterion`) — for the mediation models;
- **Instrumental-variable** identification (:func:`iv_criterion`) when an
  exogenous instrument is declared.

Full ID (Tian-Pearl) for arbitrary DAGs is out of scope.

References:
- Pearl (2009), Causality, §3.3 (Back-door and Front-door criteria)
- Shpitser & Pearl (2008), Identification of Conditional Interventional
  Distributions, §3 (the standard adjustment criterion that this matches)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dag_spec import DAGSpec, NodeType


@dataclass
class BackdoorPath:
    """A backdoor path from treatment X to outcome Y.

    `nodes` is the ordered list of node ids the path visits, starting at X and
    ending at Y. `edge_dirs[i]` is the direction of the i-th edge ('→' or '←'
    when read left-to-right along `nodes`).
    """

    nodes: list[str]
    edge_dirs: list[str]
    blocked_by: list[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return len(self.blocked_by) > 0

    def render(self) -> str:
        parts = [self.nodes[0]]
        for i, d in enumerate(self.edge_dirs):
            parts.append(" → " if d == "→" else " ← ")
            parts.append(self.nodes[i + 1])
        return "".join(parts)


@dataclass
class IdentificationReport:
    treatment: str
    outcome: str
    adjustment_set: list[str]  # node ids that close all backdoor paths
    backdoor_paths: list[BackdoorPath]  # all backdoor paths (blocked + open)
    open_paths_remaining: list[BackdoorPath]  # those still open after adjustment
    descendants_of_treatment: list[str]  # for diagnostics — must NOT be adjusted
    identifiable: bool
    notes: list[str]


# ── Graph helpers ────────────────────────────────────────────────────────────


def _ancestors(spec: DAGSpec, node_id: str) -> set[str]:
    """All ancestors (not including the node itself)."""
    parents = {e.source for e in spec.edges if e.target == node_id}
    out: set[str] = set()
    stack = list(parents)
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(e.source for e in spec.edges if e.target == n)
    return out


def _descendants(spec: DAGSpec, node_id: str) -> set[str]:
    """All descendants (not including the node itself)."""
    out: set[str] = set()
    stack = [e.target for e in spec.edges if e.source == node_id]
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(e.target for e in spec.edges if e.source == n)
    return out


def _all_simple_paths(
    spec: DAGSpec, start: str, end: str, max_len: int = 12
) -> list[list[tuple[str, str]]]:
    """All simple undirected paths from start to end, each represented as a
    list of (next_node, direction) tuples. direction is '→' if the edge goes
    from current→next, '←' if next→current.

    `max_len` caps depth to keep this tractable on rich DAGs.
    """
    # Build undirected adjacency with edge directions
    adj: dict[str, list[tuple[str, str]]] = {n.id: [] for n in spec.nodes}
    for e in spec.edges:
        adj[e.source].append((e.target, "→"))
        adj[e.target].append((e.source, "←"))

    results: list[list[tuple[str, str]]] = []

    def dfs(node: str, target: str, visited: set[str], path: list[tuple[str, str]]):
        if len(path) > max_len:
            return
        if node == target:
            results.append(list(path))
            return
        for nxt, d in adj.get(node, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append((nxt, d))
            dfs(nxt, target, visited, path)
            path.pop()
            visited.remove(nxt)

    dfs(start, end, {start}, [])
    return results


def _backdoor_paths_from_simple(
    simple_paths: list[list[tuple[str, str]]], start: str
) -> list[BackdoorPath]:
    """A simple path is a *backdoor* path iff its first edge points INTO start."""
    out: list[BackdoorPath] = []
    for p in simple_paths:
        if not p:
            continue
        first_node, first_dir = p[0]
        # first_dir is direction of edge start→first_node in original DAG;
        # backdoor means edge into `start`, i.e. first_dir == '←'.
        if first_dir != "←":
            continue
        nodes = [start] + [n for n, _ in p]
        dirs = [d for _, d in p]
        out.append(BackdoorPath(nodes=nodes, edge_dirs=dirs))
    return out


def _is_collider(path: BackdoorPath, i: int) -> bool:
    """Is the i-th internal node a collider on this path?
    A collider C on path A * C * B is one where both edges point INTO C:
    A → C ← B. Endpoints (i=0 or i=last) are not colliders.
    """
    if i <= 0 or i >= len(path.nodes) - 1:
        return False
    left = path.edge_dirs[i - 1]
    right = path.edge_dirs[i]
    return left == "→" and right == "←"


def _path_blocked_by(
    path: BackdoorPath, conditioning: set[str], spec: DAGSpec
) -> list[str]:
    """Return the conditioning-set nodes that block this path (empty list ⇒ open).

    A path is blocked by Z iff for some internal node N:
    - N is a chain or fork node on the path AND N ∈ Z, OR
    - N is a collider AND neither N nor any descendant of N is in Z
    """
    blockers: list[str] = []
    for i in range(1, len(path.nodes) - 1):
        n = path.nodes[i]
        if _is_collider(path, i):
            desc = _descendants(spec, n)
            if n not in conditioning and not (desc & conditioning):
                blockers.append(f"unconditioned collider {n}")
        else:
            if n in conditioning:
                blockers.append(n)
    return blockers


# ── Public API ───────────────────────────────────────────────────────────────


def find_backdoor_paths(
    spec: DAGSpec, treatment_id: str, outcome_id: str
) -> list[BackdoorPath]:
    """Enumerate all backdoor paths from treatment to outcome."""
    if not spec.get_node(treatment_id) or not spec.get_node(outcome_id):
        return []
    simple = _all_simple_paths(spec, treatment_id, outcome_id)
    return _backdoor_paths_from_simple(simple, treatment_id)


def propose_adjustment_set(
    spec: DAGSpec, treatment_id: str, outcome_id: str
) -> list[str]:
    """Heuristic adjustment set: parents of treatment that are ancestors of
    outcome, minus descendants of treatment. This is sufficient under the
    backdoor criterion for the common case where treatment has no descendant
    confounders; verify with `identification_report` afterward.
    """
    if not spec.get_node(treatment_id) or not spec.get_node(outcome_id):
        return []
    parents = {e.source for e in spec.edges if e.target == treatment_id}
    ancestors_y = _ancestors(spec, outcome_id)
    descendants_t = _descendants(spec, treatment_id) | {treatment_id}
    # Instruments must NEVER enter the adjustment set: conditioning on an
    # instrument blocks the exogenous variation and amplifies any unobserved
    # confounding (bias amplification), rather than removing bias.
    instrument_ids = {n.id for n in spec.instrument_nodes}
    candidate = (parents & ancestors_y) - descendants_t - instrument_ids
    return sorted(candidate)


def _directed_simple_paths(
    spec: DAGSpec, start: str, end: str, max_len: int = 12
) -> list[list[str]]:
    """All simple *directed* paths start → ... → end (following edges forward).

    ``max_len`` caps path depth to stay tractable on rich DAGs. This is a
    *correctness*-relevant bound for the front-door check: a directed treatment→
    outcome path longer than the cap would be missed. The default comfortably
    exceeds any realistic MMM mediation chain; raise it for pathologically deep
    graphs.
    """
    children: dict[str, list[str]] = {}
    for e in spec.edges:
        children.setdefault(e.source, []).append(e.target)

    results: list[list[str]] = []

    def dfs(node: str, visited: set[str], path: list[str]):
        if len(path) > max_len:
            return
        if node == end:
            results.append(list(path))
            return
        for nxt in children.get(node, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt, visited, path)
            path.pop()
            visited.remove(nxt)

    if not spec.get_node(start) or not spec.get_node(end):
        return []
    dfs(start, {start}, [start])
    return results


# ── Front-door identification ────────────────────────────────────────────────


@dataclass
class FrontdoorIdentificationReport:
    """Result of a front-door identification check for a declared mediator set."""

    treatment: str
    outcome: str
    mediators: list[str]
    intercepts_all_paths: bool  # M is on every directed T→Y path (no direct effect)
    treatment_mediator_unconfounded: bool  # no open back-door T→M
    mediator_outcome_blocked_by_treatment: bool  # back-door M→Y closed by {T}
    identifiable: bool
    notes: list[str]


def frontdoor_criterion(
    spec: DAGSpec, treatment_id: str, mediators: list[str], outcome_id: str
) -> FrontdoorIdentificationReport:
    """Check Pearl's front-door criterion for ``mediators`` between T and Y.

    A mediator set M identifies P(Y | do(T)) by the front-door formula iff:

    (a) **M intercepts every directed path T → Y** — in particular there is no
        unmediated direct effect (a bare ``T → Y`` edge breaks this);
    (b) **there is no unblocked back-door path from T to M** (T and M are not
        confounded); and
    (c) **every back-door path from M to Y is blocked by conditioning on T**.

    Reference: Pearl (2009), Causality, §3.3.2 (Front-Door Criterion).
    """
    notes: list[str] = []
    if not spec.get_node(treatment_id) or not spec.get_node(outcome_id):
        return FrontdoorIdentificationReport(
            treatment=treatment_id,
            outcome=outcome_id,
            mediators=list(mediators),
            intercepts_all_paths=False,
            treatment_mediator_unconfounded=False,
            mediator_outcome_blocked_by_treatment=False,
            identifiable=False,
            notes=["Treatment or outcome node not found in DAG."],
        )
    med_set = set(mediators)

    # (a) Every directed T→Y path must pass through a declared mediator.
    directed = _directed_simple_paths(spec, treatment_id, outcome_id)
    if not directed:
        intercepts = False
        notes.append(
            f"No directed path from '{treatment_id}' to '{outcome_id}'; there is "
            "no effect to identify via mediation."
        )
    else:
        uncovered = [p for p in directed if not (set(p[1:-1]) & med_set)]
        intercepts = not uncovered
        if uncovered:
            notes.append(
                "A directed path bypasses the declared mediators "
                f"({' → '.join(uncovered[0])}); the front door is incomplete "
                "(e.g. a direct treatment→outcome effect)."
            )

    # (b) No OPEN back-door path from T to any mediator (T and M unconfounded).
    tm_unconfounded = True
    for m in mediators:
        for p in find_backdoor_paths(spec, treatment_id, m):
            if not _path_blocked_by(p, set(), spec):
                tm_unconfounded = False
                notes.append(
                    f"Open back-door path between treatment and mediator '{m}' "
                    f"({p.render()}): treatment and mediator are confounded."
                )
                break

    # (c) Every back-door path from each mediator to Y is blocked by {T}.
    my_blocked = True
    for m in mediators:
        for p in find_backdoor_paths(spec, m, outcome_id):
            if not _path_blocked_by(p, {treatment_id}, spec):
                my_blocked = False
                notes.append(
                    f"Back-door path from mediator '{m}' to the outcome is not "
                    f"closed by conditioning on the treatment ({p.render()})."
                )
                break

    identifiable = intercepts and tm_unconfounded and my_blocked
    if identifiable:
        notes.append(
            "Front-door criterion satisfied: the effect is identified through "
            f"the mediator(s) {sorted(med_set)} via the front-door formula."
        )
    return FrontdoorIdentificationReport(
        treatment=treatment_id,
        outcome=outcome_id,
        mediators=list(mediators),
        intercepts_all_paths=intercepts,
        treatment_mediator_unconfounded=tm_unconfounded,
        mediator_outcome_blocked_by_treatment=my_blocked,
        identifiable=identifiable,
        notes=notes,
    )


# ── Instrumental-variable identification ─────────────────────────────────────


@dataclass
class IVIdentificationReport:
    """Result of an instrumental-variable identification check."""

    instrument: str
    treatment: str
    outcome: str
    is_relevant: bool  # Z influences T (directed Z → ... → T)
    is_exogenous: bool  # no open back-door path Z → Y (Z unconfounded with Y)
    satisfies_exclusion: bool  # every directed Z → Y path passes through T
    identifiable: bool
    weak_instrument_warning: str | None
    notes: list[str]


def iv_criterion(
    spec: DAGSpec, instrument_id: str, treatment_id: str, outcome_id: str
) -> IVIdentificationReport:
    """Check the graphical conditions for ``instrument_id`` to be a valid IV.

    A variable Z is a valid instrument for the effect of T on Y iff:

    - **Relevance:** Z influences T (a directed path Z → … → T exists);
    - **Exogeneity:** Z is unconfounded with Y (no open back-door path Z → Y);
    - **Exclusion:** Z affects Y only through T (every directed Z → Y path
      passes through T).

    The graph can verify *existence* but not *strength*: a relevant-but-weak
    instrument still yields unreliable estimates, so a warning is attached.
    """
    notes: list[str] = []
    for nid in (instrument_id, treatment_id, outcome_id):
        if not spec.get_node(nid):
            return IVIdentificationReport(
                instrument=instrument_id,
                treatment=treatment_id,
                outcome=outcome_id,
                is_relevant=False,
                is_exogenous=False,
                satisfies_exclusion=False,
                identifiable=False,
                weak_instrument_warning=None,
                notes=[f"Node '{nid}' not found in DAG."],
            )

    # Relevance: a directed path Z → … → T.
    is_relevant = treatment_id in _descendants(spec, instrument_id)
    if not is_relevant:
        notes.append(
            f"Instrument '{instrument_id}' has no directed path to treatment "
            f"'{treatment_id}' — it is irrelevant."
        )

    # Exclusion: every directed Z → Y path passes through T.
    z_to_y = _directed_simple_paths(spec, instrument_id, outcome_id)
    bypassing = [p for p in z_to_y if treatment_id not in p]
    satisfies_exclusion = not bypassing
    if bypassing:
        notes.append(
            "Exclusion restriction violated: a directed path from the instrument "
            f"reaches the outcome without passing through the treatment "
            f"({' → '.join(bypassing[0])})."
        )

    # Exogeneity: no OPEN back-door path Z → Y (Z not confounded with Y).
    is_exogenous = True
    for p in find_backdoor_paths(spec, instrument_id, outcome_id):
        if not _path_blocked_by(p, set(), spec):
            is_exogenous = False
            notes.append(
                f"Instrument is confounded with the outcome ({p.render()}); it is "
                "not exogenous."
            )
            break

    identifiable = is_relevant and is_exogenous and satisfies_exclusion
    weak_instrument_warning = None
    if identifiable:
        notes.append(
            f"'{instrument_id}' is a valid instrument for '{treatment_id}' → "
            f"'{outcome_id}' (relevant, exogenous, exclusion holds)."
        )
        weak_instrument_warning = (
            "The graph confirms a valid instrument but cannot assess its "
            "strength. A weak instrument (low Z→T association in the data) "
            "produces biased, high-variance estimates; check first-stage strength."
        )
    return IVIdentificationReport(
        instrument=instrument_id,
        treatment=treatment_id,
        outcome=outcome_id,
        is_relevant=is_relevant,
        is_exogenous=is_exogenous,
        satisfies_exclusion=satisfies_exclusion,
        identifiable=identifiable,
        weak_instrument_warning=weak_instrument_warning,
        notes=notes,
    )


def identification_report(
    spec: DAGSpec,
    treatment_id: str,
    outcome_id: str,
    adjustment_set: list[str] | None = None,
) -> IdentificationReport:
    """Full backdoor identification report. If `adjustment_set` is None, we
    propose one via `propose_adjustment_set` and check it.
    """
    notes: list[str] = []
    if not spec.get_node(treatment_id):
        return IdentificationReport(
            treatment=treatment_id,
            outcome=outcome_id,
            adjustment_set=[],
            backdoor_paths=[],
            open_paths_remaining=[],
            descendants_of_treatment=[],
            identifiable=False,
            notes=[f"Treatment node '{treatment_id}' not found in DAG"],
        )
    if not spec.get_node(outcome_id):
        return IdentificationReport(
            treatment=treatment_id,
            outcome=outcome_id,
            adjustment_set=[],
            backdoor_paths=[],
            open_paths_remaining=[],
            descendants_of_treatment=[],
            identifiable=False,
            notes=[f"Outcome node '{outcome_id}' not found in DAG"],
        )

    descendants_t = sorted(_descendants(spec, treatment_id))
    if adjustment_set is None:
        adjustment_set = propose_adjustment_set(spec, treatment_id, outcome_id)

    # Hard rule: descendants of treatment cannot be in adjustment set.
    bad = [n for n in adjustment_set if n in descendants_t]
    if bad:
        notes.append(
            f"Removed descendants of {treatment_id} from adjustment set "
            f"(can introduce bias): {bad}"
        )
        adjustment_set = [n for n in adjustment_set if n not in bad]

    paths = find_backdoor_paths(spec, treatment_id, outcome_id)
    z = set(adjustment_set)
    annotated: list[BackdoorPath] = []
    open_remaining: list[BackdoorPath] = []
    for p in paths:
        blockers = _path_blocked_by(p, z, spec)
        p.blocked_by = blockers
        annotated.append(p)
        if not blockers:
            open_remaining.append(p)

    identifiable = len(open_remaining) == 0
    if not identifiable:
        notes.append(
            f"{len(open_remaining)} backdoor path(s) remain open after "
            f"conditioning on {sorted(z) or '∅'}. Effect is NOT identified by "
            "this adjustment set."
        )
    elif not paths:
        notes.append(
            "No backdoor paths exist; the total effect is identified without adjustment."
        )
    else:
        notes.append(
            f"All {len(paths)} backdoor path(s) are blocked by "
            f"{sorted(z) or '∅'}. Effect is identified by backdoor adjustment."
        )

    # Useful diagnostic: warn about MEDIATOR nodes in the adjustment set.
    for nid in adjustment_set:
        node = spec.get_node(nid)
        if node and node.node_type == NodeType.MEDIATOR:
            notes.append(
                f"Adjusting for mediator '{nid}' will block part of the "
                "treatment's effect — likely not what you want for the total effect."
            )

    return IdentificationReport(
        treatment=treatment_id,
        outcome=outcome_id,
        adjustment_set=sorted(adjustment_set),
        backdoor_paths=annotated,
        open_paths_remaining=open_remaining,
        descendants_of_treatment=descendants_t,
        identifiable=identifiable,
        notes=notes,
    )


@dataclass
class DagRoleClassification:
    """Causal-role classification of every node, plus identification status.

    Produced by :func:`classify_dag_roles` to wire the backdoor identification
    result into model fitting (bad-control prevention). Role strings match the
    values of :class:`mmm_framework.config.enums.CausalControlRole` so the config
    layer can map them without importing this module.
    """

    confounders: set[str]  # node ids in the adjustment set(s)
    post_treatment: dict[str, str]  # node id -> a treatment it descends from
    colliders: dict[str, str]  # collider node id -> a treatment whose path it opens
    adjustment_set: list[str]  # union of adjustment sets across treatments
    identifiable: bool  # every (treatment, outcome) pair identified
    notes: list[str]  # human-readable identification notes

    def role_for(self, node_id: str) -> tuple[str, str | None]:
        """Return ``(role, reason)`` for ``node_id``, with a plain-language reason.

        Precedence is safety-first, so the guidance always names the most
        damaging problem first:

        1. **mediator** -- a consequence of the treatment (post-treatment). Always
           refused: conditioning on it removes part of the effect being measured.
        2. **collider** -- conditioning on it *opens* a spurious path that is
           otherwise closed. Refused.
        3. **confounder** -- a common cause that must be kept and not shrunk.
        4. **precision_control** -- a safe predictor of the KPI only.
        """
        if node_id in self.post_treatment:
            t = self.post_treatment[node_id]
            return (
                "mediator",
                f"it is a consequence of media '{t}' (post-treatment): "
                "conditioning on it would absorb part of that channel's effect",
            )
        if node_id in self.colliders:
            t = self.colliders[node_id]
            return (
                "collider",
                f"it is a common effect on a back-door path for media '{t}': "
                "conditioning on it opens a spurious association",
            )
        if node_id in self.confounders:
            return (
                "confounder",
                "it is a common cause of media and the KPI (back-door adjustment "
                "set): it is kept and given a wide, un-shrunk prior",
            )
        return (
            "precision_control",
            "it predicts the KPI but is not a common cause, mediator, or collider "
            "-- safe to include for efficiency",
        )


def classify_dag_roles(
    spec: DAGSpec,
    treatment_ids: list[str],
    outcome_id: str,
    control_ids: list[str] | None = None,
) -> DagRoleClassification:
    """Classify each node's causal role w.r.t. a set of treatments and an outcome.

    Runs the backdoor :func:`identification_report` for every treatment against
    the outcome and aggregates the result so non-experts get all three roles
    detected automatically:

    - a node in any treatment's adjustment set is a **confounder** (kept, not
      shrunk);
    - a descendant of any treatment is **post-treatment** (mediator / over-control)
      and must not be conditioned on. Because every media channel is an ancestor
      of the KPI, this also subsumes descendants of the *outcome*;
    - a node that is a **collider** whose inclusion opens a back-door path is
      refused (see below);
    - everything else is a safe **precision control**.

    Collider detection is done against the *actual* conditioning set the model
    uses -- ``control_ids`` (all controls) -- not the proposed adjustment set.
    A control is flagged as a collider only when it sits as a collider on a
    back-door path that is **open** given the full control set, i.e. conditioning
    on it genuinely opens a spurious path. A collider on a path already blocked by
    another control (e.g. a confounder) is *safe* and is **not** flagged -- this
    is what makes the detection free of the false positives that pure
    path-enumeration would produce (it also handles M-bias correctly: a collider
    is only dangerous when its blocking forks are not also conditioned on).

    Reachability (post-treatment) is exact and the collider check is evaluated
    against the real conditioning set, so both are enforced as hard refusals. The
    adjustment set is a heuristic, so a confounder role only *widens* a prior.
    Assumes a single primary outcome; multiple-KPI DAGs are classified against
    the supplied ``outcome_id``.
    """
    confounders: set[str] = set()
    post_treatment: dict[str, str] = {}
    colliders: dict[str, str] = {}
    adjustment_union: set[str] = set()
    identifiable = True
    notes: list[str] = []

    # The model conditions on every control, so that is the conditioning set we
    # must use when asking "does including this control open a back-door path?".
    conditioning = set(control_ids or [])

    for t in treatment_ids:
        rep = identification_report(spec, t, outcome_id)
        confounders |= set(rep.adjustment_set)
        adjustment_union |= set(rep.adjustment_set)
        for d in rep.descendants_of_treatment:
            post_treatment.setdefault(d, t)
        if not rep.identifiable:
            identifiable = False
        notes.extend(rep.notes)

        # Collider detection: a back-door path that is OPEN under the full control
        # set is open partly because a conditioned collider on it was activated.
        # Flag the controls responsible (the collider itself, or a control that is
        # a descendant of the collider and so activates it). Blocked paths are
        # skipped, so a collider on a path closed by a confounder is never flagged.
        for path in find_backdoor_paths(spec, t, outcome_id):
            if _path_blocked_by(path, conditioning, spec):
                continue
            for i in range(1, len(path.nodes) - 1):
                if not _is_collider(path, i):
                    continue
                c = path.nodes[i]
                if c in conditioning:
                    colliders.setdefault(c, t)
                else:
                    # A conditioned descendant of the collider activated the path.
                    for d in _descendants(spec, c) & conditioning:
                        colliders.setdefault(d, t)

    # A control can be both a confounder (blocks one path) and a collider (opens
    # another); the collider role takes precedence in ``role_for`` because opening
    # a spurious path is the more damaging error.
    return DagRoleClassification(
        confounders=confounders,
        post_treatment=post_treatment,
        colliders=colliders,
        adjustment_set=sorted(adjustment_union),
        identifiable=identifiable,
        notes=notes,
    )


def report_to_dict(rep: IdentificationReport) -> dict:
    """JSON-safe dict for transport to the frontend / agent tool output."""
    return {
        "treatment": rep.treatment,
        "outcome": rep.outcome,
        "adjustment_set": rep.adjustment_set,
        "identifiable": rep.identifiable,
        "notes": rep.notes,
        "descendants_of_treatment": rep.descendants_of_treatment,
        "backdoor_paths": [
            {"path": p.render(), "blocked_by": p.blocked_by, "nodes": p.nodes}
            for p in rep.backdoor_paths
        ],
        "open_paths_remaining": [
            {"path": p.render(), "nodes": p.nodes} for p in rep.open_paths_remaining
        ],
    }
