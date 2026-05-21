"""Causal identification: backdoor-criterion adjustment-set finder.

Limited to the backdoor case. Frontdoor and full ID (Tian-Pearl) are not
implemented — those are research-quality and out of scope for the agent.

References:
- Pearl (2009), Causality, §3.3 (Backdoor criterion)
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
    adjustment_set: list[str]              # node ids that close all backdoor paths
    backdoor_paths: list[BackdoorPath]     # all backdoor paths (blocked + open)
    open_paths_remaining: list[BackdoorPath]  # those still open after adjustment
    descendants_of_treatment: list[str]    # for diagnostics — must NOT be adjusted
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


def _all_simple_paths(spec: DAGSpec, start: str, end: str, max_len: int = 12) -> list[list[tuple[str, str]]]:
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


def _backdoor_paths_from_simple(simple_paths: list[list[tuple[str, str]]], start: str) -> list[BackdoorPath]:
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


def _path_blocked_by(path: BackdoorPath, conditioning: set[str], spec: DAGSpec) -> list[str]:
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

def find_backdoor_paths(spec: DAGSpec, treatment_id: str, outcome_id: str) -> list[BackdoorPath]:
    """Enumerate all backdoor paths from treatment to outcome."""
    if not spec.get_node(treatment_id) or not spec.get_node(outcome_id):
        return []
    simple = _all_simple_paths(spec, treatment_id, outcome_id)
    return _backdoor_paths_from_simple(simple, treatment_id)


def propose_adjustment_set(spec: DAGSpec, treatment_id: str, outcome_id: str) -> list[str]:
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
    candidate = (parents & ancestors_y) - descendants_t
    return sorted(candidate)


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
            treatment=treatment_id, outcome=outcome_id, adjustment_set=[], backdoor_paths=[],
            open_paths_remaining=[], descendants_of_treatment=[], identifiable=False,
            notes=[f"Treatment node '{treatment_id}' not found in DAG"],
        )
    if not spec.get_node(outcome_id):
        return IdentificationReport(
            treatment=treatment_id, outcome=outcome_id, adjustment_set=[], backdoor_paths=[],
            open_paths_remaining=[], descendants_of_treatment=[], identifiable=False,
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
        notes.append("No backdoor paths exist; the total effect is identified without adjustment.")
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
