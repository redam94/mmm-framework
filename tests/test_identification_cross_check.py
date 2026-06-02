"""Cross-check the hand-rolled d-separation engine against networkx (P2-6).

The back-door / front-door / IV criteria all rest on two primitives:
``_all_simple_paths`` and ``_path_blocked_by`` (collider/chain/fork blocking).
Here we derive d-separation from those primitives and assert it agrees with
``networkx.is_d_separator`` for EVERY conditioning subset on a battery of DAGs
(chains, forks, colliders, M-bias, the front-door and IV graphs). networkx is a
dev-only dependency; the module skips cleanly where it is absent.
"""

from __future__ import annotations

from itertools import chain, combinations

import pytest

from mmm_framework.dag_model_builder.dag_spec import DAGEdge, DAGNode, DAGSpec, NodeType
from mmm_framework.dag_model_builder.identification import (
    BackdoorPath,
    _all_simple_paths,
    _directed_simple_paths,
    _path_blocked_by,
)

# Dev-only cross-check dependency; skip the whole module where networkx is absent.
nx = pytest.importorskip("networkx")


def _spec(node_ids, edges) -> DAGSpec:
    return DAGSpec(
        nodes=[
            DAGNode(id=i, variable_name=i.upper(), node_type=NodeType.CONTROL)
            for i in node_ids
        ],
        edges=[DAGEdge(source=s, target=t) for s, t in edges],
    )


def _spec_to_nx(spec: DAGSpec):
    g = nx.DiGraph()
    g.add_nodes_from(n.id for n in spec.nodes)
    g.add_edges_from((e.source, e.target) for e in spec.edges)
    return g


def _our_d_separated(spec: DAGSpec, x: str, y: str, z: set[str]) -> bool:
    """d-separation derived purely from our path-blocking primitive."""
    for p in _all_simple_paths(spec, x, y):
        nodes = [x] + [n for n, _ in p]
        dirs = [d for _, d in p]
        if not _path_blocked_by(BackdoorPath(nodes=nodes, edge_dirs=dirs), z, spec):
            return False  # an open path => d-connected
    return True


def _powerset(items):
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))


# A battery of small DAGs spanning every blocking structure that matters.
DAGS = {
    "chain": (["a", "b", "c"], [("a", "b"), ("b", "c")]),
    "fork": (["a", "b", "c"], [("b", "a"), ("b", "c")]),
    "collider": (["a", "b", "c"], [("a", "c"), ("b", "c")]),
    "collider_with_descendant": (
        ["a", "b", "c", "d"],
        [("a", "c"), ("b", "c"), ("c", "d")],
    ),
    "m_bias": (
        ["t", "y", "a", "b", "m"],
        [("a", "t"), ("a", "m"), ("b", "m"), ("b", "y"), ("t", "y")],
    ),
    "frontdoor": (
        ["t", "m", "y", "u"],
        [("t", "m"), ("m", "y"), ("u", "t"), ("u", "y")],
    ),
    "iv": (["z", "t", "y", "u"], [("z", "t"), ("t", "y"), ("u", "t"), ("u", "y")]),
    "diamond": (
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
    ),
}


@pytest.mark.parametrize("name", list(DAGS))
def test_d_separation_matches_networkx_for_all_conditioning_sets(name):
    node_ids, edges = DAGS[name]
    spec = _spec(node_ids, edges)
    g = _spec_to_nx(spec)
    assert nx.is_directed_acyclic_graph(g)

    pairs = [(x, y) for x in node_ids for y in node_ids if x < y]
    for x, y in pairs:
        others = [n for n in node_ids if n not in (x, y)]
        for z_tuple in _powerset(others):
            z = set(z_tuple)
            ours = _our_d_separated(spec, x, y, z)
            theirs = nx.is_d_separator(g, {x}, {y}, z)
            assert ours == theirs, (
                f"[{name}] d-sep disagreement for {x}⊥{y} | {z}: "
                f"ours={ours}, networkx={theirs}"
            )


@pytest.mark.parametrize("name", list(DAGS))
def test_directed_simple_paths_match_networkx(name):
    # _directed_simple_paths underpins the front-door "intercepts all directed
    # paths" and IV "every directed Z->Y path passes through T" checks; cross-
    # check it against networkx.all_simple_paths on the directed graph.
    node_ids, edges = DAGS[name]
    spec = _spec(node_ids, edges)
    g = _spec_to_nx(spec)
    for x in node_ids:
        for y in node_ids:
            if x == y:
                continue
            ours = {tuple(p) for p in _directed_simple_paths(spec, x, y)}
            theirs = {tuple(p) for p in nx.all_simple_paths(g, x, y)}
            assert ours == theirs, f"[{name}] directed paths {x}->{y} disagree"
