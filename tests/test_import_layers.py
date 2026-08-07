"""Import-layer gate: the package graph may not grow new upward edges
(src-refactor PR 0.3).

The module-level package graph has exactly one cycle ({agents, platform},
closed by the single module-level import PR 10 removes). Everything else that
looks like a cycle is produced by DEFERRED imports — 600+ function-local
imports of the package's own modules — and this gate is the ratchet that stops
that number growing back: today's violations are allowlisted in
``tests/contracts/import_layer_allowlist.json`` and the allowlist may only
shrink.

What the gate counts — this is the whole design:

* every ``import``/``from … import`` resolving to a sibling top-level package
  or root module, **including function-local imports** (a module-level-only
  gate is defeated by ``def f(): from ..planning import x``, which is
  precisely the habit being capped);
* ``if TYPE_CHECKING:`` bodies are EXCLUDED — they do not execute, and
  including them manufactures a phantom
  ``{model, calibration, validation, frequentist}`` cycle;
* dynamic imports are invisible to AST and OUT OF SCOPE — the
  ``importlib.import_module`` call in ``agents/tools.py`` and the PEP-562
  ``__getattr__`` façades in ``agents``/``estimands``/``garden`` are edges
  this gate cannot see; do not trust it as complete;
* the unit is the **import occurrence**, aggregated per ``(edge, file)`` —
  counting names, statements or files gives answers that differ by up to 3×,
  and a ratchet on an undefined unit is decorative;
* an unclassified package is a **hard failure** naming the module — a silent
  default tier makes the gate useless the first time someone adds a package;
* ``__init__.py`` files are exempt as SOURCES (a façade legally re-exports),
  never as targets — an edge to a package counts however it is spelled.

Rule: ``tier(dst) <= tier(src)``. Sideways legal, upward not. A strict ``<``
would need ~8 more tiers and would ratchet on the legitimate
``model ⇄ estimands/diagnostics/calibration/mmm_extensions`` peer cluster.

Regenerate the allowlist (deliberately, reviewing the diff):
``MMM_REGEN_IMPORT_LAYERS=1 uv run pytest tests/test_import_layers.py``.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "mmm_framework"
ALLOWLIST_PATH = ROOT / "tests" / "contracts" / "import_layer_allowlist.json"
REGEN = os.environ.get("MMM_REGEN_IMPORT_LAYERS") == "1"

#: Tier map. Every entry carries the reason it sits where it sits; the
#: anti-rubber-stamp check below refuses empty reasons and non-reasons
#: ("TODO", "later", "n/a") — inherited from the #228 gates.
LAYERS: dict[str, tuple[int, str]] = {
    # ── tier 0: leaf utilities — import nothing of ours ──────────────────
    "config": (
        0,
        "pure Pydantic configs + enums; the vocabulary everything else reads",
    ),
    "utils": (0, "arviz shims, standardization, statistics; no package deps"),
    "transforms": (0, "numpy/pytensor kernels (adstock/saturation/seasonality/trend)"),
    "integrations": (0, "ad-platform/BigQuery connectors; zero outgoing package edges"),
    "security": (
        0,
        "fully isolated - no package edges in or out; placement unconstrained by evidence",
    ),
    "storage": (0, "object-store abstraction; zero outgoing package edges"),
    "data_loader.py": (0, "MFF loading; consumes config only"),
    "dataset.py": (0, "native dataset container; consumes config only"),
    # ── tier 1: the model layer and its immediate periphery ──────────────
    "model": (1, "BayesianMMM + results; sits on config/transforms/utils"),
    "estimands": (1, "declarative estimand registry; peer of model (legal sideways)"),
    "diagnostics": (1, "SBC/coverage/identification; peer of model"),
    "calibration": (1, "experiment likelihoods; peer of model"),
    "frequentist": (1, "ridge/cvxpy estimation; peer of model"),
    "mmm_extensions": (1, "Nested/MV/Combined models; peer of model"),
    "dag_model_builder": (
        1,
        "DAG -> model configuration; builds on config + model vocab",
    ),
    "datasets": (1, "bundled example data + loaders"),
    "synth": (1, "synthetic DGP worlds; consumes config + data_loader"),
    "builders": (1, "fluent config builders; consumes config + model vocab"),
    "dataset_loader.py": (1, "native dataset loading on top of dataset.py"),
    "data_preparation.py": (1, "scaling; consumes config + data_loader"),
    # ── tier 2: analysis/planning/reporting — consumers of fitted models ─
    "validation": (2, "backtest/spec-curve/refutation run fits and read reports"),
    "planning": (2, "optimizer/forecast/variance consume fitted models + finance"),
    "reporting": (2, "report generation over fitted models"),
    "eda": (2, "pre-fit data quality; reads loaders + configs"),
    "garden": (2, "Model Garden contract + compat suite over model families"),
    "finance": (2, "valuation/bridge-line vocabulary used by planning + reporting"),
    "ltv": (2, "LTV preprocessing + likelihood; consumes model layer"),
    "estimators": (
        2,
        "fully isolated - no package edges in or out; placement unconstrained by evidence",
    ),
    "excel_config": (2, "Excel config parsing/generation over config + builders"),
    "continuous_learning": (2, "geo bandit; consumes synth + planning vocab"),
    "platform": (
        2,
        "the persistence layer; the agent tools are its heaviest CLIENT - placing the store above its callers inverts reality",
    ),
    "analysis.py": (2, "counterfactual/marginal analysis over fitted models"),
    "serialization.py": (2, "save/load over model families"),
    "lineage.py": (2, "provenance records for pipelines + reports"),
    # ── tier 3: the app/orchestration layer ──────────────────────────────
    "agents": (3, "the LangGraph oracle + service modules; top-level orchestration"),
    "data_studio": (3, "upload->EDA->clean pipeline; orchestrates eda + loaders"),
    "jobs.py": (3, "async fit jobs; orchestrates model + serialization"),
    # ── tier 4: auth wraps everything ────────────────────────────────────
    "auth": (4, "org/user auth core; nothing of ours may depend on it"),
}

_BANNED_REASON = re.compile(r"\b(todo|later|tbd|n/?a|fix ?me)\b", re.I)

#: Not packages: data directories and caches that live under src/.
_SKIP_DIRS = {"api", "__pycache__"}


# ── the walker ───────────────────────────────────────────────────────────────


def _unit_of(path: Path) -> str:
    """Map a file to its LAYERS unit: package name or root-module filename."""
    rel = path.relative_to(SRC)
    if len(rel.parts) == 1:
        return rel.name  # root module, keyed by filename
    return rel.parts[0]


class _ImportCollector(ast.NodeVisitor):
    """Collect sibling-package import targets, skipping TYPE_CHECKING bodies."""

    def __init__(self, own_module_parts: list[str]) -> None:
        self.own = own_module_parts  # e.g. ["mmm_framework", "transforms", "carryover"]
        self.targets: list[str] = []  # dotted module paths under mmm_framework

    @staticmethod
    def _is_type_checking(test: ast.expr) -> bool:
        return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )

    def visit_If(self, node: ast.If) -> None:
        if self._is_type_checking(node.test):
            for orelse in node.orelse:
                self.visit(orelse)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "mmm_framework" or alias.name.startswith("mmm_framework."):
                self.targets.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0:
            mod = node.module or ""
            if mod == "mmm_framework" or mod.startswith("mmm_framework."):
                if mod == "mmm_framework":
                    # `from mmm_framework import agents, planning` — each name
                    # is its own package edge.
                    for alias in node.names:
                        self.targets.append(f"mmm_framework.{alias.name}")
                else:
                    self.targets.append(mod)
            return
        # Relative: resolve against this module's package.
        base = self.own[: len(self.own) - node.level]
        if not base or base[0] != "mmm_framework":
            return
        if node.module:
            self.targets.append(".".join(base + node.module.split(".")))
        else:
            # `from .. import planning` — names are the targets.
            for alias in node.names:
                self.targets.append(".".join(base + [alias.name]))


def collect_edges() -> tuple[dict[str, dict[str, int]], list[str]]:
    """``{"src -> dst": {src_file: occurrences}}`` for cross-unit edges, plus
    every unclassified unit encountered (a hard failure for the caller)."""
    edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unclassified: set[str] = set()

    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if rel.parts[0] in _SKIP_DIRS:
            continue
        src_unit = _unit_of(path)
        if src_unit == "__init__.py":
            continue  # the top-level façade
        if src_unit not in LAYERS:
            unclassified.add(src_unit)
            continue
        if path.name == "__init__.py":
            continue  # façades are exempt as SOURCES

        module_parts = ["mmm_framework"] + list(rel.with_suffix("").parts)
        collector = _ImportCollector(module_parts)
        collector.visit(ast.parse(path.read_text(encoding="utf-8")))

        for dotted in collector.targets:
            parts = dotted.split(".")
            if len(parts) < 2:
                continue
            head = parts[1]
            dst_unit = f"{head}.py" if (SRC / f"{head}.py").exists() else head
            if dst_unit == src_unit:
                continue
            if dst_unit not in LAYERS:
                if (SRC / head).is_dir() or (SRC / f"{head}.py").exists():
                    unclassified.add(dst_unit)
                continue  # a symbol re-exported from the root façade
            src_file = str(path.relative_to(ROOT))
            edges[f"{src_unit} -> {dst_unit}"][src_file] += 1

    return {k: dict(v) for k, v in edges.items()}, sorted(unclassified)


def violations() -> dict[str, dict[str, int]]:
    """Upward edges only: ``tier(dst) > tier(src)``."""
    edges, unclassified = collect_edges()
    assert not unclassified, (
        f"unclassified units {unclassified}: every package/root module under "
        "src/mmm_framework must have a LAYERS entry (a silent default tier "
        "makes this gate useless the first time someone adds a package)"
    )
    out: dict[str, dict[str, int]] = {}
    for edge, files in sorted(edges.items()):
        src_unit, dst_unit = edge.split(" -> ")
        if LAYERS[dst_unit][0] > LAYERS[src_unit][0]:
            out[edge] = dict(sorted(files.items()))
    return out


# ── the gate ─────────────────────────────────────────────────────────────────


def test_layer_reasons_are_substantive():
    problems = [
        name
        for name, (_tier, reason) in LAYERS.items()
        if not str(reason).strip() or _BANNED_REASON.search(str(reason))
    ]
    assert not problems, f"LAYERS entries with empty or non-reasons: {problems}"


def test_no_new_upward_imports():
    """The ratchet. Every upward edge must be inside the allowlist's per-file
    ceiling; an allowlist entry whose violation vanished must be deleted."""
    found = violations()

    if REGEN or not ALLOWLIST_PATH.exists():
        ALLOWLIST_PATH.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n")
        if REGEN:
            import pytest

            pytest.skip("allowlist regenerated; review the diff and rerun")

    allow = json.loads(ALLOWLIST_PATH.read_text())

    def _remedies(edge: str) -> str:
        src_unit, dst_unit = edge.split(" -> ")
        return (
            f"{edge} reaches tier {LAYERS[dst_unit][0]} from tier "
            f"{LAYERS[src_unit][0]}. Three legal remedies: move the symbol "
            f"down (does it belong in {dst_unit}?), invert the dependency "
            "with a Protocol, or amend LAYERS with a reviewed comment."
        )

    problems: list[str] = []
    for edge, files in sorted(found.items()):
        allowed = allow.get(edge, {})
        for f, n in sorted(files.items()):
            cap = allowed.get(f, 0)
            if n > cap:
                problems.append(
                    f"{edge} [{f}]: {n} occurrence(s), allowlisted {cap}. "
                    + _remedies(edge)
                )
    # The shrink-only half: stale allowlist entries must go.
    for edge, files in sorted(allow.items()):
        live = found.get(edge, {})
        for f, cap in sorted(files.items()):
            if live.get(f, 0) == 0:
                problems.append(
                    f"STALE allowlist entry {edge} [{f}]: the violation is "
                    "gone — delete the entry so the ratchet tightens "
                    "(MMM_REGEN_IMPORT_LAYERS=1 regenerates)"
                )
            elif live[f] < cap:
                problems.append(
                    f"LOOSE allowlist entry {edge} [{f}]: caps {cap} but only "
                    f"{live[f]} remain — lower it (regen tightens ceilings)"
                )

    assert not problems, (
        "import-layer violations (unit: import occurrences, incl. "
        "function-local; TYPE_CHECKING excluded):\n  " + "\n  ".join(problems)
    )


def test_the_one_module_level_cycle_is_the_known_one():
    """Documentation-as-test: the {agents, platform} cycle is closed by ONE
    module-level import (platform/runs.py -> agents.spec_locks). PR 10 removes
    it; when that lands, this test flips to asserting the graph is acyclic at
    module level and the allowlist entry disappears."""
    runs = (SRC / "platform" / "runs.py").read_text(encoding="utf-8")
    assert "from mmm_framework.agents.spec_locks import" in runs, (
        "the known platform -> agents module-level import is gone: PR 10 has "
        "landed — update this test to assert acyclicity and delete the "
        "allowlist entry"
    )
