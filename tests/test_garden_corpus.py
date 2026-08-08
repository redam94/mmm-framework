"""Garden corpus contract (src-refactor PR 0.4, second half).

The registry's ``garden_models.source_path`` rows live in a gitignored SQLite
file and mostly point at deleted workspaces — at export time only **4 of 19**
rows still resolved on disk (the spec measured 6 at v1.3.3; the rot continued
while nothing gated it). Without a checked-in corpus, "we preserved the
private authoring surface" is an assertion about one developer's SQLite file.

The corpus is therefore code in this repo:

* ``examples/garden_models/*.py`` — the maintained exemplars;
* ``tests/fixtures/garden_corpus/*.py`` — the recoverable registry rows
  (Black-formatted on export — AST-identical; inspected: no org identifiers).

What the test asserts, for every corpus file:

1. it LOADS through the real loader (``load_garden_class_from_path``,
   which resolves the class via ``find_garden_class``) and yields a
   class;
2. every **private base-class member the source actually calls**
   (``self._name(...)``) exists on the loaded class — this is wider than the
   promised ``AUTHORING_SURFACE`` pin in ``test_api_contracts.py`` and
   catches drift in members an author reached for beyond the documented list;
3. every module-level helper it imports from ``mmm_framework.model.base``
   still exists there.

Graph-BUILDING every corpus model is deliberately out of scope here: the
non-MMM families need bespoke data (indicator frames, transactions, survey
mediators) and each already has a dedicated fit test
(``test_cfa_garden_model.py``, ``test_lca_garden_model.py``,
``test_clv_garden_model.py``, the nested-recovery suite); the matrix's
``garden_subclass`` case builds and fingerprints the MMM-family path.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples" / "garden_models").glob("*.py"))
FIXTURES = sorted((ROOT / "tests" / "fixtures" / "garden_corpus").glob("*.py"))
CORPUS = [p for p in EXAMPLES + FIXTURES if p.name != "__init__.py"]

#: Files that are scripts/comparisons, not garden models (no loadable class).
_NOT_MODELS = {"breakout_pso_vs_bayes.py"}


def _corpus_ids():
    return [str(p.relative_to(ROOT)) for p in CORPUS if p.name not in _NOT_MODELS]


def _corpus_paths():
    return [p for p in CORPUS if p.name not in _NOT_MODELS]


def _load_class(path: Path):
    from mmm_framework.garden.loader import load_garden_class_from_path

    return load_garden_class_from_path(str(path))


class _PrivateUseCollector(ast.NodeVisitor):
    """``self._name(...)`` calls + ``from mmm_framework.model.base import``."""

    def __init__(self) -> None:
        self.self_calls: set[str] = set()
        self.base_imports: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "self"
            and f.attr.startswith("_")
            and not f.attr.startswith("__")
        ):
            self.self_calls.add(f.attr)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.endswith("model.base"):
            for alias in node.names:
                self.base_imports.add(alias.name)


def test_corpus_is_not_empty():
    """The parser-guard rule: an empty corpus must fail loudly."""
    assert len(_corpus_paths()) >= 8, (
        f"garden corpus shrank to {len(_corpus_paths())} files — the point "
        "of PR 0.4 is that this corpus exists in the repo, not in a "
        "gitignored SQLite file"
    )
    assert len(FIXTURES) >= 4, "the exported registry fixtures are gone"


def test_fixture_sources_carry_no_org_identifiers():
    pat = re.compile(r"\borg_[a-z0-9]{6,}\b|dev-org", re.I)
    hits = [p.name for p in FIXTURES if pat.search(p.read_text(encoding="utf-8"))]
    assert not hits, f"org identifiers in exported fixtures: {hits}"


@pytest.mark.parametrize("path", _corpus_paths(), ids=_corpus_ids())
def test_corpus_model_loads_and_its_private_surface_exists(path: Path):
    cls = _load_class(path)
    assert isinstance(cls, type), f"{path.name}: loader returned {cls!r}"

    collector = _PrivateUseCollector()
    collector.visit(ast.parse(path.read_text(encoding="utf-8")))

    missing = sorted(name for name in collector.self_calls if not hasattr(cls, name))
    assert not missing, (
        f"{path.name} calls private base members that no longer exist: "
        f"{missing} — a base-class refactor broke a registered garden model "
        "without any test noticing; that is the Phase 4 hazard this corpus "
        "exists to catch"
    )

    if collector.base_imports:
        import mmm_framework.model.base as MB

        gone = sorted(n for n in collector.base_imports if not hasattr(MB, n))
        assert not gone, (
            f"{path.name} imports {gone} from mmm_framework.model.base and "
            "they are gone"
        )
