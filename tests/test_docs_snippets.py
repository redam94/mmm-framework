"""Documentation code-snippet gate.

Statically verifies that Python code blocks in the hand-authored docs site
(``docs/*.html``) **and in the repo's Markdown** (``README.md``, ``CLAUDE.md``,
``technical-docs/*.md``) only reference APIs that actually exist in
``mmm_framework``:

1. Every ``from mmm_framework... import X`` / ``import mmm_framework...``
   statement must resolve: the module imports and each imported name exists.
2. Method calls on conventionally-named variables (see ``BINDING_MAP``) must
   exist on the bound class (``dir()`` plus dataclass fields).

Opting out
----------
* Put an HTML comment ``<!-- doc-snippet: skip -->`` immediately before (or
  inside) the ``<pre>``/``<div class="code-example">`` block — in Markdown, on
  the line above the fence — or
* make the first line of the snippet ``# pseudocode`` or ``# illustrative``.

Blocks that mention ``mmm_framework`` but do not parse as Python after
cleaning are reported as *soft warnings* (not hard failures); imports are
still regex-scanned and verified in that case.

See ``technical-docs/doc-snippet-testing.md`` for the full convention.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import re
import warnings
from functools import lru_cache
from pathlib import Path

import pytest
from bs4 import BeautifulSoup, Comment, NavigableString

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"

SKIP_MARKER = "doc-snippet: skip"
PSEUDOCODE_MARKERS = ("# pseudocode", "# illustrative")

# Variable-name -> (module, class) bindings for the docs' conventional names.
# Deliberately small and explicit to avoid false positives.  ``config`` is
# ambiguous (many config classes) and intentionally absent.
BINDING_MAP: dict[str, tuple[str, str]] = {
    "mmm": ("mmm_framework.model", "BayesianMMM"),
    "model": ("mmm_framework.model", "BayesianMMM"),
    "fitted_model": ("mmm_framework.model", "BayesianMMM"),
    "results": ("mmm_framework.model.results", "MMMResults"),
    "fit": ("mmm_framework.model.results", "MMMResults"),
    "contrib": ("mmm_framework.model.results", "ContributionResults"),
    "contributions": ("mmm_framework.model.results", "ContributionResults"),
    "panel": ("mmm_framework.data_loader", "PanelDataset"),
    # Extension models. Deliberately NOT called `model` in the docs: the plain
    # name is bound to BayesianMMM above, and these families have a different
    # read surface (`get_mediation_effects`, `get_cross_effects_summary`, ...).
    "nested_model": ("mmm_framework.mmm_extensions.models.nested", "NestedMMM"),
    "mv_model": (
        "mmm_framework.mmm_extensions.models.multivariate",
        "MultivariateMMM",
    ),
}

# Bindings that only apply when a guard substring appears in the snippet.
# ``result`` is only trusted in a backtest context (run_backtest in snippet);
# ``post``/``state`` only in a continuous-learning context (their names are too
# generic to bind globally without colliding with other pages).
CONDITIONAL_BINDINGS: dict[str, tuple[str, tuple[str, str]]] = {
    "result": ("run_backtest", ("mmm_framework.validation.backtest", "BacktestResult")),
    "post": (
        "continuous_learning",
        ("mmm_framework.continuous_learning.model", "Posterior"),
    ),
    "state": (
        "continuous_learning",
        ("mmm_framework.continuous_learning.loop", "LearningState"),
    ),
}

# If a mapped name is (re)assigned inside a snippet, we only keep the binding
# when the assigned value is a call to one of these producers; anything else
# (or a loop target / unpacking) drops the binding for that snippet.
TRUSTED_PRODUCERS: dict[str, set[str]] = {
    "mmm": {"BayesianMMM"},
    "model": {"BayesianMMM"},
    "fitted_model": {"BayesianMMM"},
    "results": {"MMMResults", "fit"},
    "fit": {"MMMResults", "fit"},
    "contrib": {"ContributionResults", "get_contributions"},
    "contributions": {"ContributionResults", "get_contributions"},
    "panel": {"PanelDataset", "load"},
    "result": {"BacktestResult", "run_backtest"},
    "post": {"Posterior", "fit"},  # post = cl.fit(...) is the canonical producer
    "state": {"LearningState"},
    "nested_model": {"NestedMMM"},
    "mv_model": {"MultivariateMMM"},
}

# Lines that are clearly shell, not Python.
_SHELL_PREFIXES = (
    "$ ",
    "$",
    "pip ",
    "pip3 ",
    "uv ",
    "git ",
    "% ",
    "make ",
    "redis-server",
)

# Unicode whitespace that HTML entities (&emsp;, &nbsp;, ...) decode to.
_WS_TRANSLATION = str.maketrans(
    {" ": "    ", " ": "  ", " ": " ", "\xa0": " ", "​": ""}
)


@dataclasses.dataclass
class Snippet:
    """One extracted code block."""

    file: str
    index: int  # 1-based position within the file
    source: str  # cleaned text
    explicit_python: bool  # carried a python/language-python class
    skipped: bool = False
    skip_reason: str = ""


@dataclasses.dataclass
class CheckResult:
    violations: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    checked: bool = False  # True if the block was treated as Python and analyzed


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _has_skip_comment(element) -> bool:
    """True if a ``doc-snippet: skip`` HTML comment precedes or sits inside."""
    for comment in element.find_all(string=lambda s: isinstance(s, Comment)):
        if SKIP_MARKER in comment:
            return True
    # Look back through up to 3 meaningful previous siblings.
    seen = 0
    for sib in element.previous_siblings:
        if isinstance(sib, NavigableString) and not isinstance(sib, Comment):
            if sib.strip():
                seen += 1
            continue
        if isinstance(sib, Comment):
            if SKIP_MARKER in sib:
                return True
            seen += 1
        else:
            seen += 1
        if seen >= 3:
            break
    return False


def _clean(text: str) -> str:
    """Normalize entity whitespace and drop shell/prompt lines."""
    text = text.translate(_WS_TRANSLATION)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(p) for p in _SHELL_PREFIXES):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def extract_snippets(html_path: Path) -> list[Snippet]:
    """Extract candidate Python snippets from one docs HTML page."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    snippets: list[Snippet] = []
    index = 0

    elements = []
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        if code is None:
            continue  # bare <pre>: mermaid diagrams / ASCII art in this repo
        classes = code.get("class") or []
        explicit = "python" in classes or "language-python" in classes
        if classes and not explicit:
            continue  # explicitly another language
        elements.append((pre, code, explicit))
    for div in soup.find_all("div", class_="code-example"):
        elements.append((div, div, False))

    for container, node, explicit in elements:
        index += 1
        text = _clean(node.get_text())
        if not text:
            continue
        snip = Snippet(
            file=html_path.name, index=index, source=text, explicit_python=explicit
        )
        first_line = text.splitlines()[0].strip().lower()
        if _has_skip_comment(container):
            snip.skipped, snip.skip_reason = True, "doc-snippet: skip comment"
        elif any(first_line.startswith(m) for m in PSEUDOCODE_MARKERS):
            snip.skipped, snip.skip_reason = True, "pseudocode/illustrative marker"
        snippets.append(snip)
    return snippets


#: An *unlabeled* fence is only treated as Python when it actually imports the
#: package. Merely mentioning ``mmm_framework`` is too weak a signal in
#: Markdown: CLAUDE.md's directory tree is an unlabeled fence full of
#: ``src/mmm_framework/...`` paths (the same reason the HTML extractor skips
#: bare ``<pre>`` blocks — mermaid diagrams and ASCII art).
_MD_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+mmm_framework", re.M)


def extract_md_snippets(md_path: Path) -> list[Snippet]:
    """Extract candidate Python snippets from one Markdown file.

    Mirrors :func:`extract_snippets` for fenced code blocks: a ```` ```python ````
    fence is checked strictly, an *unlabeled* fence only when it imports
    ``mmm_framework`` (the Markdown analogue of the untyped ``code-example``
    divs), and a fence labeled another language is ignored. The opt-outs are
    the same — an HTML comment on the preceding line, or a
    ``# pseudocode`` / ``# illustrative`` first line.
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    snippets: list[Snippet] = []
    index = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped.startswith("```"):
            i += 1
            continue

        # Fence length matters: a ````-fenced block may contain ``` lines.
        fence = "`" * (len(stripped) - len(stripped.lstrip("`")))
        lang = stripped[len(fence) :].strip().lower()
        body: list[str] = []
        j = i + 1
        while j < len(lines) and lines[j].strip() != fence:
            body.append(lines[j])
            j += 1

        # Nearest preceding non-blank line carries the skip comment, if any.
        k = i - 1
        while k >= 0 and not lines[k].strip():
            k -= 1
        preceding = lines[k] if k >= 0 else ""
        i = j + 1

        if lang not in ("", "python", "py", "python3"):
            continue
        text = _clean("\n".join(body))
        if not text:
            continue
        explicit = lang.startswith("py")
        if not explicit and not _MD_IMPORT_RE.search(text):
            continue

        index += 1
        snip = Snippet(
            file=md_path.name, index=index, source=text, explicit_python=explicit
        )
        first_line = text.splitlines()[0].strip().lower()
        if SKIP_MARKER in preceding:
            snip.skipped, snip.skip_reason = True, "doc-snippet: skip comment"
        elif any(first_line.startswith(m) for m in PSEUDOCODE_MARKERS):
            snip.skipped, snip.skip_reason = True, "pseudocode/illustrative marker"
        snippets.append(snip)
    return snippets


# ---------------------------------------------------------------------------
# Static API verification
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _import_module(module: str):
    """Import a module, caching both successes and failures."""
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - any import failure is a finding
        return exc


@lru_cache(maxsize=None)
def _resolve_class(module: str, cls_name: str):
    mod = _import_module(module)
    if isinstance(mod, Exception):
        return None
    return getattr(mod, cls_name, None)


@lru_cache(maxsize=None)
def _class_members(module: str, cls_name: str) -> frozenset[str]:
    cls = _resolve_class(module, cls_name)
    if cls is None:
        return frozenset()
    members = set(dir(cls))
    if dataclasses.is_dataclass(cls):
        members |= {f.name for f in dataclasses.fields(cls)}
    return frozenset(members)


def _check_import_target(module: str, names: list[str], where: str) -> list[str]:
    """Verify ``from <module> import <names>`` resolves."""
    violations = []
    mod = _import_module(module)
    if isinstance(mod, Exception):
        violations.append(f"{where}: module `{module}` does not import ({mod!r})")
        return violations
    for name in names:
        if name == "*":
            continue
        if hasattr(mod, name):
            continue
        # ``from pkg import submodule`` where submodule isn't re-exported.
        sub = _import_module(f"{module}.{name}")
        if isinstance(sub, Exception):
            violations.append(f"{where}: `{module}` has no attribute `{name}`")
    return violations


_IMPORT_FROM_RE = re.compile(
    r"^\s*from\s+(mmm_framework[\w.]*)\s+import\s+\(?([^)#\n]*(?:\n[^)#]*?)*?)\)?\s*(?:#.*)?$",
    re.M,
)
_IMPORT_RE = re.compile(r"^\s*import\s+(mmm_framework[\w.]*)", re.M)


def _regex_scan_imports(source: str, where: str) -> list[str]:
    """Fallback import verification when a block doesn't parse as Python."""
    violations = []
    # Pull multiline parenthesized import bodies too.
    for match in re.finditer(
        r"from\s+(mmm_framework[\w.]*)\s+import\s+(\(([^)]*)\)|[^\n(]*)",
        source,
    ):
        module = match.group(1)
        body = match.group(3) if match.group(3) is not None else match.group(2)
        names = [
            n.split(" as ")[0].strip()
            for n in body.replace("\n", ",").split(",")
            if n.strip() and not n.strip().startswith("#")
        ]
        names = [n for n in names if n.isidentifier()]
        violations += _check_import_target(module, names, where)
    for match in _IMPORT_RE.finditer(source):
        mod = _import_module(match.group(1))
        if isinstance(mod, Exception):
            violations.append(
                f"{where}: module `{match.group(1)}` does not import ({mod!r})"
            )
    return violations


def _effective_bindings(tree: ast.AST, source: str) -> dict[str, tuple[str, str]]:
    """BINDING_MAP minus names the snippet rebinds to something untrusted."""
    bindings = dict(BINDING_MAP)
    for name, (guard, target) in CONDITIONAL_BINDINGS.items():
        if guard in source:
            bindings[name] = target

    assigned_untrusted: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.For):
            targets, value = [node.target], None
        elif isinstance(node, ast.comprehension):
            targets, value = [node.target], None
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            targets, value = [node.optional_vars], None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in ast.walk(node.args):
                if isinstance(arg, ast.arg) and arg.arg in bindings:
                    assigned_untrusted.add(arg.arg)
            continue

        for tgt in targets:
            for sub in ast.walk(tgt):
                if not (isinstance(sub, ast.Name) and sub.id in bindings):
                    continue
                trusted = False
                if isinstance(value, ast.Call):
                    func = value.func
                    func_name = (
                        func.id
                        if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute) else ""
                    )
                    trusted = func_name in TRUSTED_PRODUCERS.get(sub.id, set())
                if not trusted:
                    assigned_untrusted.add(sub.id)

    for name in assigned_untrusted:
        bindings.pop(name, None)
    return bindings


def check_snippet(snippet: Snippet) -> CheckResult:
    """Run the static API gate on one snippet."""
    result = CheckResult()
    where = f"{snippet.file} [block {snippet.index}]"
    source = snippet.source

    try:
        tree = ast.parse(source)
    except SyntaxError:
        if "mmm_framework" in source:
            result.warnings.append(
                f"{where}: mentions mmm_framework but does not parse as Python "
                "after cleaning (imports were still regex-checked)"
            )
            result.violations += _regex_scan_imports(source, where)
            result.checked = True
        # Non-Python block (JS, console output, ...): skip silently.
        return result

    result.checked = True

    # 1. Imports.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("mmm_framework"):
                names = [alias.name for alias in node.names]
                result.violations += _check_import_target(node.module, names, where)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mmm_framework"):
                    mod = _import_module(alias.name)
                    if isinstance(mod, Exception):
                        result.violations.append(
                            f"{where}: module `{alias.name}` does not import ({mod!r})"
                        )

    # 2. Method calls on conventionally-named objects.
    bindings = _effective_bindings(tree, source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            continue
        var = func.value.id
        if var not in bindings:
            continue
        module, cls_name = bindings[var]
        members = _class_members(module, cls_name)
        if not members:
            continue  # binding class itself unresolvable; don't cascade
        if func.attr not in members:
            result.violations.append(
                f"{where}: `{var}.{func.attr}(...)` but "
                f"`{module}.{cls_name}` has no such method/field"
            )
    return result


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _docs_files() -> list[Path]:
    return sorted(DOCS_DIR.glob("*.html"))


def _md_files() -> list[Path]:
    """Markdown sources the gate covers, in reading-importance order.

    ``README.md`` is the PyPI landing page and the first thing a new user
    reads; ``CLAUDE.md`` is loaded into every agent session, so a wrong example
    there actively misleads the assistant. ``technical-docs/*.md`` are the
    internal specs. All three were unguarded until 2026-07-25, and every Python
    statement in CLAUDE.md's usage block had rotted (see #172).
    """
    return [ROOT / "README.md", ROOT / "CLAUDE.md"] + sorted(
        (ROOT / "technical-docs").glob("*.md")
    )


def _assert_no_violations(snippets: list[Snippet], hint: str) -> None:
    violations: list[str] = []
    soft_warnings: list[str] = []
    for snippet in snippets:
        if snippet.skipped:
            continue
        res = check_snippet(snippet)
        violations += res.violations
        soft_warnings += res.warnings
    for msg in soft_warnings:
        warnings.warn(f"doc-snippet soft warning: {msg}", stacklevel=1)
    assert not violations, (
        "Documentation references APIs that do not exist:\n  "
        + "\n  ".join(violations)
        + f"\n\nFix the page, or mark the block with {hint} "
        "(see technical-docs/doc-snippet-testing.md)."
    )


@pytest.mark.parametrize("html_path", _docs_files(), ids=lambda p: p.name)
def test_docs_snippets(html_path: Path) -> None:
    """Every Python code block on this docs page references real APIs."""
    _assert_no_violations(
        extract_snippets(html_path),
        "<!-- doc-snippet: skip --> / a `# pseudocode` first line",
    )


@pytest.mark.parametrize("md_path", _md_files(), ids=lambda p: p.name)
def test_markdown_snippets(md_path: Path) -> None:
    """Every Python code block in this Markdown file references real APIs."""
    _assert_no_violations(
        extract_md_snippets(md_path),
        "<!-- doc-snippet: skip --> on the line above / a `# pseudocode` first line",
    )


def test_docs_dir_exists_and_has_snippets() -> None:
    """Guard against the gate silently checking nothing."""
    files = _docs_files()
    assert files, f"No docs HTML files found under {DOCS_DIR}"
    total = sum(len(extract_snippets(f)) for f in files)
    assert total > 50, f"Suspiciously few code blocks extracted ({total})"


def test_markdown_sources_exist_and_have_snippets() -> None:
    """The same guard for Markdown — this gate's whole point is that a source
    which is never *collected* fails silently forever (#172)."""
    files = _md_files()
    missing = [p for p in files if not p.exists()]
    assert not missing, f"Declared Markdown sources do not exist: {missing}"
    for required in ("README.md", "CLAUDE.md"):
        assert any(p.name == required for p in files), f"{required} not collected"
    total = sum(len(extract_md_snippets(f)) for f in files)
    assert total > 30, f"Suspiciously few Markdown code blocks extracted ({total})"


# ---------------------------------------------------------------------------
# Self-tests: prove the checker catches fictional APIs
# ---------------------------------------------------------------------------


def _synthetic(source: str) -> Snippet:
    return Snippet(file="<synthetic>", index=1, source=source, explicit_python=True)


def test_checker_flags_fictional_import() -> None:
    res = check_snippet(_synthetic("from mmm_framework import TotallyFakeClass"))
    assert any("TotallyFakeClass" in v for v in res.violations)


def test_checker_flags_fictional_module() -> None:
    res = check_snippet(_synthetic("import mmm_framework.no_such_module"))
    assert any("no_such_module" in v for v in res.violations)


def test_checker_flags_fictional_submodule_import() -> None:
    res = check_snippet(_synthetic("from mmm_framework.fictional.deep import Whatever"))
    assert res.violations


def test_checker_flags_fictional_method() -> None:
    res = check_snippet(_synthetic("mmm.summon_dragons(budget=1e6)"))
    assert any("summon_dragons" in v for v in res.violations)


def test_checker_flags_fictional_method_in_regexfallback_block() -> None:
    # Unparseable block that still names mmm_framework: imports regex-scanned.
    res = check_snippet(
        _synthetic("from mmm_framework import NotARealThing\nthis is : not python !!")
    )
    assert any("NotARealThing" in v for v in res.violations)
    assert res.warnings  # and it lands on the soft-warning list


def test_checker_accepts_real_apis() -> None:
    src = (
        "from mmm_framework import BayesianMMM\n"
        "from mmm_framework.model.results import MMMResults\n"
        "results = mmm.fit(draws=500)\n"
        "fitted_model.predict()\n"
    )
    res = check_snippet(_synthetic(src))
    assert res.violations == []


def test_checker_ignores_rebound_names() -> None:
    # `model` rebound to something else: binding must be dropped, no flag.
    src = "model = SomeOtherThing()\nmodel.definitely_not_on_bayesianmmm()"
    res = check_snippet(_synthetic(src))
    assert res.violations == []


def test_checker_skips_unknown_variables_silently() -> None:
    res = check_snippet(_synthetic("config.totally_made_up()\nfoo.bar()"))
    assert res.violations == []


def test_checker_flags_fictional_method_on_cl_posterior() -> None:
    # `post` binds to continuous_learning.model.Posterior only when the
    # snippet mentions continuous_learning; cl.fit is a trusted producer.
    src = (
        "import mmm_framework.continuous_learning as cl\n"
        "post = cl.fit(data, channels=chs)\n"
        "post.summon_dragons()\n"
    )
    res = check_snippet(_synthetic(src))
    assert any("summon_dragons" in v for v in res.violations)


def test_checker_accepts_real_cl_posterior_and_state_apis() -> None:
    src = (
        "import mmm_framework.continuous_learning as cl\n"
        "post = cl.fit(data, channels=chs)\n"
        "post.gamma_summary()\n"
        "state = cl.LearningState(channels=chs, center=c, pairs=p, pair_signs=s)\n"
        "state.ingest(wave)\n"
        "state.recommend()\n"
    )
    res = check_snippet(_synthetic(src))
    assert res.violations == []


def test_cl_bindings_inert_without_guard() -> None:
    # Outside a continuous-learning snippet, `post`/`state` stay unbound.
    res = check_snippet(_synthetic("post.summon_dragons()\nstate.not_real()"))
    assert res.violations == []


def test_pseudocode_marker_skips_block(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text(
        '<html><body><pre><code class="python"># pseudocode\n'
        "from mmm_framework import Imaginary</code></pre></body></html>"
    )
    snips = extract_snippets(page)
    assert len(snips) == 1 and snips[0].skipped


def test_skip_comment_skips_block(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text(
        "<html><body><!-- doc-snippet: skip -->\n"
        '<pre><code class="python">from mmm_framework import Imaginary'
        "</code></pre></body></html>"
    )
    snips = extract_snippets(page)
    assert len(snips) == 1 and snips[0].skipped


def test_extraction_strips_highlight_spans_and_entities(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text(
        '<html><body><div class="code-example">'
        '<span class="keyword">from</span> mmm_framework '
        '<span class="keyword">import</span> BayesianMMM\n'
        "x = {&quot;a&quot;: 1}\n"
        '<span class="keyword">if</span> x:\n'
        '&emsp;<span class="function">print</span>(x[&quot;a&quot;] &lt; 2)'
        "</div></body></html>"
    )
    snips = extract_snippets(page)
    assert len(snips) == 1
    res = check_snippet(snips[0])
    assert res.checked and res.violations == []


# ---------------------------------------------------------------------------
# Self-tests: the Markdown extractor
# ---------------------------------------------------------------------------


def _md(tmp_path: Path, body: str) -> list[Snippet]:
    doc = tmp_path / "doc.md"
    doc.write_text(body)
    return extract_md_snippets(doc)


def test_md_extracts_python_fence(tmp_path: Path) -> None:
    snips = _md(
        tmp_path, "text\n\n```python\nfrom mmm_framework import BayesianMMM\n```\n"
    )
    assert len(snips) == 1
    assert snips[0].explicit_python
    assert check_snippet(snips[0]).violations == []


def test_md_flags_fictional_api_in_fence(tmp_path: Path) -> None:
    snips = _md(
        tmp_path, "```python\nfrom mmm_framework import TotallyFakeClass\n```\n"
    )
    assert any("TotallyFakeClass" in v for v in check_snippet(snips[0]).violations)


def test_md_ignores_other_languages(tmp_path: Path) -> None:
    body = '```bash\nuv run pytest\n```\n\n```json\n{"a": 1}\n```\n'
    assert _md(tmp_path, body) == []


def test_md_checks_unlabeled_fence_only_when_it_imports_the_package(
    tmp_path: Path,
) -> None:
    body = (
        "```\nfrom mmm_framework import BayesianMMM\n```\n\n"
        "```\nsome unrelated prose block\n```\n"
    )
    snips = _md(tmp_path, body)
    assert len(snips) == 1
    assert not snips[0].explicit_python


def test_md_ignores_a_directory_tree_that_merely_names_the_package(
    tmp_path: Path,
) -> None:
    """CLAUDE.md's repo map is an unlabeled fence full of package paths."""
    body = "```\nmmm-framework/\n├── src/mmm_framework/    # core\n```\n"
    assert _md(tmp_path, body) == []


def test_md_honors_skip_comment_above_the_fence(tmp_path: Path) -> None:
    body = (
        "<!-- doc-snippet: skip -->\n\n"
        "```python\nfrom mmm_framework import Imaginary\n```\n"
    )
    snips = _md(tmp_path, body)
    assert len(snips) == 1 and snips[0].skipped


def test_md_honors_pseudocode_marker(tmp_path: Path) -> None:
    snips = _md(tmp_path, "```python\n# pseudocode\nmmm.summon_dragons()\n```\n")
    assert len(snips) == 1 and snips[0].skipped


def test_md_respects_fence_length(tmp_path: Path) -> None:
    """A ````-fenced block may legitimately contain ``` lines."""
    body = "````markdown\n```python\nnot code, an example of markup\n```\n````\n"
    assert _md(tmp_path, body) == []
