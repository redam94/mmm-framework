# Documentation Code-Snippet Testing

**Gate:** `tests/test_docs_snippets.py` — runs in the default (fast) suite, ~2–3 s.

**Rule:** document only APIs that exist. The suite enforces imports and known-object
methods on every Python code block in `docs/*.html`.

## What the gate checks

The test extracts Python code from every hand-authored docs page:

- `<pre><code class="python">…` and `<pre><code class="language-python">…`
- unclassed `<pre><code>…` blocks (treated as Python if they parse)
- `<div class="code-example">…</div>` blocks (span-based highlighting is
  stripped; HTML entities like `&lt;`, `&quot;`, `&emsp;` are decoded —
  `&emsp;` becomes a 4-space indent)

For each block it then verifies, statically (via `ast`):

1. **Imports.** Every `from mmm_framework[.x.y] import A, B` and
   `import mmm_framework.x` must resolve: the module imports, and each name
   exists on it (or is an importable submodule). A fictional symbol fails the
   suite.
2. **Method calls on conventional variable names.** For `name.method(...)`
   where `name` is in the binding map below, `method` must exist on the bound
   class (`dir(cls)` plus dataclass fields). Unknown variable names are
   skipped silently. If a snippet reassigns a mapped name to anything other
   than its trusted producer (e.g. `model = SomeOtherThing()`), the binding is
   dropped for that snippet — no false positives from local reuse.

Shell lines (`$`, `pip`, `uv `, `git `, …) are stripped before parsing.
`# Output:` sections are comments and parse fine. A block that mentions
`mmm_framework` but does not parse as Python after cleaning produces a *soft
warning* (and a regex-based import scan still runs); blocks with no
`mmm_framework` reference that don't parse (JS, console output) are skipped.

## Variable-name binding map

| Variable name(s) | Bound class |
|---|---|
| `mmm`, `model`, `fitted_model` | `mmm_framework.model.BayesianMMM` |
| `results`, `fit` | `mmm_framework.model.results.MMMResults` |
| `contrib`, `contributions` | `mmm_framework.model.results.ContributionResults` |
| `panel` | `mmm_framework.data_loader.PanelDataset` |
| `result` (only when `run_backtest` appears in the snippet) | `mmm_framework.validation.backtest.BacktestResult` |
| `config` | **not bound** — ambiguous across config classes, always skipped |

Extend the map in `BINDING_MAP` / `CONDITIONAL_BINDINGS` (and
`TRUSTED_PRODUCERS` for the rebinding guard) when docs adopt a new
conventional name. Keep it small and unambiguous.

## Marking a block as pseudocode / skipping it

Two equivalent opt-outs:

```html
<!-- doc-snippet: skip -->
<pre><code class="python">future_api.not_yet_real()</code></pre>
```

or make the first line of the snippet a marker comment:

```python
# pseudocode  (or: # illustrative)
the_idea(not_the_api)
```

Use these sparingly — a skipped block is an unverified block. Prefer writing
the snippet against the real API.

## When the gate fails

The failure message names the page, block index, and the missing symbol.
Either the docs reference a fictional API (fix the page) or the checker is
wrong (refine `tests/test_docs_snippets.py` — e.g. the binding map). The
module contains self-tests that feed synthetic fictional snippets to the
checker, so refinements can't silently disable detection.
