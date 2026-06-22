"""Authoring support for Model Garden bespoke models — the knowledge + helpers
behind the Atelier's IDE tooling and Bayesian-modeling copilot.

Three pieces, all deterministic and dependency-light (no embeddings, no model
import at module load):

* :data:`GARDEN_AUTHORING_KNOWLEDGE` + :func:`build_copilot_system_prompt` — the
  curated, code-derived knowledge pack that turns a generic LLM into a focused
  expert on authoring ``CustomMMM`` subclasses (the oracle contract, the
  deterministic-name conventions, PyMC idioms, and the hard-won performance
  lessons).
* :func:`static_authoring_lint` — an AST-only "Problems" check for the editor
  (syntax, the garden class, the contract conventions). Host-safe: it never
  imports or executes the candidate source.
* :func:`format_source` — best-effort code formatting for the editor's *Format*
  button (``ruff format`` with a ``black`` fallback).
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from typing import Any

# --------------------------------------------------------------------------- #
# Knowledge pack — derived verbatim from garden/contract.py + model/base.py so
# the copilot teaches the REAL conventions, not a hallucinated approximation.
# --------------------------------------------------------------------------- #

GARDEN_AUTHORING_KNOWLEDGE = """\
# Model Garden — authoring a bespoke MMM (the oracle contract)

A *garden model* is a `BayesianMMM` subclass an expert authors, the agent
("oracle") tests + versions, and any project can re-fit. The clean path:
**subclass `mmm_framework.garden.CustomMMM` and override `_build_model` only.**
You inherit `fit` / `predict` / `sample_channel_contributions` /
`compute_component_decomposition` / serialization for free.

## Hard rules
- Keep the constructor signature `(panel, model_config, trend_config=None, ...)`.
  Do NOT override `__init__` — customize in `_build_model`. The agent swaps the
  class onto any project's data, so a non-standard ctor breaks re-fitting.
- `fit(method=None, random_seed=None, ...) -> MMMResults` must set `self._trace`.
- Build everything inside `with pm.Model(coords=self._build_coords()) as model:`.

## Deterministics your `_build_model` MUST register (read by the read-ops)
The reporting/agent surface reads these by NAME. Register them or the ops return
empty / degrade. All are in **standardized** KPI space except `y_obs_scaled`.
- `intercept_component`  (n_obs,)            baseline, dims="obs"
- `trend_component`      (n_obs,)            trend / latent level
- `seasonality_component`(n_obs,)            Fourier seasonality at each obs
- `channel_contributions`(n_obs, n_channel)  per-channel media contribution, dims=("obs","channel")
- `media_total`          (n_obs,)            = channel_contributions.sum(axis=1)
- `controls_total`       (n_obs,)            sum of control contributions
- `geo_component` / `product_component`      only when has_geo / has_product
- `y_obs_scaled`         (n_obs,)            = y_obs * y_std + y_mean  (ORIGINAL scale)

## Posterior parameter NAME conventions (extraction keys off these)
- `beta_<channel>`        channel coefficient — REQUIRED, or ROI/decomposition is empty.
  (If a channel's coefficient is fixed, still register a `pm.Deterministic("beta_<ch>", ...)`.)
- `adstock_alpha_<channel>` geometric carryover rate → unlocks half-life/carryover reporting.
- `sat_half_<channel>`, `sat_slope_<channel>` (Hill) or `sat_lam_<channel>` (logistic) → saturation reporting.

## pm.Data nodes predict() / sample_channel_contributions() swap (match these names)
- `X_media_raw`  (n_obs, n_channel)  normalized spend (parametric path) — build it from `self._prepare_raw_media_for_model()`.
- `X_controls`   (n_obs, n_control)  standardized controls.
- `time_idx`     (n_obs,)            period index per obs (fixed at build).

## Base-class helpers to REUSE inside your override (don't re-derive)
- `self._build_coords() -> dict`                         PyMC coords (obs/channel/control/fourier…).
- `self._prepare_raw_media_for_model() -> np.ndarray`    (n_obs, n_channel) normalized [0,1] spend.
- `self._build_channel_saturation(ch) -> (kind, params)` creates `sat_*_<ch>` RVs; pair with `_apply_saturation_pt(x, kind, params)` from `mmm_framework.model.base`.
- `self._build_trend_component(model, time_idx)`          configured trend (linear/piecewise/spline/GP).
- `self._build_control_betas(sigma)`                      `beta_controls` honoring causal roles / selection.
- helpers/RV-samplers: `_sample_from_prior_config(name, prior, default)` (honors per-channel ROI priors).

## Required instance attributes (present after __init__; you inherit them)
`channel_names`, `y_mean`, `y_std`, `_media_raw_max`, `panel`, `model_config`,
`has_geo`, `has_product`. (n_obs, n_channels, n_controls, time_idx, geo_idx,
product_idx, X_controls, n_periods, seasonality_features are also available.)

## Standardization (get this right or the scaling check fails)
Inputs/outputs are standardized: `self.y = (y_raw - y_mean) / y_std`. Components
live in standardized space; to read in original units multiply by `y_std`
(and add `y_mean` for the intercept/level). `y_obs_scaled` already un-scales.

## Bespoke configuration (settable params + non-default likelihood)
Don't hard-code tuning as class attributes — declare a **`CONFIG_SCHEMA`** (a
`pydantic.BaseModel` subclass) so your params are settable, defaulted, validated,
serialized, and rendered as a UI form. Read them via `self.model_params.<field>`
in `_build_model`:

    from pydantic import BaseModel, Field
    class AwarenessParams(BaseModel):
        number_of_trials: int = Field(default=500, gt=0)
        awareness_retention: float = 0.75
    class MyAwarenessMMM(CustomMMM):
        CONFIG_SCHEMA = AwarenessParams
        def _build_model(self):
            n = self.model_params.number_of_trials  # validated + defaulted

The spec carries overrides under `spec["model_params"]`. For a non-Gaussian KPI,
declare the family via `model_config.likelihood` (`spec["likelihood"]`, e.g.
`{"family": "binomial", "params": {"n_trials": …}}`) and WRITE the observation
node yourself in `_build_model` — the built-in additive dispatch only fits
normal/student_t (its priors assume standardized-Normal y). For a binomial
awareness model: `p = pm.math.sigmoid(mu)` then
`pm.Binomial("y_obs", n=self.model_params.number_of_trials, p=p, observed=self.y, dims="obs")`
(binomial KPIs are NOT standardized — `self.y` is the raw success count, `y_std==1`).

## PERFORMANCE — avoid pytensor.scan for recursions
A `pytensor.scan` (adstock/state-space recursion) builds a slow, GIL-holding
gradient graph; under the in-process kernel it can make a MAP/NUTS fit crawl and
**freeze the app during a compatibility test**. A geometric recursion
`Sₜ = ρ·Sₜ₋₁ + xₜ` has the closed form `Sₜ = Σ_{τ≤t} ρ^(t-τ)·x_τ` — a
lower-triangular Toeplitz matmul. Build the decay matrix once and `@` it:

    t = np.arange(n_obs); lag = t[:, None] - t[None, :]
    decay = pt.where(pt.as_tensor_variable(lag >= 0),
                     rho ** pt.as_tensor_variable(np.maximum(lag, 0)), 0.0)
    stock = decay @ media_inflow   # (n_obs, n_channel), no scan, compiles instantly

## Panels
National = a single time-ordered series (obs order == time order). For geo/product
panels, run any recursion INDEPENDENTLY per cell (`cell_idx`), never across cells.
If your model is national-only, raise a clear error on `has_geo or has_product`.

## Approximate fits
`method` ∈ {"map","advi","fullrank_advi","pathfinder"} give a fast approximate
posterior (`MMMResults.approximate=True`, R-hat/ESS None) — great for a structural
sanity check before paying for NUTS.
"""


#: Failure-mode knowledge for diagnosing a broken Atelier-NOTEBOOK cell. Used only
#: when the copilot is invoked with notebook context (a failing cell + traceback).
#: Grounded in the real ways a PyMC/MMM cell breaks in this framework's kernel.
NOTEBOOK_DIAGNOSIS_KNOWLEDGE = """\
# Diagnosing a failed notebook cell

The user runs free-form Python cells against their bespoke model in ONE warm
kernel (cells run top-to-bottom and share state — `mmm`, `results`, `df`, `data`,
`spec` persist between cells). `GardenModel` is bound to their model class; an
uploaded dataset binds as `df`; render with `show_table(df, title=...)` /
`fig.show()`. The most common failures and the fix to lead with:

## "Initial evaluation of model at starting point failed!" / a logp is `-inf`
PyMC could not evaluate the model at its start point. Read the per-variable logp
table in the error — the culprit is whichever entry is `-inf` (or wildly negative):
- **`y_obs: -inf`** → the observation distribution cannot reach the data at the
  start point. Almost always a **support / link mismatch** or a **scaling** bug:
  - A positive-only or bounded likelihood (Gamma/NegativeBinomial/Beta — look for
    a `kappa`/dispersion param) was given a mean `mu` that is ≤ 0 (or outside
    (0,1)). Put the right inverse link on the mean so it stays in support
    (`pm.math.exp` / `pt.softplus` for positive, `pm.math.sigmoid` for (0,1));
    do NOT feed a standardized (mean-0) quantity straight into a positive-only mean.
  - KPI/spend **scale**: feeding RAW spend or KPI where the model expects this
    framework's standardized space. Build media via
    `self._prepare_raw_media_for_model()` (normalized [0,1]); components live in
    standardized KPI space (× `y_std`, + `y_mean` for the level to read originals).
  - A deterministic that hits `log(≤0)` / divide-by-zero at the init values.
- **One latent param very negative (not -inf)** (e.g. an awareness innovation at
  ~ -143) → a too-tight prior fighting a centered parameterization. Reparameterize
  **non-centered** (`z ~ N(0,1)`; `x = mu + sigma * z`) so the start point is feasible.
- Fast confirm: `mmm.model.debug()` lists the offending point; a `method='map'`
  fit (`mmm.fit(method='map')`) is the cheap loop while you fix the link/priors.

## `NameError: name '...' is not defined` (`data`, `spec`, `mmm`, `df`)
Cells share one kernel and run in order. Either an earlier cell that defines it
hasn't run (use **Run all**, or run that cell first), or that earlier cell errored
so the name was never bound. `df` only exists when a dataset was uploaded.

## Slow / frozen fit, `pytensor.scan` warnings
A `scan`-based adstock/state recursion compiles a slow gradient graph and can hang
the in-process kernel. Vectorize to the lower-triangular Toeplitz matmul (see the
authoring knowledge above) — same math, compiles instantly.

## Shape / broadcast errors ("could not be broadcast", dim mismatch)
A registered deterministic has the wrong dims. `channel_contributions` is
`(n_obs, n_channel)` with `dims=("obs","channel")`; `media_total` =
`channel_contributions.sum(axis=1)` is `(n_obs,)`. Align coords with
`self._build_coords()` instead of hard-coding shapes.

## "No garden class found" / read-ops return empty
The source must expose exactly ONE `BayesianMMM` subclass (or set `GARDEN_MODEL`).
Empty ROI/decomposition usually means a missing `beta_<channel>` (register a
`pm.Deterministic("beta_<ch>", ...)` even for a fixed coefficient) or missing
`sat_*` / `adstock_alpha_*` registrations.

## Data / MFF binding
The uploaded file binds as `df`. MFF long-format has `VariableName` + a value
column (one row per variable × period); map `kpi` / `media_channels` /
`control_variables` in the `spec` to those variable names before `build_model`.
"""


def _clip(text: str, limit: int) -> str:
    """Length-cap a context blob with a visible truncation marker."""
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)\n"


#: Hard cap on sibling cells folded into the prompt context. Bounds peak memory
#: regardless of the client payload — the join below would otherwise materialize
#: the whole (untrusted, unbounded) ``other_cells`` list before ``_clip`` slices it.
_MAX_OTHER_CELLS = 40


def _notebook_context_section(notebook: dict[str, Any]) -> list[str]:
    """Render the failing-cell / notebook context block appended to the copilot
    prompt when the panel is invoked from the notebook (diagnose or chat)."""
    cell_code = _clip(str(notebook.get("cell_code") or "").strip(), 8000)
    traceback = _clip(str(notebook.get("traceback") or "").strip(), 6000)
    preview = _clip(str(notebook.get("dataset_preview") or "").strip(), 2000)
    # Slice BEFORE the comprehension/join so an oversized payload can't force an
    # O(N) pass or a giant intermediate string just to be truncated to 8000 chars.
    raw_others = (notebook.get("other_cells") or [])[:_MAX_OTHER_CELLS]
    others = [c for c in raw_others if str(c or "").strip()]
    is_error = bool(notebook.get("is_error"))

    out: list[str] = ["", "## The user's notebook cell"]
    if is_error:
        # Diagnose framing is gated on is_error ALONE (not is_error AND traceback)
        # so an errored cell with no captured output still gets the fix-it prompt.
        out += [
            "The user ran this cell against the model (above) and it FAILED. "
            "Diagnose the ROOT CAUSE in a sentence or two, give a couple of tight, "
            "specific tips, then return the COMPLETE corrected cell in a single "
            "```python fenced block so they can apply it in one click. If the real "
            "bug is in the MODEL SOURCE above (not the cell), say so plainly and "
            "return the corrected model class instead. Do not pad with generic "
            "advice.",
        ]
    else:
        out += [
            "The user is working in this notebook cell. Help with their question; "
            "when code is the answer, return a COMPLETE, runnable cell in a single "
            "```python fenced block so they can apply it in one click.",
        ]
    if cell_code:
        out += ["", "Cell:", "```python", cell_code, "```"]
    if traceback:
        out += ["", "Error / traceback:", "```", traceback, "```"]
    if preview:
        out += [
            "",
            "Dataset preview (binds as `df`):",
            "```",
            preview,
            "```",
        ]
    if others:
        joined = _clip(
            "\n\n# ── next cell ──\n".join(_clip(c, 1500) for c in others), 8000
        )
        out += [
            "",
            "Other code cells in the notebook (context — variables may come from "
            "these; cells run top-to-bottom in one shared kernel):",
            "```python",
            joined,
            "```",
        ]
    return out


def build_copilot_system_prompt(
    source_code: str | None = None,
    *,
    notebook: dict[str, Any] | None = None,
) -> str:
    """Assemble the modeling-copilot system prompt: expert persona + the live
    contract + the authoring knowledge pack + (optionally) the user's current
    editor source so suggestions are grounded in their actual code.

    When ``notebook`` is given (a failing/active notebook cell + traceback +
    dataset preview + sibling cells), the prompt gains a cell-diagnosis knowledge
    pack and a focused "fix this cell" instruction so the same copilot doubles as
    the notebook's debugging assistant."""
    try:
        from mmm_framework.garden.contract import describe_contract

        contract = describe_contract()
    except Exception:  # noqa: BLE001 — never let a contract import break the copilot
        contract = ""

    diagnosing = notebook is not None
    role_line = (
        "You are the Atelier's modeling copilot: a senior Bayesian statistician "
        "and PyMC expert who helps users author bespoke Marketing Mix Models "
        "(MMMs) for this codebase's Model Garden"
        + (
            " — and debug the cells they run against those models in the Atelier "
            "notebook."
            if diagnosing
            else "."
        )
    )

    parts = [
        role_line,
        "",
        "Your expertise: Bayesian workflow (priors, identifiability, "
        "reparameterization, prior/posterior predictive checks, divergences, "
        "non-centered parameterizations), PyMC 5 / PyTensor idioms, MMM structure "
        "(adstock/carryover, saturation/diminishing returns, hierarchical pooling, "
        "media/baseline decomposition, ROI & marginal ROI), and this framework's "
        "`CustomMMM` authoring contract.",
        "",
        "How to help:",
        "- Be concrete and correct. Prefer a working, minimal code change to a "
        "lecture. When you propose code, return a COMPLETE, runnable "
        + ("cell" if diagnosing else "module (or the full class)")
        + " in a single ```python fenced block so the user can apply it in one "
        "click.",
        "- Honor the contract below EXACTLY — register the deterministics the "
        "read-ops need and the `beta_<channel>` / `sat_*` / `adstock_alpha_*` "
        "naming, or reporting silently breaks. Explain WHY a prior or structure "
        "choice matters (identifiability, shrinkage, scale).",
        "- NEVER use `pytensor.scan` for adstock/state recursions — use the "
        "vectorized lower-triangular Toeplitz matmul shown in the knowledge pack "
        "(scan-grad compilation is slow and can freeze the in-process fit/test).",
        "- If the user's request is ambiguous or statistically risky (e.g. an "
        "unidentified latent level competing with media), say so and offer the "
        "safer default.",
        "- Keep prose tight; use short paragraphs and lists. You cannot run code "
        "or fit models — reason from the contract, the user's source, and any "
        "traceback provided.",
        "",
        contract,
        "",
        GARDEN_AUTHORING_KNOWLEDGE,
    ]
    if diagnosing:
        parts += ["", NOTEBOOK_DIAGNOSIS_KNOWLEDGE]

    src = (source_code or "").strip()
    if src:
        clipped = _clip(src, 16000)
        parts += [
            "",
            "## The user's current editor source (their model)",
            "Ground your suggestions in this code (edit it, don't restart from "
            "scratch unless asked):",
            "```python",
            clipped,
            "```",
        ]

    if diagnosing:
        parts += _notebook_context_section(notebook)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Static lint — AST only, never executes the candidate source.
# --------------------------------------------------------------------------- #

#: Deterministic names a custom `_build_model` should register for full reporting.
_EXPECTED_DETERMINISTICS = ("channel_contributions", "media_total")
_GARDEN_BASES = ("CustomMMM", "BayesianMMM", "BaseExtendedMMM")


def _problem(severity: str, message: str, line: int | None = None) -> dict[str, Any]:
    return {"severity": severity, "message": message, "line": line}


def static_authoring_lint(source_code: str) -> tuple[str | None, list[dict[str, Any]]]:
    """AST-level checks for the editor's Problems panel.

    Returns ``(class_name, problems)`` where each problem is
    ``{severity: "error"|"warning"|"info", message, line}``. ``error`` blocks a
    clean register; ``warning``/``info`` are advisory. Never raises; never
    imports or executes the source.
    """
    problems: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return None, [_problem("error", f"Syntax error: {e.msg}", e.lineno)]

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    explicit = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name):
            if any(
                isinstance(t, ast.Name) and t.id == "GARDEN_MODEL" for t in n.targets
            ):
                explicit = n.value.id

    if not classes:
        return None, [
            _problem(
                "error", "No class defined — expected a BayesianMMM/CustomMMM subclass."
            )
        ]

    cls: ast.ClassDef | None = None
    if explicit:
        cls = next((c for c in classes if c.name == explicit), None)
        if cls is None:
            problems.append(
                _problem(
                    "error",
                    f"GARDEN_MODEL points at '{explicit}', which is not defined.",
                )
            )
    elif len(classes) == 1:
        cls = classes[0]
    else:
        problems.append(
            _problem(
                "warning",
                f"{len(classes)} classes defined ({', '.join(c.name for c in classes)}); "
                "add `GARDEN_MODEL = YourClass` so the loader knows which one is the model.",
            )
        )
        cls = next(
            (c for c in classes if _bases_names(c) & set(_GARDEN_BASES)), classes[0]
        )

    if cls is None:
        return None, problems

    class_name = cls.name
    base_names = _bases_names(cls)
    if not (base_names & set(_GARDEN_BASES)):
        problems.append(
            _problem(
                "warning",
                f"'{class_name}' should subclass `CustomMMM` (recommended) or `BayesianMMM` "
                "to inherit the oracle contract.",
                cls.lineno,
            )
        )

    methods = {
        n.name: n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "__init__" in methods:
        problems.append(
            _problem(
                "warning",
                "Avoid overriding __init__ — keep the (panel, model_config, trend_config) "
                "constructor and customize in _build_model so the agent can re-fit your model.",
                methods["__init__"].lineno,
            )
        )

    # Convention checks that only apply when the user overrides the graph builder.
    if "_build_model" in methods:
        missing = [
            d
            for d in _EXPECTED_DETERMINISTICS
            if f'"{d}"' not in source_code and f"'{d}'" not in source_code
        ]
        if missing:
            problems.append(
                _problem(
                    "warning",
                    "Your _build_model does not register "
                    + ", ".join(f"`{d}`" for d in missing)
                    + " — ROI / decomposition reporting will be empty without it.",
                    methods["_build_model"].lineno,
                )
            )
        if "beta_" not in source_code:
            problems.append(
                _problem(
                    "warning",
                    "No `beta_<channel>` parameter found — channel ROI/decomposition helpers "
                    "extract contributions by this name.",
                    methods["_build_model"].lineno,
                )
            )

    if _uses_scan(tree):
        problems.append(
            _problem(
                "info",
                "pytensor.scan is used — scan-grad compilation is slow and can freeze the "
                "in-process compatibility test. Prefer a vectorized lower-triangular Toeplitz "
                "matmul for adstock/state recursions (ask the copilot to convert it).",
            )
        )

    if not problems:
        problems.append(
            _problem(
                "info",
                f"No issues found in '{class_name}'. Run the compatibility test to fit it.",
            )
        )
    return class_name, problems


def _uses_scan(tree: ast.AST) -> bool:
    """True only if the source actually IMPORTS or CALLS pytensor's scan — so a
    comment mentioning scan (e.g. one explaining why it's avoided) never trips it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "pytensor"
        ):
            if any(a.name == "scan" for a in node.names):
                return True
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "scan":
                return True
            if isinstance(fn, ast.Name) and fn.id in ("scan", "pytensor_scan"):
                return True
    return False


def _bases_names(cls: ast.ClassDef) -> set[str]:
    out: set[str] = set()
    for b in cls.bases:
        if isinstance(b, ast.Name):
            out.add(b.id)
        elif isinstance(b, ast.Attribute):
            out.add(b.attr)
    return out


# --------------------------------------------------------------------------- #
# Best-effort formatting for the editor's Format button.
# --------------------------------------------------------------------------- #


def format_source(source_code: str) -> tuple[str | None, str | None]:
    """Format Python source, returning ``(formatted, error)`` — exactly one of
    which is meaningful. Tries ``ruff format`` (fast, always in the dev env),
    then a ``black`` import, then reports that no formatter is installed."""
    ruff = shutil.which("ruff")
    if ruff:
        try:
            proc = subprocess.run(
                [ruff, "format", "-"],
                input=source_code,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode == 0:
                return proc.stdout, None
            return None, (proc.stderr.strip() or "ruff format failed").splitlines()[-1]
        except Exception:  # noqa: BLE001 — fall through to black
            pass
    try:
        import black

        return black.format_str(source_code, mode=black.Mode()), None
    except ImportError:
        return None, "No formatter installed (ruff/black)."
    except Exception as e:  # noqa: BLE001 — surface a syntax error cleanly
        return None, str(e).splitlines()[0] if str(e) else "format failed"


__all__ = [
    "GARDEN_AUTHORING_KNOWLEDGE",
    "NOTEBOOK_DIAGNOSIS_KNOWLEDGE",
    "build_copilot_system_prompt",
    "static_authoring_lint",
    "format_source",
]
