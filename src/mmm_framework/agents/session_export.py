"""Export a chat session's analysis as a standalone, runnable Python script.

The agent's ``execute_python`` runs in a STATEFUL warm kernel where ``df``,
``mmm``, ``results``, the framework convenience names, and the
``save_result``/``load_result`` helpers are *injected* — they are never written
as code. Several state-producing steps (``configure_model``, ``fit_mmm_model``,
``generate_synthetic_data``) are TOOL calls that emit no ``code_snippet`` at all.
So a naive concatenation of the captured cells would ``NameError`` when run as a
script ("works in the notebook, breaks as a file").

This generator therefore synthesizes a PREAMBLE that reconstitutes the injected
state — import the framework surface the kernel pre-binds, load the dataset as
``df``, and define the ``save_result``/``load_result`` helpers — and then
replays the session's work in conversation order. When the LangGraph checkpoint
messages are available, TOOL CALLS are reconstructed as real code:

* ``fit_mmm_model`` becomes ``build_and_fit(spec, dataset_path)`` with the
  fit's full normalized spec (from the matching ``model_run`` artifact)
  embedded, plus a commented fast-reload alternative;
* model-interpretation tools (``get_roi_metrics``, ``run_budget_scenario``, …)
  become direct calls into the same ``mmm_framework.agents.model_ops``
  functions the agent ran;
* EDA tools (``run_eda``, ``validate_data``, ``detect_outliers``) become calls
  into the kernel-free ``mmm_framework.eda`` engine;
* everything else (KB search, preferences, report generation, …) is kept as a
  commented record so no step of the session is silently dropped.

Cells that errored when they ran are MARKED, not dropped (so no work is lost
and the script doesn't lie about what succeeded). Threads without checkpoints
(synthetic/legacy threads) fall back to the artifact-only layout.

Data backing it already exists: ``execute_python`` code is persisted as ordered
``code_snippet`` artifacts, paired to ``text_output`` by ``call_id``, and fits
record a ``model_run`` artifact with ``model_path``/``dataset_path``/``spec``.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from mmm_framework.api import sessions as _sessions

logger = logging.getLogger(__name__)

# Import block that mirrors EXACTLY what execute_python pre-binds as bare names
# (see agents/tools.py), so cells that use `BayesianMMM(...)` / `ModelConfigBuilder()`
# / `mmf.analysis` resolve identically when run standalone.
_PREAMBLE_IMPORTS = """\
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mmm_framework as mmf
from mmm_framework import (
    MFFConfigBuilder,
    KPIConfigBuilder,
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    ModelConfigBuilder,
    TrendConfigBuilder,
    PriorConfigBuilder,
    BayesianMMM,
    load_mff,
)
from mmm_framework.builders.model import SeasonalityConfigBuilder
from mmm_framework.builders.prior import AdstockConfigBuilder, SaturationConfigBuilder

# bare submodule aliases (as provided inside the agent kernel)
try:
    import mmm_framework.analysis as analysis
    import mmm_framework.mmm_extensions as mmm_extensions
    import mmm_framework.reporting as reporting
    import mmm_framework.diagnostics as diagnostics
except Exception:
    pass

# analysis surfaces the exported tool calls run against: the agent's model-op
# functions (ROI, diagnostics, budget ops, …) and the kernel-free EDA engine.
# Guarded like the aliases above — this SAME string is the subprocess kernel's
# startup source (agents/kernels.py::_build_startup_source), which must never
# fail to initialize because an optional surface is missing. A script section
# that needs one of these then fails loudly at use with a NameError.
try:
    from mmm_framework.agents import model_ops
    import mmm_framework.eda as eda
except Exception:
    pass
"""

# Standalone copies of the kernel's durable-result helpers, writing to ./results
# (the kernel writes to <thread_dir>/results). Kept byte-compatible with the
# kernel: concatenate the extension (NOT Path.with_suffix, which would truncate a
# dotted name like "q4.2024" to "q4.parquet").
_PREAMBLE_HELPERS = '''\
from pathlib import Path as _Path


def save_result(name, obj):
    """Persist `obj` under `name` to ./results (parquet for tables, else pickle)."""
    _Path("results").mkdir(exist_ok=True)
    base = _Path("results") / str(name)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
        try:
            p = _Path(str(base) + ".parquet")
            frame.to_parquet(p)
            return str(p)
        except Exception:
            pass
    try:
        import cloudpickle as _pk
    except Exception:
        import pickle as _pk
    p = _Path(str(base) + ".pkl")
    with open(p, "wb") as _fh:
        _pk.dump(obj, _fh)
    return str(p)


def load_result(name):
    """Reload an object saved earlier with save_result(name)."""
    base = _Path("results") / str(name)
    pq = _Path(str(base) + ".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    pk = _Path(str(base) + ".pkl")
    if pk.exists():
        try:
            import cloudpickle as _pk
        except Exception:
            import pickle as _pk
        with open(pk, "rb") as _fh:
            return _pk.load(_fh)
    raise FileNotFoundError(
        f"No saved result named {name!r}. Available: {list_saved_results()}"
    )


def list_saved_results():
    """Names previously persisted with save_result (in ./results)."""
    d = _Path("results")
    if not d.exists():
        return []
    return sorted({p.stem for p in d.glob("*") if p.suffix in (".parquet", ".pkl")})


def show_table(df, title=None, group="repl"):
    """Standalone stand-in for the kernel's dashboard table renderer: print a
    preview instead (the dashboard only exists inside the agent session)."""
    if title:
        print(f"== {title} ==")
    try:
        print(df.head(20).to_string())
    except Exception:
        print(df)


def run_op(op, mmm=None, results=None, **kwargs):
    """Run an agent model-op standalone: call the same function the agent's
    tool ran, print its markdown content (or error), and return the raw
    result dict ({"content", "dashboard", "error", ...})."""
    res = op(mmm, results, **kwargs)
    if isinstance(res, dict):
        if res.get("error"):
            print(f"[model op error] {res['error']}")
        elif res.get("content"):
            print(res["content"])
    return res
'''


# ── timeline extraction (checkpointed LangGraph messages → tool events) ───────


@dataclass
class ToolEvent:
    """One tool call from the conversation, paired with its result."""

    call_id: str
    name: str
    args: dict = field(default_factory=dict)
    result_text: str = ""
    is_error: bool = False


def _msg_type(m: Any) -> str:
    t = getattr(m, "type", None)
    if t is None and isinstance(m, dict):
        t = m.get("type") or m.get("role")
    return str(t or "")


def _msg_content_text(m: Any) -> str:
    c = getattr(m, "content", None)
    if c is None and isinstance(m, dict):
        c = m.get("content")
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if isinstance(c, str):
        return c
    return "" if c is None else str(c)


def _tool_calls_of(m: Any) -> list[tuple[str, dict, str]]:
    """(name, args, id) triples off an AI message — dict OR object tool_calls
    (same tolerance as api/main.py::_repair_orphan_tool_calls)."""
    tcs = getattr(m, "tool_calls", None)
    if tcs is None and isinstance(m, dict):
        tcs = m.get("tool_calls")
    out: list[tuple[str, dict, str]] = []
    for tc in tcs or []:
        if isinstance(tc, dict):
            name, args, tcid = tc.get("name"), tc.get("args"), tc.get("id")
        else:
            name = getattr(tc, "name", None)
            args = getattr(tc, "args", None)
            tcid = getattr(tc, "id", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        out.append(
            (
                str(name or ""),
                dict(args) if isinstance(args, dict) else {},
                str(tcid or ""),
            )
        )
    return out


def extract_timeline(messages: list | None) -> list[tuple[str, Any]]:
    """The conversation as ordered ``("question", str)`` / ``("tool", ToolEvent)``
    items. Duck-typed over LangChain message objects AND plain dicts; any
    extraction failure degrades to an empty timeline (the artifact-only
    fallback), never an exception.
    """
    try:
        results: dict[str, tuple[str, bool]] = {}
        for m in messages or []:
            if _msg_type(m) != "tool":
                continue
            tcid = getattr(m, "tool_call_id", None)
            if tcid is None and isinstance(m, dict):
                tcid = m.get("tool_call_id")
            if not tcid:
                continue
            status = getattr(m, "status", None)
            if status is None and isinstance(m, dict):
                status = m.get("status")
            results[str(tcid)] = (_msg_content_text(m), str(status or "") == "error")

        timeline: list[tuple[str, Any]] = []
        for m in messages or []:
            t = _msg_type(m)
            if t == "human":
                text = _msg_content_text(m).strip()
                if text:
                    timeline.append(("question", text))
            elif t == "ai":
                for name, args, tcid in _tool_calls_of(m):
                    if not name:
                        continue
                    rtext, is_err = results.get(tcid, ("", False))
                    timeline.append(
                        (
                            "tool",
                            ToolEvent(
                                call_id=tcid,
                                name=name,
                                args=args,
                                result_text=rtext,
                                is_error=is_err,
                            ),
                        )
                    )
        return timeline
    except Exception:
        logger.warning(
            "session export: could not extract the message timeline; "
            "falling back to the artifact-only layout",
            exc_info=True,
        )
        return []


# ── rendering helpers ─────────────────────────────────────────────────────────


def _comment_block(text: str, max_lines: int = 12, label: str = "Output") -> str:
    """Render `text` as a truncated, commented preview block."""
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return ""
    shown = lines[:max_lines]
    out = ["#   " + ln for ln in shown]
    if len(lines) > max_lines:
        out.append(f"#   ... ({len(lines) - max_lines} more line(s))")
    return f"# {label}:\n" + "\n".join(out)


def _truncate(text: str, limit: int = 300) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _docstring_safe(text: str) -> str:
    """Neutralize content interpolated into the generated module docstring —
    a session name / scope containing ``\"\"\"`` (or stray backslash escapes)
    must not terminate the docstring or inject statements into the script."""
    return str(text).replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _comment_safe(text: str) -> str:
    """One-line, newline-free text safe to embed in a generated # comment
    (a \\n or \\r would end the comment and leak the rest as code)."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _resolve_dataset(thread_id: str, run_payloads: list[dict]) -> str | None:
    """The dataset the session worked on — from the latest fit that recorded
    one (not only the very last run), else a registered data file."""
    for payload in reversed(run_payloads):
        if payload.get("dataset_path"):
            return payload["dataset_path"]
    try:
        for f in _sessions.list_files(thread_id):
            name = str(f.get("name", "")).lower()
            if name.endswith((".csv", ".parquet")):
                return f.get("path")
    except Exception:
        pass
    return None


def _fit_errored(ev: ToolEvent) -> bool:
    """A fit whose ToolMessage carries the error text (fit_mmm_model reports
    failures in-content, without status='error')."""
    if ev.is_error:
        return True
    head = (ev.result_text or "").lstrip()
    return head.startswith("Error fitting model") or head.startswith("Error:")


# Parses the auto-save line build_and_fit puts in the fit summary
# (agents/fitting.py: `summary += f" Auto-saved as **{run_name}**."`) — keep in
# sync; when the text drifts, matching degrades to the nth-fit↔nth-run pairing.
_RUN_NAME_RE = re.compile(r"Auto-saved as \*\*([^*]+)\*\*")


def _match_fit_runs(
    fit_events: list[ToolEvent], run_payloads: list[dict]
) -> dict[int, dict]:
    """Pair each fit event with its model_run artifact payload.

    Precedence: run_name parsed from the ToolMessage's "Auto-saved as
    **run_X**" text → nth-successful-fit ↔ nth-run → latest run.
    Keyed by ``id(event)``.
    """
    matched: dict[int, dict] = {}
    remaining = list(run_payloads)
    for ev in fit_events:
        m = _RUN_NAME_RE.search(ev.result_text or "")
        if not m:
            continue
        for p in remaining:
            if p.get("run_name") == m.group(1):
                matched[id(ev)] = p
                remaining.remove(p)
                break
    unnamed = [
        ev for ev in fit_events if id(ev) not in matched and not _fit_errored(ev)
    ]
    for ev, p in zip(unnamed, remaining):
        matched[id(ev)] = p
    if run_payloads:
        for ev in fit_events:
            if id(ev) not in matched and not _fit_errored(ev):
                matched[id(ev)] = run_payloads[-1]
    return matched


# ── Tier-2: generic model-op tools ────────────────────────────────────────────


def _map_no_args(args: dict) -> dict:
    return {}


def _map_get_estimands(args: dict) -> dict | None:
    requested: list = list(args.get("estimands") or [])
    custom = args.get("custom_estimands")
    if custom:
        try:
            parsed = json.loads(custom)
        except Exception:
            return None
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return None
        requested.extend(parsed)
    return {"estimands": requested} if requested else {}


def _map_calibration_check(args: dict) -> dict:
    return {
        "n_sims": int(args.get("n_sims", 32)),
        "L": int(args.get("posterior_draws", 100)),
        "sampler": str(args.get("sampler", "numpyro")),
    }


def _map_coverage_check(args: dict) -> dict:
    return {
        "n_sims": int(args.get("n_sims", 16)),
        "L": int(args.get("posterior_draws", 150)),
        "sampler": str(args.get("sampler", "numpyro")),
        "truth": str(args.get("truth", "auto")),
    }


def _map_prior_predictive(args: dict) -> dict:
    return {"n_samples": int(args.get("n_samples", 500))}


def _map_leave_one_out(args: dict) -> dict | None:
    comp = args.get("component_to_drop")
    return {"component_to_drop": str(comp)} if comp else None


def _map_budget_scenario(args: dict) -> dict | None:
    raw = args.get("spend_changes")
    if isinstance(raw, dict):
        return {"spend_changes": raw}
    try:
        changes = json.loads(raw or "")
    except Exception:
        return None
    return {"spend_changes": changes} if isinstance(changes, dict) else None


def _map_marginal_analysis(args: dict) -> dict:
    chans = None
    raw = args.get("channels")
    if isinstance(raw, list):
        chans = raw
    elif raw:
        try:
            chans = json.loads(raw)
        except Exception:
            chans = None
    return {
        "spend_increase_pct": float(args.get("spend_increase_pct", 10.0)),
        "channels": chans,
    }


def _map_budget_optimizer(args: dict) -> dict:
    kw: dict = {
        "min_multiplier": float(args.get("min_multiplier", 0.0)),
        "max_multiplier": float(args.get("max_multiplier", 2.0)),
    }
    if args.get("total_budget") is not None:
        kw["total_budget"] = float(args["total_budget"])
    if args.get("budget_change_pct") is not None:
        kw["budget_change_pct"] = float(args["budget_change_pct"])
    if args.get("channel_bounds"):
        kw["bounds"] = args["channel_bounds"]
    return kw


def _map_save_model(args: dict) -> dict | None:
    name = args.get("name")
    return {"name": str(name)} if name else None


# tool name → (model_ops op name, arg mapper, extra kwargs EXPRESSION string).
# The mapper replicates the small transforms the agent tool does before
# dispatching (json.loads of string args, key renames); returning None demotes
# the call to a comment. The extra expression is appended verbatim to the call
# for kwargs that must reference script variables (spec/dataset_path).
# KEEP IN SYNC with the tool bodies in agents/tools.py + agents/causal_tools.py
# — a renamed tool/op or changed transform must be mirrored here or the export
# silently demotes (or misrenders) that step.
_OP_TOOLS: dict[str, tuple[str, Callable[[dict], dict | None], str | None]] = {
    "get_roi_metrics": ("roi_metrics", _map_no_args, None),
    "validate_model": ("validate_model", _map_no_args, None),
    "get_estimands": ("compute_estimands", _map_get_estimands, None),
    "get_component_decomposition": ("component_decomposition", _map_no_args, None),
    "get_model_diagnostics": ("model_diagnostics", _map_no_args, None),
    "get_adstock_weights": ("adstock_weights", _map_no_args, None),
    "get_saturation_curves": ("saturation_curves", _map_no_args, None),
    "run_posterior_predictive_checks": (
        "posterior_predictive_checks",
        _map_no_args,
        None,
    ),
    "run_residual_diagnostics": ("residual_diagnostics", _map_no_args, None),
    "run_channel_diagnostics": ("channel_diagnostics", _map_no_args, None),
    "run_refutation_suite": ("refutation_suite", _map_no_args, None),
    "run_cross_validation": ("cross_validation", _map_no_args, None),
    "run_calibration_check": (
        "simulation_based_calibration",
        _map_calibration_check,
        "spec=spec, dataset_path=dataset_path",
    ),
    "run_coverage_check": (
        "recovery_coverage_check",
        _map_coverage_check,
        "spec=spec, dataset_path=dataset_path",
    ),
    "prior_predictive_check": (
        "prior_predictive_check",
        _map_prior_predictive,
        "spec=spec, dataset_path=dataset_path",
    ),
    "leave_one_out_decomposition": ("leave_one_out", _map_leave_one_out, None),
    "run_budget_scenario": ("budget_scenario", _map_budget_scenario, None),
    "run_marginal_analysis": ("marginal_analysis", _map_marginal_analysis, None),
    "run_budget_optimizer": ("optimize_budget", _map_budget_optimizer, None),
    "save_fitted_model": ("save_model", _map_save_model, None),
}

# Tier-1 EDA tools rendered against the kernel-free mmm_framework.eda engine.
_EDA_TOOLS = frozenset({"run_eda", "validate_data", "detect_outliers"})

# Ops that legitimately run BEFORE any fit (allow_unfitted: they build an
# unfitted model from spec + dataset). Pre-fit they replay against
# `session_spec` (the session's first fitted spec) instead of being demoted.
_PREFIT_OPS = frozenset(
    {"prior_predictive_check", "run_calibration_check", "run_coverage_check"}
)

# Mirrors agents/eda_tools.py::EDA_ANALYSES (not imported — eda_tools pulls
# langchain at module scope and this module must stay import-light for the
# kernel startup reuse). Keep the two tuples in sync.
_EDA_ANALYSES = (
    "profile",
    "distributions",
    "correlation",
    "collinearity",
    "spend_share",
    "seasonality",
    "kpi_vs_media",
    "stationarity",
)

_MODEL_REF_RE = re.compile(r"\b(mmm|results)\b")


# ── section renderers ─────────────────────────────────────────────────────────


def _spec_literal(spec: dict) -> str:
    """The spec as JSON text safe to embed inside r''' ... ''' (single quotes
    become \\u0027 escapes, which json.loads decodes back)."""
    return json.dumps(spec, indent=2, default=str).replace("'", "\\u0027")


def _add_panel_build(add: Callable[[str], None]) -> None:
    """Emit the panel rebuild for a model reload. Extended-flavor saves load
    panel-free (their arrays ride the pickle), so a panel-build failure falls
    back to None — a CORE save with panel=None then raises a clear TypeError
    from MMMSerializer.load instead of silently mis-loading."""
    add("try:")
    add("    panel = load_mff(dataset_path, _mff_config_from_spec(spec))")
    add("except Exception:")
    add("    panel = None  # extended-flavor saves reload panel-free")


def _render_fit(
    add: Callable[[str], None],
    ev: ToolEvent,
    run_payload: dict | None,
    *,
    have_dataset: bool,
    pulled_in: bool = False,
    dataset_ctx: dict | None = None,
) -> bool:
    """Render one fit_mmm_model call. Returns True when a runnable model
    section (defining ``mmm``/``results``/``spec``) was emitted.

    ``dataset_ctx`` tracks the currently-bound dataset basename across fit
    sections — a session may fit against several datasets, and each fit must
    replay against ITS dataset, not the last one loaded.
    """
    run_name = _comment_safe((run_payload or {}).get("run_name") or "model")
    note = " (pulled in from an earlier turn so this slice runs)" if pulled_in else ""
    add(f"# ── Fit (fit_mmm_model → {run_name}){note} " + "─" * 10)
    if _fit_errored(ev):
        add("# NOTE: this fit FAILED when it ran in the session — kept as a record.")
        preview = _comment_block(ev.result_text, max_lines=6, label="Error")
        if preview:
            add(preview)
        return False
    if not have_dataset:
        add("# NOTE: no dataset file could be resolved for this session, so the fit")
        add("#       cannot be reproduced standalone. Copy the dataset next to this")
        add("#       script, set `dataset_path`, and re-export.")
        return False

    run_ds = (run_payload or {}).get("dataset_path")
    if dataset_ctx is not None and run_ds:
        run_base = os.path.basename(str(run_ds))
        if run_base and run_base != dataset_ctx.get("base"):
            reader = (
                "read_parquet" if run_base.lower().endswith(".parquet") else "read_csv"
            )
            add("# This fit ran against a DIFFERENT dataset file — rebind it")
            add(f"# (session path was: {_comment_safe(run_ds)}).")
            add(f"dataset_path = {run_base!r}")
            add(f"df = pd.{reader}(dataset_path)")
            dataset_ctx["base"] = run_base

    spec = (run_payload or {}).get("spec")
    model_path = (run_payload or {}).get("model_path")
    if isinstance(spec, dict) and spec:
        if spec.get("garden_ref"):
            add(
                "# WARNING: this fit used a bespoke Model-Garden model "
                f"({spec.get('garden_ref')!r})."
            )
            add(
                "#          Its source file lives in the agent workspace — copy it next"
            )
            add("#          to this script / install it before running this section.")
        add("spec = json.loads(")
        add(f"    r'''{_spec_literal(spec)}'''")
        add(")")
        add("from mmm_framework.agents.fitting import build_and_fit")
        add("")
        add("# build_and_fit replays the session's fit exactly — panel build, priors,")
        add("# experiment calibration, and the sampler settings in spec['inference']")
        add("# (method/chains/draws/tune/target_accept/seed). It auto-saves the run")
        add("# under ./mmm_models/.")
        add("mmm, results, fit_info = build_and_fit(spec, dataset_path)")
        if model_path:
            add("# Fast alternative — reload the model this session already saved")
            add("# instead of refitting (keep the saved run directory alongside;")
            add("# extended-flavor saves load panel-free — pass panel=None):")
            add("# from mmm_framework.serialization import MMMSerializer")
            add("# from mmm_framework.agents.fitting import _mff_config_from_spec")
            add("# panel = load_mff(dataset_path, _mff_config_from_spec(spec))")
            add(f"# mmm = MMMSerializer.load({str(model_path)!r}, panel)")
            add("# results = None  # the reload restores the model + posterior trace")
        return True
    if model_path:
        meta_path = str(model_path).rstrip("/") + "/run_metadata.json"
        add("# The session artifact carried no spec — reload the saved run instead")
        add(f"# (keep the {str(model_path)!r} directory alongside this script).")
        add("from mmm_framework.serialization import MMMSerializer")
        add("from mmm_framework.agents.fitting import _mff_config_from_spec")
        add("")
        add(f"with open({meta_path!r}) as _f:")
        add('    spec = json.load(_f)["spec"]')
        _add_panel_build(add)
        add(f"mmm = MMMSerializer.load({str(model_path)!r}, panel)")
        add("results = None  # the reload restores the model + posterior trace")
        return True
    add("# NOTE: no saved model_run could be matched to this fit — not exportable.")
    preview = _comment_block(ev.result_text, max_lines=6)
    if preview:
        add(preview)
    return False


def _render_cell(
    add: Callable[[str], None],
    code: str,
    out: dict,
    ev: ToolEvent | None,
    include_output_previews: bool,
) -> None:
    if out.get("is_error") or (ev is not None and ev.is_error):
        add("# NOTE: this cell raised an error when it ran in the session")
        add("#       (kept verbatim for fidelity — it may need fixing to re-run).")
    add(code if code else "# (empty cell)")
    if include_output_previews and out.get("stdout"):
        preview = _comment_block(out["stdout"])
        if preview:
            add(preview)


# The variable roles for EDA: the current fit spec when one has been defined,
# else session_spec (the session's first fitted spec — closest stand-in for
# the working spec the session's EDA actually used; None → heuristic roles).
_EDA_PANEL_LINE = (
    "panel = eda.load_eda_panel("
    'dataset_path, spec if "spec" in globals() else session_spec)'
)


def _add_series_loop(
    add: Callable[[str], None], var_list_expr: str, indent: str = ""
) -> None:
    """Emit the per-variable period-axis series aggregation run_eda uses."""
    add(f"{indent}for _var in {var_list_expr}:")
    add(f"{indent}    _series = panel.df_wide[_var].astype(float)")
    add(f"{indent}    if panel.dims:")
    add(
        f"{indent}        "
        "_series = _series.groupby(level=panel.date_col).sum(min_count=1)"
    )


def _render_run_eda(add: Callable[[str], None], ev: ToolEvent) -> None:
    analyses = ev.args.get("analyses")
    requested = [a for a in (analyses or list(_EDA_ANALYSES)) if a in _EDA_ANALYSES]
    add("# ── EDA (run_eda) " + "─" * 40)
    add(_EDA_PANEL_LINE)
    if "profile" in requested:
        add("print(eda.profile_panel(panel).to_string())")
    if "distributions" in requested:
        add("try:")
        add("    from mmm_framework.eda.charts import fig_distributions")
        add("")
        add("    fig_distributions(panel).show()")
        add("except Exception as _exc:")
        add('    print("distributions chart failed:", _exc)')
    if "correlation" in requested or "collinearity" in requested:
        add("coll = eda.collinearity_analysis(panel)")
        if "correlation" in requested:
            add('print(coll["correlation"].round(2).to_string())')
        if "collinearity" in requested:
            add('print("VIF:", coll["vif"])')
            add('print("High VIF (weakly identified):", coll["high_vif"])')
    if "spend_share" in requested:
        add("share = eda.spend_share(panel)")
        add('print("Spend totals:", share["totals"])')
        add('print("Spend shares:", share["shares"], "HHI:", share["hhi"])')
    if "seasonality" in requested:
        add("try:")
        add("    from mmm_framework.eda.charts import fig_decomposition")
        add("")
        add("    _period = eda.seasonal_period_for_freq(panel.freq)")
        _add_series_loop(add, "[v for v in [panel.kpi] if v]", indent="    ")
        add("        _dec = eda.decompose_series(_series, _period, variable=_var)")
        add('        print(_var, "trend strength", _dec.trend_strength,')
        add('              "seasonal strength", _dec.seasonal_strength)')
        add("        fig_decomposition(_dec).show()")
        add("except Exception as _exc:")
        add('    print("seasonality decomposition failed:", _exc)')
    if "kpi_vs_media" in requested:
        add("try:")
        add("    from mmm_framework.eda.charts import fig_kpi_vs_media")
        add("")
        add("    if panel.kpi and panel.media:")
        add("        fig_kpi_vs_media(panel).show()")
        add("except Exception as _exc:")
        add('    print("kpi_vs_media chart failed:", _exc)')
    if "stationarity" in requested:
        _add_series_loop(add, "[v for v in [panel.kpi, *panel.media] if v]")
        add("    print(_var, eda.stationarity_tests(_series))")


def _render_validate_data(add: Callable[[str], None], ev: ToolEvent) -> None:
    add("# ── Data validation (validate_data) " + "─" * 22)
    add(_EDA_PANEL_LINE)
    add("report = eda.validate_dataset(panel)")
    add('print("Data validation:", "PASSED" if report.passed else "FAILED")')
    add("for _issue in report.issues:")
    add(
        '    print(f"[{_issue.severity}] {_issue.check} '
        "{_issue.variable or ''}: {_issue.message}\")"
    )


def _render_detect_outliers(add: Callable[[str], None], ev: ToolEvent) -> None:
    sensitivity = str(ev.args.get("sensitivity") or "default")
    variables = ev.args.get("variables")
    add("# ── Outlier detection (detect_outliers) " + "─" * 18)
    add(_EDA_PANEL_LINE)
    call = (
        "outlier_report = eda.detect_outliers("
        f"panel, eda.OutlierConfig.for_sensitivity({sensitivity!r})"
    )
    if variables:
        call += f", {list(variables)!r}"
    add(call + ")")
    add("for _flag in outlier_report.flags:")
    add("    print(_flag)")
    add("for _action in outlier_report.actions or []:")
    add("    print(_action)")


def _render_op(
    add: Callable[[str], None],
    ev: ToolEvent,
    *,
    have_model: bool,
    have_session_spec: bool,
    include_output_previews: bool,
) -> None:
    op_name, mapper, extra = _OP_TOOLS[ev.name]
    add(f"# ── {_comment_safe(ev.name)} → model_ops.{op_name} " + "─" * 20)
    prefit = ev.name in _PREFIT_OPS and not have_model
    if not have_model and not (prefit and have_session_spec):
        if ev.name in _PREFIT_OPS:
            add("# NOTE: this pre-fit check ran against the session's working spec,")
            add("#       which the export cannot reconstruct here — kept as a record.")
        else:
            add("# NOTE: no fitted model at this point in the session — this tool")
            add("#       call cannot run standalone; kept as a record only.")
        if ev.args:
            add(f"#   args: {_truncate(json.dumps(ev.args, default=str))}")
        return
    try:
        kwargs = mapper(ev.args or {})
        if kwargs is not None:
            # repr() must round-trip as a Python literal — json.loads accepts
            # NaN/Infinity, whose reprs are bare names that would NameError.
            ast.literal_eval(repr(kwargs))
    except Exception:
        kwargs = None
    if kwargs is None:
        add("# NOTE: this call's arguments could not be reconstructed — kept as a")
        add("#       record only.")
        add(f"#   args: {_truncate(json.dumps(ev.args, default=str))}")
        return
    if prefit:
        add("# Pre-fit check, replayed against session_spec (the spec of this")
        add("# session's first fit — the working spec at the time may have differed).")
        parts = ["None", "None"]
    else:
        parts = ["mmm", "results"]
    parts += [f"{k}={v!r}" for k, v in kwargs.items()]
    if extra:
        parts.append(
            extra.replace("spec=spec", "spec=session_spec") if prefit else extra
        )
    add(f"res_{op_name} = run_op(model_ops.{op_name}, {', '.join(parts)})")
    if include_output_previews and ev.result_text:
        preview = _comment_block(ev.result_text, max_lines=8)
        if preview:
            add(preview)


def _render_comment_tool(
    add: Callable[[str], None], ev: ToolEvent, include_output_previews: bool
) -> None:
    add(f"# ── Tool call (not exportable as code): {_comment_safe(ev.name)} " + "─" * 6)
    if ev.args:
        add(f"#   args: {_truncate(json.dumps(ev.args, default=str))}")
    if include_output_previews and ev.result_text:
        preview = _comment_block(ev.result_text, max_lines=6)
        if preview:
            add(preview)


# ── turn slicing / scope ──────────────────────────────────────────────────────


def _split_turns(
    timeline: list[tuple[str, Any]],
) -> list[tuple[str | None, list[ToolEvent]]]:
    """Group the timeline into (question, tool events) turns. Events before the
    first question (rare) get a question-less leading turn."""
    turns: list[tuple[str | None, list[ToolEvent]]] = []
    current_q: str | None = None
    current_events: list[ToolEvent] = []
    started = False
    for kind, item in timeline:
        if kind == "question":
            if started:
                turns.append((current_q, current_events))
            current_q, current_events, started = item, [], True
        else:
            started = True
            current_events.append(item)
    if started:
        turns.append((current_q, current_events))
    return turns


def _select_turns(turns: list, scope: str) -> list[int]:
    """Turn indices selected by ``scope``: "all" | "last" | "turn:<k>" (1-based).
    Anything invalid selects everything (never raises)."""
    n_turns = len(turns)
    s = str(scope or "all").strip().lower()
    if s == "last" and n_turns:
        # The last turn WITH tool activity — a trailing prose-only turn
        # ("thanks, looks good") must not export an empty script.
        for i in range(n_turns - 1, -1, -1):
            if turns[i][1]:
                return [i]
        return [n_turns - 1]
    if s.startswith("turn:"):
        try:
            k = int(s.split(":", 1)[1])
        except (ValueError, IndexError):
            return list(range(n_turns))
        if 1 <= k <= n_turns:
            return [k - 1]
    return list(range(n_turns))


def _event_needs_model(ev: ToolEvent, cell_code: str | None) -> bool:
    if ev.name in _OP_TOOLS:
        return True
    if ev.name == "execute_python":
        code = cell_code if cell_code is not None else str(ev.args.get("code") or "")
        return bool(_MODEL_REF_RE.search(code))
    return False


# ── main entry point ──────────────────────────────────────────────────────────


def build_session_script(
    thread_id: str,
    *,
    include_output_previews: bool = True,
    messages: list | None = None,
    scope: str = "all",
) -> str:
    """Build a standalone, runnable Python script reproducing the session's work.

    Args:
        thread_id: the session/thread to export.
        include_output_previews: append each cell's/tool's captured output as a
            short commented block (a lab-notebook feel); set False for code only.
        messages: the thread's checkpointed LangGraph messages. When provided,
            tool calls (fits, model ops, EDA) are reconstructed as runnable code
            in conversation order; ``None``/empty falls back to the artifact-only
            layout (synthetic/legacy threads).
        scope: ``"all"`` (default), ``"last"`` (final user turn onward), or
            ``"turn:<k>"`` (1-based). Invalid values fall back to ``"all"``.
    """
    arts = _sessions.list_artifacts(thread_id)
    session = _sessions.get_session(thread_id) or {}
    name = session.get("name") or thread_id

    code_cells = [a for a in arts if a.get("kind") == "code_snippet"]
    cells_by_call_id = {
        a["payload"].get("call_id"): a["payload"]
        for a in code_cells
        if a.get("payload", {}).get("call_id")
    }
    outputs = {
        a["payload"].get("call_id"): a["payload"]
        for a in arts
        if a.get("kind") == "text_output"
    }
    model_runs = [a for a in arts if a.get("kind") == "model_run"]
    run_payloads = [a["payload"] for a in model_runs]
    latest_run = run_payloads[-1] if run_payloads else None
    dataset_path = _resolve_dataset(thread_id, run_payloads)

    timeline = extract_timeline(messages) if messages else []

    lines: list[str] = []
    add = lines.append

    # ── Header ──────────────────────────────────────────────────────────────
    add('"""')
    add(f"MMM session export — {_docstring_safe(name)}")
    add(f"thread_id: {_docstring_safe(thread_id)}")
    add(f"generated: {datetime.now(timezone.utc).isoformat()}")
    if str(scope or "all").strip().lower() not in ("", "all"):
        add(f"scope: {_docstring_safe(scope)}")
    add("")
    add("Reproduces the analysis done by the agent in this session. The PREAMBLE")
    add("below reconstitutes state the agent's TOOLS injected (the dataset as")
    add("`df`, the fitted model, helpers) — those were never written as code.")
    if timeline:
        add("The session's work then replays in conversation order: model fits")
        add("become build_and_fit(spec, dataset_path) with the exact fitted spec")
        add("embedded (sampler settings live in spec['inference']; build_and_fit")
        add("auto-saves each run to ./mmm_models), interpretation tools become")
        add("mmm_framework.agents.model_ops calls, EDA tools become")
        add("mmm_framework.eda calls, and python cells run verbatim. Tool calls")
        add("with no code equivalent are kept as commented records.")
    else:
        add("The CELLS then run in execution order, exactly as they did in the")
        add("session's stateful kernel.")
    add('"""')
    add("")
    add(_PREAMBLE_IMPORTS)

    # ── Dataset ─────────────────────────────────────────────────────────────
    if dataset_path:
        base = os.path.basename(str(dataset_path))
        reader = "read_parquet" if base.lower().endswith(".parquet") else "read_csv"
        add("# Dataset the session worked on. Place this file next to the script")
        add(f"# (session path was: {_comment_safe(dataset_path)}).")
        add(f"dataset_path = {base!r}")
        add(f"df = pd.{reader}(dataset_path)")
        add("")

    if not timeline:
        _append_artifact_only_body(
            add,
            code_cells=code_cells,
            outputs=outputs,
            latest_run=latest_run,
            have_dataset=bool(dataset_path),
            include_output_previews=include_output_previews,
        )
        return "\n".join(lines).rstrip("\n") + "\n"

    # ── Helpers ─────────────────────────────────────────────────────────────
    add(_PREAMBLE_HELPERS)

    # ── Conversation replay ─────────────────────────────────────────────────
    fit_events = [
        ev for kind, ev in timeline if kind == "tool" and ev.name == "fit_mmm_model"
    ]
    fit_runs = _match_fit_runs(fit_events, run_payloads)

    # session_spec: the first fitted spec — pre-fit sections (EDA variable
    # roles, prior-predictive/SBC checks) replay against it as the closest
    # reconstructable stand-in for the session's working spec at that moment.
    session_spec: dict | None = None
    if dataset_path:
        for ev in fit_events:
            payload = fit_runs.get(id(ev)) or {}
            if isinstance(payload.get("spec"), dict) and payload["spec"]:
                session_spec = payload["spec"]
                break
        if session_spec is None and latest_run:
            spec_candidate = latest_run.get("spec")
            if isinstance(spec_candidate, dict) and spec_candidate:
                session_spec = spec_candidate
    add("")
    if session_spec is not None:
        add("# The session's model spec (from its first fit). Pre-fit sections —")
        add("# EDA variable roles, prior checks — use it as the closest stand-in")
        add("# for the working spec at that point in the conversation.")
        add("session_spec = json.loads(")
        add(f"    r'''{_spec_literal(session_spec)}'''")
        add(")")
    else:
        add("session_spec = None  # no fitted spec was recorded for this session")

    dataset_ctx = (
        {"base": os.path.basename(str(dataset_path))} if dataset_path else None
    )

    turns = _split_turns(timeline)
    selected = _select_turns(turns, scope)
    selected_set = set(selected)
    if len(selected) != len(turns):
        add("")
        add(f"# NOTE: scope-limited export — {len(turns) - len(selected)} other")
        add("#       turn(s) omitted. Export with scope=all for the full session.")

    # A slice that references the model pulls in the most recent prior fit so
    # the exported code still runs.
    pulled_fit: ToolEvent | None = None
    if selected and selected[0] > 0:
        slice_events = [ev for i in selected for ev in turns[i][1]]
        has_fit = any(
            ev.name == "fit_mmm_model" and not _fit_errored(ev) for ev in slice_events
        )
        needs_model = any(
            _event_needs_model(ev, (cells_by_call_id.get(ev.call_id) or {}).get("code"))
            for ev in slice_events
        )
        if needs_model and not has_fit:
            for i in range(selected[0] - 1, -1, -1):
                for ev in reversed(turns[i][1]):
                    if ev.name == "fit_mmm_model" and not _fit_errored(ev):
                        pulled_fit = ev
                        break
                if pulled_fit is not None:
                    break

    have_model = False
    if pulled_fit is not None:
        add("")
        have_model = _render_fit(
            add,
            pulled_fit,
            fit_runs.get(id(pulled_fit)),
            have_dataset=bool(dataset_path),
            pulled_in=True,
            dataset_ctx=dataset_ctx,
        )

    seen_exec_call_ids: set[str] = {
        ev.call_id
        for kind, ev in timeline
        if kind == "tool" and ev.name == "execute_python" and ev.call_id
    }

    for turn_no, (question, events) in enumerate(turns, 1):
        if (turn_no - 1) not in selected_set:
            continue
        q = _comment_safe(question or "(session start)")
        if len(q) > 64:
            q = q[:61] + "..."
        add("")
        add("# " + "═" * 70)
        add(f'# Turn {turn_no} — "{q}"')
        add("# " + "═" * 70)
        if not events:
            add("# (no tool activity this turn)")
        for ev in events:
            add("")
            if ev.name == "fit_mmm_model":
                # (pulled_fit always comes from an UNselected earlier turn —
                # _select_turns returns either all turns or a single one — so a
                # double render is impossible here.)
                ok = _render_fit(
                    add,
                    ev,
                    fit_runs.get(id(ev)),
                    have_dataset=bool(dataset_path),
                    dataset_ctx=dataset_ctx,
                )
                have_model = have_model or ok
            elif ev.name == "execute_python":
                cell = cells_by_call_id.get(ev.call_id) or {}
                code = (cell.get("code") or ev.args.get("code") or "").rstrip("\n")
                out = outputs.get(ev.call_id, {})
                add("# ── Cell (execute_python) " + "─" * 32)
                if not have_model and _MODEL_REF_RE.search(code or ""):
                    add("# NOTE: this cell uses `mmm`/`results`, which the export")
                    add("#       could not reconstruct above (no runnable fit at this")
                    add("#       point) — it will NameError until that is repaired.")
                _render_cell(add, code, out, ev, include_output_previews)
            elif ev.name in _EDA_TOOLS:
                if not dataset_path:
                    add(
                        "# ── Tool call (not exportable as code): "
                        f"{_comment_safe(ev.name)} " + "─" * 6
                    )
                    add("# NOTE: no dataset file could be resolved for this session.")
                    continue
                if ev.name == "run_eda":
                    _render_run_eda(add, ev)
                elif ev.name == "validate_data":
                    _render_validate_data(add, ev)
                else:
                    _render_detect_outliers(add, ev)
            elif ev.name in _OP_TOOLS:
                _render_op(
                    add,
                    ev,
                    have_model=have_model,
                    have_session_spec=session_spec is not None,
                    include_output_previews=include_output_previews,
                )
            else:
                _render_comment_tool(add, ev, include_output_previews)

    # ── Appendix: cells whose call_ids the checkpoint no longer carries ─────
    orphan_cells = [
        a
        for a in code_cells
        if a.get("payload", {}).get("call_id") not in seen_exec_call_ids
    ]
    if orphan_cells:
        add("")
        add("# " + "═" * 70)
        add("# Appendix — cells not present in the conversation checkpoint")
        add("# (kept so no captured work is lost; run order within the appendix")
        add("#  follows capture order)")
        add("# " + "═" * 70)
        for i, cell in enumerate(orphan_cells, 1):
            payload = cell.get("payload", {})
            code = (payload.get("code") or "").rstrip("\n")
            out = outputs.get(payload.get("call_id"), {})
            add("")
            add(f"# ───────────────────────── Appendix[{i}] ─────────────────────────")
            _render_cell(add, code, out, None, include_output_previews)

    return "\n".join(lines).rstrip("\n") + "\n"


def _append_artifact_only_body(
    add: Callable[[str], None],
    *,
    code_cells: list[dict],
    outputs: dict,
    latest_run: dict | None,
    have_dataset: bool,
    include_output_previews: bool,
) -> None:
    """Today's artifact-only layout (threads without checkpoints): model reload
    preamble + helpers + every captured cell in order."""
    # ── Fitted model ────────────────────────────────────────────────────────
    if latest_run and latest_run.get("model_path"):
        model_path = str(latest_run["model_path"])
        add("# Fitted model auto-saved by fit_mmm_model in this session. Keep the")
        add(f"# '{_comment_safe(model_path)}' directory alongside this script.")
        if have_dataset:
            add("from mmm_framework.serialization import MMMSerializer")
            add("from mmm_framework.agents.fitting import _mff_config_from_spec")
            add("")
            spec = latest_run.get("spec")
            if isinstance(spec, dict) and spec:
                add("spec = json.loads(")
                add(f"    r'''{_spec_literal(spec)}'''")
                add(")")
            else:
                meta_path = model_path.rstrip("/") + "/run_metadata.json"
                add(f"with open({meta_path!r}) as _f:")
                add('    spec = json.load(_f)["spec"]')
            _add_panel_build(add)
            add(f"mmm = MMMSerializer.load({model_path!r}, panel)")
            add("results = None  # the reload restores the model + posterior trace")
        else:
            add("# NOTE: no dataset file could be resolved for this session — a core")
            add("#       model reload needs the panel rebuilt from the dataset. Copy")
            add("#       the dataset next to this script, load it as `df`, then:")
            add("#   from mmm_framework.serialization import MMMSerializer")
            add("#   from mmm_framework.agents.fitting import _mff_config_from_spec")
            add(
                f"#   with open({(model_path.rstrip('/') + '/run_metadata.json')!r}) as _f:"
            )
            add('#       spec = json.load(_f)["spec"]')
            add("#   panel = load_mff(dataset_path, _mff_config_from_spec(spec))")
            add(f"#   mmm = MMMSerializer.load({model_path!r}, panel)")
        add("")

    # ── Helpers ─────────────────────────────────────────────────────────────
    add(_PREAMBLE_HELPERS)

    # ── Cells ───────────────────────────────────────────────────────────────
    add("")
    add("# " + "═" * 70)
    add("# Session cells (execute_python, in order)")
    add("# " + "═" * 70)
    if not code_cells:
        add("")
        add("# (this session ran no execute_python cells)")
    for i, cell in enumerate(code_cells, 1):
        payload = cell.get("payload", {})
        code = (payload.get("code") or "").rstrip("\n")
        out = outputs.get(payload.get("call_id"), {})
        add("")
        add(f"# ───────────────────────── In[{i}] ─────────────────────────")
        _render_cell(add, code, out, None, include_output_previews)
