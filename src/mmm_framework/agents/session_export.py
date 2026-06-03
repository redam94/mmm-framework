"""Export a chat session's Python work as a standalone, runnable script.

The agent's ``execute_python`` runs in a STATEFUL warm kernel where ``df``,
``mmm``, ``results``, the framework convenience names, and the
``save_result``/``load_result`` helpers are *injected* — they are never written
as code. Several state-producing steps (``configure_model``, ``fit_mmm_model``,
``generate_synthetic_data``) are TOOL calls that emit no ``code_snippet`` at all.
So a naive concatenation of the captured cells would ``NameError`` when run as a
script ("works in the notebook, breaks as a file").

This generator therefore synthesizes a PREAMBLE that reconstitutes the injected
state — import the framework surface the kernel pre-binds, load the dataset as
``df``, load the fitted model from disk as ``mmm``/``results``, and define the
``save_result``/``load_result`` helpers — and only then appends every cell in
execution order. Cells that errored when they ran are MARKED, not dropped (so no
work is lost and the script doesn't lie about what succeeded). The result is a
real, portable reproduction of the session's analysis.

Data backing it already exists: ``execute_python`` code is persisted as ordered
``code_snippet`` artifacts, paired to ``text_output`` by ``call_id``, and fits
record a ``model_run`` artifact with ``model_path``/``dataset_path``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from mmm_framework.api import sessions as _sessions

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
'''


def _comment_block(text: str, max_lines: int = 12) -> str:
    """Render `text` as a truncated, commented preview block."""
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return ""
    shown = lines[:max_lines]
    out = ["#   " + ln for ln in shown]
    if len(lines) > max_lines:
        out.append(f"#   ... ({len(lines) - max_lines} more line(s))")
    return "# Output:\n" + "\n".join(out)


def _resolve_dataset(thread_id: str, latest_run: dict | None) -> str | None:
    """The dataset the session worked on — from the latest fit, else a file."""
    if latest_run and latest_run.get("dataset_path"):
        return latest_run["dataset_path"]
    try:
        for f in _sessions.list_files(thread_id):
            name = str(f.get("name", "")).lower()
            if name.endswith((".csv", ".parquet")):
                return f.get("path")
    except Exception:
        pass
    return None


def build_session_script(
    thread_id: str, *, include_output_previews: bool = True
) -> str:
    """Build a standalone, runnable Python script reproducing the session's work.

    Args:
        thread_id: the session/thread to export.
        include_output_previews: append each cell's captured stdout as a short
            commented block (a lab-notebook feel); set False for code only.
    """
    arts = _sessions.list_artifacts(thread_id)
    session = _sessions.get_session(thread_id) or {}
    name = session.get("name") or thread_id

    code_cells = [a for a in arts if a.get("kind") == "code_snippet"]
    outputs = {
        a["payload"].get("call_id"): a["payload"]
        for a in arts
        if a.get("kind") == "text_output"
    }
    model_runs = [a for a in arts if a.get("kind") == "model_run"]
    latest_run = model_runs[-1]["payload"] if model_runs else None
    dataset_path = _resolve_dataset(thread_id, latest_run)

    lines: list[str] = []
    add = lines.append

    # ── Header ──────────────────────────────────────────────────────────────
    add('"""')
    add(f"MMM session export — {name}")
    add(f"thread_id: {thread_id}")
    add(f"generated: {datetime.now(timezone.utc).isoformat()}")
    add("")
    add("Reproduces the Python analysis done by the agent in this session. The")
    add("PREAMBLE below reconstitutes state the agent's TOOLS injected (the")
    add("dataset as `df`, the fitted model as `mmm`/`results`, helpers) — those")
    add("were never written as code — and the CELLS then run in execution order,")
    add("exactly as they did in the session's stateful kernel.")
    add('"""')
    add("")
    add(_PREAMBLE_IMPORTS)

    # ── Dataset ─────────────────────────────────────────────────────────────
    if dataset_path:
        base = os.path.basename(str(dataset_path))
        reader = "read_parquet" if base.lower().endswith(".parquet") else "read_csv"
        add("# Dataset the session worked on. Place this file next to the script")
        add(f"# (session path was: {dataset_path}).")
        add(f"dataset_path = {base!r}")
        add(f"df = pd.{reader}(dataset_path)")
        add("")

    # ── Fitted model ────────────────────────────────────────────────────────
    if latest_run and latest_run.get("model_path"):
        add("# Fitted model auto-saved by fit_mmm_model in this session. Keep the")
        add(f"# '{latest_run['model_path']}' directory alongside this script.")
        add("from mmm_framework import MMMSerializer")
        add(f"mmm, results = MMMSerializer().load({latest_run['model_path']!r})")
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
        code = (cell.get("payload", {}).get("code") or "").rstrip("\n")
        out = outputs.get(cell.get("payload", {}).get("call_id"), {})
        add("")
        add(f"# ───────────────────────── In[{i}] ─────────────────────────")
        if out.get("is_error"):
            add("# NOTE: this cell raised an error when it ran in the session")
            add("#       (kept verbatim for fidelity — it may need fixing to re-run).")
        add(code if code else "# (empty cell)")
        if include_output_previews and out.get("stdout"):
            preview = _comment_block(out["stdout"])
            if preview:
                add(preview)

    return "\n".join(lines).rstrip("\n") + "\n"
