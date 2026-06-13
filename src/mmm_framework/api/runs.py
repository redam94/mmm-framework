"""MLflow-style run tracking over the existing model_run artifacts.

Every fit already persists a ``model_run`` artifact carrying the full
normalized spec, the dataset path, and the summary; ``fit_mmm_model``
additionally stamps a dataset fingerprint, a spec hash, the parent run id, and
a snapshot of the assumption stack at fit time. This module materializes that
into a LINEAGE timeline: for each run, what changed vs its predecessor (spec
leaves, dataset, assumptions) — the provenance needed to audit the process or
write the final report.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from mmm_framework.agents.spec_locks import flatten_leaves

from . import sessions as sessions_store


def data_fingerprint(path: str | None) -> dict[str, Any] | None:
    """Cheap content identity for a dataset file: md5 + size + row count."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    n_lines = 0
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                n_lines += chunk.count(b"\n")
        return {
            "md5": h.hexdigest()[:12],
            "size_bytes": os.path.getsize(path),
            "n_rows": max(0, n_lines - 1),  # minus header
            "path": str(path),
        }
    except Exception:
        return None


def spec_hash(spec: dict | None) -> str | None:
    if not spec:
        return None
    return hashlib.md5(
        json.dumps(spec, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _spec_leaf_diff(old_spec: dict | None, new_spec: dict | None) -> list[dict]:
    """Changed/added/removed spec leaves between two runs, as
    ``{path, old, new}`` (None = absent)."""
    old_leaves = flatten_leaves(old_spec or {})
    new_leaves = flatten_leaves(new_spec or {})
    changes = []
    for path in sorted(set(old_leaves) | set(new_leaves)):
        old_v, new_v = old_leaves.get(path), new_leaves.get(path)
        if old_v != new_v:
            changes.append({"path": path, "old": old_v, "new": new_v})
    return changes


def _assumptions_delta(
    old_assumptions: list[dict] | None, new_assumptions: list[dict] | None
) -> list[dict]:
    """Assumptions added or re-versioned between two runs (the rationale layer
    of the lineage: WHY the spec changed)."""
    old_v = {a.get("key"): a.get("version") for a in (old_assumptions or [])}
    delta = []
    for a in new_assumptions or []:
        key = a.get("key")
        if key not in old_v:
            delta.append({**a, "change": "added"})
        elif a.get("version") != old_v[key]:
            delta.append({**a, "change": "revised"})
    return delta


def build_run_timeline(project_id: str | None = None) -> list[dict[str, Any]]:
    """All model runs (newest first), each annotated with its diff vs the
    chronologically previous run in the same scope."""
    sessions = sessions_store.list_sessions(project_id=project_id)
    runs: list[dict] = []
    for s in sessions:
        tid = s["thread_id"]
        for art in sessions_store.list_artifacts(tid):
            if art.get("kind") != "model_run":
                continue
            p = art.get("payload", {})
            runs.append(
                {
                    "artifact_id": art["id"],
                    "run_id": p.get("run_id"),
                    "run_name": p.get("run_name"),
                    "thread_id": tid,
                    "session_name": s.get("name"),
                    "project_id": s.get("project_id"),
                    "created_at": art.get("created_at"),
                    "timestamp_iso": p.get("timestamp_iso"),
                    "kpi": p.get("kpi"),
                    "channels": p.get("channels", []),
                    "controls": p.get("controls", []),
                    "trend": p.get("trend"),
                    "seasonality": p.get("seasonality"),
                    "inference": p.get("inference"),
                    "n_obs": p.get("n_obs"),
                    "summary": p.get("summary"),
                    "report_path": p.get("report_path"),
                    "model_path": p.get("model_path"),
                    "data_fingerprint": p.get("data_fingerprint"),
                    "diagnostics": p.get("diagnostics"),
                    "spec_hash": p.get("spec_hash"),
                    "parent_run_id": p.get("parent_run_id"),
                    "assumptions": p.get("assumptions") or [],
                    "_spec": p.get("spec"),
                }
            )
    runs.sort(key=lambda r: r.get("created_at") or 0)

    prev: dict | None = None
    for r in runs:
        if prev is None:
            r["changes"] = {
                "spec_changes": [],
                "data_changed": False,
                "assumptions_delta": [
                    {**a, "change": "added"} for a in r["assumptions"]
                ],
                "baseline": True,
            }
        else:
            old_fp = (prev.get("data_fingerprint") or {}).get("md5")
            new_fp = (r.get("data_fingerprint") or {}).get("md5")
            r["changes"] = {
                "spec_changes": _spec_leaf_diff(prev.get("_spec"), r.get("_spec")),
                "data_changed": bool(old_fp != new_fp and (old_fp or new_fp)),
                "assumptions_delta": _assumptions_delta(
                    prev.get("assumptions"), r.get("assumptions")
                ),
                "baseline": False,
            }
        prev = r

    for r in runs:
        r.pop("_spec", None)  # full specs are heavy; the diff carries the info
    runs.reverse()
    return runs


def run_timeline_markdown(
    project_id: str | None = None, max_runs: int | None = None
) -> str:
    """The timeline as markdown — the provenance section of a final report.

    ``max_runs`` keeps only the most recent N runs (used by the agent tool to
    bound how much lands in the LLM's context); ``None`` returns the full
    lineage (used for reports).
    """
    runs = build_run_timeline(project_id)
    if not runs:
        return "No model runs recorded yet."
    omitted = 0
    if max_runs is not None and len(runs) > max_runs:
        # build_run_timeline returns newest-first; keep the most recent N.
        omitted = len(runs) - max_runs
        runs = runs[:max_runs]
    lines = ["### Model Run Lineage", ""]
    if omitted:
        lines.append(
            f"_Showing the {max_runs} most recent runs; {omitted} earlier "
            f"run(s) omitted._"
        )
        lines.append("")
    for r in reversed(runs):  # chronological for a report
        when = r.get("timestamp_iso") or ""
        lines.append(f"**{r.get('run_name') or r.get('run_id')}** ({when})")
        lines.append(
            f"- KPI `{r.get('kpi')}`; channels: {', '.join(r.get('channels') or [])}; "
            f"trend: {r.get('trend')}; n_obs: {r.get('n_obs')}"
        )
        fp = r.get("data_fingerprint") or {}
        if fp:
            lines.append(
                f"- Data: `{fp.get('md5')}` ({fp.get('n_rows'):,} rows)"
                if fp.get("n_rows") is not None
                else f"- Data: `{fp.get('md5')}`"
            )
        ch = r.get("changes") or {}
        if ch.get("baseline"):
            lines.append("- Baseline run (first fit in this scope).")
        else:
            if ch.get("data_changed"):
                lines.append("- ⚠️ Dataset changed vs previous run.")
            for c in ch.get("spec_changes", [])[:12]:
                lines.append(f"  - `{c['path']}`: {c['old']!r} → {c['new']!r}")
            extra = len(ch.get("spec_changes", [])) - 12
            if extra > 0:
                lines.append(f"  - … and {extra} more spec change(s)")
        for a in (ch.get("assumptions_delta") or [])[:8]:
            lines.append(
                f"  - assumption {a.get('change')}: `{a.get('key')}` (v{a.get('version')})"
                + (f" — {a.get('rationale')}" if a.get("rationale") else "")
            )
        lines.append("")
    return "\n".join(lines)
