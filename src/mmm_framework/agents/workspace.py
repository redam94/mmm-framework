"""Scoped workspace directories for agent output + knowledge-base sources.

The agent needs a place to write generated files (reports, CSVs, plots) that is
scoped to the active session, so files can be listed, grepped, and downloaded —
instead of everything landing in the server process CWD with fixed names. The
knowledge base needs a per-project source directory.

Layout (root = ``$MMM_AGENT_WORKSPACE`` or ``<cwd>/agent_workspace``)::

    <root>/threads/<thread_id>/        per-session output dir (execute_python chdir target)
    <root>/projects/<project_id>/kb/   per-project KB source files

``uploads/<thread_id>/`` (the pre-existing dataset-upload dir) is left untouched.

All path construction goes through here so traversal guards and the download
allow-list have a single source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from mmm_framework.agents.runtime import get_current_thread

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]")


def _safe_segment(value: str) -> str:
    """Sanitise a single path segment (thread/project id) — defence in depth."""
    seg = _SAFE_SEGMENT.sub("_", str(value or "").strip()) or "_"
    return seg[:128]


def workspace_root() -> Path:
    root = os.environ.get("MMM_AGENT_WORKSPACE")
    base = Path(root) if root else Path.cwd() / "agent_workspace"
    return base.resolve()


def thread_dir(thread_id: str | None = None) -> Path:
    """Per-session output directory (created on demand)."""
    tid = _safe_segment(thread_id or get_current_thread())
    d = workspace_root() / "threads" / tid
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_kb_dir(project_id: str) -> Path:
    """Per-project knowledge-base source directory (created on demand)."""
    pid = _safe_segment(project_id)
    d = workspace_root() / "projects" / pid / "kb"
    d.mkdir(parents=True, exist_ok=True)
    return d


def uploads_dir(thread_id: str) -> Path:
    """The legacy dataset-upload directory (``uploads/<thread_id>/``)."""
    d = Path("uploads") / _safe_segment(thread_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_store_dir() -> Path:
    """Content-addressed store for captured Plotly figures (shared, dedup'd)."""
    d = workspace_root() / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def store_plot(fig_json: dict) -> str:
    """Write a Plotly figure JSON to the content-addressed store once and return
    its id (a hash of the content). Identical figures collapse to one file, and
    the id can be served with an immutable cache header so the browser caches it
    permanently."""
    payload = json.dumps(fig_json, sort_keys=True, default=str)
    pid = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    path = plot_store_dir() / f"{pid}.json"
    if not path.exists():
        path.write_text(payload, encoding="utf-8")
    return pid


def plot_path(plot_id: str) -> Path | None:
    """Resolve a stored plot id to its on-disk JSON path (or None)."""
    p = plot_store_dir() / f"{_safe_segment(plot_id)}.json"
    return p if p.exists() else None


def allowed_roots() -> list[Path]:
    """Directories a download/read is permitted to touch (resolved, absolute)."""
    roots = [
        workspace_root(),
        (Path.cwd() / "uploads").resolve(),
        (Path.cwd() / "mmm_models").resolve(),
        (Path.cwd() / "mmm_configs").resolve(),
        Path.cwd().resolve(),  # legacy fixed-name reports written to CWD
    ]
    # de-dup while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def is_within(path: str | Path, *roots: Path) -> bool:
    """True if ``path`` resolves inside one of ``roots`` (or the allow-list)."""
    try:
        p = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    candidates = list(roots) or allowed_roots()
    for root in candidates:
        try:
            p.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def safe_join(root: Path, relpath: str) -> Path:
    """Join ``relpath`` onto ``root``, raising ``ValueError`` on traversal."""
    root = root.resolve()
    target = (root / (relpath or "")).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {relpath!r}") from exc
    return target


def snapshot_dir(directory: Path) -> dict[str, float]:
    """Map of file path -> mtime for every file under ``directory`` (recursive)."""
    out: dict[str, float] = {}
    if not directory.exists():
        return out
    for p in directory.rglob("*"):
        if p.is_file():
            try:
                out[str(p)] = p.stat().st_mtime
            except OSError:
                continue
    return out


def diff_new_files(directory: Path, before: dict[str, float]) -> list[Path]:
    """Files created or modified under ``directory`` since the ``before`` snapshot."""
    after = snapshot_dir(directory)
    changed: list[Path] = []
    for path, mtime in after.items():
        if path not in before or before[path] != mtime:
            changed.append(Path(path))
    return sorted(changed)


def register_generated_files(
    thread_id: str,
    before: dict[str, float],
    kind: str = "export",
    exclude_dirs: tuple[str, ...] = (),
) -> list[dict]:
    """Register files written to the thread workspace into ``data_files``.

    Returns the list of registered file records (so callers can surface
    download ids). Skips files already registered at the same path+size.
    ``exclude_dirs`` names subdirectories of the thread workspace whose files
    should NOT be registered (e.g. ``("results",)`` — internal save_result
    snapshots that are reloaded by name, not user-facing deliverables).
    """
    from mmm_framework.api import sessions as sessions_store

    d = thread_dir(thread_id)
    new_files = diff_new_files(d, before)
    if exclude_dirs:
        roots = [(d / sub).resolve() for sub in exclude_dirs]
        new_files = [
            p
            for p in new_files
            if not any(p.resolve().is_relative_to(r) for r in roots)
        ]
    if not new_files:
        return []

    existing = {f.get("path"): f for f in sessions_store.list_files(thread_id)}
    registered: list[dict] = []
    for p in new_files:
        try:
            size = p.stat().st_size
        except OSError:
            size = None
        prior = existing.get(str(p))
        if prior is not None and prior.get("size_bytes") == size:
            continue  # unchanged, already tracked
        rec = sessions_store.register_file(
            thread_id=thread_id,
            path=str(p),
            name=p.name,
            kind=kind,
            size_bytes=size,
            preview=None,
            meta={"source": "execute_python"},
        )
        registered.append(rec)
    return registered
