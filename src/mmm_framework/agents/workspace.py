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


def project_data_dir(project_id: str) -> Path:
    """Per-project data directory for scheduled-sync snapshots (created on demand)."""
    pid = _safe_segment(project_id)
    d = workspace_root() / "projects" / pid / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def garden_dir(org_id: str, name: str, version: str | int) -> Path:
    """Canonical, ORG-scoped store for a registered Model Garden model's
    artifacts (``model.py`` source + ``manifest.json`` + optional reference fit).

    Lives outside any thread (``<root>/garden/<org>/<name>/<version>/``) so it is
    shared across a tenant's projects by construction. ``workspace_root()`` is
    already an allowed download/read root, so no allow-list change is needed.
    """
    d = (
        workspace_root()
        / "garden"
        / _safe_segment(org_id)
        / _safe_segment(name)
        / _safe_segment(str(version))
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def garden_loaded_dir(
    name: str, version: str | int, thread_id: str | None = None
) -> Path:
    """Per-session copy of a loaded garden model's source.

    ``load_garden_model`` copies the canonical source here so the kernel can
    import it — including the container kernel, which only mounts the thread
    workspace, never the org-level :func:`garden_dir`.
    """
    d = (
        thread_dir(thread_id)
        / "garden_loaded"
        / _safe_segment(name)
        / _safe_segment(str(version))
    )
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


# A single captured figure is untrusted egress from the kernel: cap its size
# (default 5 MiB; tune with MMM_PLOT_MAX_BYTES) and keep only the real Plotly
# figure keys so a cell can't smuggle arbitrary payloads through the plot MIME.
_PLOT_MAX_BYTES = int(os.environ.get("MMM_PLOT_MAX_BYTES", str(5 * 1024 * 1024)))
_PLOT_ALLOWED_KEYS = ("data", "layout", "frames", "config")


def store_plot(fig_json: dict, thread_id: str | None = None) -> str:
    """Write a captured Plotly figure to the content-addressed store and return
    its id.

    The figure is **untrusted egress** from the kernel (Phase 3 PR-E.3): it is
    schema-validated (must be a ``{"data": [...], ...}`` figure dict; extra
    top-level keys are dropped) and size-capped (``MMM_PLOT_MAX_BYTES``) — a
    violation raises ``ValueError`` so the caller drops it (never stores or
    serves arbitrary bytes). The id is **salted with ``thread_id``** so it is not
    guessable from figure content across sessions — the id IS the capability,
    since ``GET /plots/{id}`` has no other tenant ACL — and identical figures
    don't dedup across tenants. Within one session, identical figures still
    collapse to one id, so the immutable-cache browser behavior is preserved."""
    if not isinstance(fig_json, dict) or not isinstance(fig_json.get("data"), list):
        raise ValueError("not a Plotly figure (expected a dict with a 'data' list)")
    fig = {k: fig_json[k] for k in _PLOT_ALLOWED_KEYS if k in fig_json}
    payload = json.dumps(fig, sort_keys=True, default=str)
    if len(payload.encode("utf-8")) > _PLOT_MAX_BYTES:
        raise ValueError(f"figure exceeds MMM_PLOT_MAX_BYTES ({_PLOT_MAX_BYTES} bytes)")
    salt = f"{thread_id or ''}\x00"
    pid = hashlib.sha256((salt + payload).encode("utf-8")).hexdigest()[:24]
    path = plot_store_dir() / f"{pid}.json"
    if not path.exists():
        path.write_text(payload, encoding="utf-8")
    return pid


def plot_path(plot_id: str) -> Path | None:
    """Resolve a stored plot id to its on-disk JSON path (or None)."""
    p = plot_store_dir() / f"{_safe_segment(plot_id)}.json"
    return p if p.exists() else None


def table_store_dir() -> Path:
    """Content-addressed store for structured table payloads (shared, dedup'd)."""
    d = workspace_root() / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Tables are untrusted egress from the kernel, same as plots: schema-filtered,
# size-capped, thread-salted. Rows never travel through the SSE stream — only
# the id ref does; the payload is served once via GET /tables/{id}.
_TABLE_MAX_BYTES = int(os.environ.get("MMM_TABLE_MAX_BYTES", str(1024 * 1024)))
_TABLE_ALLOWED_KEYS = (
    "title",
    "columns",
    "rows",
    "total_rows",
    "truncated",
    "source",
    "group",
)


def store_table(table_json: dict, thread_id: str | None = None) -> str:
    """Write a structured table payload to the content-addressed store and
    return its id.

    Same trust model as :func:`store_plot`: the payload is schema-filtered
    (must be a dict with a ``rows`` list; extra top-level keys are dropped) and
    size-capped (``MMM_TABLE_MAX_BYTES``) — a violation raises ``ValueError``
    so the caller drops it. The id is salted with ``thread_id`` because the id
    IS the capability for ``GET /tables/{id}``."""
    if not isinstance(table_json, dict) or not isinstance(table_json.get("rows"), list):
        raise ValueError("not a table payload (expected a dict with a 'rows' list)")
    table = {k: table_json[k] for k in _TABLE_ALLOWED_KEYS if k in table_json}
    payload = json.dumps(table, sort_keys=True, default=str)
    if len(payload.encode("utf-8")) > _TABLE_MAX_BYTES:
        raise ValueError(
            f"table exceeds MMM_TABLE_MAX_BYTES ({_TABLE_MAX_BYTES} bytes)"
        )
    salt = f"{thread_id or ''}\x00"
    tid = hashlib.sha256((salt + payload).encode("utf-8")).hexdigest()[:24]
    path = table_store_dir() / f"{tid}.json"
    if not path.exists():
        path.write_text(payload, encoding="utf-8")
    return tid


def table_path(table_id: str) -> Path | None:
    """Resolve a stored table id to its on-disk JSON path (or None)."""
    p = table_store_dir() / f"{_safe_segment(table_id)}.json"
    return p if p.exists() else None


def report_path(name: str, thread_id: str | None = None) -> Path:
    """Where an agent report HTML file lives (PR-F.6). Hosted: per-session under
    the workspace (an allowed root, so it survives dropping ``Path.cwd()`` and is
    tenant-scoped). Dev: the legacy CWD location, unchanged."""
    from mmm_framework.agents.profile import is_hosted

    leaf = Path(name).name
    return (thread_dir(thread_id) / leaf) if is_hosted() else (Path.cwd() / leaf)


def allowed_roots() -> list[Path]:
    """Directories a download/read is permitted to touch (resolved, absolute)."""
    from mmm_framework.agents.profile import is_hosted

    roots = [
        workspace_root(),
        (Path.cwd() / "uploads").resolve(),
        (Path.cwd() / "mmm_models").resolve(),
        (Path.cwd() / "mmm_configs").resolve(),
    ]
    if not is_hosted():
        # Legacy fixed-name reports written to CWD. Dropped in the hosted profile
        # (reports go per-session under the workspace via report_path()).
        roots.append(Path.cwd().resolve())
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
