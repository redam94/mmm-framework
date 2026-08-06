"""Regenerate the three hand-maintained API-surface artifacts, together.

Run from the repo root::

    uv run python scripts/sync_api_surface.py          # write all three + docs
    uv run python scripts/sync_api_surface.py --check  # exit 1 if any is stale

Artifacts (issue #228 — before this script, every new endpoint invalidated all
three with no tooling, and the checked-in ``openapi.json`` had drifted 11
operations behind the live app):

1. ``tests/contracts/rest_routes.json`` — the (method, path) contract set,
   written in canonical ``(path, method)`` order.
2. ``docs/shared/openapi.json`` — ``app.openapi()`` with the ``info`` block
   overridden: ``version`` comes from ``mmm_framework.__version__`` (the app
   object says "0.1.0") and the provenance ``description`` is re-injected —
   a naive re-export silently downgrades the version the docs render.
3. ``docs/tools/build_rest_docs.py::EXPECTED_OPS`` — rewritten to the live
   operation count, then the REST-docs build runs so ``docs/rest-api.html``
   regenerates and its own asserts (op count, group classification, HTML
   balance) act as the gate.

The auth gate: the script REFUSES to write when any ``/projects/{project_id}``
route lacks a ``_proj_read``/``_proj_write``/``_proj_admin`` dependency —
automating the bump otherwise removes the one moment someone notices a route
shipped without a tenant guard. (The first run of this gate found one:
``POST /projects/{project_id}/plan-of-record`` carried only a rate limit.)

``tests/test_api_surface_sync.py`` runs ``--check`` in-process, so CI fails
when any artifact drifts.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTES_PATH = ROOT / "tests" / "contracts" / "rest_routes.json"
OPENAPI_PATH = ROOT / "docs" / "shared" / "openapi.json"
BUILD_REST_DOCS = ROOT / "docs" / "tools" / "build_rest_docs.py"
REST_HTML = ROOT / "docs" / "rest-api.html"

#: The provenance line the checked-in spec carries; app.openapi() has none.
OPENAPI_DESCRIPTION = (
    "MMM Framework agent API (mmm-framework-server). "
    "Generated from the live FastAPI app."
)

_PROJ_GUARDS = ("_proj_read", "_proj_write", "_proj_admin")


def _app():
    from mmm_framework_server.main import app

    return app


def unguarded_project_routes() -> list[tuple[str, str]]:
    """Every ``/projects/{project_id}`` route with no ``_proj_*`` dependency.

    Matching is by identity against the module-level ``Depends`` singletons —
    each guard factory returns a fresh closure, so ``id()`` is precise. The
    bare collection routes (``/projects`` list/create) legitimately carry no
    per-project guard, hence the ``{project_id}`` prefix.
    """
    from fastapi.routing import APIRoute

    from mmm_framework_server import main as M

    guard_ids = {id(getattr(M, name).dependency) for name in _PROJ_GUARDS}
    out: list[tuple[str, str]] = []
    for r in M.app.routes:
        if not isinstance(r, APIRoute):
            continue
        if not r.path.startswith("/projects/{project_id}"):
            continue
        if any(id(d.dependency) in guard_ids for d in r.dependencies):
            continue
        for method in sorted(r.methods - {"HEAD", "OPTIONS"}):
            out.append((method, r.path))
    return sorted(out)


def build_spec() -> dict:
    """``app.openapi()`` with the hand-maintained ``info`` block re-applied."""
    import mmm_framework

    spec = _app().openapi()
    spec["info"]["version"] = mmm_framework.__version__
    spec["info"]["description"] = OPENAPI_DESCRIPTION
    return spec


def render_routes(spec: dict) -> str:
    pairs = sorted(
        {
            (path, method.upper())
            for path, ops in spec["paths"].items()
            for method in ops
        }
    )
    data = [{"method": m, "path": p} for p, m in pairs]
    return json.dumps(data, indent=2) + "\n"


def render_openapi(spec: dict) -> str:
    # The checked-in convention: indent=1, no trailing newline.
    return json.dumps(spec, indent=1)


def op_count(spec: dict) -> int:
    return sum(len(ops) for ops in spec["paths"].values())


def render_build_rest_docs(n_ops: int) -> str:
    src = BUILD_REST_DOCS.read_text(encoding="utf-8")
    new, n = re.subn(r"^EXPECTED_OPS = \d+", f"EXPECTED_OPS = {n_ops}", src, flags=re.M)
    if n != 1:
        raise SystemExit(
            f"expected exactly one 'EXPECTED_OPS = <n>' line in "
            f"{BUILD_REST_DOCS}, found {n}"
        )
    return new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any artifact is stale, writing nothing",
    )
    args = ap.parse_args(argv)

    unguarded = unguarded_project_routes()
    if unguarded:
        for method, path in unguarded:
            print(f"UNGUARDED: {method} {path} has no _proj_* dependency")
        print(
            "Refusing to sync: every /projects/{project_id} route must carry "
            "a tenant guard (_proj_read/_proj_write/_proj_admin). Fix the "
            "route(s) above first."
        )
        return 1

    spec = build_spec()
    targets = {
        ROUTES_PATH: render_routes(spec),
        OPENAPI_PATH: render_openapi(spec),
        BUILD_REST_DOCS: render_build_rest_docs(op_count(spec)),
    }
    stale = {
        path: content
        for path, content in targets.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    }

    if args.check:
        if stale:
            for path in stale:
                print(f"STALE: {path.relative_to(ROOT)}")
            print("Run: uv run python scripts/sync_api_surface.py")
            return 1
        print("API-surface artifacts are in sync.")
        return 0

    # Diff of ADDED routes, with their guard story, before anything writes —
    # this is the moment a human sees what shipped.
    if ROUTES_PATH.exists():
        old = {
            (r["method"], r["path"])
            for r in json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
        }
        new = {(m.upper(), p) for p, ops in spec["paths"].items() for m in ops}
        for method, path in sorted(new - old, key=lambda t: (t[1], t[0])):
            print(f"ADDED: {method} {path}")
        for method, path in sorted(old - new, key=lambda t: (t[1], t[0])):
            print(f"REMOVED (breaking!): {method} {path}")

    changed = 0
    for path, content in targets.items():
        if path in stale:
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
            changed += 1
    if BUILD_REST_DOCS in stale or not REST_HTML.exists() or changed:
        # Regenerate docs/rest-api.html; its own asserts are part of the gate.
        res = subprocess.run(
            [sys.executable, str(BUILD_REST_DOCS)],
            cwd=str(BUILD_REST_DOCS.parent.parent),
            capture_output=True,
            text=True,
        )
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        if res.returncode != 0:
            return res.returncode
    if not changed:
        print("API-surface artifacts already in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
