"""The API-surface sync gate (issue #228).

Three hand-maintained artifacts used to drift independently — the checked-in
``openapi.json`` was 11 operations behind the live app when this gate landed.
``scripts/sync_api_surface.py`` regenerates all three together; what these pin:

* the checked-in artifacts match the live app (the ``--check`` CI mirror);
* the auth gate refuses when a ``/projects/{project_id}`` route carries no
  ``_proj_*`` tenant guard — its first run caught a real one
  (``POST /projects/{project_id}/plan-of-record`` had only a rate limit);
* the renderers are deterministic, so a second run writes nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sync():
    spec = importlib.util.spec_from_file_location(
        "sync_api_surface", ROOT / "scripts" / "sync_api_surface.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_artifacts_are_in_sync():
    """The CI mirror of ``--check``: every checked-in artifact matches what
    the live app would regenerate. Fix with ``make api-sync``."""
    sync = _sync()
    spec = sync.build_spec()
    stale = []
    for path, content in {
        sync.ROUTES_PATH: sync.render_routes(spec),
        sync.OPENAPI_PATH: sync.render_openapi(spec),
        sync.BUILD_REST_DOCS: sync.render_build_rest_docs(sync.op_count(spec)),
    }.items():
        if path.read_text(encoding="utf-8") != content:
            stale.append(str(path.relative_to(ROOT)))
    assert not stale, f"stale API-surface artifacts {stale}: run `make api-sync`"


def test_every_project_route_carries_a_tenant_guard():
    """The auth gate, standalone: no ``/projects/{project_id}`` route may ship
    without ``_proj_read``/``_proj_write``/``_proj_admin``. A rate limit is not
    a tenant guard."""
    sync = _sync()
    unguarded = sync.unguarded_project_routes()
    assert unguarded == [], (
        "routes without a _proj_* dependency (any authenticated principal "
        f"could cross tenants): {unguarded}"
    )


def test_openapi_version_is_the_package_version():
    """A naive ``app.openapi()`` re-export writes the FastAPI default
    ("0.1.0") into the docs provenance line; the renderer must override it."""
    import mmm_framework

    sync = _sync()
    assert sync.build_spec()["info"]["version"] == mmm_framework.__version__


def test_renderers_are_deterministic():
    sync = _sync()
    spec = sync.build_spec()
    assert sync.render_routes(spec) == sync.render_routes(sync.build_spec())
    assert sync.render_openapi(spec) == sync.render_openapi(sync.build_spec())


def test_routes_json_is_canonically_sorted():
    """Hand-appended entries drift out of order (7 had, before #228); the
    regenerated file is sorted by (path, method) so diffs stay minimal."""
    import json

    sync = _sync()
    rows = json.loads(sync.ROUTES_PATH.read_text(encoding="utf-8"))
    keys = [(r["path"], r["method"]) for r in rows]
    assert keys == sorted(keys)
