"""MMM_SESSIONS_DB overrides the sessions/auth SQLite location.

The sessions DB used to be a fixed package-local path, forcing deployments to
symlink it onto persistent storage. ``resolve_db_path`` (platform/sessions.py)
and the mirrored ``_resolve_default_db_path`` (auth/store.py) both honor the
env var so all writers land in the same file.

Since the 2026-07-24 packaging split (``mmm_framework.api`` service layer →
``mmm_framework.platform``), the no-env default prefers a pre-existing legacy
DB at the old ``src/mmm_framework/api/sessions.db`` location so the move does
not orphan local dev state; otherwise it is platform-package-local.
"""

from __future__ import annotations

from pathlib import Path

from mmm_framework.platform import sessions as S
from mmm_framework.auth import store as auth_store


def _expected_default() -> Path:
    legacy = Path(S.__file__).resolve().parent.parent / "api" / "sessions.db"
    if legacy.exists():
        return legacy
    return Path(S.__file__).resolve().parent / "sessions.db"


def test_default_is_package_local_with_legacy_fallback(monkeypatch):
    monkeypatch.delenv("MMM_SESSIONS_DB", raising=False)
    assert S.resolve_db_path() == _expected_default()
    # auth/store.py duplicates the logic (stdlib-self-contained) — the two
    # resolvers must agree on the SAME file or state silently forks.
    assert auth_store._resolve_default_db_path() == S.resolve_db_path()


def test_env_override_points_both_stores_at_the_same_file(monkeypatch, tmp_path):
    target = tmp_path / "state" / "sessions.db"
    monkeypatch.setenv("MMM_SESSIONS_DB", str(target))
    assert S.resolve_db_path() == target
    assert auth_store._resolve_default_db_path() == target


def test_env_override_expands_user(monkeypatch):
    monkeypatch.setenv("MMM_SESSIONS_DB", "~/mmm-state/sessions.db")
    assert S.resolve_db_path() == Path.home() / "mmm-state" / "sessions.db"


def test_blank_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MMM_SESSIONS_DB", "   ")
    assert S.resolve_db_path() == _expected_default()


def test_conn_creates_missing_parent_dir(monkeypatch, tmp_path):
    # An env-pointed path in a fresh directory (e.g. an empty persistent disk)
    # must work without manual mkdir.
    monkeypatch.setattr(S, "DB_PATH", tmp_path / "fresh" / "dir" / "sessions.db")
    S.init_db()
    assert (tmp_path / "fresh" / "dir" / "sessions.db").exists()
