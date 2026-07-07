"""MMM_SESSIONS_DB overrides the sessions/auth SQLite location.

The sessions DB used to be a fixed package-local path
(``src/mmm_framework/api/sessions.db``), forcing deployments to symlink it onto
persistent storage. ``resolve_db_path`` (api/sessions.py) and the mirrored
``_resolve_default_db_path`` (auth/store.py) both honor the env var so all
writers land in the same file.
"""

from __future__ import annotations

from pathlib import Path

from mmm_framework.api import sessions as S
from mmm_framework.auth import store as auth_store


def test_default_is_package_local(monkeypatch):
    monkeypatch.delenv("MMM_SESSIONS_DB", raising=False)
    assert S.resolve_db_path() == Path(S.__file__).parent / "sessions.db"
    assert (
        auth_store._resolve_default_db_path()
        == Path(auth_store.__file__).resolve().parents[1] / "api" / "sessions.db"
    )


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
    assert S.resolve_db_path() == Path(S.__file__).parent / "sessions.db"


def test_conn_creates_missing_parent_dir(monkeypatch, tmp_path):
    # An env-pointed path in a fresh directory (e.g. an empty persistent disk)
    # must work without manual mkdir.
    monkeypatch.setattr(S, "DB_PATH", tmp_path / "fresh" / "dir" / "sessions.db")
    S.init_db()
    assert (tmp_path / "fresh" / "dir" / "sessions.db").exists()
