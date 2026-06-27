"""Data Studio HTTP endpoints — upload/state/pipeline/eda/commit/discard wiring,
filename safety, 400 on a bad op, and the no-LLM commit state write."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    import mmm_framework.api.main as main

    monkeypatch.setattr(main, "DB_PATH", S.DB_PATH)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


def _csv_bytes(n=60, seed=0) -> bytes:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "week": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
            "sales": 1000 + np.arange(n) * 5 + rng.normal(0, 20, n),
            "tv_spend": np.abs(rng.normal(50, 10, n)),
            "search_spend": np.abs(rng.normal(30, 8, n)),
            "notes": ["x"] * n,
        }
    )
    df.loc[n // 2, "tv_spend"] = 900.0
    return df.to_csv(index=False).encode()


def _session(client) -> str:
    pid = client.post("/projects", json={"name": "P"}).json()["project_id"]
    return client.post("/sessions", json={"name": "s", "project_id": pid}).json()[
        "thread_id"
    ]


def _upload(client, tid, name="sales.csv", data=None):
    return client.post(
        f"/data-studio/{tid}/upload",
        files={"file": (name, io.BytesIO(data or _csv_bytes()), "text/csv")},
    )


def test_upload_returns_columns_and_roles(client):
    tid = _session(client)
    r = _upload(client, tid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "sales" in body["columns"] and body["inferred_roles"]["sales"] == "kpi"
    assert body["n_rows"] == 60 and body["preview_rows"]
    # GET hydration reflects the staged file
    g = client.get(f"/data-studio/{tid}").json()
    assert g["staging"]["raw"]["name"] == "sales.csv"
    assert g["staging"]["committed"] is False


def test_upload_filename_is_traversal_safe(client):
    tid = _session(client)
    r = _upload(client, tid, name="../../../evil.csv")
    assert r.status_code == 200, r.text
    from mmm_framework.data_studio import service as Svc

    raw = Svc.read_manifest(tid)["raw"]
    assert "/" not in raw["name"] and ".." not in raw["name"]
    assert "data_studio/raw" in raw["path"].replace("\\", "/")


def test_pipeline_bad_op_returns_400(client):
    tid = _session(client)
    _upload(client, tid)
    r = client.put(f"/data-studio/{tid}/pipeline", json={"steps": [{"op": "teleport"}]})
    assert r.status_code == 400
    # a valid pipeline drops the column
    r = client.put(
        f"/data-studio/{tid}/pipeline",
        json={"steps": [{"op": "drop_columns", "columns": ["notes"]}]},
    )
    assert r.status_code == 200, r.text
    assert "notes" not in r.json()["columns"]


def test_eda_returns_inline_figures(client):
    tid = _session(client)
    _upload(client, tid)
    r = client.post(
        f"/data-studio/{tid}/eda", json={"analyses": ["overview", "outliers"]}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "overview" in body["analyses"]
    fig = body["analyses"]["overview"]["figures"][0]
    assert "data" in fig and "layout" in fig
    assert body["outlier_suggestions"]


def test_commit_sets_dataset_path_no_messages(client):
    tid = _session(client)
    _upload(client, tid)
    r = client.post(f"/data-studio/{tid}/commit", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset_path"].endswith("data_studio_dataset.csv")
    assert body["model_spec"]["kpi"] == "sales"

    # the working dataset + spec landed on agent state, with NO chat messages
    state = client.get(f"/state/{tid}").json()
    assert state["dashboard_data"]["dataset"]["rows"] > 0
    assert state["dashboard_data"]["data_studio"]["committed"] is True
    assert state.get("messages", []) == []
    # GET hydration now reports committed
    assert client.get(f"/data-studio/{tid}").json()["staging"]["committed"] is True


def test_discard_clears_pointer(client):
    tid = _session(client)
    _upload(client, tid)
    r = client.post(f"/data-studio/{tid}/discard", json={})
    assert r.status_code == 200 and r.json()["discarded"] is True
    assert client.get(f"/data-studio/{tid}").json()["staging"] is None
