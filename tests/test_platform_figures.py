"""Platform-reported attribution ingestion + triangulation fold (issue #120).

The triangulation engine (reporting/triangulation.py) already accepts a platform
dict; this covers the persistence slot: the registry table, the CSV/JSON parser,
the per-channel reduction, the triangulation auto-fold, and the REST surface.
"""

from __future__ import annotations

import pytest

from mmm_framework.api import triangulation as TRI


# ---------------------------------------------------------------------------
# parser (pure)
# ---------------------------------------------------------------------------
class TestParsePlatform:
    def test_csv_with_metadata(self):
        recs = TRI.parse_platform_records(
            b"channel,value,source,attribution_window,incremental\n"
            b"Search,9.0,Meta,7-day click,\nTV,2.5,Meta,,true\n",
            "p.csv",
        )
        by = {r["channel"]: r for r in recs}
        assert by["Search"]["value"] == 9.0
        assert by["Search"]["source"] == "Meta"
        assert by["Search"]["attribution_window"] == "7-day click"
        assert by["TV"]["incremental"] is True

    def test_json_list_and_map(self):
        assert TRI.parse_platform_records(
            b'[{"channel":"TV","value":3}]', "p.json"
        ) == [{"channel": "TV", "value": 3}]
        assert TRI.parse_platform_records(b'{"Search":9.0}', "p.json") == [
            {"channel": "Search", "value": 9.0}
        ]

    def test_json_nested_map(self):
        recs = TRI.parse_platform_records(
            b'{"Search":{"value":9,"source":"Meta"}}', "p.json"
        )
        assert recs == [{"channel": "Search", "value": 9, "source": "Meta"}]


# ---------------------------------------------------------------------------
# store + reduction + triangulation fold
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


class TestPlatformStore:
    def test_defaults_non_incremental(self, store):
        pid = store.create_project("P")["project_id"]
        figs = store.upsert_platform_figures(pid, [{"channel": "TV", "value": 3.0}])
        assert figs[0]["incremental"] is False
        assert figs[0]["metric"] == "roas"

    def test_incremental_flag_honored(self, store):
        pid = store.create_project("P")["project_id"]
        figs = store.upsert_platform_figures(
            pid, [{"channel": "TV", "value": 3.0, "incremental": True}]
        )
        assert figs[0]["incremental"] is True

    def test_upsert_overwrites_channel_source(self, store):
        pid = store.create_project("P")["project_id"]
        store.upsert_platform_figures(
            pid, [{"channel": "TV", "value": 3.0, "source": "Meta"}]
        )
        store.upsert_platform_figures(
            pid, [{"channel": "TV", "value": 5.0, "source": "Meta"}]
        )
        rows = store.list_platform_figures(pid)
        assert len(rows) == 1 and rows[0]["value"] == 5.0

    def test_delete_by_channel(self, store):
        pid = store.create_project("P")["project_id"]
        store.upsert_platform_figures(
            pid,
            [
                {"channel": "TV", "value": 3.0},
                {"channel": "Search", "value": 9.0},
            ],
        )
        assert store.delete_platform_figures(pid, channel="Search") == 1
        assert {r["channel"] for r in store.list_platform_figures(pid)} == {"TV"}

    def test_platform_dict_one_per_channel_newest(self, store, monkeypatch):
        pid = store.create_project("P")["project_id"]
        store.upsert_platform_figures(
            pid, [{"channel": "TV", "value": 3.0, "source": "Meta"}]
        )
        d = TRI.platform_dict_for_project(pid)
        assert d == {
            "TV": {
                "value": 3.0,
                "metric": "roas",
                "attribution_window": None,
                "incremental": False,
            }
        }

    def test_build_triangulation_folds_stored_platform(self, store):
        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s", project_id=pid)["thread_id"]
        store.add_artifact(
            tid,
            "model_run",
            {
                "run_id": "r1",
                "kpi": "sales",
                "channels": ["Search"],
                "model_kind": "mmm",
                "estimands": [
                    {
                        "estimand": "contribution_roi",
                        "channel": "Search",
                        "kind": "roi",
                        "status": "ok",
                        "mean": 4.0,
                        "hdi_low": 3.0,
                        "hdi_high": 5.0,
                        "units": "ROI",
                    }
                ],
            },
        )
        store.upsert_platform_figures(pid, [{"channel": "Search", "value": 9.0}])
        tri = TRI.build_project_triangulation(pid)
        assert tri["sources_available"]["platform"] == 1
        search = next(c for c in tri["channels"] if c["channel"] == "Search")
        assert search["agreement"] == "platform-inflated"
        assert any(s["source"] == "platform" for s in search["sources"])


# ---------------------------------------------------------------------------
# endpoints (TestClient)
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    import mmm_framework.api.main as main
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def project(client):
    return client.post("/projects", json={"name": "P"}).json()["project_id"]


class TestEndpoints:
    def test_upload_list_delete(self, client, project):
        r = client.post(
            f"/projects/{project}/platform-figures",
            files={
                "file": (
                    "p.csv",
                    b"channel,value,source\nSearch,9.0,Meta\nTV,2.5,Meta\n",
                    "text/csv",
                )
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["ingested"] == 2

        figs = client.get(f"/projects/{project}/platform-figures").json()["figures"]
        assert {f["channel"] for f in figs} == {"Search", "TV"}
        assert all(f["incremental"] is False for f in figs)  # tagged last-touch

        assert (
            client.delete(f"/projects/{project}/platform-figures?channel=TV").json()[
                "deleted"
            ]
            == 1
        )
        assert (
            len(client.get(f"/projects/{project}/platform-figures").json()["figures"])
            == 1
        )

    def test_bad_upload_400(self, client, project):
        r = client.post(
            f"/projects/{project}/platform-figures",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert r.status_code == 400

    def test_uploaded_figures_reach_triangulation_endpoint(self, client, project):
        # seed an MMM run so the triangulation endpoint has an MMM source
        from mmm_framework.api import sessions as S

        tid = S.create_session("s", project_id=project)["thread_id"]
        S.add_artifact(
            tid,
            "model_run",
            {
                "run_id": "r1",
                "kpi": "sales",
                "channels": ["Search"],
                "model_kind": "mmm",
                "estimands": [
                    {
                        "estimand": "contribution_roi",
                        "channel": "Search",
                        "kind": "roi",
                        "status": "ok",
                        "mean": 4.0,
                        "hdi_low": 3.0,
                        "hdi_high": 5.0,
                        "units": "ROI",
                    }
                ],
            },
        )
        client.post(
            f"/projects/{project}/platform-figures",
            files={"file": ("p.csv", b"channel,value\nSearch,9.0\n", "text/csv")},
        )
        tri = client.get(f"/projects/{project}/triangulation").json()
        assert tri["sources_available"]["platform"] == 1
        search = next(c for c in tri["channels"] if c["channel"] == "Search")
        assert search["agreement"] == "platform-inflated"

    def test_404_unknown_project(self, client):
        assert client.get("/projects/nope/platform-figures").status_code == 404
