"""Tests for the structured-tables pipeline + preferences/branding/templates.

Covers: the content-addressed table store, the table builders, the list-aware
dashboard reducer, table emission from model ops / EDA / execute_python
(show_table), the /tables endpoint + SSE stripping, the preferences store and
branding endpoints, brand extraction (pure-HTML parsing + SSRF guard),
branded plots/reports, list_templates, build_model_from_dag, and the
UI outlier-apply endpoint. No network, no LLM."""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the session store + checkpointer + workspace at a temp location."""
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as ss

    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.db")
    ss.init_db()
    return ss


@pytest.fixture()
def client(store, tmp_path, monkeypatch):
    import mmm_framework.api.main as main

    monkeypatch.setattr(main, "DB_PATH", store.DB_PATH)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


# ── table store ──────────────────────────────────────────────────────────────


def _table(rows=None, **over):
    base = {
        "title": "T",
        "columns": [{"key": "a", "label": "A", "type": "number"}],
        "rows": rows if rows is not None else [{"a": 1}],
        "total_rows": 1,
        "truncated": False,
        "source": "test",
        "group": "results",
    }
    base.update(over)
    return base


def test_table_store_content_addressed_salted_and_capped(store, monkeypatch):
    from mmm_framework.agents import workspace as W

    t = _table()
    a1 = W.store_table(t, "threadA")
    a2 = W.store_table(dict(t), "threadA")
    b1 = W.store_table(t, "threadB")
    assert a1 == a2  # within-session dedup
    assert a1 != b1  # thread-salted ids — no cross-tenant dedup/guessing
    assert W.table_path(a1) is not None and W.table_path("nope") is None

    # schema: must have a rows list; extra top-level keys are stripped
    with pytest.raises(ValueError):
        W.store_table({"title": "x"}, "threadA")
    with pytest.raises(ValueError):
        W.store_table("not a dict", "threadA")
    pid = W.store_table(_table(evil="x" * 50), "threadA")
    assert "evil" not in W.table_path(pid).read_text()

    # size cap
    monkeypatch.setattr(W, "_TABLE_MAX_BYTES", 50)
    with pytest.raises(ValueError):
        W.store_table(_table(rows=[{"a": "y" * 200}]), "threadA")


def test_df_to_table_json_truncates_and_sanitizes():
    import numpy as np
    import pandas as pd

    from mmm_framework.agents.tables import TABLE_ROW_CAP, df_to_table_json

    df = pd.DataFrame({"ch": [f"c{i}" for i in range(500)], "roi": [1.5] * 500})
    df.loc[0, "roi"] = np.nan
    t = df_to_table_json(df, title="ROI", source="s", group="results")
    assert t["total_rows"] == 500 and t["truncated"] is True
    assert len(t["rows"]) == TABLE_ROW_CAP
    assert t["rows"][0]["roi"] is None  # NaN -> None (strict JSON)
    types = {c["key"]: c["type"] for c in t["columns"]}
    assert types == {"ch": "string", "roi": "number"}

    # non-Range index folds into columns
    dfi = pd.DataFrame({"v": [1, 2]}, index=pd.Index(["a", "b"], name="ch"))
    ti = df_to_table_json(dfi, title="x", source="s")
    assert "ch" in ti["rows"][0]


def test_publish_tables_refs_and_drops(store):
    from mmm_framework.agents.tables import publish_tables

    dd = {}
    refs, dropped = publish_tables([_table(), {"bad": True}], dd, "threadA")
    assert len(refs) == 1 and dropped == 1
    assert set(refs[0]) == {"id", "title", "source", "group", "ts"}
    assert dd["tables"] == refs  # appended as refs, never rows


def test_merge_dashboard_unions_ref_lists():
    from mmm_framework.agents.state import _merge_dashboard

    a = {"plots": [{"id": "p1"}], "tables": [{"id": "t1"}], "x": 1}
    b = {"plots": [{"id": "p2"}], "tables": [{"id": "t1"}, {"id": "t2"}], "x": 2}
    m = _merge_dashboard(a, b)
    assert [p["id"] for p in m["plots"]] == ["p1", "p2"]  # both survive
    assert [t["id"] for t in m["tables"]] == ["t1", "t2"]  # deduped by id
    assert m["x"] == 2  # scalars stay last-writer-wins
    # legacy inline figures (no id) are tolerated
    m2 = _merge_dashboard({"plots": [{"data": []}]}, {"plots": [{"id": "p"}]})
    assert len(m2["plots"]) == 2
    # explicit clear escape hatch
    assert _merge_dashboard({"plots": [{"id": "p"}]}, {"plots": None})["plots"] is None


# ── emission paths ───────────────────────────────────────────────────────────


def test_execute_python_show_table_emits_refs_not_rows(store):
    from mmm_framework.agents import tools as T

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    cfg = {"configurable": {"thread_id": sess["thread_id"]}}
    code = (
        "import pandas as pd\n"
        "show_table(pd.DataFrame({'a':[1,2],'b':['x','y']}), title='My Table')\n"
        "print('done')"
    )
    cmd = T.execute_python.func(
        state={"dashboard_data": {}}, code=code, tool_call_id="c", config=cfg
    )
    tables = cmd.update["dashboard_data"]["tables"]
    assert len(tables) == 1
    assert tables[0]["title"] == "My Table" and tables[0]["group"] == "repl"
    assert "rows" not in tables[0]  # ref only — rows live in the store
    assert "formatted table" in cmd.update["messages"][0].content


def test_modelop_tables_cross_the_dispatch_boundary(store):
    from mmm_framework.agents.tools import _modelop_command
    from mmm_framework.agents.runtime import set_current_thread

    set_current_thread("threadZ")
    res = {
        "content": "### x",
        "dashboard": {"roi_metrics": [{"channel": "TV"}]},
        "error": None,
        "tables": [_table(title="ROI by Channel", source="get_roi_metrics")],
    }
    cmd = _modelop_command(res, {"dashboard_data": {}}, "tc")
    dd = cmd.update["dashboard_data"]
    assert dd["roi_metrics"] == [{"channel": "TV"}]
    assert len(dd["tables"]) == 1 and dd["tables"][0]["title"] == "ROI by Channel"
    assert "table(s) rendered" in cmd.update["messages"][0].content


def test_message_dashboard_strips_ref_lists():
    from mmm_framework.api.main import _message_dashboard

    combined = {"plots": [1], "tables": [2], "model_spec": {"kpi": "Sales"}}
    assert _message_dashboard(combined) == {"model_spec": {"kpi": "Sales"}}


def test_tables_endpoint_serves_immutable_and_404s(client, store):
    from mmm_framework.agents import workspace as W

    tid = W.store_table(_table(), "threadA")
    r = client.get(f"/tables/{tid}")
    assert r.status_code == 200
    assert r.json()["rows"] == [{"a": 1}]
    assert "immutable" in r.headers.get("cache-control", "")
    assert client.get("/tables/doesnotexist").status_code == 404


# ── preferences & branding ───────────────────────────────────────────────────


def test_preferences_crud_and_project_branding(store):
    store.set_preference("global", "fav_palette", "corporate")
    assert store.get_preference("global", "fav_palette") == "corporate"
    store.set_preference("global", "fav_palette", "warm")  # upsert
    assert store.list_preferences("global") == {"fav_palette": "warm"}
    assert store.delete_preference("global", "fav_palette") is True
    assert store.get_preference("global", "fav_palette") is None

    b = {"client_name": "Acme", "colors": {"palette": ["#0a3d62"]}, "confirmed": True}
    store.set_project_branding("proj1", b)
    assert store.get_project_branding("proj1")["client_name"] == "Acme"
    assert store.get_project_branding("proj2") is None


def test_branding_endpoints_roundtrip_and_validation(client, store):
    pid = client.post("/projects", json={"name": "Acme"}).json()["project_id"]

    assert client.get(f"/projects/{pid}/branding").json() == {}
    good = {
        "client_name": "Acme",
        "colors": {"primary": "#0A3D62", "palette": ["#0a3d62", "#e58e26"]},
        "footer_text": "Prepared for Acme",
    }
    r = client.put(f"/projects/{pid}/branding", json=good)
    assert r.status_code == 200
    saved = r.json()
    assert saved["confirmed"] is True and saved["source"] == "manual"
    assert saved["colors"]["primary"] == "#0a3d62"  # normalized lowercase
    assert client.get(f"/projects/{pid}/branding").json()["client_name"] == "Acme"

    bad = {"colors": {"primary": "not-a-color"}}
    assert client.put(f"/projects/{pid}/branding", json=bad).status_code == 422
    assert client.get("/projects/nope/branding").status_code == 404


def test_global_preferences_endpoint_and_hosted_403(client, store, monkeypatch):
    r = client.put("/preferences", json={"key": "currency", "value": "EUR"})
    assert r.status_code == 200
    assert client.get("/preferences").json()["preferences"]["currency"] == "EUR"

    monkeypatch.setenv("MMM_AGENT_HOSTED", "1")
    r = client.put("/preferences", json={"key": "x", "value": 1})
    assert r.status_code == 403


def test_preference_tools_and_branding_recall(store):
    from mmm_framework.agents import tools as T

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    cfg = {"configurable": {"thread_id": sess["thread_id"]}}

    cmd = T.save_preference.func(
        key="branding",
        value=json.dumps(
            {
                "client_name": "Acme",
                "colors": {"primary": "#0a3d62", "palette": ["#0a3d62"]},
                "confirmed": True,
            }
        ),
        scope="project",
        state={"dashboard_data": {}},
        tool_call_id="c1",
        config=cfg,
    )
    assert "Saved preference" in cmd.update["messages"][0].content
    assert store.get_project_branding(proj["project_id"])["client_name"] == "Acme"

    cmd = T.get_preferences.func(
        state={"dashboard_data": {}}, tool_call_id="c2", config=cfg
    )
    content = cmd.update["messages"][0].content
    assert "Acme" in content and "#0a3d62" in content
    assert cmd.update["dashboard_data"]["branding"]["client_name"] == "Acme"

    # invalid branding payload is rejected, not stored
    cmd = T.save_preference.func(
        key="branding",
        value=json.dumps({"colors": {"primary": "blue"}}),
        scope="project",
        state={"dashboard_data": {}},
        tool_call_id="c3",
        config=cfg,
    )
    assert "Invalid branding" in cmd.update["messages"][0].content


def test_apply_brand_colors_remaps_design_palette():
    from mmm_framework.agents.branding import apply_brand_colors
    from mmm_framework.agents.tools import _PALETTE

    branding = {
        "colors": {"palette": ["#111111", "#222222"]},
        "confirmed": True,
    }
    fig = {
        "data": [
            {"marker": {"color": _PALETTE[0]}},
            {"line": {"color": _PALETTE[1]}},
            {"marker": {"color": "#bada55"}},  # custom color: untouched
        ],
        "layout": {},
    }
    out = apply_brand_colors(fig, branding)
    assert out["layout"]["colorway"] == ["#111111", "#222222"]
    assert out["data"][0]["marker"]["color"] == "#111111"
    assert out["data"][1]["line"]["color"] == "#222222"
    assert out["data"][2]["marker"]["color"] == "#bada55"


def test_branded_plots_only_when_confirmed(store):
    """Unconfirmed (extracted) branding must NOT restyle plots."""
    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    tid = sess["thread_id"]
    cfg = {"configurable": {"thread_id": tid}}
    code = "import plotly.express as px; px.bar(x=['a'], y=[1]).show()"

    store.set_project_branding(
        proj["project_id"],
        {"colors": {"palette": ["#101010"]}, "confirmed": False},
    )
    cmd = T.execute_python.func(
        state={"dashboard_data": {}}, code=code, tool_call_id="c1", config=cfg
    )
    pid = cmd.update["dashboard_data"]["plots"][0]["id"]
    assert "#101010" not in W.plot_path(pid).read_text()

    store.set_project_branding(
        proj["project_id"],
        {"colors": {"palette": ["#101010"]}, "confirmed": True},
    )
    cmd = T.execute_python.func(
        state={"dashboard_data": {}}, code=code, tool_call_id="c2", config=cfg
    )
    pid2 = cmd.update["dashboard_data"]["plots"][0]["id"]
    assert "#101010" in W.plot_path(pid2).read_text()


def test_apply_branding_html_swaps_palette_logo_footer():
    from mmm_framework.agents.report_builder import _PALETTE, apply_branding_html

    html = (
        f"<style>a{{color:{_PALETTE[0]}}}</style>"
        '<div class="logo">MMM Framework<span>Project Report</span></div>'
        '<div class="footer">MMM Framework — Bayesian Marketing Mix Modelling'
        " · Generated today</div>"
    )
    out = apply_branding_html(
        html,
        {
            "colors": {"palette": ["#0a3d62"]},
            "client_name": "Acme",
            "logo_url": "https://acme.com/logo.png",
            "footer_text": "Prepared for Acme — Confidential",
        },
    )
    assert _PALETTE[0] not in out and "#0a3d62" in out
    assert "Acme<span>" in out and "logo.png" in out
    assert "Prepared for Acme — Confidential" in out
    # no branding -> identity
    assert apply_branding_html(html, None) == html


def _extract_json_array(s: str, start: int) -> str:
    """Bracket-balanced JSON array starting at ``s[start] == '['``."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]


def _extra_chart_datapoints(html: str, id_prefix: str) -> dict[str, int]:
    """Map each ``Plotly.newPlot("<id_prefix>...")`` div to its total x/y points."""
    import re

    out: dict[str, int] = {}
    for m in re.finditer(r'Plotly\.newPlot\(\s*"([^"]+)"\s*,\s*(?=\[)', html):
        div_id = m.group(1)
        if not div_id.startswith(id_prefix):
            continue
        arr = _extract_json_array(html, m.end())
        cleaned = (
            arr.replace("NaN", "null")
            .replace("-Infinity", "null")
            .replace("Infinity", "null")
        )
        traces = json.loads(cleaned)
        out[div_id] = sum(
            len(t.get(k, []))
            for t in traces
            for k in ("x", "y", "z", "values")
            if isinstance(t.get(k), list)
        )
    return out


def test_report_builder_hydrates_plot_refs_into_additional_charts(store):
    """Regression: ``dashboard_data['plots']`` holds thin ``{id, title}`` refs
    (the heavy figure lives in the content-addressed plot store). The project
    report + internal slides must inline those figures into the "Additional
    Charts" section — otherwise every extra chart renders as an empty graph.
    Unresolvable refs are dropped, never emitted as an empty ``<div>``."""
    from mmm_framework.agents import workspace as W
    from mmm_framework.agents.report_builder import (
        _hydrate_plots,
        generate_html_report,
        generate_html_slides,
    )

    tid = "threadR"
    fig_a = {
        "data": [{"type": "scatter", "x": [1, 2, 3], "y": [4, 5, 6]}],
        "layout": {"title": {"text": "Spend vs Response"}},
    }
    fig_b = {  # figure carries no title -> the ref title must be adopted
        "data": [{"type": "bar", "x": ["TV", "Search"], "y": [10, 20]}],
        "layout": {},
    }
    refs = [
        {"id": W.store_plot(fig_a, tid), "title": "Spend vs Response"},
        {"id": W.store_plot(fig_b, tid), "title": "Channel Contribution"},
        {"id": "deadbeefmissing000000000", "title": "Broken"},  # unresolvable
    ]

    # _hydrate_plots: 2 resolved (missing dropped), ref title backfilled.
    hydrated = _hydrate_plots(refs)
    assert len(hydrated) == 2
    assert all(isinstance(f["data"], list) and f["data"] for f in hydrated)
    assert hydrated[1]["layout"]["title"]["text"] == "Channel Contribution"
    # An already-inline figure (store-write-failure fallback) passes through.
    inline = {"data": [{"x": [1], "y": [2]}], "layout": {}}
    assert _hydrate_plots([inline]) == [inline]
    assert _hydrate_plots(None) == []

    dashboard = {
        "dataset": {"rows": 52},
        "model_spec": {"kpi": "Sales", "media_channels": [{"name": "TV"}]},
        "roi_metrics": [{"channel": "TV", "roi_mean": 2.0}],
        "diagnostics": {"converged": True, "rhat_max": 1.0, "divergences": 0},
        "plots": refs,
    }

    report = generate_html_report("R", "07 July 2026", dashboard, [])
    pts = _extra_chart_datapoints(report, "chart-extra-")
    assert len(pts) == 2, "the missing ref must be dropped, not rendered empty"
    assert all(v > 0 for v in pts.values()), f"empty additional charts: {pts}"
    assert "Additional Charts" in report

    slides = generate_html_slides("R", "07 July 2026", dashboard, [], client_mode=False)
    spts = _extra_chart_datapoints(slides, "slide-extra-")
    assert spts and all(v > 0 for v in spts.values()), f"empty slide charts: {spts}"


# ── brand extraction ─────────────────────────────────────────────────────────


_FIXTURE_HTML = """<html><head>
<title>Acme Corp — Marketing Excellence</title>
<meta property="og:site_name" content="Acme Corp"/>
<meta name="theme-color" content="#0a3d62"/>
<meta property="og:image" content="/img/logo.png"/>
<link rel="icon" href="/favicon.ico"/>
<style>
.btn { background: #e58e26; font-family: Montserrat, sans-serif; }
.hero { color: #e58e26; background: #0a3d62; }
.grey { color: #f8f8f8; background: #333333; border-color: #cccccc; }
body { font-family: 'Open Sans', serif; }
</style>
</head><body>
<div style="color: rgb(60, 99, 130)">hi</div>
</body></html>"""


def test_parse_brand_html_extracts_brand_not_chrome():
    from mmm_framework.agents.brand_extract import parse_brand_html

    b = parse_brand_html(_FIXTURE_HTML, "https://acme.com/")
    assert b["client_name"] == "Acme Corp"  # og:site_name wins over <title>
    assert b["colors"]["primary"] == "#0a3d62"  # theme-color first
    assert "#e58e26" in b["colors"]["palette"]
    # greys / near-white / near-black are filtered out
    for grey in ("#f8f8f8", "#333333", "#cccccc"):
        assert grey not in b["colors"]["palette"]
    assert b["logo_url"] == "https://acme.com/img/logo.png"  # absolutized
    assert b["fonts"]["heading"] == "Montserrat"
    assert b["source"] == "extracted" and b["confirmed"] is False


def test_brand_ssrf_guard_blocks_private_targets(monkeypatch):
    from mmm_framework.agents.brand_extract import (
        BrandExtractError,
        _assert_public_http,
    )

    for bad in (
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/",
        "http://user:pw@example.com/",
        "http://example.com:8080/",
    ):
        with pytest.raises(BrandExtractError):
            _assert_public_http(bad)

    # public-looking hostname that RESOLVES private (DNS rebind / internal DNS)
    import socket

    def fake_getaddrinfo(host, port, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(BrandExtractError):
        _assert_public_http("https://internal.example.com/")


def test_brand_extract_hosted_gate(monkeypatch):
    from mmm_framework.agents.brand_extract import BrandExtractError, _assert_enabled

    monkeypatch.setenv("MMM_AGENT_HOSTED", "1")
    monkeypatch.delenv("MMM_BRAND_FETCH_ALLOW", raising=False)
    with pytest.raises(BrandExtractError):
        _assert_enabled()
    monkeypatch.setenv("MMM_BRAND_FETCH_ALLOW", "1")
    _assert_enabled()  # explicitly allowed


def test_extract_tool_saves_unconfirmed_proposal(store, monkeypatch):
    from mmm_framework.agents import brand_extract as BX
    from mmm_framework.agents import tools as T

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    cfg = {"configurable": {"thread_id": sess["thread_id"]}}

    monkeypatch.setattr(
        BX,
        "_fetch",
        lambda url, **kw: _FIXTURE_HTML,
    )
    cmd = T.extract_brand_from_website.func(
        url="https://acme.com",
        state={"dashboard_data": {}},
        tool_call_id="c",
        config=cfg,
    )
    content = cmd.update["messages"][0].content
    assert "#0a3d62" in content and "proposed" in content.lower()
    saved = store.get_project_branding(proj["project_id"])
    assert saved and saved["confirmed"] is False and saved["source"] == "extracted"
    assert cmd.update["dashboard_data"]["branding"]["client_name"] == "Acme Corp"


def test_extract_endpoint_uses_guard(client, store):
    pid = client.post("/projects", json={"name": "A"}).json()["project_id"]
    r = client.post(
        f"/projects/{pid}/branding/extract", json={"url": "http://127.0.0.1/x"}
    )
    assert r.status_code == 400
    assert "non-public" in r.json()["detail"] or "refusing" in r.json()["detail"]


# ── templates & DAG-derived spec ─────────────────────────────────────────────


def test_list_templates_kinds(store, tmp_path, monkeypatch):
    from mmm_framework.agents import tools as T

    monkeypatch.chdir(tmp_path)  # isolate mmm_configs scan
    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    cfg = {"configurable": {"thread_id": sess["thread_id"]}}

    os.makedirs("mmm_configs", exist_ok=True)
    with open("mmm_configs/uk_q1.json", "w") as f:
        json.dump({"kpi": "Sales", "media_channels": [{"name": "TV"}]}, f)
    store.set_project_branding(
        proj["project_id"], {"colors": {"palette": ["#0a3d62"]}, "confirmed": True}
    )
    store.add_kb_document(
        proj["project_id"],
        name="q1 report template.html",
        path="(x)",
        kind="text",
        status="ready",
        meta={"template": True},
    )

    out = T.list_templates.func(config=cfg)
    assert "`client`" in out and "`minimal`" in out  # report templates
    assert "`corporate`" in out and "`brand`" in out  # palettes incl. branding
    assert "uk_q1" in out  # saved configs
    assert "q1 report template.html" in out  # tagged KB docs


def test_build_model_from_dag_commits_and_honors_locks(store):
    from mmm_framework.agents.causal_tools import build_model_from_dag

    dag_spec = {
        "nodes": [
            {"id": "sales", "variable_name": "Sales", "node_type": "kpi"},
            {"id": "tv", "variable_name": "TV", "node_type": "media"},
            {"id": "dig", "variable_name": "Digital", "node_type": "media"},
            {"id": "price", "variable_name": "Price", "node_type": "control"},
        ],
        "edges": [
            {"source": "tv", "target": "sales"},
            {"source": "dig", "target": "sales"},
            {"source": "price", "target": "sales"},
        ],
    }
    state = {
        "dashboard_data": {"dag": {"spec": dag_spec, "validation": {"valid": True}}},
        "model_spec": {
            "kpi": "Old",
            "media_channels": [{"name": "TV", "adstock": {"l_max": 12}}],
            "control_variables": [],
        },
        "locked_fields": [],
    }
    cmd = build_model_from_dag.func(state=state, tool_call_id="t1")
    spec = cmd.update["model_spec"]
    assert spec["kpi"] == "Sales"
    assert [c["name"] for c in spec["media_channels"]] == ["TV", "Digital"]
    assert spec["media_channels"][0]["adstock"] == {"l_max": 12}  # preserved
    assert cmd.update["model_status"] == "configured"

    # locked field -> pending confirmation, not silently applied
    state["locked_fields"] = ["kpi"]
    cmd = build_model_from_dag.func(state=state, tool_call_id="t2")
    assert cmd.update["model_spec"]["kpi"] == "Old"
    assert any(p["path"] == "kpi" for p in cmd.update["pending_spec_changes"])

    # no DAG -> friendly error
    cmd = build_model_from_dag.func(state={"dashboard_data": {}}, tool_call_id="t3")
    assert "No causal DAG" in cmd.update["messages"][0].content


def test_new_tools_registered_and_schema_hides_injected_args(store):
    from mmm_framework.agents.tools import TOOLS

    names = {t.name for t in TOOLS}
    for expected in {
        "get_preferences",
        "save_preference",
        "list_templates",
        "extract_brand_from_website",
        "build_model_from_dag",
    }:
        assert expected in names, f"missing tool {expected}"

    extract = next(t for t in TOOLS if t.name == "extract_brand_from_website")
    assert "config" not in extract.args and "url" in extract.args
    save = next(t for t in TOOLS if t.name == "save_preference")
    assert "config" not in save.args and "key" in save.args


# ── EDA envelope + outlier apply endpoint ────────────────────────────────────


def test_eda_envelope_action_lifecycle():
    from mmm_framework.agents.eda_tools import _update_eda_envelope

    dd = {}
    _update_eda_envelope(
        dd,
        issues=[{"severity": "warning", "check": "gaps", "message": "m"}],
        actions=[
            {"action_id": "a1", "strategy": "winsorize", "status": "proposed"},
            {"action_id": "a2", "strategy": "dummy", "status": "proposed"},
        ],
        damaged=["TV"],
    )
    assert dd["eda"]["issues"][0]["check"] == "gaps"
    assert dd["eda"]["normalization_damaged"] == ["TV"]
    _update_eda_envelope(dd, applied_ids=["a1"])
    statuses = {a["action_id"]: a["status"] for a in dd["eda"]["outlier_actions"]}
    assert statuses == {"a1": "applied", "a2": "proposed"}
    assert dd["eda"]["updated_at"] > 0


def test_outlier_apply_endpoint_validates_and_reports(client, store):
    """The UI endpoint reuses the tool core: with no outlier report it returns
    a clear 400 (and never appends chat messages)."""
    pid = client.post("/projects", json={"name": "A"}).json()["project_id"]
    tid = client.post("/sessions", json={"name": "s", "project_id": pid}).json()[
        "thread_id"
    ]
    r = client.post(f"/outliers/{tid}/apply", json={"action_ids": ["a1"]})
    assert r.status_code == 400
    assert "detect_outliers" in r.json()["error"]
