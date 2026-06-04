"""Tests for the agent knowledge-base + workspace upgrade.

Covers the new primitives wired into the agent: the scoped workspace directory,
the thread-scoped model cache, project identity, knowledge-base storage +
retrieval, the new agent tools, and the new API endpoints (projects, KB,
downloads). Embeddings are stubbed so the suite needs no network/credentials.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the session store + checkpointer + workspace at a temp location."""
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as ss

    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.db")
    ss.init_db()
    return ss


# ── workspace ────────────────────────────────────────────────────────────────


def test_workspace_dirs_and_traversal_guard(store, monkeypatch):
    from mmm_framework.agents import workspace as W

    td = W.thread_dir("threadA")
    assert td.exists() and td.name == "threadA"
    # safe_join allows nested paths
    assert W.safe_join(td, "sub/out.csv").name == "out.csv"
    # ...but blocks escapes
    with pytest.raises(ValueError):
        W.safe_join(td, "../../etc/passwd")
    # is_within: a workspace file passes, /etc/hosts fails
    f = td / "x.txt"
    f.write_text("hi")
    assert W.is_within(str(f))
    assert not W.is_within("/etc/hosts")


def test_register_generated_files(store, monkeypatch):
    from mmm_framework.agents import workspace as W

    td = W.thread_dir("threadB")
    before = W.snapshot_dir(td)
    (td / "report.csv").write_text("a,b\n1,2\n")
    regs = W.register_generated_files("threadB", before)
    assert len(regs) == 1 and regs[0]["name"] == "report.csv"
    # idempotent: unchanged file is not re-registered
    again = W.register_generated_files("threadB", W.snapshot_dir(td))
    assert again == []


def test_plot_store_is_content_addressed(store):
    from mmm_framework.agents import workspace as W

    fig = {"data": [{"type": "bar", "x": ["a"], "y": [1]}], "layout": {"title": "t"}}
    pid = W.store_plot(fig)
    # same content -> same id (dedup); resolvable to disk
    assert W.store_plot(dict(fig)) == pid
    assert W.plot_path(pid) is not None
    # different content -> different id
    assert W.store_plot({"data": [], "layout": {}}) != pid
    assert W.plot_path("nope") is None


def test_store_plot_thread_salt_and_validation(store, monkeypatch):
    """Phase 3 PR-E.3: plot ids are salted per session (not cross-tenant
    guessable / no cross-tenant dedup), the payload is schema-validated +
    size-capped, and extra top-level keys are stripped."""
    import pytest as _pytest

    from mmm_framework.agents import workspace as W

    fig = {"data": [{"type": "bar", "x": ["a"], "y": [1]}], "layout": {"title": "t"}}
    a1 = W.store_plot(fig, "threadA")
    a2 = W.store_plot(dict(fig), "threadA")
    b1 = W.store_plot(fig, "threadB")
    assert a1 == a2  # within-session dedup preserved (immutable caching)
    assert a1 != b1  # different session -> different id (no cross-tenant dedup)
    assert W.store_plot(fig) != a1  # unsalted differs again
    assert W.plot_path(a1) is not None and W.plot_path(b1) is not None

    # untrusted egress: non-figure payloads are rejected
    with _pytest.raises(ValueError):
        W.store_plot({"not": "a figure"}, "threadA")
    with _pytest.raises(ValueError):
        W.store_plot("just a string", "threadA")

    # extra top-level keys are dropped (no smuggling through the plot channel)
    pid = W.store_plot({"data": [], "layout": {}, "evil": "x" * 50}, "threadA")
    assert "evil" not in W.plot_path(pid).read_text()

    # oversize is rejected (caller drops it)
    monkeypatch.setattr(W, "_PLOT_MAX_BYTES", 50)
    with _pytest.raises(ValueError):
        W.store_plot({"data": [{"big": "y" * 200}], "layout": {}}, "threadA")


def test_execute_python_emits_plot_refs_not_inline_json(store):
    from mmm_framework.agents import tools as T

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    cfg = {"configurable": {"thread_id": sess["thread_id"]}}
    code = "import plotly.express as px; px.bar(x=['a'], y=[1]).show()"
    cmd = T.execute_python.func(
        state={"dashboard_data": {}}, code=code, tool_call_id="c", config=cfg
    )
    plots = cmd.update["dashboard_data"]["plots"]
    assert len(plots) == 1
    # lightweight ref, NOT the full figure JSON
    assert "id" in plots[0] and "data" not in plots[0]


def test_execute_python_drops_oversize_plot_with_notice(store, monkeypatch):
    """An oversize/invalid captured figure is dropped (not stored, not inlined),
    and the user is told rather than silently losing a chart (PR-E.3)."""
    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W

    monkeypatch.setattr(W, "_PLOT_MAX_BYTES", 10)  # force rejection
    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    cfg = {"configurable": {"thread_id": sess["thread_id"]}}
    code = "import plotly.express as px; px.bar(x=['a'], y=[1]).show()"
    cmd = T.execute_python.func(
        state={"dashboard_data": {}}, code=code, tool_call_id="c", config=cfg
    )
    assert cmd.update["dashboard_data"].get("plots", []) == []  # dropped, not stored
    assert "omitted" in cmd.update["messages"][0].content  # user is informed


def test_execute_python_writes_downloadable_and_inputs_readable(
    store, tmp_path, monkeypatch
):
    """Unified workspace: everything execute_python writes (even a BARE relative
    name) lands in the session workspace and is registered for download, and
    workspace files (generated/uploaded data) are readable by name."""
    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W

    monkeypatch.chdir(tmp_path)  # a server cwd distinct from the workspace
    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    tid = sess["thread_id"]
    cfg = {"configurable": {"thread_id": tid}}
    cwd_before = os.path.realpath(os.getcwd())

    # a dataset already in the workspace is readable by its bare name
    (W.thread_dir(tid) / "synthetic_mff_data.csv").write_text(
        "date,sales\n2024-01-01,100\n"
    )
    r = T.execute_python.func(
        state={"dashboard_data": {}},
        code="import pandas as pd; print('rows', len(pd.read_csv('synthetic_mff_data.csv')))",
        tool_call_id="c1",
        config=cfg,
    )
    assert "rows 1" in r.update["messages"][0].content  # NOT FileNotFoundError

    # a BARE relative write is auto-registered for download (the key behaviour)
    T.execute_python.func(
        state={"dashboard_data": {}},
        code="import pandas as pd; pd.DataFrame({'a':[1,2]}).to_csv('out.csv', index=False); print('ok')",
        tool_call_id="c2",
        config=cfg,
    )
    assert "out.csv" in [f["name"] for f in store.list_files(tid)]
    assert (W.thread_dir(tid) / "out.csv").exists()
    # it did NOT leak into the server cwd, and the cwd is restored
    assert not (tmp_path / "out.csv").exists()
    assert os.path.realpath(os.getcwd()) == cwd_before


def test_generate_synthetic_data_writes_to_workspace(store):
    """generate_synthetic_data writes into the workspace, exposes an absolute
    dataset_path, and registers the file for download."""
    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    tid = sess["thread_id"]
    cfg = {"configurable": {"thread_id": tid}}

    cmd = T.generate_synthetic_data.func(
        state={"dashboard_data": {}}, n_weeks=8, tool_call_id="g", config=cfg
    )
    dp = cmd.update["dataset_path"]
    assert os.path.isabs(dp) and dp.endswith("synthetic_mff_data.csv")
    assert (W.thread_dir(tid) / "synthetic_mff_data.csv").exists()
    assert "synthetic_mff_data.csv" in [f["name"] for f in store.list_files(tid)]


def test_fit_spec_accepts_bare_string_channels():
    """Regression: fit must tolerate media_channels/control_variables given as
    bare strings (what weaker models emit) — previously 'string indices must be
    integers'."""
    from mmm_framework.agents.tools import _normalize_spec_vars

    spec = {
        "kpi": "Sales",
        "media_channels": ["TV", "Digital", "Paid_Social"],
        "control_variables": ["Price_Index", "Distribution"],
    }
    _normalize_spec_vars(spec)
    assert spec["media_channels"] == [
        {"name": "TV"},
        {"name": "Digital"},
        {"name": "Paid_Social"},
    ]
    assert spec["control_variables"] == [
        {"name": "Price_Index"},
        {"name": "Distribution"},
    ]
    # the line that used to crash now works
    assert spec["media_channels"][0]["name"] == "TV"

    # dict form is preserved; malformed entries are dropped
    spec2 = {
        "media_channels": [{"name": "TV", "adstock": {"l_max": 8}}, None, 5, {"x": 1}]
    }
    _normalize_spec_vars(spec2)
    assert spec2["media_channels"] == [{"name": "TV", "adstock": {"l_max": 8}}]


# ── LM Studio (local OpenAI-compatible endpoint) ─────────────────────────────


def test_lmstudio_provider_builds_and_routes():
    from mmm_framework.agents.llm import (
        ModelConfig,
        build_llm,
        describe_active_config,
        list_lmstudio_models,
    )

    cfg = ModelConfig(provider="lmstudio", model="qwen2.5-7b-instruct")
    llm = build_llm(config=cfg)
    base = str(
        getattr(llm, "openai_api_base", "") or getattr(llm.root_client, "base_url", "")
    )
    assert type(llm).__name__ == "ChatOpenAI" and "localhost:1234" in base
    assert llm.model_name == "qwen2.5-7b-instruct"

    # a model override stays on the lmstudio endpoint (does NOT re-route to a cloud provider)
    llm2 = build_llm(config=cfg, model_name="llama-3.1-8b-instruct", api_key="x")
    assert (
        type(llm2).__name__ == "ChatOpenAI"
        and llm2.model_name == "llama-3.1-8b-instruct"
    )

    # custom base_url honored
    cfg2 = ModelConfig(
        provider="lmstudio", model="m", base_url="http://host.local:4321/v1"
    )
    llm3 = build_llm(config=cfg2)
    assert "host.local:4321" in str(
        getattr(llm3, "openai_api_base", "")
        or getattr(llm3.root_client, "base_url", "")
    )

    d = describe_active_config(cfg)
    assert d["requires_api_key"] is False and d["is_local_endpoint"] is True
    assert d["base_url"].endswith("/v1")

    # a per-request base_url override (X-Base-Url) retargets the local endpoint
    llm4 = build_llm(config=cfg, base_url="http://10.0.0.5:1234/v1")
    assert "10.0.0.5:1234" in str(
        getattr(llm4, "openai_api_base", "")
        or getattr(llm4.root_client, "base_url", "")
    )

    # ...but a base_url override is IGNORED for a cloud provider (no SSRF redirect)
    cloud = ModelConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="x")
    llm5 = build_llm(config=cloud, base_url="http://evil.example/v1")
    assert type(llm5).__name__ == "ChatAnthropic"

    # discovery degrades gracefully when LM Studio isn't running
    assert list_lmstudio_models(cfg, base_url="http://127.0.0.1:59999/v1") == []


def test_client_provider_override():
    """X-Provider lets a non-Vertex deployment switch provider entirely, but a
    Vertex-locked server ignores it (ADC stays authoritative)."""
    from mmm_framework.agents.llm import ModelConfig, build_llm

    def base_of(llm):
        return str(
            getattr(llm, "openai_api_base", "")
            or getattr(getattr(llm, "root_client", None), "base_url", "")
        )

    anthro = ModelConfig(provider="anthropic", model="claude-sonnet-4-6")
    # direct server -> switch to LM Studio
    llm = build_llm(
        config=anthro,
        provider="lmstudio",
        base_url="http://localhost:1234/v1",
        model_name="qwen2.5-7b",
    )
    assert type(llm).__name__ == "ChatOpenAI" and "1234" in base_of(llm)
    # direct server -> switch to OpenAI with a client key
    llm = build_llm(
        config=anthro, provider="openai", api_key="sk-x", model_name="gpt-4o"
    )
    assert type(llm).__name__ == "ChatOpenAI" and llm.model_name == "gpt-4o"
    # Vertex server is locked: X-Provider is ignored (stays on Vertex/ADC)
    vtx = ModelConfig(
        provider="vertex_anthropic", model="claude@x", location="us-east5"
    )
    llm = build_llm(config=vtx, provider="lmstudio", base_url="http://evil/v1")
    assert type(llm).__name__ == "ChatAnthropicVertex"
    # junk provider falls through to model-name inference (no crash)
    llm = build_llm(config=anthro, provider="nope", api_key="x", model_name="gpt-4o")
    assert type(llm).__name__ == "ChatOpenAI"


# ── thread-scoped model cache ────────────────────────────────────────────────


def test_model_cache_is_thread_scoped_and_bounded():
    from mmm_framework.agents import runtime as rt

    rt.set_current_thread("t1")
    rt.MODEL_CACHE["fitted_model"] = "A"
    rt.set_current_thread("t2")
    rt.MODEL_CACHE["fitted_model"] = "B"
    rt.set_current_thread("t1")
    assert rt.MODEL_CACHE.get("fitted_model") == "A"
    rt.set_current_thread("t2")
    assert rt.MODEL_CACHE.get("fitted_model") == "B"
    # LRU bound = 2: touching a third thread evicts the least-recently-used
    rt.set_current_thread("t3")
    rt.MODEL_CACHE["fitted_model"] = "C"
    assert (
        "t1" not in rt.MODEL_CACHE.thread_ids() or len(rt.MODEL_CACHE.thread_ids()) <= 2
    )


# ── projects + resolution ────────────────────────────────────────────────────


def test_projects_crud_and_resolution(store):
    assert any(p["project_id"] == "default" for p in store.list_projects())
    proj = store.create_project("Acme", "client")
    pid = proj["project_id"]
    sess = store.create_session(name="s", project_id=pid)
    assert store.resolve_project_id(sess["thread_id"]) == pid
    assert store.resolve_project_id("ghost") == "default"
    # delete reassigns sessions and refuses the default project
    assert store.delete_project(pid) is True
    assert store.resolve_project_id(sess["thread_id"]) == "default"
    assert store.delete_project("default") is False


# ── knowledge base ───────────────────────────────────────────────────────────


def test_kb_chunk_store_and_cosine_search(store, monkeypatch):
    from mmm_framework.agents import knowledge_base as kb

    proj = store.create_project("KB")
    pid = proj["project_id"]
    chunks = kb.chunk_text(
        "alpha adstock.\n\n" + "beta saturation. " * 40, size=200, overlap=40
    )
    assert len(chunks) >= 2

    doc = store.add_kb_document(pid, "n.md", "/tmp/n.md", "markdown", 10)
    vecs = [[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]]
    rows = []
    for i, (c, v) in enumerate(zip(chunks[:3], vecs)):
        blob, dim = kb._to_blob(v)
        rows.append((i, c, blob, dim))
    store.add_kb_chunks(doc["id"], pid, rows)
    store.set_kb_document_status(doc["id"], "ready", n_chunks=len(rows))

    monkeypatch.setattr(kb, "embed_query", lambda q, cfg=None: [0.95, 0.05, 0, 0])
    res = kb.search(pid, "tv adstock", top_k=2)
    assert res and res[0]["chunk_index"] in (0, 2)
    assert res[0]["score"] > 0.9

    assert store.list_kb_documents(pid)[0]["status"] == "ready"
    assert store.delete_kb_document(doc["id"]) is True
    assert store.iter_kb_chunks(pid) == []


def test_extract_text_formats(tmp_path):
    from mmm_framework.agents import knowledge_base as kb

    md = tmp_path / "a.md"
    md.write_text("# Title\nbody")
    assert "body" in kb.extract_text(md)
    assert kb.kind_for("x.pdf") == "pdf" and kb.kind_for("y.csv") == "csv"


# ── agent tools ──────────────────────────────────────────────────────────────


def test_new_tools_registered_and_schema_hides_injected_args(store):
    from mmm_framework.agents.tools import TOOLS

    names = {t.name for t in TOOLS}
    for expected in {
        "library_reference",
        "search_knowledge_base",
        "list_knowledge_base",
        "list_workspace_files",
        "read_workspace_file",
        "grep_workspace",
        "query_past_results",
        "run_budget_scenario",
        "run_marginal_analysis",
        "define_analysis_plan",
        "check_spec_divergence",
    }:
        assert expected in names, f"missing tool {expected}"

    grep = next(t for t in TOOLS if t.name == "grep_workspace")
    # injected config must NOT be exposed to the model
    assert "config" not in grep.args and "pattern" in grep.args


def test_workspace_tools_functional(store, monkeypatch):
    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    cfg = {"configurable": {"thread_id": sess["thread_id"]}}
    W.thread_dir(sess["thread_id"]).joinpath("notes.txt").write_text(
        "alpha adstock\nbeta\n"
    )

    assert "notes.txt" in T.list_workspace_files.invoke({}, config=cfg)
    assert "adstock" in T.grep_workspace.invoke({"pattern": "adstock"}, config=cfg)
    assert "alpha adstock" in T.read_workspace_file.invoke(
        {"path": "notes.txt"}, config=cfg
    )
    assert "menu" in T.library_reference.invoke({}).lower()


# ── API endpoints ────────────────────────────────────────────────────────────


@pytest.fixture()
def client(store, tmp_path, monkeypatch):
    import mmm_framework.api.main as main

    monkeypatch.setattr(main, "DB_PATH", store.DB_PATH)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


def test_api_projects_kb_and_downloads(client, store):
    # default project exists
    r = client.get("/projects")
    assert r.status_code == 200
    assert any(p["project_id"] == "default" for p in r.json()["projects"])

    pid = client.post("/projects", json={"name": "Acme"}).json()["project_id"]
    tid = client.post("/sessions", json={"name": "s", "project_id": pid}).json()[
        "thread_id"
    ]
    assert client.get(f"/sessions?project_id={pid}").json()["total"] == 1

    # empty KB list/search take no embedding call
    assert client.get(f"/projects/{pid}/kb").json()["total"] == 0
    assert (
        client.get(f"/projects/{pid}/kb/search", params={"q": "x"}).json()["results"]
        == []
    )

    # generated file download (inside workspace = allowed)
    from mmm_framework.agents import workspace as W

    fp = W.thread_dir(tid) / "out.csv"
    fp.write_text("a,b\n1,2\n")
    rec = store.register_file(tid, str(fp), "out.csv", "export", 8)
    assert client.get(f"/workspace/{tid}/files").json()["total"] == 1
    dl = client.get(f"/files/{rec['id']}/download")
    assert dl.status_code == 200 and dl.content == b"a,b\n1,2\n"

    # a path outside the allow-list is refused
    bad = store.register_file(tid, "/etc/hosts", "hosts", "export", 1)
    assert client.get(f"/files/{bad['id']}/download").status_code == 403


# ── Phase 3 PR-E.2: path / TOCTOU hardening ──────────────────────────────────


def test_safe_upload_name_flattens_and_guards():
    from mmm_framework.api.main import _safe_upload_name

    assert _safe_upload_name("../../etc/passwd", "d.bin") == "passwd"
    assert _safe_upload_name("/abs/evil.csv", "d.bin") == "evil.csv"
    assert _safe_upload_name("sub/dir/x.csv", "d.bin") == "x.csv"
    for bad in ("..", ".", "", None):
        assert _safe_upload_name(bad, "d.bin") == "d.bin"
    assert _safe_upload_name("normal.csv", "d.bin") == "normal.csv"


def test_download_rejects_symlink_escape(client, store):
    """A symlink inside the workspace pointing OUTSIDE the allowed roots is
    refused — is_within resolves the link target before serving (PR-E.2)."""
    import os

    from mmm_framework.agents import workspace as W

    if not os.path.exists("/etc/hosts"):
        pytest.skip("/etc/hosts not present")
    pid = client.post("/projects", json={"name": "Sym"}).json()["project_id"]
    tid = client.post("/sessions", json={"name": "s", "project_id": pid}).json()[
        "thread_id"
    ]
    link = W.thread_dir(tid) / "sneaky.csv"
    try:
        os.symlink("/etc/hosts", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    rec = store.register_file(tid, str(link), "sneaky.csv", "export", 1)
    assert client.get(f"/files/{rec['id']}/download").status_code == 403


def test_report_endpoints_serve_and_missing(client, monkeypatch, tmp_path):
    """The report endpoints serve via the TOCTOU-safe path now; missing keeps the
    helpful 404 JSON, present serves inline html (cwd is still an allowed root)."""
    monkeypatch.chdir(tmp_path)  # isolate from any repo-root agent_mmm_report.html
    r = client.get("/report")
    assert r.status_code == 404 and "error" in r.json()
    (tmp_path / "agent_mmm_report.html").write_text("<html>hi</html>")
    r2 = client.get("/report")
    assert r2.status_code == 200
    assert r2.headers["content-type"].startswith("text/html")
    assert r2.content == b"<html>hi</html>"
    # download variant: same bytes, attachment disposition
    r3 = client.get("/report/download")
    assert r3.status_code == 200
    assert "attachment" in r3.headers.get("content-disposition", "")
    assert r3.content == b"<html>hi</html>"


def test_upload_filename_traversal_is_flattened(client, store):
    """A crafted upload filename can't escape the session dir — it's flattened to
    a basename and safe_join'd (PR-E.2)."""
    from mmm_framework.agents import workspace as W

    pid = client.post("/projects", json={"name": "U"}).json()["project_id"]
    tid = client.post("/sessions", json={"name": "s", "project_id": pid}).json()[
        "thread_id"
    ]
    r = client.post(
        f"/upload?thread_id={tid}",
        files={"file": ("../../evil.csv", b"x,y\n1,2\n", "text/csv")},
    )
    assert r.status_code == 200
    td = W.thread_dir(tid)
    assert (td / "evil.csv").exists()  # landed inside the session dir
    assert not (td.parent.parent / "evil.csv").exists()  # nothing escaped


# ── warm-kernel namespace persistence ────────────────────────────────────────


def _new_thread(store):
    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    tid = sess["thread_id"]
    return tid, {"configurable": {"thread_id": tid}}


def test_namespace_persists_across_calls(store):
    """A variable defined in one execute_python call is visible in the next."""
    from mmm_framework.agents import tools as T

    tid, cfg = _new_thread(store)
    T.execute_python.func(
        state={"dashboard_data": {}},
        code="x = 41; y = pd.DataFrame({'a': [1, 2, 3]})",
        tool_call_id="c1",
        config=cfg,
    )
    r = T.execute_python.func(
        state={"dashboard_data": {}},
        code="print('x is', x + 1); print('len', len(y))",
        tool_call_id="c2",
        config=cfg,
    )
    out = r.update["messages"][0].content
    assert "NameError" not in out
    assert "x is 42" in out and "len 3" in out


def test_namespace_isolated_per_thread(store):
    """Variables don't leak between sessions (thread-scoped buckets)."""
    from mmm_framework.agents import tools as T

    tid_a, cfg_a = _new_thread(store)
    tid_b, cfg_b = _new_thread(store)
    T.execute_python.func(
        state={"dashboard_data": {}}, code="secret = 7", tool_call_id="a", config=cfg_a
    )
    r = T.execute_python.func(
        state={"dashboard_data": {}},
        code="print(secret)",
        tool_call_id="b",
        config=cfg_b,
    )
    assert "NameError" in r.update["messages"][0].content


def test_reset_namespace_clears_vars_with_hint(store):
    """reset_namespace wipes user vars; a missing name yields a self-healing hint."""
    from mmm_framework.agents import tools as T

    tid, cfg = _new_thread(store)
    T.execute_python.func(
        state={"dashboard_data": {}}, code="z = 99", tool_call_id="c1", config=cfg
    )
    T.reset_namespace.func(tool_call_id="r", config=cfg)
    r = T.execute_python.func(
        state={"dashboard_data": {}}, code="print(z)", tool_call_id="c2", config=cfg
    )
    out = r.update["messages"][0].content
    assert "NameError" in out
    assert "Hint:" in out and "`z`" in out


def test_save_and_load_result_survives_reset(store):
    """save_result persists to disk; load_result reloads it after a kernel reset."""
    from mmm_framework.agents import tools as T

    tid, cfg = _new_thread(store)
    T.execute_python.func(
        state={"dashboard_data": {}},
        code="save_result('mytbl', pd.DataFrame({'a': [1, 2]}))",
        tool_call_id="c1",
        config=cfg,
    )
    T.reset_namespace.func(tool_call_id="r", config=cfg)
    r = T.execute_python.func(
        state={"dashboard_data": {}},
        code="t = load_result('mytbl'); print('rows', len(t)); print(list_saved_results())",
        tool_call_id="c2",
        config=cfg,
    )
    out = r.update["messages"][0].content
    assert "NameError" not in out
    assert "rows 2" in out and "mytbl" in out


def test_dataset_autobinds_as_df(store):
    """The active dataset is auto-loaded as `df` even on a cold kernel."""
    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W

    tid, cfg = _new_thread(store)
    (W.thread_dir(tid) / "data.csv").write_text(
        "date,sales\n2024-01-01,100\n2024-01-08,120\n"
    )
    r = T.execute_python.func(
        state={"dashboard_data": {}, "dataset_path": "data.csv"},
        code="print('cols', list(df.columns)); print('n', len(df))",
        tool_call_id="c1",
        config=cfg,
    )
    out = r.update["messages"][0].content
    assert "NameError" not in out
    assert "cols ['date', 'sales']" in out and "n 2" in out


def test_save_load_handles_dotted_names(store):
    """Names with dots (e.g. 'q4.2024') must not collide via suffix truncation."""
    from mmm_framework.agents import tools as T

    tid, cfg = _new_thread(store)
    r = T.execute_python.func(
        state={"dashboard_data": {}},
        code=(
            "save_result('q4.2024', pd.DataFrame({'a': [1]}))\n"
            "save_result('q4.2023', pd.DataFrame({'a': [2, 2]}))\n"
            "print('n2024', len(load_result('q4.2024')))\n"
            "print('n2023', len(load_result('q4.2023')))\n"
            "print(sorted(list_saved_results()))"
        ),
        tool_call_id="c1",
        config=cfg,
    )
    out = r.update["messages"][0].content
    assert "NameError" not in out and "Error executing" not in out
    assert "n2024 1" in out and "n2023 2" in out  # distinct, no collision
    assert "['q4.2023', 'q4.2024']" in out


def test_namespace_precedence_system_wins_user_persists(store):
    """System names re-win every call; user-defined names persist alongside."""
    from mmm_framework.agents import tools as T

    tid, cfg = _new_thread(store)
    T.execute_python.func(
        state={"dashboard_data": {}},
        code="pd = 5; keep = 123",  # shadow a system name + define a user name
        tool_call_id="c1",
        config=cfg,
    )
    r = T.execute_python.func(
        state={"dashboard_data": {}},
        code="print('has_df', hasattr(pd, 'DataFrame')); print('keep', keep)",
        tool_call_id="c2",
        config=cfg,
    )
    out = r.update["messages"][0].content
    assert "has_df True" in out  # pd restored to the module
    assert "keep 123" in out  # user var survived


def test_df_reloads_when_dataset_changes(store):
    """`df` refreshes when the active dataset_path changes (new upload)."""
    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W

    tid, cfg = _new_thread(store)
    wd = W.thread_dir(tid)
    (wd / "d1.csv").write_text("a\n1\n")
    (wd / "d2.csv").write_text("a\n1\n2\n3\n")

    r1 = T.execute_python.func(
        state={"dashboard_data": {}, "dataset_path": "d1.csv"},
        code="print('n', len(df))",
        tool_call_id="c1",
        config=cfg,
    )
    assert "n 1" in r1.update["messages"][0].content

    r2 = T.execute_python.func(
        state={"dashboard_data": {}, "dataset_path": "d2.csv"},
        code="print('n', len(df))",
        tool_call_id="c2",
        config=cfg,
    )
    assert "n 3" in r2.update["messages"][0].content  # refreshed to the new dataset


def test_save_result_not_registered_as_download(store):
    """save_result snapshots (results/) stay out of the user-facing Files tab."""
    from mmm_framework.agents import tools as T

    tid, cfg = _new_thread(store)
    T.execute_python.func(
        state={"dashboard_data": {}},
        code="save_result('internal', pd.DataFrame({'a': [1]}))",
        tool_call_id="c1",
        config=cfg,
    )
    names = [f["name"] for f in store.list_files(tid)]
    assert not any(n.startswith("internal") for n in names)


# ── session → portable .py export ────────────────────────────────────────────


def test_session_export_runs_standalone(store, tmp_path, monkeypatch):
    """The exported script reconstitutes injected state (df) and actually RUNS —
    not a code dump that NameErrors. This is the test that makes it real."""
    from mmm_framework.agents.session_export import build_session_script

    tid, _ = _new_thread(store)
    # a dataset the session 'worked on', registered like an upload
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n3,4\n5,6\n")
    store.register_file(tid, str(data), "data.csv", "dataset", data.stat().st_size)
    # a cell that depends on the INJECTED `df` and the INJECTED save_result helper
    store.add_artifact(
        tid,
        "code_snippet",
        {
            "call_id": "c1",
            "code": "print('rows', len(df))\nsave_result('head', df.head(2))",
        },
    )
    store.add_artifact(
        tid, "text_output", {"call_id": "c1", "stdout": "rows 3", "is_error": False}
    )

    script = build_session_script(tid)
    # preamble reconstitutes what the kernel injected
    assert "import mmm_framework as mmf" in script
    assert "df = pd.read_csv(dataset_path)" in script
    assert "def save_result(" in script

    # …and it executes cleanly with the dataset present in cwd
    monkeypatch.chdir(tmp_path)
    g: dict = {}
    exec(compile(script, "<export>", "exec"), g)  # must not raise
    saved = list((tmp_path / "results").glob("head.*"))
    assert saved, "save_result() in the exported cell did not produce a file"


def test_session_export_reconstitutes_fitted_model_and_marks_errors(store):
    """A fit (a TOOL call, no code_snippet) must appear in the preamble as a model
    load; an errored cell is marked, not dropped."""
    from mmm_framework.agents.session_export import build_session_script

    tid, _ = _new_thread(store)
    store.add_artifact(
        tid,
        "model_run",
        {
            "run_name": "run_X",
            "model_path": "mmm_models/run_X",
            "dataset_path": "/abs/data.csv",
        },
    )
    store.add_artifact(
        tid, "code_snippet", {"call_id": "c1", "code": "print(results.summary())"}
    )
    store.add_artifact(
        tid,
        "text_output",
        {"call_id": "c1", "stdout": "NameError: results", "is_error": True},
    )

    script = build_session_script(tid)
    assert "MMMSerializer().load('mmm_models/run_X')" in script  # fit reconstituted
    assert "print(results.summary())" in script  # cell kept
    assert "raised an error" in script  # errored cell marked, not dropped


# ── PR1: shared kernel helpers (extracted for InProcess/Subprocess parity) ────


def test_format_execution_error_invariants():
    """The load-bearing 'Error executing code' substring + NameError hint must be
    produced by the shared formatter (api/main.py keys is_error off it;
    session_export marks cells with it; both kernel impls will reuse it)."""
    from mmm_framework.agents.tools import format_execution_error

    plain = format_execution_error("Traceback...\nValueError: x")
    assert plain.startswith("Error executing code:\n")
    assert "Hint:" not in plain

    named = format_execution_error("tb", is_name_error=True, missing_name="tbl")
    assert "Error executing code" in named
    assert "`tbl`" in named and "load_result" in named

    anon = format_execution_error("tb", is_name_error=True, missing_name=None)
    assert "a variable" in anon


def test_normalize_figure_applies_palette():
    """The extracted _normalize_figure remaps default Plotly colors to the design
    palette and sets the colorway (same behavior both kernel impls must produce)."""
    import plotly.graph_objects as go
    from mmm_framework.agents.tools import _normalize_figure, _PALETTE

    fig = go.Figure(go.Bar(x=["a", "b"], y=[1, 2], marker_color="#636efa"))
    _normalize_figure(fig)
    # default plotly blue (#636efa) maps to the first palette color
    assert fig.data[0].marker.color.lower() == _PALETTE[0].lower()
    assert tuple(c.lower() for c in fig.layout.colorway) == tuple(
        c.lower() for c in _PALETTE
    )


# ── regression: read tools tolerate bare-string channels/controls ────────────


def test_read_tools_tolerate_bare_string_vars(store, tmp_path, monkeypatch):
    """get_current_config/save_config/load_config must not crash with 'string
    indices must be integers' when model_spec has bare-string channels/controls
    (weaker models emit the simple form). Reproduces the live /chat crash."""
    from mmm_framework.agents import tools as T

    monkeypatch.chdir(tmp_path)  # save_config writes under cwd/mmm_configs
    tid, cfg = _new_thread(store)
    spec = {
        "kpi": "sales",
        "media_channels": ["TV", "Digital"],  # bare strings (not dicts)
        "control_variables": ["price", "holiday"],  # the case that crashed
    }
    state = {"model_spec": spec, "dashboard_data": {}}

    # get_current_config — the reported crash site
    r = T.get_current_config.func(state=state, tool_call_id="c1")
    txt = r.update["messages"][0].content
    assert "string indices" not in txt
    assert "`TV`" in txt and "price" in txt and "holiday" in txt

    # save_config then load_config round-trip with bare-string vars
    s = T.save_config.func(state=state, name="bare", tool_call_id="c2")
    assert "string indices" not in s.update["messages"][0].content
    loaded = T.load_config.func(
        state={"dashboard_data": {}}, name="bare", tool_call_id="c3"
    )
    lt = loaded.update["messages"][0].content
    assert "string indices" not in lt and "TV" in lt and "price" in lt


@pytest.mark.slow
def test_fit_save_load_roundtrip(store, tmp_path, monkeypatch):
    """End-to-end proof of the Phase 2 PR-C bug fixes: a real fit now actually
    auto-saves to disk (the auto-save call was broken), and the saved model
    loads back and serves an interpretation tool."""
    import json

    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W

    monkeypatch.chdir(tmp_path)  # mmm_models/ lands here, not the repo
    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    tid = sess["thread_id"]
    cfg = {"configurable": {"thread_id": tid}}

    T.generate_synthetic_data.func(
        state={"dashboard_data": {}}, n_weeks=30, tool_call_id="g", config=cfg
    )
    ds = str(W.thread_dir(tid) / "synthetic_mff_data.csv")
    spec = {
        "kpi": "Sales",
        "media_channels": [
            {"name": n} for n in ("TV", "Digital", "Paid_Social", "Radio")
        ],
        "control_variables": [{"name": "Price_Index"}, {"name": "Distribution"}],
        "time_granularity": "weekly",
        "inference": {"chains": 1, "draws": 50, "tune": 50, "target_accept": 0.8},
    }

    fit = T.fit_mmm_model.func(
        state={"dashboard_data": {}},
        dataset_path=ds,
        model_spec=json.dumps(spec),
        tool_call_id="f",
        config=cfg,
    )
    summary = fit.update["messages"][0].content
    assert "fitted successfully" in summary.lower()
    assert (
        "Auto-save failed" not in summary and "Auto-saved" in summary
    )  # the bug is fixed
    run_name = fit.update["dashboard_data"]["model_run"]["run_name"]
    model_dir = tmp_path / "mmm_models" / run_name
    assert (model_dir / "metadata.json").exists()  # the model is actually on disk

    # cold the cache, then load from disk and interpret
    T._MODEL_CACHE.clear_thread(tid)
    loaded = T.load_fitted_model.func(
        state={"model_spec": spec, "dataset_path": ds},
        name=run_name,
        tool_call_id="l",
        config=cfg,
    )
    assert "loaded" in loaded.update["messages"][0].content.lower()
    roi = T.get_roi_metrics.func(
        state={"dashboard_data": {}}, tool_call_id="r", config=cfg
    )
    assert "No fitted model" not in roi.update["messages"][0].content


@pytest.mark.slow
def test_subprocess_fit_then_interpret_removes_boundary(store):
    """The Phase-2 milestone: fit IN the subprocess kernel, then a model op finds
    the model there (no longer the Phase-1 'no model in subprocess' boundary)."""
    pytest.importorskip("jupyter_client")
    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W
    from mmm_framework.agents.kernels import SubprocessKernel

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    tid = sess["thread_id"]
    cfg = {"configurable": {"thread_id": tid}}
    T.generate_synthetic_data.func(
        state={"dashboard_data": {}}, n_weeks=30, tool_call_id="g", config=cfg
    )
    ds = str(W.thread_dir(tid) / "synthetic_mff_data.csv")
    spec = {
        "kpi": "Sales",
        "media_channels": [
            {"name": n} for n in ("TV", "Digital", "Paid_Social", "Radio")
        ],
        "control_variables": [{"name": "Price_Index"}, {"name": "Distribution"}],
        "time_granularity": "weekly",
        "inference": {"chains": 1, "draws": 50, "tune": 50, "target_accept": 0.8},
    }

    k = SubprocessKernel()
    try:
        info = k.fit(spec, ds)
        assert "error" not in info, info
        assert "fitted successfully" in info["summary"].lower()
        # the model is now a kernel global -> run_model_op finds it (boundary gone)
        roi = k.run_model_op("roi_metrics", {})
        assert roi.get("error") is None, roi
        assert "ROI" in (roi.get("content") or "")
    finally:
        k.shutdown()


@pytest.mark.slow
def test_subprocess_cold_reload_after_eviction(store):
    """PR-C.3 exit: fit in a subprocess kernel, EVICT it (a 2nd session under a
    1-kernel cap), then interpret session A again -> the respawned (cold) kernel
    rehydrates the model from disk and still works."""
    pytest.importorskip("jupyter_client")
    import json

    from mmm_framework.agents import tools as T
    from mmm_framework.agents import workspace as W
    from mmm_framework.agents.kernels import KernelManager, SubprocessKernel

    proj = store.create_project("P")
    sA = store.create_session(name="a", project_id=proj["project_id"])["thread_id"]
    sB = store.create_session(name="b", project_id=proj["project_id"])["thread_id"]
    T.generate_synthetic_data.func(
        state={"dashboard_data": {}},
        n_weeks=30,
        tool_call_id="g",
        config={"configurable": {"thread_id": sA}},
    )
    ds = str(W.thread_dir(sA) / "synthetic_mff_data.csv")
    spec = {
        "kpi": "Sales",
        "media_channels": [
            {"name": n} for n in ("TV", "Digital", "Paid_Social", "Radio")
        ],
        "control_variables": [{"name": "Price_Index"}, {"name": "Distribution"}],
        "time_granularity": "weekly",
        "inference": {"chains": 1, "draws": 50, "tune": 50, "target_accept": 0.8},
    }

    mgr = KernelManager("subprocess", {"subprocess": SubprocessKernel})
    mgr._max = 1  # one live kernel -> spawning B evicts A
    try:
        kA = mgr.get_or_spawn(sA)
        assert "error" not in kA.fit(spec, ds)
        assert kA.run_model_op("roi_metrics", {}).get("error") is None  # warm

        mgr.get_or_spawn(sB).run_model_op("roi_metrics", {})  # evicts (shuts down) A

        kA2 = mgr.get_or_spawn(sA)  # NEW cold kernel for A
        assert kA2 is not kA
        roi = kA2.run_model_op("roi_metrics", {})  # must rehydrate A's model from disk
        assert roi.get("error") is None, roi
        assert "ROI" in (roi.get("content") or "")
    finally:
        mgr.shutdown_all()


# ── Phase 3 PR-F.6: hosted profile activation ─────────────────────────────────


def test_hosted_profile_toggle_and_kernel_default(monkeypatch):
    from mmm_framework.agents import profile

    monkeypatch.delenv("MMM_AGENT_HOSTED", raising=False)
    monkeypatch.delenv("MMM_AGENT_KERNEL", raising=False)
    assert not profile.is_hosted()
    assert profile.default_kernel_impl() == "inprocess"
    monkeypatch.setenv("MMM_AGENT_HOSTED", "1")
    assert profile.is_hosted()
    assert profile.default_kernel_impl() == "container"  # sandboxed by default
    monkeypatch.setenv("MMM_AGENT_KERNEL", "subprocess")  # explicit wins
    assert profile.default_kernel_impl() == "subprocess"


def test_report_path_and_allowed_roots_drop_cwd_when_hosted(monkeypatch, tmp_path):
    from pathlib import Path

    from mmm_framework.agents import workspace as W

    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("MMM_AGENT_HOSTED", raising=False)
    # dev: legacy CWD report path; CWD is an allowed root
    assert W.report_path("r.html").parent == Path.cwd()
    assert Path.cwd().resolve() in W.allowed_roots()
    # hosted: per-session under the workspace; CWD dropped from the allow-roots
    monkeypatch.setenv("MMM_AGENT_HOSTED", "1")
    rp = W.report_path("r.html", "sessA")
    assert str(W.thread_dir("sessA")) in str(rp)
    assert Path.cwd().resolve() not in W.allowed_roots()
    assert W.workspace_root() in W.allowed_roots()


def test_chat_rejects_guessable_thread_when_hosted(client, store, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_HOSTED", "1")
    # the guessable default and any client-invented id are refused (403), with no
    # silent auto-create
    assert (
        client.post(
            "/chat", json={"message": "hi", "thread_id": "default_thread"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/chat", json={"message": "hi", "thread_id": "made-up-not-minted"}
        ).status_code
        == 403
    )
    # a server-minted session is NOT rejected by the guard (it exists in the store)
    pid = client.post("/projects", json={"name": "P"}).json()["project_id"]
    tid = client.post("/sessions", json={"name": "s", "project_id": pid}).json()[
        "thread_id"
    ]
    assert store.get_session(tid) is not None
