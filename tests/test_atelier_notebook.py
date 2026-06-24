"""Tests for the Atelier notebook (api/main.py model-garden/notebook/*): the
cell-execution worker (source staging + GardenModel binding + plot/table/error
output mapping), the non-blocking cell job + poll endpoint with org-scoping, and
the persisted-notebook doc round-trip (seeded starter -> save -> reload -> upsert).

Uses the default in-process kernel (no extra service) and avoids a real PyMC fit
so the suite stays fast."""

from __future__ import annotations

import asyncio
import json

import pytest

# A trivial-but-valid bespoke model: a single BayesianMMM subclass, so
# find_garden_class resolves it without a GARDEN_MODEL hint.
GOOD_SRC = (
    "from mmm_framework.garden import CustomMMM\n"
    "\n"
    "class DemoModel(CustomMMM):\n"
    '    """Trivial bespoke model for the notebook tests."""\n'
    "    pass\n"
)
# Imports cleanly but exposes no resolvable garden class -> setup cell errors.
BAD_SRC = "x = 1  # no BayesianMMM subclass here\n"


@pytest.fixture()
def main(tmp_path, monkeypatch):
    """The api.main module wired to a temp sessions DB + temp workspace."""
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    from mmm_framework.api import main as M

    # Fresh per-test source-rev cache so staging always (re)runs the setup cell.
    M._NOTEBOOK_SOURCE_REV.clear()
    return M


def _live_req(
    main, code, *, name="DemoModel", rev="r1", src=GOOD_SRC, dataset_path=None
):
    return {
        "tid": main._notebook_tid("dev-org", name, None),
        "name": name,
        "ver_seg": "draft",
        "source_code": src,
        "source_path": None,
        "source_rev": rev,
        "dataset_path": dataset_path,
        "code": code,
    }


class TestCellWorker:
    """_notebook_cell_sync: the synchronous kernel worker behind the cell job."""

    def test_binds_garden_model_and_prints(self, main):
        res = main._notebook_cell_sync(
            _live_req(main, "print('class is', GardenModel.__name__)")
        )
        assert res["is_error"] is False
        assert "DemoModel" in res["stdout"]

    def test_show_table_yields_table_ref(self, main):
        res = main._notebook_cell_sync(
            _live_req(
                main,
                "import pandas as pd\n"
                "show_table(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), title='Mini')",
            )
        )
        assert res["is_error"] is False
        assert len(res["tables"]) == 1
        assert res["tables"][0]["id"]

    def test_fig_show_yields_plot_ref(self, main):
        res = main._notebook_cell_sync(
            _live_req(
                main,
                "import plotly.express as px\n"
                "px.bar(x=[1, 2, 3], y=[4, 5, 6], title='Bars').show()",
            )
        )
        assert res["is_error"] is False
        assert len(res["plots"]) == 1
        assert res["plots"][0]["id"]
        assert res["plots"][0]["title"] == "Bars"

    def test_runtime_error_is_flagged(self, main):
        res = main._notebook_cell_sync(_live_req(main, "raise ValueError('boom')"))
        assert res["is_error"] is True
        assert "boom" in res["stdout"]

    def test_bad_source_surfaces_setup_error(self, main):
        res = main._notebook_cell_sync(
            _live_req(main, "print('unreached')", src=BAD_SRC, rev="bad1")
        )
        assert res["is_error"] is True
        assert res.get("setup_error") is True
        # rev NOT cached on a setup failure -> a fixed source re-imports next run.
        assert main._NOTEBOOK_SOURCE_REV.get(_live_req(main, "")["tid"]) != "bad1"

    def test_user_model_persists_across_cells(self, main):
        """A model the user fits in one cell survives into the next (warm kernel;
        the per-call header never clobbers user-assigned `mmm`)."""
        main._notebook_cell_sync(_live_req(main, "my_state = 41"))
        res = main._notebook_cell_sync(_live_req(main, "print('val', my_state + 1)"))
        assert res["is_error"] is False
        assert "val 42" in res["stdout"]

    def test_source_staged_as_non_py_file(self, main):
        """Regression: the model source is staged as a NON-.py file so a dev
        server on `uvicorn --reload` does NOT restart on every cell (a restart
        wipes the warm kernel and loses cross-cell vars like `data`)."""
        import os
        from mmm_framework.agents import workspace as _ws

        res = main._notebook_cell_sync(_live_req(main, "print(GardenModel.__name__)"))
        assert res["is_error"] is False
        tid = main._notebook_tid("dev-org", "DemoModel", None)
        staged_dir = _ws.garden_loaded_dir("DemoModel", "draft", tid)
        files = os.listdir(staged_dir)
        assert files, "source was not staged"
        assert not any(f.endswith(".py") for f in files), f"staged a .py file: {files}"


class TestCellEndpoint:
    """POST /model-garden/notebook/cell -> poll GET .../cell/{job_id}."""

    async def _run_to_completion(self, main, body, principal=None):
        principal = principal or main._DEV_PRINCIPAL
        resp = await main.start_notebook_cell(body, principal)
        job_id = json.loads(resp.body)["job_id"]
        for _ in range(200):  # ~10s budget; a print cell finishes in well under
            poll = await main.get_notebook_cell(job_id, principal)
            payload = json.loads(poll.body)
            if payload["status"] in ("done", "error"):
                return job_id, payload
            await asyncio.sleep(0.05)
        raise AssertionError("cell job did not finish in time")

    @pytest.mark.asyncio
    async def test_cell_job_lifecycle(self, main):
        body = main.NotebookCellRequest(
            name="DemoModel",
            version=None,
            source_code=GOOD_SRC,
            source_rev="r1",
            code="print('hi from', GardenModel.__name__)",
        )
        _job_id, payload = await self._run_to_completion(main, body)
        assert payload["status"] == "done"
        assert payload["result"]["is_error"] is False
        assert "DemoModel" in payload["result"]["stdout"]

    @pytest.mark.asyncio
    async def test_poll_is_org_scoped(self, main):
        from fastapi import HTTPException
        from mmm_framework.auth.models import AuthContext, Role

        body = main.NotebookCellRequest(
            name="DemoModel", source_code=GOOD_SRC, source_rev="r1", code="print('x')"
        )
        job_id, _ = await self._run_to_completion(main, body)
        other = AuthContext(
            user_id="u2", org_id="other-org", email="o@x", org_role=Role.OWNER
        )
        with pytest.raises(HTTPException) as ei:
            await main.get_notebook_cell(job_id, other)
        assert ei.value.status_code == 404

    @pytest.mark.asyncio
    async def test_viewer_cannot_run_cells(self, main):
        from fastapi import HTTPException
        from mmm_framework.auth.models import AuthContext, Role

        viewer = AuthContext(
            user_id="v", org_id="dev-org", email="v@x", org_role=Role.VIEWER
        )
        body = main.NotebookCellRequest(name="DemoModel", code="print('x')")
        with pytest.raises(HTTPException) as ei:
            await main.start_notebook_cell(body, viewer)
        assert ei.value.status_code == 403


class TestDiagnosisPrompt:
    """build_copilot_system_prompt(notebook=...): the cell-diagnosis grounding the
    Notebook copilot uses to fix failed cells."""

    def test_plain_prompt_has_no_diagnosis_pack(self):
        from mmm_framework.agents.garden_authoring import build_copilot_system_prompt

        prompt = build_copilot_system_prompt("class M: pass")
        assert "Diagnosing a failed notebook cell" not in prompt
        # The base authoring knowledge + persona are still present.
        assert "modeling copilot" in prompt

    def test_diagnose_prompt_grounds_on_cell_traceback_and_data(self):
        from mmm_framework.agents.garden_authoring import build_copilot_system_prompt

        prompt = build_copilot_system_prompt(
            "class M:\n    pass\n",
            notebook={
                "cell_code": "mmm.fit(method='map')  # MY_FAILING_CALL",
                "traceback": "SamplingError: Initial evaluation ... y_obs: -inf",
                "dataset_preview": "date,Sales,TV\n2020-01-01,100,5",
                "other_cells": ["data = load()  # SIBLING_CELL", ""],
                "is_error": True,
            },
        )
        # The diagnosis knowledge pack is included...
        assert "Diagnosing a failed notebook cell" in prompt
        assert "Initial evaluation of model at starting point failed" in prompt
        # ...and the actual cell, traceback, dataset, and sibling cells are grounded.
        assert "MY_FAILING_CALL" in prompt
        assert "y_obs: -inf" in prompt
        assert "2020-01-01,100,5" in prompt
        assert "SIBLING_CELL" in prompt
        # Diagnosis framing: fix the failing cell.
        assert "FAILED" in prompt
        assert "corrected cell" in prompt

    def test_non_error_notebook_context_is_assistive_not_diagnostic(self):
        from mmm_framework.agents.garden_authoring import build_copilot_system_prompt

        prompt = build_copilot_system_prompt(
            "class M: pass",
            notebook={"cell_code": "df.head()", "is_error": False},
        )
        # Diagnosis knowledge still loaded (notebook-aware), but the "FAILED"
        # diagnose instruction is not used for a non-error turn.
        assert "Diagnosing a failed notebook cell" in prompt
        assert "working in this notebook cell" in prompt
        assert "and it FAILED" not in prompt

    def test_diagnose_framing_survives_empty_traceback(self):
        """An errored cell with no captured output still gets the fix-it framing
        (gated on is_error alone, not is_error AND a non-empty traceback)."""
        from mmm_framework.agents.garden_authoring import build_copilot_system_prompt

        prompt = build_copilot_system_prompt(
            "class M: pass",
            notebook={"cell_code": "mmm.fit()", "traceback": "", "is_error": True},
        )
        assert "it FAILED" in prompt
        assert "working in this notebook cell" not in prompt
        # No traceback block when there's nothing to show.
        assert "Error / traceback:" not in prompt

    def test_other_cells_are_capped(self):
        """A huge other_cells payload is capped before the join (memory bound),
        regardless of what the (untrusted) client sends."""
        from mmm_framework.agents.garden_authoring import build_copilot_system_prompt

        prompt = build_copilot_system_prompt(
            "class M: pass",
            notebook={
                "cell_code": "x = 1",
                "is_error": False,
                "other_cells": [f"CELL_{i}" for i in range(5000)],
            },
        )
        # Only the first _MAX_OTHER_CELLS (40) survive into the prompt.
        assert "CELL_0" in prompt
        assert "CELL_39" in prompt
        assert "CELL_40" not in prompt
        assert "CELL_4999" not in prompt

    def test_long_context_is_truncated(self):
        from mmm_framework.agents.garden_authoring import build_copilot_system_prompt

        prompt = build_copilot_system_prompt(
            "class M: pass",
            notebook={
                "cell_code": "x = 1\n" * 5000,
                "traceback": "boom\n" * 5000,
                "is_error": True,
            },
        )
        assert "… (truncated)" in prompt

    def test_request_model_accepts_notebook_context(self, main):
        req = main.GardenCopilotRequest(
            messages=[{"role": "user", "content": "fix it"}],
            source_code="class M: pass",
            notebook={
                "cell_code": "mmm.fit()",
                "traceback": "boom",
                "is_error": True,
                "other_cells": ["a = 1"],
            },
        )
        assert req.notebook is not None
        assert req.notebook.is_error is True
        assert req.notebook.cell_code == "mmm.fit()"
        # Backward compatible: a turn with no notebook context still parses.
        plain = main.GardenCopilotRequest(source_code="x = 1")
        assert plain.notebook is None


class TestNotebookDoc:
    """GET/PUT /model-garden/notebook: seeded starter, save, reload, upsert."""

    @pytest.mark.asyncio
    async def test_get_seeds_starter_when_absent(self, main):
        resp = await main.get_notebook("BrandNew", main._DEV_PRINCIPAL, None)
        doc = json.loads(resp.body)
        assert doc.get("seeded") is True
        kinds = [c["type"] for c in doc["cells"]]
        assert kinds[0] == "markdown" and "code" in kinds

    @pytest.mark.asyncio
    async def test_save_then_reload_round_trip(self, main):
        cells = [{"id": "c1", "type": "code", "source": "print(1)", "outputs": None}]
        save = await main.save_notebook(
            main.NotebookSaveRequest(name="Saver", version=None, cells=cells),
            main._DEV_PRINCIPAL,
        )
        saved_id = json.loads(save.body)["id"]
        resp = await main.get_notebook("Saver", main._DEV_PRINCIPAL, None)
        doc = json.loads(resp.body)
        assert not doc.get("seeded")
        assert doc["cells"] == cells

        # Second save UPSERTS the same artifact (singleton per notebook).
        save2 = await main.save_notebook(
            main.NotebookSaveRequest(name="Saver", version=None, cells=cells + cells),
            main._DEV_PRINCIPAL,
        )
        assert json.loads(save2.body)["id"] == saved_id
        tid = main._notebook_tid("dev-org", "Saver", None)
        from mmm_framework.api import sessions as S

        docs = [a for a in S.list_artifacts(tid) if a["kind"] == "atelier_notebook"]
        assert len(docs) == 1  # never duplicated


class TestDemoNotebookSeeding:
    """A model's curated ``manifest["demo_notebook"]`` seeds the Atelier notebook
    in preference to the generic starter; absent, the starter is used; a persisted
    user doc always wins."""

    def _register(self, demo_notebook):
        from mmm_framework.agents.garden_registry import register_garden_model_core

        return register_garden_model_core(
            org_id="dev-org",
            source_code=GOOD_SRC,
            name="DemoModel",
            demo_notebook=demo_notebook,
        )

    @pytest.mark.asyncio
    async def test_get_seeds_model_demo_notebook(self, main):
        demo = [
            {
                "id": "d1",
                "type": "markdown",
                "source": "# Curated demo",
                "outputs": None,
            },
            {"id": "d2", "type": "code", "source": "print('hi')", "outputs": None},
        ]
        self._register(demo)
        resp = await main.get_notebook("DemoModel", main._DEV_PRINCIPAL, None)
        doc = json.loads(resp.body)
        assert doc.get("seeded") is True
        assert doc["cells"] == demo  # the curated cells, not the generic starter

    @pytest.mark.asyncio
    async def test_get_falls_back_to_starter_without_demo(self, main):
        self._register(None)  # registered, but no demo_notebook
        resp = await main.get_notebook("DemoModel", main._DEV_PRINCIPAL, None)
        doc = json.loads(resp.body)
        assert doc.get("seeded") is True
        assert doc["cells"][0]["type"] == "markdown" and len(doc["cells"]) > 1
        assert all("Curated demo" not in c.get("source", "") for c in doc["cells"])

    @pytest.mark.asyncio
    async def test_persisted_doc_wins_over_demo(self, main):
        self._register([{"id": "d1", "type": "code", "source": "x=1", "outputs": None}])
        cells = [{"id": "c1", "type": "code", "source": "print(2)", "outputs": None}]
        await main.save_notebook(
            main.NotebookSaveRequest(name="DemoModel", version=None, cells=cells),
            main._DEV_PRINCIPAL,
        )
        resp = await main.get_notebook("DemoModel", main._DEV_PRINCIPAL, None)
        doc = json.loads(resp.body)
        assert not doc.get("seeded")
        assert doc["cells"] == cells  # the user's saved doc, not the demo


class TestDemoNotebookCellsCompile:
    """The curated walkthrough notebooks shipped by the register scripts are
    syntax-valid Python (a cheap guard against authoring typos)."""

    @pytest.mark.parametrize(
        "module_name",
        ["register_awareness_garden_model", "register_lca_garden_model"],
    )
    def test_demo_cells_compile(self, module_name):
        import importlib
        import os
        import sys

        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
        )
        mod = importlib.import_module(module_name)
        nb = mod.DEMO_NOTEBOOK
        assert nb and nb[0]["type"] == "markdown"
        for cell in nb:
            assert cell["type"] in ("markdown", "code")
            if cell["type"] == "code":
                compile(cell["source"], cell["id"], "exec")  # valid Python


class TestCopilotChat:
    """GET/PUT /model-garden/copilot/chat: stateful, per-(model, version,
    surface) chat persistence — load empty, save, reload, upsert singleton,
    clear (empty messages), surface + version isolation, and the message cap."""

    @pytest.mark.asyncio
    async def test_get_empty_when_absent(self, main):
        resp = await main.get_copilot_chat("BrandNew", main._DEV_PRINCIPAL, None, "editor")
        doc = json.loads(resp.body)
        assert doc["messages"] == []
        assert doc["surface"] == "editor"

    @pytest.mark.asyncio
    async def test_save_reload_and_upsert(self, main):
        msgs = [
            {"id": "u1", "role": "user", "content": "why unidentified?"},
            {"id": "a1", "role": "assistant", "content": "collinear spend…"},
        ]
        save = await main.save_copilot_chat(
            main.CopilotChatSaveRequest(name="M", version=2, surface="editor", messages=msgs),
            main._DEV_PRINCIPAL,
        )
        first_id = json.loads(save.body)["id"]
        resp = await main.get_copilot_chat("M", main._DEV_PRINCIPAL, 2, "editor")
        assert json.loads(resp.body)["messages"] == msgs

        # Second save upserts the SAME artifact (singleton per key).
        save2 = await main.save_copilot_chat(
            main.CopilotChatSaveRequest(name="M", version=2, surface="editor", messages=msgs[:1]),
            main._DEV_PRINCIPAL,
        )
        assert json.loads(save2.body)["id"] == first_id
        from mmm_framework.api import sessions as S

        tid = main._copilot_tid("dev-org", "M", 2, "editor")
        docs = [a for a in S.list_artifacts(tid) if a["kind"] == "copilot_chat"]
        assert len(docs) == 1
        assert json.loads((await main.get_copilot_chat("M", main._DEV_PRINCIPAL, 2, "editor")).body)["messages"] == msgs[:1]

    @pytest.mark.asyncio
    async def test_clear_with_empty_messages(self, main):
        await main.save_copilot_chat(
            main.CopilotChatSaveRequest(
                name="M", version=1, surface="editor",
                messages=[{"id": "u1", "role": "user", "content": "hi"}],
            ),
            main._DEV_PRINCIPAL,
        )
        await main.save_copilot_chat(
            main.CopilotChatSaveRequest(name="M", version=1, surface="editor", messages=[]),
            main._DEV_PRINCIPAL,
        )
        resp = await main.get_copilot_chat("M", main._DEV_PRINCIPAL, 1, "editor")
        assert json.loads(resp.body)["messages"] == []

    @pytest.mark.asyncio
    async def test_surface_and_version_isolation(self, main):
        editor_msg = [{"id": "e", "role": "user", "content": "editor turn"}]
        nb_msg = [{"id": "n", "role": "user", "content": "notebook turn"}]
        await main.save_copilot_chat(
            main.CopilotChatSaveRequest(name="M", version=1, surface="editor", messages=editor_msg),
            main._DEV_PRINCIPAL,
        )
        await main.save_copilot_chat(
            main.CopilotChatSaveRequest(name="M", version=1, surface="notebook", messages=nb_msg),
            main._DEV_PRINCIPAL,
        )
        # Different surface → different chat.
        assert json.loads((await main.get_copilot_chat("M", main._DEV_PRINCIPAL, 1, "editor")).body)["messages"] == editor_msg
        assert json.loads((await main.get_copilot_chat("M", main._DEV_PRINCIPAL, 1, "notebook")).body)["messages"] == nb_msg
        # Different version → empty.
        assert json.loads((await main.get_copilot_chat("M", main._DEV_PRINCIPAL, 9, "editor")).body)["messages"] == []

    @pytest.mark.asyncio
    async def test_message_cap(self, main):
        many = [{"id": f"m{i}", "role": "user", "content": str(i)} for i in range(250)]
        await main.save_copilot_chat(
            main.CopilotChatSaveRequest(name="M", version=1, surface="editor", messages=many),
            main._DEV_PRINCIPAL,
        )
        resp = await main.get_copilot_chat("M", main._DEV_PRINCIPAL, 1, "editor")
        kept = json.loads(resp.body)["messages"]
        assert len(kept) == main._COPILOT_CHAT_MAX_MESSAGES
        assert kept[-1]["content"] == "249"  # newest kept

    @pytest.mark.asyncio
    async def test_unknown_surface_defaults_to_editor(self, main):
        # A junk surface normalizes to "editor" (so it can't fork a third store).
        await main.save_copilot_chat(
            main.CopilotChatSaveRequest(
                name="M", version=1, surface="weird",
                messages=[{"id": "u", "role": "user", "content": "x"}],
            ),
            main._DEV_PRINCIPAL,
        )
        resp = await main.get_copilot_chat("M", main._DEV_PRINCIPAL, 1, "editor")
        assert json.loads(resp.body)["messages"][0]["content"] == "x"


class TestGardenLint:
    """POST /model-garden/lint: real ruff diagnostics merged with the AST
    contract checks, with line/column spans for inline markers."""

    @pytest.mark.asyncio
    async def test_undefined_name_is_an_error_with_position(self, main):
        src = (
            "from mmm_framework.garden import CustomMMM\n"
            "\n"
            "class M(CustomMMM):\n"
            "    def _build_model(self):\n"
            "        return undefined_thing\n"
        )
        resp = await main.garden_lint_endpoint(
            main.GardenSourceRequest(source_code=src), main._DEV_PRINCIPAL
        )
        out = json.loads(resp.body)
        assert out["ok"] is False
        f821 = [p for p in out["problems"] if p.get("code") == "F821"]
        assert f821, out["problems"]
        assert f821[0]["severity"] == "error"
        assert f821[0]["line"] == 5 and f821[0]["column"] is not None

    @pytest.mark.asyncio
    async def test_clean_source_is_ok(self, main):
        src = (
            "from mmm_framework.garden import CustomMMM\n"
            "import pymc as pm\n"
            "import pytensor.tensor as pt\n"
            "\n"
            "class CleanMMM(CustomMMM):\n"
            "    def _build_model(self) -> pm.Model:\n"
            "        with pm.Model(coords=self._build_coords()) as model:\n"
            '            b = pm.Gamma("beta_tv", mu=1.0, sigma=1.0)\n'
            '            pm.Deterministic("channel_contributions", b * pt.ones(self.n_obs))\n'
            '            pm.Deterministic("media_total", b * pt.ones(self.n_obs))\n'
            "        return model\n"
            "\n"
            "GARDEN_MODEL = CleanMMM\n"
        )
        resp = await main.garden_lint_endpoint(
            main.GardenSourceRequest(source_code=src), main._DEV_PRINCIPAL
        )
        out = json.loads(resp.body)
        assert out["ok"] is True
        assert not any(p["severity"] == "error" for p in out["problems"])
