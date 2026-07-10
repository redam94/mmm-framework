"""Session export reconstructs TOOL CALLS as runnable Python (U3).

The export used to concatenate only ``execute_python`` cells; fits and every
interpretation tool ran through opaque tool calls that vanished from the
download. These tests pin the new checkpoint-driven replay:

* ``fit_mmm_model`` → ``build_and_fit(spec, dataset_path)`` with the fitted
  spec embedded (+ a commented corrected fast-reload; the old broken
  ``MMMSerializer().load(path)`` form is gone);
* model-op tools → ``run_op(model_ops.<op>, mmm, results, ...)`` with the same
  arg transforms the agent tools do (JSON string args parsed, keys renamed);
* EDA tools → kernel-free ``mmm_framework.eda`` calls;
* everything else → commented records;
* turn headers / scope slicing / artifact-only fallback;
* and the exec() end-to-end gate: the generated script actually RUNS.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mmm_framework.agents.session_export import build_session_script, extract_timeline


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the session store + workspace at a temp location."""
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as ss

    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.db")
    ss.init_db()
    return ss


def _new_thread(store):
    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    return sess["thread_id"]


SPEC = {
    "kpi": "Sales",
    "media_channels": [{"name": "TV"}, {"name": "Digital"}],
    "control_variables": [],
    "trend": {"type": "linear"},
    "seasonality": {"yearly": 2, "monthly": 0, "weekly": 0},
    "inference": {"method": "map", "metrics_draws": 0},
}


def _seed_run(store, tid, dataset_path, run_name="run_A", spec=SPEC, model_path=None):
    store.add_artifact(
        tid,
        "model_run",
        {
            "run_name": run_name,
            "model_path": model_path or f"mmm_models/{run_name}",
            "dataset_path": str(dataset_path),
            "spec": spec,
        },
    )


def _fit_messages(question="Fit a model", run_name="run_A", call_id="f1"):
    return [
        HumanMessage(content=question),
        AIMessage(
            content="",
            tool_calls=[{"name": "fit_mmm_model", "args": {}, "id": call_id}],
        ),
        ToolMessage(
            content=f"Model fitted successfully! Auto-saved as **{run_name}**.",
            tool_call_id=call_id,
        ),
    ]


def _tool_turn(question, name, args, call_id, result="ok"):
    return [
        HumanMessage(content=question),
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}]),
        ToolMessage(content=result, tool_call_id=call_id),
    ]


# ── 1. fit rendering ──────────────────────────────────────────────────────────


def test_fit_renders_build_and_fit_with_embedded_spec(store):
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")

    script = build_session_script(tid, messages=_fit_messages())

    # embedded spec JSON, parsed at run time
    assert "spec = json.loads(" in script
    assert '"kpi": "Sales"' in script
    # the fit replays through build_and_fit
    assert "from mmm_framework.agents.fitting import build_and_fit" in script
    assert "mmm, results, fit_info = build_and_fit(spec, dataset_path)" in script
    # corrected fast-reload offered as a commented alternative
    assert "# panel = load_mff(dataset_path, _mff_config_from_spec(spec))" in script
    assert "# mmm = MMMSerializer.load('mmm_models/run_A', panel)" in script
    # the old broken single-arg form is gone
    assert "MMMSerializer().load(" not in script


def test_fit_spec_roundtrips_through_the_emitted_json(store):
    tid = _new_thread(store)
    spec = {**SPEC, "note": "client's Q4 'holiday' plan"}  # quotes must survive
    _seed_run(store, tid, "/abs/data.csv", spec=spec)

    script = build_session_script(tid, messages=_fit_messages())
    # pull the emitted r'''...''' literal back out and parse it
    start = script.index("r'''") + 4
    end = script.index("'''", start)
    assert json.loads(script[start:end]) == spec


def test_garden_ref_fit_gets_a_warning(store):
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv", spec={**SPEC, "garden_ref": "org/my_model"})

    script = build_session_script(tid, messages=_fit_messages())
    assert "WARNING" in script and "Model-Garden" in script


# ── 2. model-op rendering ─────────────────────────────────────────────────────


def test_model_ops_render_as_run_op_calls(store):
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    messages = (
        _fit_messages()
        + _tool_turn("roi?", "get_roi_metrics", {}, "r1")
        + _tool_turn(
            "what if?",
            "run_budget_scenario",
            {"spend_changes": '{"TV": 0.2, "Digital": -0.1}'},
            "b1",
        )
    )

    script = build_session_script(tid, messages=messages)
    assert "res_roi_metrics = run_op(model_ops.roi_metrics, mmm, results)" in script
    # the JSON string arg is parsed into a dict literal, as the agent tool does
    assert (
        "run_op(model_ops.budget_scenario, mmm, results, "
        "spend_changes={'TV': 0.2, 'Digital': -0.1})" in script
    )


def test_budget_optimizer_arg_renames(store):
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    messages = _fit_messages() + _tool_turn(
        "optimize",
        "run_budget_optimizer",
        {"total_budget": 1000.0, "channel_bounds": {"TV": [1.0, 1.0]}},
        "o1",
    )

    script = build_session_script(tid, messages=messages)
    assert "run_op(model_ops.optimize_budget, mmm, results, " in script
    assert "total_budget=1000.0" in script
    # channel_bounds is renamed to the op's `bounds` kwarg
    assert "bounds={'TV': [1.0, 1.0]}" in script
    assert "channel_bounds=" not in script


def test_op_with_unparseable_args_demotes_to_comment(store):
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    messages = _fit_messages() + _tool_turn(
        "what if?", "run_budget_scenario", {"spend_changes": "not-json"}, "b1"
    )

    script = build_session_script(tid, messages=messages)
    assert "run_op(model_ops.budget_scenario" not in script
    assert "could not be reconstructed" in script


def test_op_before_any_fit_becomes_a_note(store):
    tid = _new_thread(store)
    messages = _tool_turn("roi?", "get_roi_metrics", {}, "r1")

    script = build_session_script(tid, messages=messages)
    assert "run_op(model_ops.roi_metrics" not in script
    assert "no fitted model at this point in the session" in script


def test_prefit_check_replays_against_session_spec(store):
    """prior_predictive_check / run_calibration_check are allow_unfitted pre-fit
    ops — run BEFORE the fit they must still export as runnable code, replayed
    against session_spec (the session's first fitted spec)."""
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    messages = _tool_turn(
        "check priors", "prior_predictive_check", {"n_samples": 200}, "p1"
    ) + _fit_messages(question="now fit")

    script = build_session_script(tid, messages=messages)
    assert "session_spec = json.loads(" in script
    assert (
        "res_prior_predictive_check = run_op(model_ops.prior_predictive_check, "
        "None, None, n_samples=200, spec=session_spec, dataset_path=dataset_path)"
        in script
    )
    # …and the ordering holds: the pre-fit check comes before the fit section
    assert script.index("res_prior_predictive_check") < script.index(
        "mmm, results, fit_info = build_and_fit"
    )


def test_prefit_check_without_any_fit_is_a_record(store):
    """No fit in the whole session → no session_spec → the pre-fit check is
    kept as a record with an accurate note (it isn't 'needs a fitted model')."""
    tid = _new_thread(store)
    messages = _tool_turn("check priors", "prior_predictive_check", {}, "p1")

    script = build_session_script(tid, messages=messages)
    assert "run_op(model_ops.prior_predictive_check" not in script
    assert "working spec" in script


def test_nan_op_args_are_demoted_not_emitted(store):
    """json.loads accepts NaN/Infinity, whose reprs are not Python literals —
    such args must demote the call to a comment, never emit `nan`."""
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    messages = _fit_messages() + _tool_turn(
        "what if?", "run_budget_scenario", {"spend_changes": '{"TV": NaN}'}, "b1"
    )

    script = build_session_script(tid, messages=messages)
    assert "run_op(model_ops.budget_scenario" not in script
    assert "could not be reconstructed" in script


def test_header_injection_is_neutralized(store):
    """A session name or scope containing triple quotes must not terminate the
    generated module docstring (script stays syntactically valid Python)."""
    proj = store.create_project("P")
    sess = store.create_session(
        name='evil""" \nimport os  # injected', project_id=proj["project_id"]
    )
    tid = sess["thread_id"]
    store.add_artifact(tid, "code_snippet", {"call_id": "c1", "code": "x = 1"})

    for scope in ("all", 'turn:1"""\nimport os'):
        script = build_session_script(tid, messages=[], scope=scope)
        compile(script, "<export>", "exec")  # must not SyntaxError


def test_kernel_startup_source_still_compiles():
    """The subprocess kernel reuses _PREAMBLE_IMPORTS/_PREAMBLE_HELPERS as its
    startup source — the export's new imports must never break kernel init."""
    from mmm_framework.agents.kernels import _build_startup_source

    src = _build_startup_source()
    compile(src, "<kernel-startup>", "exec")
    # the new analysis-surface imports stay guarded (kernel-safe)
    imports_block = src.split("def save_result")[0]
    assert "from mmm_framework.agents import model_ops" in imports_block
    guarded = imports_block.split("from mmm_framework.agents import model_ops")[0]
    assert guarded.rstrip().endswith("try:")


# ── 3. EDA rendering ──────────────────────────────────────────────────────────


def test_eda_renders_kernel_free_calls(store, tmp_path):
    tid = _new_thread(store)
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n")
    store.register_file(tid, str(data), "data.csv", "dataset", 8)
    messages = _tool_turn(
        "explore",
        "run_eda",
        {"analyses": ["profile", "collinearity", "bogus_analysis"]},
        "e1",
    )

    script = build_session_script(tid, messages=messages)
    assert "panel = eda.load_eda_panel(dataset_path" in script
    assert "print(eda.profile_panel(panel).to_string())" in script
    assert "coll = eda.collinearity_analysis(panel)" in script
    # unknown analyses are ignored, not rendered
    assert "bogus_analysis" not in script


def test_validate_data_and_detect_outliers_render(store, tmp_path):
    tid = _new_thread(store)
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n")
    store.register_file(tid, str(data), "data.csv", "dataset", 8)
    messages = _tool_turn("check", "validate_data", {}, "v1") + _tool_turn(
        "outliers", "detect_outliers", {"sensitivity": "high"}, "o1"
    )

    script = build_session_script(tid, messages=messages)
    assert "report = eda.validate_dataset(panel)" in script
    assert (
        "eda.detect_outliers(panel, eda.OutlierConfig.for_sensitivity('high'))"
        in script
    )


# ── 4. comment tier ───────────────────────────────────────────────────────────


def test_non_exportable_tools_become_comments(store):
    tid = _new_thread(store)
    messages = _tool_turn(
        "search", "search_knowledge_base", {"query": "adstock"}, "k1", result="found it"
    ) + _tool_turn("prefs", "save_preference", {"key": "brand", "value": "acme"}, "p1")

    script = build_session_script(tid, messages=messages)
    assert "# ── Tool call (not exportable as code): search_knowledge_base" in script
    assert "# ── Tool call (not exportable as code): save_preference" in script
    assert "run_op(model_ops." not in script  # no code emitted for them
    assert "adstock" in script  # args preview kept


# ── 5. turn headers + scope ───────────────────────────────────────────────────


def _two_turn_session(store):
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    store.add_artifact(
        tid, "code_snippet", {"call_id": "c1", "code": "print(mmm.n_obs)"}
    )
    messages = _fit_messages(question="Fit a model please") + _tool_turn(
        "now analyze it",
        "execute_python",
        {"code": "print(mmm.n_obs)"},
        "c1",
    )
    return tid, messages


def test_turn_headers_and_ordering(store):
    tid, messages = _two_turn_session(store)
    script = build_session_script(tid, messages=messages)

    assert '# Turn 1 — "Fit a model please"' in script
    assert '# Turn 2 — "now analyze it"' in script
    # the fit renders before the cell that uses it
    assert script.index("build_and_fit(spec, dataset_path)") < script.index(
        "print(mmm.n_obs)"
    )


def test_scope_last_pulls_in_the_prior_fit(store):
    tid, messages = _two_turn_session(store)
    script = build_session_script(tid, messages=messages, scope="last")

    assert "# Turn 1 —" not in script
    assert '# Turn 2 — "now analyze it"' in script
    # the turn-2 cell references mmm, so the fit section is pulled in
    assert "mmm, results, fit_info = build_and_fit(spec, dataset_path)" in script
    assert "pulled in from an earlier turn" in script


def test_scope_turn_k_and_bogus_scope(store):
    tid, messages = _two_turn_session(store)

    only_first = build_session_script(tid, messages=messages, scope="turn:1")
    assert "# Turn 1 —" in only_first
    assert "# Turn 2 —" not in only_first

    bogus = build_session_script(tid, messages=messages, scope="turn:banana")
    assert "# Turn 1 —" in bogus and "# Turn 2 —" in bogus
    bogus2 = build_session_script(tid, messages=messages, scope="nonsense")
    assert "# Turn 1 —" in bogus2 and "# Turn 2 —" in bogus2


# ── 6. fallbacks ──────────────────────────────────────────────────────────────


def test_no_messages_falls_back_to_artifact_layout(store):
    tid = _new_thread(store)
    store.add_artifact(tid, "code_snippet", {"call_id": "c1", "code": "print('hi')"})
    store.add_artifact(
        tid, "text_output", {"call_id": "c1", "stdout": "hi", "is_error": False}
    )

    script = build_session_script(tid, messages=[])
    assert "# Session cells (execute_python, in order)" in script
    assert "In[1]" in script
    assert "print('hi')" in script
    assert "Turn 1" not in script


def test_plain_dict_messages_are_tolerated(store):
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    messages = [
        {"type": "human", "content": "fit it"},
        {
            "type": "ai",
            "content": "",
            "tool_calls": [{"name": "fit_mmm_model", "args": {}, "id": "f1"}],
        },
        {
            "type": "tool",
            "content": "Model fitted successfully! Auto-saved as **run_A**.",
            "tool_call_id": "f1",
        },
    ]
    timeline = extract_timeline(messages)
    assert [k for k, _ in timeline] == ["question", "tool"]
    script = build_session_script(tid, messages=messages)
    assert "mmm, results, fit_info = build_and_fit(spec, dataset_path)" in script


def test_errored_fit_becomes_a_comment(store):
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    messages = [
        HumanMessage(content="fit it"),
        AIMessage(
            content="", tool_calls=[{"name": "fit_mmm_model", "args": {}, "id": "f1"}]
        ),
        ToolMessage(content="Error fitting model: no such column", tool_call_id="f1"),
    ]

    script = build_session_script(tid, messages=messages)
    assert "mmm, results, fit_info = build_and_fit" not in script
    assert "FAILED when it ran in the session" in script
    assert "no such column" in script


def test_multi_dataset_fits_rebind_their_own_dataset(store):
    """A session that fit against TWO datasets must replay each fit against
    ITS dataset, not the last one loaded."""
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/first.csv", run_name="run_A")
    _seed_run(store, tid, "/abs/second.csv", run_name="run_B")
    messages = _fit_messages(
        question="fit on first", run_name="run_A", call_id="f1"
    ) + _fit_messages(question="fit on second", run_name="run_B", call_id="f2")

    script = build_session_script(tid, messages=messages)
    # preamble binds the latest-known dataset…
    assert "dataset_path = 'second.csv'" in script
    # …and the first fit rebinds to the dataset IT actually used
    assert "dataset_path = 'first.csv'" in script
    assert "DIFFERENT dataset file" in script
    first_rebind = script.index("dataset_path = 'first.csv'")
    fit_a = script.index("# ── Fit (fit_mmm_model → run_A)")
    fit_b = script.index("# ── Fit (fit_mmm_model → run_B)")
    assert fit_a < first_rebind < fit_b


def test_scope_last_skips_trailing_prose_only_turn(store):
    """A closing "thanks!" turn with no tool calls must not make scope=last
    export an empty script — the last turn WITH activity is selected."""
    tid = _new_thread(store)
    _seed_run(store, tid, "/abs/data.csv")
    messages = _fit_messages(question="fit a model") + [
        HumanMessage(content="thanks, looks good!"),
        AIMessage(content="You're welcome."),
    ]

    script = build_session_script(tid, messages=messages, scope="last")
    assert "mmm, results, fit_info = build_and_fit(spec, dataset_path)" in script
    assert '"thanks, looks good!"' not in script


def test_model_referencing_cell_without_model_is_flagged(store):
    """A cell that uses `mmm` when no fit could be reconstructed must carry a
    note explaining the coming NameError, like the op sections do."""
    tid = _new_thread(store)
    store.add_artifact(
        tid, "code_snippet", {"call_id": "c1", "code": "print(mmm.n_obs)"}
    )
    messages = _tool_turn(
        "inspect", "execute_python", {"code": "print(mmm.n_obs)"}, "c1"
    )

    script = build_session_script(tid, messages=messages)
    assert "print(mmm.n_obs)" in script  # cell kept verbatim
    assert "could not reconstruct" in script  # …but flagged


def test_appendix_collects_cells_missing_from_checkpoint(store):
    tid = _new_thread(store)
    # a cell artifact whose call_id no longer appears in the (trimmed) checkpoint
    store.add_artifact(
        tid, "code_snippet", {"call_id": "gone", "code": "print('orphan')"}
    )
    messages = _tool_turn("hello", "get_session_status", {}, "s1")

    script = build_session_script(tid, messages=messages)
    assert "Appendix — cells not present in the conversation checkpoint" in script
    assert "print('orphan')" in script


# ── 7. exec() end-to-end (fast, mocked fit + op) ──────────────────────────────


def test_exported_script_executes_end_to_end(store, tmp_path, monkeypatch, capsys):
    tid = _new_thread(store)
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n3,4\n")
    _seed_run(store, tid, data)
    store.add_artifact(
        tid,
        "code_snippet",
        {"call_id": "c1", "code": "print('channels', mmm.channel_names)"},
    )
    messages = (
        _fit_messages()
        + _tool_turn("roi?", "get_roi_metrics", {}, "r1")
        + _tool_turn(
            "inspect",
            "execute_python",
            {"code": "print('channels', mmm.channel_names)"},
            "c1",
        )
    )

    class DummyMMM:
        channel_names = ["TV", "Digital"]

    import mmm_framework.agents.fitting as fitting
    import mmm_framework.agents.model_ops as model_ops

    monkeypatch.setattr(
        fitting, "build_and_fit", lambda spec, dataset_path: (DummyMMM(), object(), {})
    )
    monkeypatch.setattr(
        model_ops,
        "roi_metrics",
        lambda mmm, results=None, **kw: {"content": "ROI OK", "error": None},
    )

    script = build_session_script(tid, messages=messages)
    monkeypatch.chdir(tmp_path)
    exec(compile(script, "<export>", "exec"), {})  # must not raise

    out = capsys.readouterr().out
    assert "ROI OK" in out
    assert "channels ['TV', 'Digital']" in out


# ── 8. exec() end-to-end (slow, REAL MAP fit + real roi_metrics) ─────────────


def _write_mff(path, n=40):
    periods = pd.date_range("2021-01-04", periods=n, freq="W-MON")
    dims = {
        "Geography": None,
        "Product": None,
        "Campaign": None,
        "Outlet": None,
        "Creative": None,
    }
    rows = []
    for i, p in enumerate(periods):
        iso = p.strftime("%Y-%m-%d")
        rows.append(
            {
                **dims,
                "Period": iso,
                "VariableName": "Sales",
                "VariableValue": 1000 + 10 * i + (i % 4) * 25,
            }
        )
        rows.append(
            {
                **dims,
                "Period": iso,
                "VariableName": "TV",
                "VariableValue": 100 + (i % 5) * 20,
            }
        )
        rows.append(
            {
                **dims,
                "Period": iso,
                "VariableName": "Digital",
                "VariableValue": 80 + (i % 3) * 15,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.mark.slow
def test_exported_script_runs_a_real_map_fit(store, tmp_path, monkeypatch, capsys):
    """The unmocked script end-to-end: real build_and_fit (MAP) + real
    roi_metrics from the generated code."""
    tid = _new_thread(store)
    dataset_path = _write_mff(tmp_path / "data.csv")
    spec = {
        "kpi": "Sales",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "trend": {"type": "linear"},
        "seasonality": {"yearly": 0, "monthly": 0, "weekly": 0},
        "inference": {"method": "map", "metrics_draws": 0},
    }
    _seed_run(store, tid, dataset_path, spec=spec, model_path=None)
    messages = _fit_messages() + _tool_turn("roi?", "get_roi_metrics", {}, "r1")

    # Skip the (slow, best-effort) HTML report step of build_and_fit.
    import mmm_framework.reporting.generator as gen

    monkeypatch.setattr(
        gen,
        "ReportBuilder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped in test")),
    )

    script = build_session_script(tid, messages=messages)
    monkeypatch.chdir(tmp_path)
    exec(compile(script, "<export>", "exec"), {})  # must not raise

    out = capsys.readouterr().out
    assert "ROI" in out  # real roi_metrics markdown printed by run_op
