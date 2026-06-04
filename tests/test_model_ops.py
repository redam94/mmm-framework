"""Tests for the model-op registry + dispatch (Phase 2, PR-A).

These lock the pure-refactor behavior without needing a (slow) real fit: the
model_ops error-as-data contract, the Command-building helper, the no-model
dispatch path, and the save_fitted_model latent-bug fix (correct serializer
call signature).
"""

from mmm_framework.agents import model_ops as M


def test_ops_registry_complete():
    assert set(M.OPS) == {
        "roi_metrics",
        "component_decomposition",
        "model_diagnostics",
        "adstock_weights",
        "saturation_curves",
    }


def test_ops_return_error_as_data_on_bad_model():
    """An op never raises for a compute failure — it returns the error as data so
    it can cross the future kernel boundary (PR-B)."""
    for name, op in M.OPS.items():
        r = op(object())  # not a real model -> compute fails inside
        assert r["content"] is None, name
        assert r["dashboard"] == {}, name
        assert isinstance(r["error"], str) and r["error"], name


def test_modelop_command_success_merges_dashboard():
    from mmm_framework.agents.tools import _modelop_command

    res = {"content": "hi", "dashboard": {"roi_metrics": [{"a": 1}]}, "error": None}
    cmd = _modelop_command(res, {"dashboard_data": {"x": 1}}, "tc")
    assert cmd.update["messages"][0].content == "hi"
    assert cmd.update["dashboard_data"] == {"x": 1, "roi_metrics": [{"a": 1}]}


def test_modelop_command_error_has_no_dashboard():
    from mmm_framework.agents.tools import _modelop_command

    cmd = _modelop_command(
        {"content": None, "dashboard": {}, "error": "boom"}, {}, "tc"
    )
    assert cmd.update["messages"][0].content == "boom"
    assert "dashboard_data" not in cmd.update


def test_interpretation_tools_no_model_message():
    from mmm_framework.agents import tools as T

    cfg = {"configurable": {"thread_id": "t_nomodel"}}
    T._MODEL_CACHE.clear_thread("t_nomodel")  # ensure no model on this thread
    for tool in (
        T.get_roi_metrics,
        T.get_component_decomposition,
        T.get_model_diagnostics,
        T.get_adstock_weights,
        T.get_saturation_curves,
    ):
        cmd = tool.func(state={"dashboard_data": {}}, tool_call_id="t", config=cfg)
        assert "No fitted model" in cmd.update["messages"][0].content


def test_save_fitted_model_calls_serializer_with_correct_signature(
    monkeypatch, tmp_path
):
    """Regression for the latent bug: the prior call passed `results` as `path`
    (`save(fitted, results, save_dir)`) and ALWAYS raised -> 'Save failed'. The
    classmethod is `save(model, path, ...)`; verify the corrected call."""
    from mmm_framework.agents import tools as T
    import mmm_framework.serialization as S

    captured = {}

    class _FakeSerializer:
        @staticmethod
        def save(model, path, *a, **k):
            captured["call"] = (model, str(path))

    monkeypatch.setattr(S, "MMMSerializer", _FakeSerializer)
    monkeypatch.chdir(tmp_path)
    T.set_current_thread("t_save_fix")
    T._MODEL_CACHE["fitted_model"] = "MODEL_OBJ"

    cmd = T.save_fitted_model.func(
        state={},
        name="v1",
        tool_call_id="t",
        config={"configurable": {"thread_id": "t_save_fix"}},
    )
    out = cmd.update["messages"][0].content
    assert "Save failed" not in out and "v1" in out
    # (model, path) — NOT (model, results, path)
    assert captured["call"] == ("MODEL_OBJ", "mmm_models/v1")
    T._MODEL_CACHE.clear_thread("t_save_fix")
