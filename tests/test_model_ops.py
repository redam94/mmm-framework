"""Tests for the model-op registry + dispatch (Phase 2, PR-A).

These lock the pure-refactor behavior without needing a (slow) real fit: the
model_ops error-as-data contract, the Command-building helper, the no-model
dispatch path, and the save_fitted_model latent-bug fix (correct serializer
call signature).
"""

import inspect

from mmm_framework.agents import model_ops as M


def test_ops_registry_complete():
    assert set(M.OPS) == {
        "roi_metrics",
        "compute_estimands",
        "garden_compat",
        "garden_tune_suggestions",
        "component_decomposition",
        "model_diagnostics",
        "adstock_weights",
        "saturation_curves",
        "budget_scenario",
        "marginal_analysis",
        "prior_predictive_check",
        "leave_one_out",
        "save_model",
        "optimize_budget",
        "experiment_design",
        "experiment_priorities",
        # Parameterized ops (need design/dataset params, not just a model);
        # exercised in their own wiring tests.
        "experiment_economics",
        "experiment_optimizer",
        "identify_structural_parameters",
        # Validation / verification ops (Phase 1)
        "posterior_predictive_checks",
        "residual_diagnostics",
        "channel_diagnostics",
        "refutation_suite",
        "cross_validation",
        "validate_model",
    }


def _is_model_only_op(op) -> bool:
    """An op callable as ``op(model)`` — its only required parameter is the
    model. Parameterized ops (which also require design/dataset params) take a
    different calling convention and are covered by their own wiring tests."""
    required = [
        p
        for p in inspect.signature(op).parameters.values()
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY, p.KEYWORD_ONLY)
    ]
    return len(required) <= 1


def test_analysis_tools_no_model_message():
    """run_budget_scenario / run_marginal_analysis now dispatch through the kernel;
    with no model they return the no-model error via run_model_op."""
    from mmm_framework.agents import tools as T

    cfg = {"configurable": {"thread_id": "t_analysis"}}
    T._MODEL_CACHE.clear_thread("t_analysis")
    b = T.run_budget_scenario.func(
        spend_changes='{"TV": 0.1}', config=cfg, tool_call_id="t"
    )
    assert "No fitted model" in b.update["messages"][0].content
    m = T.run_marginal_analysis.func(config=cfg, tool_call_id="t")
    assert "No fitted model" in m.update["messages"][0].content
    # bad JSON is still caught BEFORE the model dispatch (input validation)
    bad = T.run_budget_scenario.func(
        spend_changes="{not json", config=cfg, tool_call_id="t"
    )
    assert "Could not parse" in bad.update["messages"][0].content


def test_ops_return_error_as_data_on_bad_model():
    """An op never raises for a compute failure — it returns the error as data so
    it can cross the future kernel boundary (PR-B). Checked for the model-only
    ops; parameterized ops have their own wiring tests."""
    # validate_model is a BATTERY: it never returns an _err, it degrades each
    # sub-check to an "Error" row and always returns a verdict table.
    _battery_ops = {"validate_model"}
    checked = 0
    for name, op in M.OPS.items():
        if name in _battery_ops or not _is_model_only_op(op):
            continue
        r = op(object())  # not a real model -> compute fails inside
        assert r["content"] is None, name
        assert r["dashboard"] == {}, name
        assert isinstance(r["error"], str) and r["error"], name
        checked += 1
    assert checked >= 10  # guard against the filter silently skipping everything


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


# ── Phase 2 PR-C: panel helper + load_fitted_model fix ────────────────────────


def test_mff_config_from_spec_builds():
    from mmm_framework.agents.tools import _mff_config_from_spec

    spec = {
        "kpi": "Sales",
        "media_channels": [
            {
                "name": "TV",
                "adstock": {"type": "weibull", "l_max": 4},
                "saturation": {"type": "logistic"},
            }
        ],
        "control_variables": [{"name": "Price"}],
        "time_granularity": "monthly",
    }
    assert type(_mff_config_from_spec(spec)).__name__ == "MFFConfig"


def test_load_fitted_model_passes_panel_and_sets_cache(monkeypatch, tmp_path):
    """Regression for the latent bug: load(save_dir) omitted the REQUIRED panel
    and mis-unpacked a single return -> always 'Load failed'. Now it rebuilds the
    panel and calls load(save_dir, panel)."""
    import os

    import mmm_framework.serialization as S
    from mmm_framework.agents import tools as T

    monkeypatch.setattr(T, "_mff_config_from_spec", lambda spec: "MFFCFG")
    monkeypatch.setattr(T, "load_mff", lambda path, cfg: "PANEL")
    captured = {}

    class _FakeSer:
        @staticmethod
        def load(path, panel, *a, **k):
            captured["call"] = (str(path), panel)
            return "LOADED_MODEL"

    monkeypatch.setattr(S, "MMMSerializer", _FakeSer)
    monkeypatch.chdir(tmp_path)
    os.makedirs("mmm_models/v1", exist_ok=True)
    T.set_current_thread("t_load")
    T._MODEL_CACHE.clear_thread("t_load")

    cmd = T.load_fitted_model.func(
        state={"model_spec": {"kpi": "Sales"}, "dataset_path": "data.csv"},
        name="v1",
        tool_call_id="t",
        config={"configurable": {"thread_id": "t_load"}},
    )
    out = cmd.update["messages"][0].content
    assert "Load failed" not in out and "loaded" in out.lower()
    assert captured["call"] == ("mmm_models/v1", "PANEL")  # the panel is now passed
    T.set_current_thread("t_load")
    assert T._MODEL_CACHE.get("fitted_model") == "LOADED_MODEL"
    T._MODEL_CACHE.clear_thread("t_load")


def test_causal_tools_no_model_message():
    """prior_predictive_check / leave_one_out_decomposition dispatch through the
    kernel. leave_one_out needs a fitted model; prior_predictive_check runs
    pre-fit from spec+dataset, so with NEITHER a model NOR a configured spec it
    returns its own actionable error (not the fit-first message)."""
    from mmm_framework.agents import causal_tools as C
    from mmm_framework.agents import tools as T

    cfg = {"configurable": {"thread_id": "t_causal"}}
    T._MODEL_CACHE.clear_thread("t_causal")
    p = C.prior_predictive_check.func(
        n_samples=10, config=cfg, state={}, tool_call_id="t"
    )
    msg = p.update["messages"][0].content
    assert "Configure a model and load a dataset first" in msg
    assert "fit the model first" not in msg  # must NOT tell the agent to fit
    lo = C.leave_one_out_decomposition.func(
        component_to_drop="TV", config=cfg, state={}, tool_call_id="t"
    )
    assert "No fitted model" in lo.update["messages"][0].content


def _write_synth_mff(tmp_path):
    import sys
    import os

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
    )
    from ex_model_workflow import generate_synthetic_mff

    df = generate_synthetic_mff(n_weeks=30)
    path = str(tmp_path / "mff.csv")
    df.to_csv(path, index=False)
    return path


def test_prior_predictive_check_runs_prefit_from_spec(tmp_path):
    """The whole point of the check: it must work BEFORE any fit, building an
    unfitted model graph from the active spec + dataset."""
    path = _write_synth_mff(tmp_path)
    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
    }
    res = M.prior_predictive_check(
        None, None, n_samples=20, spec=spec, dataset_path=path
    )
    assert not res.get("error"), res.get("error")
    assert "pre-fit" in res["content"]
    assert res["assumption"]["key"] == "prior_predictive_check"
    summary = res["dashboard"]["prior_predictive_summary"]
    assert summary["samples"] > 0


# ── Latent controls (Trend/Seasonality listed as control variables) ───────────


def test_partition_latent_controls():
    from mmm_framework.agents.tools import _partition_latent_controls

    ds = {"Sales", "TV", "Price_Index", "Trend"}
    # "Trend" exists in THIS dataset -> stays a real control; "Seasonality"
    # doesn't -> diverted; "Bogus" -> missing
    real, latent, missing = _partition_latent_controls(
        ["Price_Index", "Trend", "Seasonality", "Bogus"], ds
    )
    assert real == ["Price_Index", "Trend"]
    assert latent == [("Seasonality", "seasonality")]
    assert missing == ["Bogus"]
    # no dataset to check: latent diverted by name, unknown assumed real
    real, latent, missing = _partition_latent_controls(["trend", "X"], None)
    assert real == ["X"] and latent == [("trend", "trend")] and missing == []


def test_configure_model_diverts_latent_controls_and_rejects_missing(tmp_path):
    """'Trend'/'Seasonality' are not dataset variables — configure_model maps
    them onto the built-in components instead of letting load_mff fail later;
    genuinely missing variables are rejected up front."""
    from mmm_framework.agents import tools as T

    path = _write_synth_mff(tmp_path)
    state = {"dataset_path": path, "locked_fields": [], "dashboard_data": {}}

    cmd = T.configure_model.func(
        state=state,
        kpi="Sales",
        kpi_level="national",
        media_channels=["TV"],
        control_variables=["Price_Index", "Trend", "Seasonality"],
        tool_call_id="t",
    )
    spec = cmd.update["model_spec"]
    assert [c["name"] for c in spec["control_variables"]] == ["Price_Index"]
    assert spec["trend"] == {"type": "linear"}
    assert spec["seasonality"] == {"yearly": 2}
    assert "diverted" in cmd.update["messages"][0].content

    bad = T.configure_model.func(
        state=state,
        kpi="Sales",
        kpi_level="national",
        media_channels=["TV"],
        control_variables=["Bogus"],
        tool_call_id="t",
    )
    assert "model_spec" not in bad.update
    assert "not found in the dataset" in bad.update["messages"][0].content

    bad_kpi = T.configure_model.func(
        state=state,
        kpi="Revenue",
        kpi_level="national",
        media_channels=["TV"],
        control_variables=[],
        tool_call_id="t",
    )
    assert "model_spec" not in bad_kpi.update


def test_build_model_hints_on_latent_controls(tmp_path):
    """A stale spec that still lists Trend/Seasonality as controls fails the
    build with an actionable hint, not just the raw loader error."""
    import pytest

    from mmm_framework.agents.fitting import build_model

    path = _write_synth_mff(tmp_path)
    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}],
        "control_variables": [{"name": "Trend"}, {"name": "Seasonality"}],
    }
    with pytest.raises(ValueError, match="built-in `trend` / `seasonality`"):
        build_model(spec, path)


def test_seasonality_amplitude_and_trend_priors_wire_into_graph(tmp_path):
    """spec.priors.seasonality controls the Fourier-coefficient prior sigma
    (the seasonal amplitude prior, previously hardcoded 0.3), per-component
    overrides win, and piecewise now honors the base-slope growth prior."""
    path = _write_synth_mff(tmp_path)
    from mmm_framework.agents.fitting import build_model

    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}],
        "control_variables": [],
        "trend": {"type": "piecewise", "n_changepoints": 5},
        "seasonality": {"yearly": 4, "monthly": 2},
        "priors": {
            "trend": {"growth_prior_mu": 0.1, "growth_prior_sigma": 0.3},
            "seasonality": {"prior_sigma": 0.5, "yearly_prior_sigma": 0.8},
        },
    }
    mmm = build_model(spec, path)
    sc = mmm.seasonality_config
    assert sc.prior_sigma_for("yearly") == 0.8
    assert sc.prior_sigma_for("monthly") == 0.5  # falls back to shared sigma
    assert mmm.trend_config.growth_prior_mu == 0.1  # piecewise base slope
    # the sigmas actually land on the PyMC RVs
    model = mmm.model
    assert abs(float(model["season_yearly"].owner.inputs[-1].eval()) - 0.8) < 1e-9
    assert abs(float(model["season_monthly"].owner.inputs[-1].eval()) - 0.5) < 1e-9


def test_experiment_priorities_tool_no_model_message():
    """compute_experiment_priorities dispatches through the kernel; with no
    fitted model it returns the no-model error, and the host-side evidence
    lookup must not blow up without a session/project."""
    from mmm_framework.agents import tools as T

    cfg = {"configurable": {"thread_id": "t_priorities"}}
    T._MODEL_CACHE.clear_thread("t_priorities")
    cmd = T.compute_experiment_priorities.func(config=cfg, tool_call_id="t")
    assert "No fitted model" in cmd.update["messages"][0].content


def test_intercept_prior_wires_into_graph(tmp_path):
    """spec.priors.intercept controls the intercept Normal (previously
    hardcoded Normal(0, 0.5), so 'tightening the intercept prior' was a
    silent no-op the prior predictive check could never reflect)."""
    path = _write_synth_mff(tmp_path)
    from mmm_framework.agents.fitting import build_model

    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}],
        "control_variables": [],
        "priors": {"intercept": {"mu": -0.5, "sigma": 0.1}},
    }
    mmm = build_model(spec, path)
    assert mmm.model_config.intercept_prior_mu == -0.5
    assert mmm.model_config.intercept_prior_sigma == 0.1
    # mu and sigma actually land on the PyMC RV
    model = mmm.model
    assert abs(float(model["intercept"].owner.inputs[-2].eval()) + 0.5) < 1e-9
    assert abs(float(model["intercept"].owner.inputs[-1].eval()) - 0.1) < 1e-9


def test_unconsumed_prior_path_validator():
    """Every priors.* write must be one build_model actually reads; anything
    else is named and rejected with the valid alternatives."""
    from mmm_framework.agents.fitting import unconsumed_prior_path as v

    spec = {
        "media_channels": [{"name": "TV"}],
        "control_variables": [{"name": "Price"}],
    }
    # consumed paths pass
    assert v(["priors", "intercept", "mu"], -0.5, spec) is None
    assert v(["priors", "intercept"], {"mu": -0.5, "sigma": 0.3}, spec) is None
    assert v(["priors", "seasonality", "yearly_prior_sigma"], 0.8, spec) is None
    assert v(["priors", "trend", "growth_prior_mu"], 0.1, spec) is None
    prior = {"distribution": "normal", "params": {"mu": 0, "sigma": 1}}
    assert v(["priors", "media", "TV", "coefficient"], prior, spec) is None
    assert (
        v(["priors", "media", "TV", "coefficient", "params", "mu"], 0.2, spec) is None
    )
    assert v(["priors", "controls", "Price", "allow_negative"], False, spec) is None
    # silently-dropped paths are rejected
    assert "never reads" in v(["priors", "intercept", "scale"], 1.0, spec)
    assert "never reads" in v(["priors", "intercept"], prior, spec)  # wrong shape
    assert "never reads" in v(["priors", "baseline"], {"mu": 0}, spec)
    assert "never reads" in v(["priors", "media", "TV", "roi"], prior, spec)
    assert "unknown media channel" in v(
        ["priors", "media", "Radio", "coefficient"], prior, spec
    )
    assert "unknown control" in v(
        ["priors", "controls", "Promo", "coefficient"], prior, spec
    )


def test_update_model_setting_rejects_unconsumed_priors(tmp_path):
    """The tool-level guard: an unread priors.* write must fail loudly instead
    of committing to the spec and changing nothing downstream."""
    from mmm_framework.agents import tools as T

    spec = {
        "kpi": "Sales",
        "media_channels": [{"name": "TV"}],
        "control_variables": [],
    }
    state = {"model_spec": spec, "locked_fields": [], "dashboard_data": {}}

    bad = T.update_model_setting.func(
        state=state,
        setting_path="priors.intercept",
        value={"distribution": "normal", "params": {"mu": -4, "sigma": 0.25}},
        tool_call_id="t",
    )
    assert "model_spec" not in bad.update
    msg = bad.update["messages"][0].content
    assert "Rejected" in msg and "mu" in msg and "sigma" in msg

    ok = T.update_model_setting.func(
        state=state,
        setting_path="priors.intercept.sigma",
        value=0.3,
        tool_call_id="t",
    )
    assert "model_spec" in ok.update


def test_seasonality_components_respect_spec_and_frequency(tmp_path):
    """monthly/weekly were previously ignored entirely, and a monthly-only spec
    silently enabled the builder's yearly=2 default. Weekly seasonality is not
    representable in weekly data -> warned and skipped, never silently built."""
    import warnings as _w

    path = _write_synth_mff(tmp_path)
    from mmm_framework.agents.fitting import build_model

    base = {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}],
        "control_variables": [],
    }
    monthly_only = build_model(dict(base, seasonality={"monthly": 2}), path)
    assert set(monthly_only.seasonality_features) == {"monthly"}

    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        weekly_on_weekly = build_model(dict(base, seasonality={"weekly": 3}), path)
    assert weekly_on_weekly.seasonality_features == {}
    assert any("cannot be represented" in str(x.message) for x in rec)

    # Nyquist clamp: monthly order 4 at period ~4.33 -> clamped to 2 (4 columns)
    with _w.catch_warnings(record=True) as rec2:
        _w.simplefilter("always")
        clamped = build_model(dict(base, seasonality={"monthly": 4}), path)
    assert clamped.seasonality_features["monthly"].shape[1] == 4
    assert any("clamping" in str(x.message) for x in rec2)


def test_bayesian_workflow_reference_tool():
    """The agent's methodology reference: registered, complete, topic-filterable."""
    from mmm_framework.agents.tools import TOOLS, bayesian_workflow_reference

    assert any(t.name == "bayesian_workflow_reference" for t in TOOLS)
    full = bayesian_workflow_reference.func()
    # the gates the agent must enforce are stated with their thresholds
    for needle in ("R-hat < 1.01", "ESS > 400", "5%", "marginal ROAS"):
        assert needle in full, needle
    # topic filter narrows to matching sections; unmatched falls back to full
    filtered = bayesian_workflow_reference.func(topic="diagnostics")
    assert "R-hat < 1.01" in filtered and len(filtered) < len(full)
    assert bayesian_workflow_reference.func(topic="zzz") == full
