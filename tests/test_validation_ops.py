"""Validation/verification/plotting agent ops (Phase 1).

Covers: the model-op plot publishing plumbing (model_ops can now return themed
figures), graceful degradation on a non-fitted model, and — under @slow — the
real validators (PPC / residuals / channel / refutation / cross-validation /
the validate_model battery) against a fitted BayesianMMM.
"""

from __future__ import annotations

import pytest

from mmm_framework.agents import model_ops as MO

# ---------------------------------------------------------------------------
# plot publishing plumbing (fast)
# ---------------------------------------------------------------------------


def _min_fig() -> dict:
    return {"data": [{"type": "scatter", "x": [1, 2, 3], "y": [1, 4, 9]}], "layout": {}}


def test_publish_modelop_plots_stores_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path))
    from mmm_framework.agents.tools import _publish_modelop_plots

    dd: dict = {}
    note = _publish_modelop_plots(
        [{"title": "My chart", "figure": _min_fig()}], dd, "thread-x"
    )
    assert dd["plots"] and "id" in dd["plots"][0]
    assert dd["plots"][0]["title"] == "My chart"
    assert "1 chart" in note


def test_publish_modelop_plots_drops_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path))
    from mmm_framework.agents.tools import _publish_modelop_plots

    dd: dict = {}
    note = _publish_modelop_plots([{"title": "bad", "figure": "not-a-dict"}], dd, "t")
    assert dd["plots"] == []
    assert "omitted" in note


# ---------------------------------------------------------------------------
# PPC check p-value convention (fast — no fit)
# ---------------------------------------------------------------------------


def test_ppc_checks_pass_when_observed_is_central():
    """One-sided Bayesian p-value: an observed statistic at the median of the
    replicated distribution (the best possible fit) must PASS.

    Regression: the old two-tailed folding mapped a central statistic to
    p ≈ 1.0, which the 0.05 < p < 0.95 band then flagged as a failure — the
    better the fit, the more likely the check failed.
    """
    import numpy as np

    from mmm_framework.validation.posterior_predictive import (
        AutocorrelationCheck,
        MeanCheck,
        SkewnessCheck,
        VarianceCheck,
    )

    # seed chosen so the observed draw is comfortably central on all four
    # statistics (any single draw can legitimately land in a 5% tail)
    rng = np.random.default_rng(4)
    y_rep = rng.normal(814.0, 136.0, size=(1000, 156))
    y_obs = rng.normal(814.0, 136.0, size=156)

    for check in (MeanCheck(), VarianceCheck(), SkewnessCheck(), AutocorrelationCheck()):
        res = check.compute(y_obs, y_rep, significance_level=0.05)
        assert res.passed, f"{res.check_name} failed with p={res.p_value}"
        # one-sided convention: central observed statistic → p near 0.5
        assert 0.05 < res.p_value < 0.95


def test_ppc_variance_check_fails_on_real_misfit():
    """Under-dispersed replicates vs observed → p in a tail → Fail."""
    import numpy as np

    from mmm_framework.validation.posterior_predictive import VarianceCheck

    rng = np.random.default_rng(0)
    y_rep = rng.normal(814.0, 50.0, size=(1000, 156))  # too little variance
    y_obs = rng.normal(814.0, 136.0, size=156)

    res = VarianceCheck().compute(y_obs, y_rep, significance_level=0.05)
    assert not res.passed
    assert res.p_value < 0.05 or res.p_value > 0.95


# ---------------------------------------------------------------------------
# graceful degradation (fast — no fit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        "posterior_predictive_checks",
        "residual_diagnostics",
        "channel_diagnostics",
        "refutation_suite",
    ],
)
def test_ops_error_gracefully_on_unfitted(op):
    """A non-model input returns an error string, never raises."""
    res = MO.OPS[op](object())
    assert res["error"] is not None
    assert res["content"] is None


def test_validate_model_never_raises_on_unfitted():
    """The battery degrades to Error rows rather than aborting."""
    res = MO.validate_model(object())
    assert res["error"] is None  # battery itself always succeeds
    rows = res["tables"][0]["rows"]
    checks = {r["check"] for r in rows}
    assert {
        "Convergence",
        "Posterior predictive",
        "Residuals",
        "Channel identifiability",
        "Confounding robustness",
    } <= checks
    # every sub-check resolves to a known verdict — none escapes as an exception
    assert all(r["verdict"] in ("Pass", "Warn", "Error") for r in rows)


# ---------------------------------------------------------------------------
# real validators against a fitted model (slow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_mmm():
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
    from mmm_framework.synth import dgp

    panel = dgp.build("clean", seed=7, n_weeks=120).panel()
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=120,
        n_tune=120,
        use_parametric_adstock=True,
    )
    mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
    results = mmm.fit(random_seed=11, progressbar=False)
    return mmm, results


@pytest.mark.slow
def test_ppc_op_real(fitted_mmm):
    mmm, results = fitted_mmm
    res = MO.OPS["posterior_predictive_checks"](mmm, results)
    assert res["error"] is None
    assert res["tables"] and res["tables"][0]["rows"]
    assert "validation_ppc" in res["dashboard"]
    # at least one PPC figure rendered
    assert res.get("plots")


@pytest.mark.slow
def test_residual_op_real(fitted_mmm):
    mmm, results = fitted_mmm
    res = MO.OPS["residual_diagnostics"](mmm, results)
    assert res["error"] is None
    assert res["tables"][0]["rows"]
    assert res.get("plots")  # residual/acf/qq figures


@pytest.mark.slow
def test_channel_op_real(fitted_mmm):
    mmm, results = fitted_mmm
    res = MO.OPS["channel_diagnostics"](mmm, results)
    assert res["error"] is None
    assert "validation_channels" in res["dashboard"]


@pytest.mark.slow
def test_refutation_op_real(fitted_mmm):
    mmm, results = fitted_mmm
    res = MO.OPS["refutation_suite"](mmm, results)
    assert res["error"] is None
    rows = res["tables"][0]["rows"]
    assert rows and "Robustness Value" in rows[0]


@pytest.mark.slow
def test_validate_model_battery_real(fitted_mmm):
    mmm, results = fitted_mmm
    res = MO.validate_model(mmm, results)
    assert res["error"] is None
    rows = res["tables"][0]["rows"]
    # a real fit should produce at least some Pass verdicts (not all Error)
    verdicts = [r["verdict"] for r in rows]
    assert "Pass" in verdicts
    assert "Error" not in verdicts  # all checks should run on a real MMM


@pytest.mark.slow
def test_cross_validation_op_real(fitted_mmm):
    mmm, results = fitted_mmm
    res = MO.cross_validation(mmm, results, horizon=8, max_origins=1, draws=60, tune=60)
    assert res["error"] is None
    assert res["tables"][0]["rows"]  # model + baseline accuracy rows
