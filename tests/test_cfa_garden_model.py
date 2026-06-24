"""End-to-end proof that a non-MMM family (Bayesian CFA) rides the full garden /
estimand / serialization pipeline: it builds, fits, recovers a planted 2-factor
loading structure, surfaces fit-index estimands through the same
``evaluate_estimands`` path as an MMM's ROI, round-trips, and passes the
capability-gated compatibility suite."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from bayesian_cfa import BayesianCFA, CFAConfig, synthetic_cfa_panel  # noqa: E402


def _build(n=300, **params):
    panel, true_load = synthetic_cfa_panel(n=n)
    mp = {"n_factors": 2, "factor_assignment": [0, 0, 0, 1, 1, 1], **params}
    mmm = BayesianCFA(
        panel, ModelConfig(), TrendConfig(type=TrendType.NONE), model_params=mp
    )
    return mmm, true_load


class TestStructure:
    def test_is_non_mmm_with_config_schema(self):
        from mmm_framework.garden.contract import is_mmm_model, model_kind

        assert BayesianCFA.__garden_model_kind__ == "cfa"
        assert model_kind(BayesianCFA) == "cfa"
        assert not is_mmm_model(BayesianCFA)
        assert BayesianCFA.CONFIG_SCHEMA is CFAConfig

    def test_prepare_data_assembles_indicators(self):
        mmm, _ = _build()
        assert mmm.channel_names == []  # non-MMM: no channels
        assert mmm.n_indicators == 6
        assert mmm.indicators.shape == (300, 6)
        assert mmm.model_params.n_factors == 2  # defaulted/validated config

    def test_passes_non_mmm_contract(self):
        from mmm_framework.garden.contract import validate_class, validate_instance

        assert validate_class(BayesianCFA) == []
        mmm, _ = _build()
        assert validate_instance(mmm) == []  # no channel attrs required

    def test_bad_factor_assignment_rejected(self):
        mmm, _ = _build(factor_assignment=[0, 0, 0, 1, 1])  # wrong length
        with pytest.raises(ValueError, match="factor_assignment"):
            _ = mmm.model  # builds lazily


@pytest.mark.slow
class TestEndToEnd:
    @pytest.fixture(scope="class")
    def fitted(self):
        mmm, true_load = _build(n=400)
        mmm.fit(method="map", random_seed=7)
        return mmm, true_load

    def test_recovers_planted_loadings(self, fitted):
        mmm, true_load = fitted
        summary = mmm.factor_loadings_summary()
        assert list(summary["factor"]) == ["F1", "F1", "F1", "F2", "F2", "F2"]
        # Every loading recovered within tolerance of the planted value.
        assert np.allclose(summary["loading"].values, true_load, atol=0.2)

    def test_fit_index_estimands_evaluate(self, fitted):
        mmm, _ = fitted
        res = mmm.evaluate_estimands()  # DEFAULT_ESTIMANDS: srmr, cov_fit
        assert {"srmr", "cov_fit"} <= set(res)
        assert res["srmr"].status == "ok" and np.isfinite(res["srmr"].mean)
        assert res["srmr"].mean < 0.1  # good fit on well-specified data
        assert res["cov_fit"].status == "ok" and res["cov_fit"].mean > 0.8

    def test_named_loading_estimand(self, fitted):
        from mmm_framework.estimands.registry import factor_loading

        mmm, true_load = fitted
        res = mmm.evaluate_estimands([factor_loading("x1", var="loading_x1")])
        assert res["x1"].status == "ok"
        assert abs(res["x1"].mean - true_load) < 0.2

    def test_capabilities_expose_latents(self, fitted):
        mmm, _ = fitted
        caps = mmm.model_capabilities()
        assert "HAS_LATENT:srmr" in caps
        assert "HAS_LATENT:cov_fit" in caps
        assert "HAS_CHANNELS" not in caps  # no channels

    def test_serialization_round_trip(self, fitted, tmp_path):
        from mmm_framework.serialization import MMMSerializer

        mmm, _ = fitted
        path = str(tmp_path / "cfa_model")
        MMMSerializer.save(mmm, path)
        loaded = MMMSerializer.load(path, mmm.panel)
        assert type(loaded).__name__ == "BayesianCFA"
        assert loaded.model_params.n_factors == 2
        assert loaded._trace is not None
        # The reloaded model still evaluates its estimands.
        res = loaded.evaluate_estimands()
        assert res["srmr"].status == "ok"


@pytest.mark.slow
def test_compat_suite_passes_for_cfa():
    """The capability-gated compatibility suite: a CFA passes the blocking tiers
    (static/build/fit/instance/trace) with the MMM-only tiers (scaling/ops_smoke/
    accuracy) marked skipped."""
    from mmm_framework.garden.compat import run_compatibility_check

    report = run_compatibility_check(
        BayesianCFA,
        scenarios=("clean",),
        fit_method="map",
        n_weeks=60,
        check_carryover=False,
    )
    tiers = {t["name"]: t for t in report["tiers"]}
    assert report["blocking_passed"], report.get("summary")
    # MMM-only tiers are skipped, not failed.
    for name in ("scaling", "ops_smoke"):
        assert tiers[name]["skipped"], (name, tiers.get(name))
    # Core structural tiers actually ran and passed.
    for name in ("static", "build", "fit", "instance", "trace"):
        assert tiers[name]["passed"] and not tiers[name]["skipped"], (
            name,
            tiers.get(name),
        )
