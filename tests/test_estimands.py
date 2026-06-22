"""Tests for the declarative estimand subsystem.

Three layers:

* **Spec** — serialize/round-trip every built-in + the worked instances; pure,
  fast (no model).
* **Capabilities** — duck-typed flag detection + capability-gated degrade.
* **Equivalence** (the bit-stability gate) — the engine reproduces all four
  legacy notions to the bit on a fixed-seed fit:
  ``compute_roi_with_uncertainty`` (#2), ``compute_channel_roi`` (#1),
  ``compute_marginal_contributions`` (#3), ``compute_counterfactual_contributions``
  (#4); plus the in-graph ``build_estimand_expr`` (#4-graph).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.estimands import Estimand, EstimandResult, registry
from mmm_framework.estimands.capabilities import (
    HAS_CHANNELS,
    HAS_CONTRIBUTION_DETERMINISTIC,
    HAS_CONTRIBUTIONS,
    model_capabilities,
)
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

# =============================================================================
# Spec round-trip
# =============================================================================


class TestSpecRoundTrip:
    def test_every_builtin_round_trips(self):
        for est in registry.all_builtins():
            again = Estimand.from_dict(json.loads(json.dumps(est.to_dict())))
            assert again == est, est.name

    def test_builtin_names_and_kinds(self):
        names = set(registry.BUILTINS)
        assert {
            "contribution_roi",
            "counterfactual_roi",
            "marginal_roas",
            "contribution",
            "awareness_lift",
            "cost_per_conversion",
        } <= names

    def test_default_names_are_the_mmm_three(self):
        assert registry.DEFAULT_NAMES == [
            "contribution_roi",
            "marginal_roas",
            "contribution",
        ]

    def test_user_declared_estimand_round_trips(self):
        # A bespoke estimand a user might attach per-test (windowed).
        d = registry.get("marginal_roas").to_dict()
        d["name"] = "tv_q4_mroas"
        d["window"] = {"start": 10, "end": 20}
        parsed = Estimand.from_dict(d)
        assert parsed.window.as_tuple() == (10, 20)
        assert parsed.name == "tv_q4_mroas"
        assert Estimand.from_dict(parsed.to_dict()) == parsed

    def test_result_round_trips(self):
        r = EstimandResult(
            name="roi:TV",
            kind="roi",
            mean=1.2,
            hdi_low=0.5,
            hdi_high=2.0,
            extra={"prob_positive": 0.9},
        )
        assert EstimandResult(**r.to_dict()) == r


# =============================================================================
# Capability gating (no fit required)
# =============================================================================


class TestCapabilities:
    def test_defaults_filter_by_capability(self):
        full = {HAS_CHANNELS, HAS_CONTRIBUTIONS, HAS_CONTRIBUTION_DETERMINISTIC}
        assert [e.name for e in registry.defaults_for(full)] == registry.DEFAULT_NAMES
        # No in-graph deterministic -> the dashboard ROI drops out.
        no_det = {HAS_CHANNELS, HAS_CONTRIBUTIONS}
        assert "contribution_roi" not in [e.name for e in registry.defaults_for(no_det)]
        # A non-MMM model with no channels gets nothing.
        assert registry.defaults_for(set()) == []

    def test_duck_typed_detection_on_stub(self):
        class Stub:
            channel_names = ["A"]

            def compute_counterfactual_contributions(self):  # pragma: no cover
                ...

        caps = model_capabilities(Stub())
        assert HAS_CHANNELS in caps and HAS_CONTRIBUTIONS in caps
        assert HAS_CONTRIBUTION_DETERMINISTIC not in caps  # unfitted stub


# =============================================================================
# In-graph estimand equivalence (#4-graph) — no fit needed
# =============================================================================


class TestGraphEstimandEquivalence:
    def test_graph_matches_legacy_build_estimand_expr(self):
        import pytensor.tensor as pt

        from mmm_framework.calibration.likelihood import (
            ExperimentEstimand,
            build_estimand_expr as legacy,
        )
        from mmm_framework.estimands.graph import build_estimand_expr as engine

        cw = pt.as_tensor_variable(3.5)
        cwp = pt.as_tensor_variable(3.85)
        for est in (
            ExperimentEstimand.CONTRIBUTION,
            ExperimentEstimand.ROAS,
            ExperimentEstimand.MROAS,
        ):
            a = legacy(
                est,
                contrib_window=cw,
                spend_window=12.0,
                scale=2.0,
                contrib_window_pert=cwp,
                lift=0.1,
            ).eval()
            b = engine(
                est,
                contrib_window=cw,
                spend_window=12.0,
                scale=2.0,
                contrib_window_pert=cwp,
                lift=0.1,
            ).eval()
            assert np.isclose(a, b), est

    def test_legacy_now_delegates_to_engine(self):
        # After step 11 the calibration shim is the engine function (re-exported).
        from mmm_framework.calibration import likelihood as L
        from mmm_framework.estimands import graph

        assert L.build_estimand_expr is graph.build_estimand_expr


# =============================================================================
# Post-hoc evaluator equivalence (the bit-stability gate) — requires a fit
# =============================================================================


@pytest.fixture(scope="module")
def fitted_model():
    periods = pd.date_range("2021-01-04", periods=40, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(5)
    t = np.arange(n)
    tv = np.abs(rng.normal(100, 25, n))
    digital = np.abs(rng.normal(80, 20, n))
    y = pd.Series(
        1000
        + 10.0 * t
        + 50.0 * np.sin(2 * np.pi * t / 52)
        + 1.0 * tv
        + 0.5 * digital
        + rng.normal(0, 20, n),
        name="Sales",
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": digital}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV", "Digital"],
            controls=None,
        ),
        index=periods,
        config=config,
    )
    model = BayesianMMM(
        panel,
        ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=2,
            n_draws=150,
            n_tune=150,
            target_accept=0.85,
        ),
        TrendConfig(type=TrendType.LINEAR),
    )
    model.fit(random_seed=0)
    return model


SEED = 0


@pytest.mark.slow
class TestEstimandEquivalence:
    def test_capabilities_on_fitted_mmm(self, fitted_model):
        caps = fitted_model.model_capabilities()
        assert {HAS_CHANNELS, HAS_CONTRIBUTIONS, HAS_CONTRIBUTION_DETERMINISTIC} <= caps

    def test_contribution_roi_equals_dashboard(self, fitted_model):
        from mmm_framework.reporting.helpers import compute_roi_with_uncertainty

        legacy = compute_roi_with_uncertainty(fitted_model, hdi_prob=0.94).set_index(
            "channel"
        )
        res = fitted_model.evaluate_estimands(
            estimands=[registry.get("contribution_roi")], random_seed=SEED
        )
        for ch in fitted_model.channel_names:
            r = res[f"contribution_roi:{ch}"]
            assert np.isclose(r.mean, legacy.loc[ch, "roi_mean"], rtol=1e-9)
            assert np.isclose(r.hdi_low, legacy.loc[ch, "roi_hdi_low"], rtol=1e-9)
            assert np.isclose(r.hdi_high, legacy.loc[ch, "roi_hdi_high"], rtol=1e-9)
            assert np.isclose(
                r.extra["prob_profitable"], legacy.loc[ch, "prob_profitable"]
            )
            assert np.isclose(r.extra["prob_positive"], legacy.loc[ch, "prob_positive"])

    def test_counterfactual_roi_equals_compute_channel_roi(self, fitted_model):
        from mmm_framework.analysis import MMMAnalyzer

        legacy = (
            MMMAnalyzer(fitted_model)
            .compute_channel_roi(random_seed=SEED)
            .set_index("Channel")
        )
        res = fitted_model.evaluate_estimands(
            estimands=[registry.get("counterfactual_roi")], random_seed=SEED
        )
        for ch in fitted_model.channel_names:
            r = res[f"counterfactual_roi:{ch}"]
            assert np.isclose(r.mean, legacy.loc[ch, "ROI"], rtol=1e-9)
            assert np.isclose(
                r.extra["contribution_mean"],
                legacy.loc[ch, "Total Contribution"],
                rtol=1e-9,
            )
            assert np.isclose(
                r.extra["contribution_hdi_low"],
                legacy.loc[ch, "Contribution HDI Low"],
                rtol=1e-9,
            )
            assert np.isclose(
                r.extra["contribution_hdi_high"],
                legacy.loc[ch, "Contribution HDI High"],
                rtol=1e-9,
            )
            assert np.isclose(
                r.extra["contribution_pct"], legacy.loc[ch, "Contribution %"], rtol=1e-9
            )

    def test_marginal_roas_equals_compute_marginal(self, fitted_model):
        legacy = fitted_model.compute_marginal_contributions(
            spend_increase_pct=10.0, hdi_prob=0.94, random_seed=SEED
        ).set_index("Channel")
        res = fitted_model.evaluate_estimands(
            estimands=[registry.get("marginal_roas")], random_seed=SEED
        )
        for ch in fitted_model.channel_names:
            r = res[f"marginal_roas:{ch}"]
            assert np.isclose(r.mean, legacy.loc[ch, "Marginal ROAS"], rtol=1e-9)
            assert np.isclose(
                r.extra["contribution_mean"],
                legacy.loc[ch, "Marginal Contribution"],
                rtol=1e-9,
            )
            assert np.isclose(
                r.hdi_low, legacy.loc[ch, "Marginal ROAS HDI Low"], rtol=1e-9
            )
            assert np.isclose(
                r.hdi_high, legacy.loc[ch, "Marginal ROAS HDI High"], rtol=1e-9
            )

    def test_contribution_equals_counterfactual_total(self, fitted_model):
        legacy = fitted_model.compute_counterfactual_contributions(random_seed=SEED)
        res = fitted_model.evaluate_estimands(
            estimands=[registry.get("contribution")], random_seed=SEED
        )
        for ch in fitted_model.channel_names:
            r = res[f"contribution:{ch}"]
            assert np.isclose(r.mean, legacy.total_contributions[ch], rtol=1e-9)

    def test_default_set_and_new_demonstrators(self, fitted_model):
        res = fitted_model.evaluate_estimands(random_seed=SEED)
        # MMM defaults expanded per channel.
        for ch in fitted_model.channel_names:
            assert res[f"contribution_roi:{ch}"].status == "ok"
            assert res[f"marginal_roas:{ch}"].status == "ok"
            assert res[f"contribution:{ch}"].status == "ok"
        # Demonstrators: a no-denominator lift and an inverted ratio.
        extra = fitted_model.evaluate_estimands(
            estimands=[
                registry.get("awareness_lift"),
                registry.get("cost_per_conversion"),
            ],
            random_seed=SEED,
        )
        for ch in fitted_model.channel_names:
            assert extra[f"awareness_lift:{ch}"].mean is not None
            assert extra[f"cost_per_conversion:{ch}"].mean is not None

    def test_unsupported_capability_degrades(self, fitted_model):
        bogus = registry.get("contribution")
        bogus.required_capabilities = ["HAS_FACTOR_LOADINGS"]
        res = fitted_model.evaluate_estimands(estimands=[bogus], random_seed=SEED)
        # Wildcard expands per channel; each is unsupported (never raises).
        statuses = {r.status for r in res.values()}
        assert statuses == {"unsupported"}

    def test_agent_op_compute_estimands(self, fitted_model):
        from mmm_framework.agents.model_ops import compute_estimands

        out = compute_estimands(fitted_model, random_seed=SEED)
        assert out["error"] is None
        rows = out["dashboard"]["estimands"]
        names = {r["estimand"] for r in rows}
        assert {"contribution_roi", "marginal_roas", "contribution"} <= names
        assert all(r["status"] == "ok" for r in rows)
        assert out["tables"] and out["tables"][0]["title"] == "Estimands"


@pytest.mark.slow
class TestEstimandThreading:
    def test_fit_time_population_and_serialization(self, tmp_path):
        # Declare an estimand, refit so it is realized at fit time, save/load,
        # and confirm declared_estimands round-trips.
        import json

        from mmm_framework.serialization import MMMSerializer

        fm = _small_fit(declared=[registry.get("contribution_roi")])
        assert fm.declared_estimands  # set on the instance
        # Best-effort fit-time population landed the per-channel results.
        assert any(
            k.startswith("contribution_roi:") for k in fm._last_results.estimands
        )

        path = tmp_path / "model"
        MMMSerializer.save(fm, str(path))
        # Save side: declared_estimands persisted in metadata with a schema_version.
        meta = json.loads((path / "metadata.json").read_text())
        assert [e["name"] for e in meta["declared_estimands"]] == ["contribution_roi"]
        assert meta["declared_estimands"][0]["schema_version"]
        # Load side: reconstructed onto the instance.
        loaded = MMMSerializer.load(str(path), fm.panel, rebuild_model=False)
        assert [e.name for e in loaded.declared_estimands] == ["contribution_roi"]

    def test_fitting_spec_estimand_dicts_round_trip(self):
        # build_model wires spec["estimands"] -> declared_estimands via the same
        # Estimand.from_dict path tested here (the full build_model integration is
        # exercised end-to-end via the running app).
        from mmm_framework.estimands.spec import Estimand

        spec_estimands = [registry.get("marginal_roas").to_dict()]
        parsed = [Estimand.from_dict(e) for e in spec_estimands]
        assert parsed[0].name == "marginal_roas"
        # A malformed entry raises (build_model wraps this in a clear ValueError).
        with pytest.raises(Exception):
            Estimand.from_dict({"not": "an estimand"})


def _small_fit(declared=None):
    periods = pd.date_range("2021-01-04", periods=30, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(7)
    tv = np.abs(rng.normal(100, 25, n))
    y = pd.Series(1000 + 1.0 * tv + rng.normal(0, 20, n), name="Sales")
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV"],
            controls=[],
        ),
        index=periods,
        config=config,
    )
    model = BayesianMMM(
        panel,
        ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=2,
            n_draws=100,
            n_tune=100,
            target_accept=0.85,
        ),
        TrendConfig(type=TrendType.LINEAR),
    )
    if declared:
        model.declared_estimands = list(declared)
    res = model.fit(random_seed=0)
    model._last_results = res
    return model
