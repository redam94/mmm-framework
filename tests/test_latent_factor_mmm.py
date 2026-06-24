"""End-to-end proof of the JOINT latent-factor MMM (``LatentFactorMMM``): an MMM
that estimates a latent "economic health" factor from indicator columns IN THE
SAME GRAPH and conditions on it.

It validates the two things that matter:

* **Recovery** — the joint model recovers the planted factor (high correlation
  with the truth) and the loading SIGNS (incl. a negative loading), through one
  shared latent that informs both the indicator block and the KPI.
* **De-confounding (the headline)** — because economic health confounds spend
  and sales, a naive MMM that omits the indicators over-credits the
  demand-chasing channels; the joint model recovers their ROI markedly closer to
  the causal truth.

Plus the hybrid report (channel/ROI sections AND a factor-loadings section) and a
serialization round-trip. Fits use NUTS (MAP is too unstable for a 150-parameter
latent model); the slow tests fit once via a class-scoped fixture.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.model import BayesianMMM, TrendConfig
from mmm_framework.model.trend_config import TrendType

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from latent_factor_mmm import (  # noqa: E402
    LatentFactorMMM,
    LatentFactorParams,
    economic_health_dataset,
)

# NUTS settings: 4 chains + enough tuning so the latent AR(1) factor converges
# (a 2-chain quick fit leaves the per-period factor under-mixed).
_FIT = dict(draws=500, tune=1000, chains=4, target_accept=0.95, random_seed=11)


def _roas(mmm: BayesianMMM, spend_sum: dict[str, float]) -> dict[str, float]:
    """Per-channel total-effect ROAS (causal counterfactual) — the exact estimand
    the synthetic ``true_roas`` answer key uses."""
    contrib = mmm.compute_counterfactual_contributions(compute_uncertainty=False)
    return {
        ch: float(contrib.total_contributions[ch] / spend_sum[ch])
        for ch in mmm.channel_names
    }


class TestStructure:
    """Fast structural checks (no fit)."""

    def test_is_hybrid_mmm_with_latent_structure(self):
        from mmm_framework.garden.contract import (
            has_latent_structure,
            is_mmm_model,
            model_kind,
        )

        # It IS an MMM (channels/ROI) AND carries latent structure (loadings).
        assert model_kind(LatentFactorMMM) == "mmm"
        assert is_mmm_model(LatentFactorMMM) is True
        assert has_latent_structure(LatentFactorMMM) is True
        assert LatentFactorMMM.CONFIG_SCHEMA is LatentFactorParams

    def test_prepare_data_splits_indicators_from_controls(self):
        dataset, _ = economic_health_dataset(seed=14)
        mmm = LatentFactorMMM(dataset, ModelConfig(), TrendConfig(type=TrendType.NONE))
        # Channels are the media; the 4 econ indicators are a SEPARATE block, not
        # MMM controls (only Price is a control).
        assert mmm.channel_names == ["TV", "Search", "Social", "Display"]
        assert mmm.n_indicators == 4
        assert mmm.indicators.shape[1] == 4
        assert list(mmm.indicator_names) == [
            "gdp_growth",
            "consumer_confidence",
            "unemployment",
            "retail_sales",
        ]
        assert mmm.control_names == ["Price"]  # indicators excluded from controls
        # Period-axis measurement matrix (national: one row per period).
        assert mmm.indicators_by_period.shape == (mmm.n_periods, 4)

    def test_passes_mmm_contract(self):
        from mmm_framework.garden.contract import validate_class, validate_instance

        assert validate_class(LatentFactorMMM) == []
        dataset, _ = economic_health_dataset(seed=14)
        mmm = LatentFactorMMM(dataset, ModelConfig(), TrendConfig(type=TrendType.NONE))
        assert validate_instance(mmm) == []  # MMM attrs required AND present

    def test_requires_indicator_role(self):
        # A plain MFF panel cannot carry indicators -> fail fast at construction.
        from mmm_framework.synth import dgp

        sc = dgp.make_economic_health(seed=14)
        with pytest.raises(ValueError, match="indicator|INDICATOR|roles"):
            LatentFactorMMM(sc.panel(), ModelConfig(), TrendConfig(type=TrendType.NONE))

    def test_both_factor_dynamics_build(self):
        dataset, _ = economic_health_dataset(seed=14)
        # static is the default; ar1 is the alternative — both build lazily.
        for dyn in ("static", "ar1"):
            mmm = LatentFactorMMM(
                dataset,
                ModelConfig(),
                TrendConfig(type=TrendType.NONE),
                model_params={"factor_dynamics": dyn},
            )
            assert mmm.model is not None, dyn
            assert "economic_health" in mmm.model.named_vars, dyn

    def test_suggest_initvals_warm_start(self):
        # The PCA warm-start seeds the factor + loadings (sign-aligned to the
        # positive anchor) — the fix for factor-model multimodality.
        dataset, _ = economic_health_dataset(seed=14)
        mmm = LatentFactorMMM(dataset, ModelConfig(), TrendConfig(type=TrendType.NONE))
        init = mmm.suggest_initvals()
        assert init["loading_anchor"] > 0
        assert init["econ_innovation"].shape == (mmm.n_periods,)
        assert init["loading_rest"].shape == (mmm.n_indicators - 1,)


@pytest.mark.slow
class TestEndToEnd:
    @pytest.fixture(scope="class")
    def fitted(self):
        dataset, sc = economic_health_dataset(seed=14)
        spend_sum = {ch: float(sc.spend[ch].sum()) for ch in sc.channels}

        # Default factor_dynamics="static" -> NUTS auto-warm-starts at the
        # indicator PCA (a free per-period factor is multimodal; the warm start
        # keeps every chain in the indicator-dominated basin).
        joint = LatentFactorMMM(
            dataset, ModelConfig(), TrendConfig(type=TrendType.NONE)
        )
        joint.fit(**_FIT)

        # Naive baseline: a plain MMM on the SAME data WITHOUT the indicators
        # (the confounded model the joint one is meant to beat).
        naive = BayesianMMM(sc.panel(), ModelConfig(), TrendConfig(type=TrendType.NONE))
        naive.fit(**_FIT)

        return joint, naive, sc, spend_sum, dataset

    def test_recovers_loading_signs(self, fitted):
        joint, _, sc, _, _ = fitted
        summary = joint.factor_loadings_summary()
        loadings = dict(zip(summary["indicator"], summary["loading"]))
        truth = sc.notes["true_loadings"]
        # Every indicator's loading recovers the planted SIGN (incl. the negative
        # loading on unemployment) and is non-trivial in magnitude.
        for name, true_lam in truth.items():
            assert np.sign(loadings[name]) == np.sign(true_lam), (name, loadings[name])
            assert abs(loadings[name]) > 0.1, (name, loadings[name])

    def test_recovers_latent_factor_series(self, fitted):
        joint, _, sc, _, _ = fitted
        e_hat = joint._trace.posterior["economic_health"].mean(("chain", "draw")).values
        # Factor identified up to sign -> compare on |corr|. Pinned tightly by the
        # indicators, so it tracks the truth closely.
        r = abs(np.corrcoef(e_hat, sc.notes["latent_econ"])[0, 1])
        assert r > 0.9, r

    def test_deconfounds_media_roi(self, fitted):
        """HEADLINE: the joint model recovers the demand-chasing channels' ROI
        closer to the causal truth than a naive MMM that omits economic health."""
        joint, naive, sc, spend_sum, _ = fitted
        truth = sc.true_roas
        chasers = sc.notes["chasers"]  # Search, Social — where the naive bias bites
        jr = _roas(joint, spend_sum)
        nr = _roas(naive, spend_sum)

        def rel_err(roas):
            return float(np.mean([abs(roas[c] - truth[c]) / truth[c] for c in chasers]))

        joint_err, naive_err = rel_err(jr), rel_err(nr)
        # The de-confounding win: lower mean relative error on the chasers.
        assert joint_err < naive_err, (joint_err, naive_err, jr, nr)
        # And the correction is in the right DIRECTION: the joint pulls each
        # chaser's ROI down from the naive (confounding-inflated) estimate.
        for c in chasers:
            assert jr[c] < nr[c], (c, jr[c], nr[c])

    def test_declared_estimands_evaluate(self, fitted):
        joint, _, _, _, _ = fitted
        res = joint.evaluate_estimands()  # contribution_roi, marginal_roas, loadings…
        # De-biased ROI present for every channel.
        assert all(f"contribution_roi:{c}" in res for c in joint.channel_names)
        # The latent level + named loadings surface through the estimand engine.
        assert res["economic_health_level"].status == "ok"
        assert res["loading_unemployment"].status == "ok"
        assert res["loading_unemployment"].mean < 0  # the negative loading

    def test_capabilities_expose_latents(self, fitted):
        joint, _, _, _, _ = fitted
        caps = joint.model_capabilities()
        assert "HAS_LATENT:economic_health" in caps
        assert "HAS_LATENT:loading_unemployment" in caps
        assert "HAS_CHANNELS" in caps  # it IS an MMM

    def test_hybrid_report_has_both_sections(self, fitted):
        from mmm_framework.reporting import MMMReportGenerator

        joint, _, _, _, _ = fitted
        gen = MMMReportGenerator(model=joint)
        html = gen.render()
        assert gen.data.model_kind == "mmm"  # still MMM-gated
        assert gen.data.factor_loadings  # latent fields filled by the MMM extractor
        assert 'id="factor-analysis"' in html  # latent section ON
        assert 'id="channel-roi"' in html  # MMM section also ON (hybrid)
        assert 'id="decomposition"' in html
        assert "Economic Health Factor" in html  # bundle.latent_section_title

    def test_serialization_round_trip(self, fitted, tmp_path):
        from mmm_framework.serialization import MMMSerializer

        joint, _, _, _, dataset = fitted
        path = str(tmp_path / "lfm_model")
        MMMSerializer.save(joint, path)
        # Reload against the role-tagged Dataset (carries the INDICATOR columns the
        # measurement block rebuilds from) — NOT a bare PanelDataset.
        loaded = MMMSerializer.load(path, dataset)
        assert type(loaded).__name__ == "LatentFactorMMM"
        assert loaded.model_params.factor_dynamics == "static"
        assert loaded._trace is not None
        assert len(loaded.factor_loadings_summary()) == 4
