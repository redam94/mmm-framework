"""Experiment (incrementality / ROAS) calibration for the extension models.

Each extension model (MultivariateMMM, NestedMMM, CombinedMMM) folds an
experiment into its PyMC graph as a likelihood on the model-implied estimand,
built from the channel's effective coefficient and saturated spend.

The estimand math is verified against the *independent in-graph counterfactual*:
zeroing the channel's spend (``pm.set_data``) and differencing the exposed ``mu``
Deterministic at the same parameter draws gives the channel's contribution
without re-using the coefficient assembly under test -- so a wrong mediated/
routed coefficient is caught (a coef-vs-coef recompute would be circular).
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.basic import ancestors

from mmm_framework.calibration import ExperimentEstimand, ExperimentMeasurement
from mmm_framework.mmm_extensions.components.transforms import logistic_saturation_pt
from mmm_framework.transforms.adstock_pt import parametric_adstock_pt
from mmm_framework.mmm_extensions.builders import (
    MediatorConfigBuilder,
    MultivariateModelConfigBuilder,
    NestedModelConfigBuilder,
    OutcomeConfigBuilder,
    cannibalization_effect,
)
from mmm_framework.mmm_extensions.config import CombinedModelConfig
from mmm_framework.mmm_extensions.models import CombinedMMM, MultivariateMMM, NestedMMM

N = 52
PERIODS = pd.date_range("2021-01-04", periods=N, freq="W-MON")
WINDOW = (str(PERIODS[10].date()), str(PERIODS[30].date()))
WINDOW_MASK = np.asarray(
    (PERIODS >= pd.to_datetime(WINDOW[0])) & (PERIODS <= pd.to_datetime(WINDOW[1]))
)
CHANNELS = ["tv", "digital", "social"]


def _media():
    rng = np.random.default_rng(5)
    return np.abs(rng.normal(100, 50, (N, 3)))


def _estimand_node(model, prefix):
    names = [
        k
        for k in model.named_vars
        if k.startswith(prefix) and k.endswith("_model_estimand")
    ]
    assert names, f"no estimand node with prefix {prefix!r}"
    return names[0]


def _counterfactual(
    model,
    est_node,
    mu_name,
    X,
    c,
    k,
    estimand,
    *,
    lift=None,
    spend=None,
    seed=0,
    draws=6,
):
    """Independent oracle: estimand from differencing the exposed ``mu``.

    For contribution/ROAS the channel is zeroed; for mROAS its window spend is
    scaled by ``(1 + lift)``. Same-seed ``pm.draw`` reproduces identical
    parameter draws across the two data settings (verified separately), so the
    difference isolates the channel's (marginal) effect.
    """
    X_alt = X.copy()
    if estimand is ExperimentEstimand.MROAS:
        X_alt[WINDOW_MASK, c] *= 1.0 + lift
    else:
        X_alt[:, c] = 0.0
    with model:
        est, mu_base = pm.draw(
            [model[est_node], model[mu_name]], draws=draws, random_seed=seed
        )
        pm.set_data({"X_media": X_alt})
        (mu_alt,) = pm.draw([model[mu_name]], draws=draws, random_seed=seed)
        pm.set_data({"X_media": X})

    def col(arr):
        return arr[:, WINDOW_MASK, k] if arr.ndim == 3 else arr[:, WINDOW_MASK]

    if estimand is ExperimentEstimand.MROAS:
        contrib = (col(mu_alt) - col(mu_base)).sum(axis=1)
        return est, contrib / (lift * spend)
    contrib = (col(mu_base) - col(mu_alt)).sum(axis=1)
    if estimand is ExperimentEstimand.ROAS:
        return est, contrib / spend
    return est, contrib


# =============================================================================
# Fixtures: one model of each type
# =============================================================================


def _multivariate():
    X = _media()
    rng = np.random.default_rng(1)
    outcomes = {
        "sales_a": 1000 + X @ np.array([2.0, 1.0, 0.5]) + rng.normal(0, 50, N),
        "sales_b": 800 + X @ np.array([1.0, 2.0, 0.3]) + rng.normal(0, 40, N),
    }
    cfg = (
        MultivariateModelConfigBuilder()
        .add_outcome(
            OutcomeConfigBuilder("sales_a", column="sales_a")
            .with_positive_media_effects(sigma=0.5)
            .build()
        )
        .add_outcome(
            OutcomeConfigBuilder("sales_b", column="sales_b")
            .with_positive_media_effects(sigma=0.5)
            .build()
        )
        .add_cross_effect(cannibalization_effect(source="sales_b", target="sales_a"))
        .build()
    )
    return X, MultivariateMMM(X, outcomes, CHANNELS, cfg, index=PERIODS)


def _nested():
    X = _media()
    rng = np.random.default_rng(2)
    y = 1000 + X @ np.array([2.0, 1.0, 0.5]) + rng.normal(0, 50, N)
    med1 = (
        MediatorConfigBuilder("awareness")
        .fully_latent()
        .with_positive_media_effect(sigma=1.0)
        .with_direct_effect(sigma=0.5)
        .build()
    )
    med2 = (
        MediatorConfigBuilder("traffic")
        .fully_latent()
        .with_positive_media_effect(sigma=1.0)
        .build()
    )
    cfg = (
        NestedModelConfigBuilder()
        .add_mediator(med1)
        .add_mediator(med2)
        .map_channels_to_mediator("awareness", ["tv", "digital"])
        .map_channels_to_mediator("traffic", ["social"])
        .build()
    )
    return X, NestedMMM(X, y, CHANNELS, cfg, index=PERIODS)


def _nested_small():
    # A minimal single-mediator nested model for the slow fit/serialize tests --
    # just enough to exercise the logp/NUTS path (correctness is covered by the
    # fresh-process oracle tests). Keeping the graph small avoids PyTensor's
    # C kernel-argument limit when compiled in a worker that has accumulated
    # state from earlier tests in the suite.
    X = _media()
    rng = np.random.default_rng(2)
    y = 1000 + X @ np.array([2.0, 1.0, 0.5]) + rng.normal(0, 50, N)
    cfg = (
        NestedModelConfigBuilder()
        .add_mediator(
            MediatorConfigBuilder("awareness")
            .fully_latent()
            .with_positive_media_effect(sigma=1.0)
            .build()
        )
        .map_channels_to_mediator("awareness", ["tv"])
        .build()
    )
    return X, NestedMMM(X, y, CHANNELS, cfg, index=PERIODS)


def _combined():
    X = _media()
    rng = np.random.default_rng(5)
    outcomes = {
        "sales_a": 1000 + X @ np.array([2.0, 1.0, 0.5]) + rng.normal(0, 50, N),
        "sales_b": 800 + X @ np.array([1.0, 2.0, 0.3]) + rng.normal(0, 40, N),
    }
    nested = (
        NestedModelConfigBuilder()
        .add_mediator(
            MediatorConfigBuilder("awareness")
            .fully_latent()
            .with_positive_media_effect(sigma=1.0)
            .build()
        )
        .map_channels_to_mediator("awareness", ["tv", "digital"])
        .build()
    )
    mv = (
        MultivariateModelConfigBuilder()
        .add_outcome(
            OutcomeConfigBuilder("sales_a", column="sales_a")
            .with_positive_media_effects(sigma=0.5)
            .build()
        )
        .add_outcome(
            OutcomeConfigBuilder("sales_b", column="sales_b")
            .with_positive_media_effects(sigma=0.5)
            .build()
        )
        .add_cross_effect(cannibalization_effect(source="sales_b", target="sales_a"))
        .build()
    )
    # awareness routes ONLY to sales_a.
    cfg = CombinedModelConfig(
        nested=nested,
        multivariate=mv,
        mediator_to_outcome_map={"awareness": ("sales_a",)},
    )
    return X, CombinedMMM(X, outcomes, CHANNELS, cfg, index=PERIODS)


# =============================================================================
# MultivariateMMM
# =============================================================================


class TestMultivariate:
    def test_nodes_added_per_estimand(self):
        X, mv = _multivariate()
        mv.add_experiment_calibration(
            [
                ExperimentMeasurement("tv", WINDOW, 5e4, 8e3, outcome="sales_a"),
                ExperimentMeasurement(
                    "tv",
                    WINDOW,
                    2.5,
                    0.4,
                    outcome="sales_a",
                    estimand=ExperimentEstimand.ROAS,
                ),
                ExperimentMeasurement(
                    "tv",
                    WINDOW,
                    1.8,
                    0.5,
                    outcome="sales_a",
                    estimand=ExperimentEstimand.MROAS,
                    spend_lift_pct=10.0,
                ),
            ]
        )
        observed = {rv.name for rv in mv.model.observed_RVs}
        for est in ("contribution", "roas", "mroas"):
            assert any(o.startswith(f"experiment_tv_sales_a_{est}_") for o in observed)

    @pytest.mark.parametrize("estimand", list(ExperimentEstimand))
    def test_counterfactual_oracle(self, estimand):
        X, mv = _multivariate()
        c, k = CHANNELS.index("tv"), 0  # sales_a
        spend = float(X[WINDOW_MASK, c].sum())
        kwargs = {"outcome": "sales_a", "estimand": estimand}
        if estimand is ExperimentEstimand.MROAS:
            kwargs["spend_lift_pct"] = 10.0
        mv.add_experiment_calibration(
            [ExperimentMeasurement("tv", WINDOW, 2.0, 0.5, **kwargs)]
        )
        model = mv.model
        node = _estimand_node(model, f"experiment_tv_sales_a_{estimand.value}_")
        est, cf = _counterfactual(
            model, node, "mu", X, c, k, estimand, lift=0.10, spend=spend, seed=11
        )
        np.testing.assert_allclose(est, cf, rtol=1e-4, atol=1e-2)

    def test_unknown_outcome_skipped(self):
        X, mv = _multivariate()
        mv.add_experiment_calibration(
            [ExperimentMeasurement("tv", WINDOW, 5e4, 8e3, outcome="nope")]
        )
        with pytest.warns(UserWarning, match="no model handle"):
            model = mv.model
        assert not any(rv.name.startswith("experiment_") for rv in model.observed_RVs)

    def test_missing_outcome_on_multioutcome_skipped(self):
        # No outcome set -> key is the bare channel, which multi-outcome handles
        # (keyed by (channel, outcome)) do not contain.
        X, mv = _multivariate()
        mv.add_experiment_calibration([ExperimentMeasurement("tv", WINDOW, 5e4, 8e3)])
        with pytest.warns(UserWarning, match="no model handle"):
            mv.model


# =============================================================================
# NestedMMM (mediated coefficient assembly)
# =============================================================================


class TestNested:
    @pytest.mark.parametrize("channel", ["tv", "social"])
    def test_contribution_counterfactual_oracle(self, channel):
        # tv: mediated (awareness) + direct; social: mediated-only (traffic).
        X, nm = _nested()
        c = CHANNELS.index(channel)
        nm.add_experiment_calibration(
            [ExperimentMeasurement(channel, WINDOW, 4e4, 6e3)]
        )
        model = nm.model
        node = _estimand_node(model, f"experiment_{channel}_contribution_")
        est, cf = _counterfactual(
            model, node, "mu", X, c, 0, ExperimentEstimand.CONTRIBUTION, seed=21
        )
        np.testing.assert_allclose(est, cf, rtol=1e-4, atol=1e-2)

    def test_roas_counterfactual_oracle(self):
        X, nm = _nested()
        c = CHANNELS.index("social")
        spend = float(X[WINDOW_MASK, c].sum())
        nm.add_experiment_calibration(
            [
                ExperimentMeasurement(
                    "social", WINDOW, 2.0, 0.4, estimand=ExperimentEstimand.ROAS
                )
            ]
        )
        model = nm.model
        node = _estimand_node(model, "experiment_social_roas_")
        est, cf = _counterfactual(
            model, node, "mu", X, c, 0, ExperimentEstimand.ROAS, spend=spend, seed=33
        )
        np.testing.assert_allclose(est, cf, rtol=1e-4, atol=1e-5)

    def test_single_outcome_uses_bare_channel_key(self):
        # NestedMMM is single-outcome: an experiment with no outcome resolves.
        X, nm = _nested()
        nm.add_experiment_calibration([ExperimentMeasurement("tv", WINDOW, 4e4, 6e3)])
        assert any(rv.name.startswith("experiment_tv") for rv in nm.model.observed_RVs)


# =============================================================================
# CombinedMMM (routed mediated + direct, per outcome)
# =============================================================================


class TestCombined:
    @pytest.mark.parametrize("outcome", ["sales_a", "sales_b"])
    def test_routing_counterfactual_oracle(self, outcome):
        # tv -> sales_a flows direct + via awareness; tv -> sales_b is direct
        # only (awareness routes to sales_a). The counterfactual catches a coef
        # that ignores the mediator->outcome routing.
        X, cm = _combined()
        c = CHANNELS.index("tv")
        k = cm.outcome_names.index(outcome)
        cm.add_experiment_calibration(
            [ExperimentMeasurement("tv", WINDOW, 4e4, 6e3, outcome=outcome)]
        )
        model = cm.model
        node = _estimand_node(model, f"experiment_tv_{outcome}_contribution_")
        est, cf = _counterfactual(
            model, node, "mu", X, c, k, ExperimentEstimand.CONTRIBUTION, seed=44
        )
        np.testing.assert_allclose(est, cf, rtol=1e-4, atol=1e-2)


# =============================================================================
# Shared behaviour (mask guards, setter, no-experiments)
# =============================================================================


class TestSharedBehaviour:
    def test_no_experiments_adds_no_nodes(self):
        X, mv = _multivariate()
        assert not any(
            rv.name.startswith("experiment_") for rv in mv.model.observed_RVs
        )

    def test_out_of_range_window_skipped(self):
        X, mv = _multivariate()
        mv.add_experiment_calibration(
            [
                ExperimentMeasurement(
                    "tv", ("2018-01-01", "2018-06-01"), 5e4, 8e3, outcome="sales_a"
                )
            ]
        )
        with pytest.warns(UserWarning, match="outside the data"):
            model = mv.model
        assert not any(rv.name.startswith("experiment_") for rv in model.observed_RVs)

    def test_holdout_regions_skipped(self):
        X, nm = _nested()
        nm.add_experiment_calibration(
            [ExperimentMeasurement("tv", WINDOW, 4e4, 6e3, holdout_regions=["west"])]
        )
        with pytest.warns(UserWarning, match="require a geo model"):
            model = nm.model
        assert not any(rv.name.startswith("experiment_") for rv in model.observed_RVs)

    def test_setter_rebuilds_model(self):
        X, nm = _nested()
        n_before = len(nm.model.observed_RVs)
        nm.add_experiment_calibration([ExperimentMeasurement("tv", WINDOW, 4e4, 6e3)])
        assert len(nm.model.observed_RVs) == n_before + 1


# =============================================================================
# End-to-end (slow): the experiment node must compile + sample under NUTS
# alongside the model's own likelihood (MvNormal/LKJ for multi-outcome, Normal
# for nested) -- pm.draw only exercises the forward graph, not the logp path.
# =============================================================================


# =============================================================================
# Media pipeline: degeneracy fixed + adstock actually applied
# =============================================================================


class TestMediaPipeline:
    def test_transform_is_not_degenerate(self):
        # normalize -> geometric adstock -> logistic saturation must show real
        # curvature across the spend range -- not be pinned at ~1, which is what
        # logistic_saturation on *raw* (unnormalized) spend produced.
        spend = np.linspace(0.0, 200.0, 50)
        scale = spend.max()
        x = pt.vector("x")
        out = logistic_saturation_pt(
            parametric_adstock_pt(x / scale, "geometric", 8, alpha=0.5, normalize=True),
            3.0,
        )
        v = pytensor.function([x], out)(spend)
        assert v.std() > 0.05  # varies across the spend range
        assert v.max() < 0.999  # not saturated at the flat tail

    def test_alpha_is_ancestor_of_mu_nested(self):
        # Permanent guard against the "alpha created but never applied" bug:
        # every channel's carryover RV must actually feed the expected value.
        X, nm = _nested()
        model = nm.model
        anc = set(ancestors([model["mu"]]))
        for ch in CHANNELS:
            assert model[f"alpha_{ch}"] in anc

    def test_alpha_is_ancestor_of_mu_multivariate(self):
        # MultivariateMMM shares adstock by default (one alpha_shared).
        X, mv = _multivariate()
        model = mv.model
        assert model["alpha_shared"] in set(ancestors([model["mu"]]))


# =============================================================================
# Serialization round-trip (model + experiments + trace)
# =============================================================================


class TestExtensionSerialization:
    def test_roundtrip_preserves_experiments(self, tmp_path):
        X, mv = _multivariate()
        exp = ExperimentMeasurement("tv", WINDOW, 5e4, 8e3, outcome="sales_a")
        mv.add_experiment_calibration([exp])
        mv.model  # build once (graph is dropped on save)
        mv.save(tmp_path)
        loaded = MultivariateMMM.load(tmp_path)
        assert loaded.experiments == [exp]
        assert any(
            rv.name.startswith("experiment_tv_sales_a")
            for rv in loaded.model.observed_RVs
        )

    @pytest.mark.slow
    def test_roundtrip_preserves_trace(self, tmp_path):
        X, nm = _nested_small()
        nm.add_experiment_calibration([ExperimentMeasurement("tv", WINDOW, 4e4, 6e3)])
        nm.fit(draws=40, tune=40, chains=1, random_seed=0, progressbar=False)
        nm.save(tmp_path)
        loaded = NestedMMM.load(tmp_path)
        assert loaded._trace is not None
        assert "alpha_tv" in loaded._trace.posterior
        assert len(loaded.experiments) == 1


@pytest.mark.slow
class TestExtensionFit:
    def test_multivariate_fits_with_experiment(self):
        X, mv = _multivariate()
        mv.add_experiment_calibration(
            [ExperimentMeasurement("tv", WINDOW, 5e4, 8e3, outcome="sales_a")]
        )
        mv.fit(draws=40, tune=40, chains=1, random_seed=0, progressbar=False)
        assert any("model_estimand" in v for v in mv.trace.posterior.data_vars)

    def test_nested_fits_with_experiment(self):
        X, nm = _nested_small()
        nm.add_experiment_calibration(
            [
                ExperimentMeasurement(
                    "tv", WINDOW, 2.0, 0.4, estimand=ExperimentEstimand.ROAS
                )
            ]
        )
        nm.fit(draws=40, tune=40, chains=1, random_seed=0, progressbar=False)
        assert any("model_estimand" in v for v in nm.trace.posterior.data_vars)
