"""Tests for likelihood-based (in-graph) experiment calibration.

These cover the new ``mmm_framework.calibration.likelihood`` route, which folds
an experimental result (contribution / ROAS / marginal ROAS) into the PyMC graph
as a likelihood on the model-implied estimand.

Fast tests cover the pure helpers, the dataclass validation/serialization, and
the in-graph estimand math -- the latter verified with ``pm.draw`` (prior draws,
no MCMC) against independent NumPy recomputation, so the PyTensor expression is
checked rather than asserted by inspection. The end-to-end fit/predict/serialize
flow is marked ``slow``.
"""

import math

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from mmm_framework.calibration import (
    ExperimentEstimand,
    ExperimentMeasurement,
    attach_experiment_likelihood,
    lognormal_sigma_from_moments,
)
from mmm_framework.config import (
    AdstockConfig,
    ControlVariableConfig,
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.transforms import geometric_adstock_2d
from mmm_framework.transforms.adstock import adstock_weights, apply_adstock
from mmm_framework.validation.results import LiftTestResult

PERIOD = ("2021-03-01", "2021-06-30")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def periods():
    return pd.date_range("2021-01-04", periods=40, freq="W-MON")


def _make_panel(periods, *, tv_adstock=None):
    n = len(periods)
    rng = np.random.default_rng(7)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=["Price"],
    )
    tv = np.abs(rng.normal(100, 30, n))
    dig = np.abs(rng.normal(80, 20, n))
    y = pd.Series(1000 + 2.5 * tv + 1.0 * dig + rng.normal(0, 40, n), name="Sales")
    tv_kwargs = {"name": "TV", "dimensions": [DimensionType.PERIOD]}
    if tv_adstock is not None:
        tv_kwargs["adstock"] = tv_adstock
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(**tv_kwargs),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    return PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": dig}),
        X_controls=pd.DataFrame({"Price": 10 + rng.normal(0, 0.5, n)}),
        coords=coords,
        index=periods,
        config=config,
    )


def _make_geo_panel(periods):
    geos = ["A", "B"]
    rng = np.random.default_rng(3)
    idx = pd.MultiIndex.from_product([periods, geos], names=["Period", "Geography"])
    n = len(idx)
    tv = np.abs(rng.normal(100, 30, n))
    dig = np.abs(rng.normal(80, 20, n))
    y = pd.Series(1000 + 2.5 * tv + rng.normal(0, 40, n), name="Sales", index=idx)
    coords = PanelCoordinates(
        periods=periods,
        geographies=geos,
        products=None,
        channels=["TV", "Digital"],
        controls=None,
    )
    config = MFFConfig(
        kpi=KPIConfig(
            name="Sales",
            dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
        ),
        media_channels=[
            MediaChannelConfig(
                name="TV",
                dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
            ),
            MediaChannelConfig(
                name="Digital",
                dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
            ),
        ],
    )
    return PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": dig}, index=idx),
        X_controls=None,
        coords=coords,
        index=idx,
        config=config,
    )


@pytest.fixture
def model_config():
    return ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=40,
        n_tune=40,
        target_accept=0.85,
    )


def _parametric_config():
    return ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=40,
        n_tune=40,
        use_parametric_adstock=True,
    )


def _estimand_node(model, channel, estimand):
    prefix = f"experiment_{channel}_{estimand}_"
    names = [
        k
        for k in model.named_vars
        if k.startswith(prefix) and k.endswith("_model_estimand")
    ]
    assert names, f"no estimand deterministic for {channel}/{estimand}"
    return names[0]


# =============================================================================
# Pure helpers
# =============================================================================


class TestLognormalHelper:
    def test_moment_match(self):
        value, se = 2.5, 0.4
        sigma = lognormal_sigma_from_moments(value, se)
        assert sigma == pytest.approx(math.sqrt(math.log1p((se / value) ** 2)))

    def test_cv_recovered(self):
        # sigma_log reproduces the natural-scale CV = sqrt(exp(sigma**2)-1).
        sigma = lognormal_sigma_from_moments(10.0, 2.0)
        cv = math.sqrt(math.exp(sigma**2) - 1)
        assert cv == pytest.approx(0.2, rel=1e-9)

    @pytest.mark.parametrize("value,se", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)])
    def test_rejects_nonpositive(self, value, se):
        with pytest.raises(ValueError):
            lognormal_sigma_from_moments(value, se)


# =============================================================================
# Dataclass validation / serialization
# =============================================================================


class TestExperimentMeasurement:
    def test_defaults_to_contribution(self):
        m = ExperimentMeasurement("TV", PERIOD, value=1e5, se=1e4)
        assert m.estimand is ExperimentEstimand.CONTRIBUTION

    def test_mroas_requires_spend_lift(self):
        with pytest.raises(ValueError, match="spend_lift_pct"):
            ExperimentMeasurement(
                "TV", PERIOD, value=2.0, se=0.5, estimand=ExperimentEstimand.MROAS
            )

    def test_mroas_with_spend_lift_ok(self):
        m = ExperimentMeasurement(
            "TV",
            PERIOD,
            value=2.0,
            se=0.5,
            estimand=ExperimentEstimand.MROAS,
            spend_lift_pct=10.0,
        )
        assert m.spend_lift_pct == 10.0

    @pytest.mark.parametrize("se", [0.0, -1.0, float("nan")])
    def test_rejects_bad_se(self, se):
        with pytest.raises(ValueError, match="se"):
            ExperimentMeasurement("TV", PERIOD, value=1.0, se=se)

    def test_lognormal_requires_positive_value(self):
        with pytest.raises(ValueError, match="lognormal"):
            ExperimentMeasurement(
                "TV", PERIOD, value=-1.0, se=0.5, distribution="lognormal"
            )

    def test_rejects_bad_distribution(self):
        with pytest.raises(ValueError, match="distribution"):
            ExperimentMeasurement(
                "TV", PERIOD, value=1.0, se=0.5, distribution="cauchy"
            )

    def test_rejects_nonpositive_spend_override(self):
        with pytest.raises(ValueError, match="spend"):
            ExperimentMeasurement("TV", PERIOD, value=1.0, se=0.5, spend=0.0)

    def test_mroas_rejects_spend_override(self):
        # The mROAS denominator is derived from observed spend x lift, so an
        # independent spend override would desynchronise it from the perturbation.
        with pytest.raises(ValueError, match="not supported for MROAS"):
            ExperimentMeasurement(
                "TV",
                PERIOD,
                value=2.0,
                se=0.5,
                estimand=ExperimentEstimand.MROAS,
                spend_lift_pct=10.0,
                spend=5000.0,
            )

    def test_estimand_accepts_str(self):
        m = ExperimentMeasurement("TV", PERIOD, value=2.0, se=0.5, estimand="roas")
        assert m.estimand is ExperimentEstimand.ROAS

    def test_roundtrip_to_from_dict(self):
        m = ExperimentMeasurement(
            "TV",
            PERIOD,
            value=2.0,
            se=0.5,
            estimand=ExperimentEstimand.MROAS,
            spend_lift_pct=10.0,
            holdout_regions=["A"],
            distribution="lognormal",
        )
        m2 = ExperimentMeasurement.from_dict(m.to_dict())
        assert m2 == m

    def test_from_lift_test_bridge(self):
        lt = LiftTestResult("TV", PERIOD, measured_lift=1.2e5, lift_se=2e4)
        m = ExperimentMeasurement.from_lift_test(lt)
        assert m.estimand is ExperimentEstimand.CONTRIBUTION
        assert m.value == pytest.approx(1.2e5)
        assert m.se == pytest.approx(2e4)

    def test_node_name_uses_explicit_name(self):
        m = ExperimentMeasurement("TV", PERIOD, value=1.0, se=0.5, name="geo_lift_q1")
        assert m.default_node_name(3) == "geo_lift_q1"


# =============================================================================
# Graph wiring (no MCMC)
# =============================================================================


class TestGraphWiring:
    def test_experiments_none_is_byte_identical(self, periods, model_config):
        base = BayesianMMM(_make_panel(periods), model_config, TrendConfig())
        with_none = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=None
        )
        assert set(with_none.model.named_vars) == set(base.model.named_vars)

    def test_each_estimand_adds_observed_node(self, periods, model_config):
        exps = [
            ExperimentMeasurement(
                "TV",
                PERIOD,
                value=5e4,
                se=8e3,
                estimand=ExperimentEstimand.CONTRIBUTION,
            ),
            ExperimentMeasurement(
                "TV", PERIOD, value=2.5, se=0.4, estimand=ExperimentEstimand.ROAS
            ),
            ExperimentMeasurement(
                "TV",
                PERIOD,
                value=1.8,
                se=0.5,
                estimand=ExperimentEstimand.MROAS,
                spend_lift_pct=10.0,
            ),
        ]
        model = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=exps
        ).model
        observed = {rv.name for rv in model.observed_RVs}
        for est in ("contribution", "roas", "mroas"):
            assert any(o.startswith(f"experiment_TV_{est}_") for o in observed)

    def test_add_experiment_calibration_rebuilds(self, periods, model_config):
        m = BayesianMMM(_make_panel(periods), model_config, TrendConfig())
        n_before = len(m.model.observed_RVs)
        m.add_experiment_calibration(
            [ExperimentMeasurement("TV", PERIOD, value=5e4, se=8e3)]
        )
        assert len(m.model.observed_RVs) == n_before + 1

    def test_unknown_channel_skipped_with_warning(self, periods, model_config):
        exps = [ExperimentMeasurement("Radio", PERIOD, value=1.0, se=0.5)]
        m = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=exps
        )
        with pytest.warns(UserWarning, match="unknown channel"):
            model = m.model
        assert not any(rv.name.startswith("experiment_") for rv in model.observed_RVs)

    def test_empty_window_skipped_with_warning(self, periods, model_config):
        # An out-of-range integer window selects no observations.
        exps = [ExperimentMeasurement("TV", (1000, 2000), 1e4, 1e3)]
        m = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=exps
        )
        with pytest.warns(UserWarning, match="no observations"):
            m.model


# =============================================================================
# In-graph estimand correctness (oracle vs NumPy)
# =============================================================================


@pytest.mark.parametrize("parametric", [False, True])
class TestEstimandOracle:
    def _build(self, periods, parametric, exps):
        if parametric:
            cfg = _parametric_config()
            panel = _make_panel(periods, tv_adstock=AdstockConfig.geometric())
        else:
            cfg = ModelConfig(
                inference_method=InferenceMethod.BAYESIAN_PYMC,
                n_chains=1,
                n_draws=40,
                n_tune=40,
            )
            panel = _make_panel(periods)
        return BayesianMMM(
            panel, cfg, TrendConfig(type=TrendType.LINEAR), experiments=exps
        )

    def test_contribution_equals_channel_contributions_sum(self, periods, parametric):
        exps = [ExperimentMeasurement("TV", PERIOD, value=5e4, se=8e3)]
        m = self._build(periods, parametric, exps)
        model = m.model
        start, end = m._period_to_indices(PERIOD)
        mask = (m.time_idx >= start) & (m.time_idx <= end)
        tv = m.channel_names.index("TV")
        node = _estimand_node(model, "TV", "contribution")
        with model:
            cc, est = pm.draw(
                [model["channel_contributions"], model[node]], draws=6, random_seed=1
            )
        expected = cc[:, mask, tv].sum(axis=1) * m.y_std
        np.testing.assert_allclose(est, expected, rtol=1e-5, atol=1e-3)

    def test_roas_equals_contribution_over_spend(self, periods, parametric):
        exps = [
            ExperimentMeasurement(
                "TV", PERIOD, value=2.5, se=0.4, estimand=ExperimentEstimand.ROAS
            )
        ]
        m = self._build(periods, parametric, exps)
        model = m.model
        start, end = m._period_to_indices(PERIOD)
        mask = (m.time_idx >= start) & (m.time_idx <= end)
        tv = m.channel_names.index("TV")
        spend = float(m.X_media_raw[mask, tv].sum())
        node = _estimand_node(model, "TV", "roas")
        with model:
            cc, est = pm.draw(
                [model["channel_contributions"], model[node]], draws=6, random_seed=2
            )
        expected = (cc[:, mask, tv].sum(axis=1) * m.y_std) / spend
        np.testing.assert_allclose(est, expected, rtol=1e-5, atol=1e-6)

    def test_mroas_matches_numpy_readstock(self, periods, parametric):
        lift = 0.10
        exps = [
            ExperimentMeasurement(
                "TV",
                PERIOD,
                value=2.0,
                se=0.5,
                estimand=ExperimentEstimand.MROAS,
                spend_lift_pct=lift * 100,
            )
        ]
        m = self._build(periods, parametric, exps)
        model = m.model
        start, end = m._period_to_indices(PERIOD)
        mask = (m.time_idx >= start) & (m.time_idx <= end)
        tv = m.channel_names.index("TV")
        spend = float(m.X_media_raw[mask, tv].sum())
        node = _estimand_node(model, "TV", "mroas")

        if parametric:
            cfg = m._get_adstock_config("TV")
            xin = m._prepare_raw_media_for_model()[:, tv]
            xin_p = xin.copy()
            xin_p[mask] *= 1 + lift
            with model:
                b, s, alpha, est = pm.draw(
                    [
                        model["beta_TV"],
                        model["sat_lam_TV"],
                        model["adstock_alpha_TV"],
                        model[node],
                    ],
                    draws=6,
                    random_seed=3,
                )
            for i in range(len(est)):
                w = adstock_weights(
                    "geometric", cfg.l_max, alpha=alpha[i], normalize=cfg.normalize
                )
                a_obs = apply_adstock(xin, w)
                a_pert = apply_adstock(xin_p, w)
                c_obs = (b[i] * (1 - np.exp(np.clip(-s[i] * a_obs, -20, 0))))[
                    mask
                ].sum()
                c_pert = (b[i] * (1 - np.exp(np.clip(-s[i] * a_pert, -20, 0))))[
                    mask
                ].sum()
                expect = ((c_pert - c_obs) * m.y_std) / (lift * spend)
                assert est[i] == pytest.approx(expect, rel=1e-4, abs=1e-5)
        else:
            xlow, xhigh = m._prepare_media_data_for_model()
            xlow_tv, xhigh_tv = xlow[:, tv], xhigh[:, tv]
            xp = m.X_media_raw.copy()
            xp[mask, tv] *= 1 + lift
            maxv = m._media_max["TV"] + 1e-8
            pl = geometric_adstock_2d(xp, m.adstock_alphas[0])[:, tv] / maxv
            ph = geometric_adstock_2d(xp, m.adstock_alphas[-1])[:, tv] / maxv
            with model:
                b, s, mix, est = pm.draw(
                    [
                        model["beta_TV"],
                        model["sat_lam_TV"],
                        model["adstock_TV"],
                        model[node],
                    ],
                    draws=6,
                    random_seed=3,
                )
            for i in range(len(est)):
                a_obs = (1 - mix[i]) * xlow_tv + mix[i] * xhigh_tv
                a_pert = (1 - mix[i]) * pl + mix[i] * ph
                c_obs = (b[i] * (1 - np.exp(np.clip(-s[i] * a_obs, -20, 0))))[
                    mask
                ].sum()
                c_pert = (b[i] * (1 - np.exp(np.clip(-s[i] * a_pert, -20, 0))))[
                    mask
                ].sum()
                expect = ((c_pert - c_obs) * m.y_std) / (lift * spend)
                assert est[i] == pytest.approx(expect, rel=1e-4, abs=1e-5)

    def test_spend_override_changes_roas_denominator(self, periods, parametric):
        override = 1234.0
        exps = [
            ExperimentMeasurement(
                "TV",
                PERIOD,
                value=2.5,
                se=0.4,
                estimand=ExperimentEstimand.ROAS,
                spend=override,
            )
        ]
        m = self._build(periods, parametric, exps)
        model = m.model
        start, end = m._period_to_indices(PERIOD)
        mask = (m.time_idx >= start) & (m.time_idx <= end)
        tv = m.channel_names.index("TV")
        node = _estimand_node(model, "TV", "roas")
        with model:
            cc, est = pm.draw(
                [model["channel_contributions"], model[node]], draws=4, random_seed=4
            )
        expected = (cc[:, mask, tv].sum(axis=1) * m.y_std) / override
        np.testing.assert_allclose(est, expected, rtol=1e-5, atol=1e-6)


# =============================================================================
# lognormal likelihood node
# =============================================================================


class TestLognormalNode:
    def test_lognormal_observed_is_log_value(self, periods, model_config):
        exps = [
            ExperimentMeasurement(
                "TV",
                PERIOD,
                value=2.5,
                se=0.4,
                estimand=ExperimentEstimand.ROAS,
                distribution="lognormal",
            )
        ]
        model = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=exps
        ).model
        node = next(
            rv for rv in model.observed_RVs if rv.name.startswith("experiment_TV_roas")
        )
        observed = model.rvs_to_values[node].eval()
        assert float(observed) == pytest.approx(np.log(2.5))


# =============================================================================
# Geo holdout masking
# =============================================================================


class TestGeoHoldout:
    def test_holdout_restricts_mask_to_geo(self, periods, model_config):
        exp = ExperimentMeasurement(
            "TV", PERIOD, value=2e4, se=3e3, holdout_regions=["A"]
        )
        m = BayesianMMM(
            _make_geo_panel(periods), model_config, TrendConfig(), experiments=[exp]
        )
        m.model  # trigger build
        mask = m._experiment_obs_mask(exp)
        geo_a = m.geo_names.index("A")
        assert mask is not None
        assert np.all(m.geo_idx[mask] == geo_a)
        assert mask.sum() > 0

    def test_unknown_geo_skipped(self, periods, model_config):
        exp = ExperimentMeasurement(
            "TV", PERIOD, value=2e4, se=3e3, holdout_regions=["Z"]
        )
        m = BayesianMMM(
            _make_geo_panel(periods), model_config, TrendConfig(), experiments=[exp]
        )
        with pytest.warns(UserWarning, match="unknown holdout_regions"):
            model = m.model
        assert not any(rv.name.startswith("experiment_") for rv in model.observed_RVs)

    def test_holdout_on_nongeo_model_skipped(self, periods, model_config):
        # A geo-restricted measurement on a non-geo model would be fit against the
        # national contribution, so it is skipped rather than silently mis-scaled.
        exp = ExperimentMeasurement(
            "TV", PERIOD, value=2e4, se=3e3, holdout_regions=["A"]
        )
        m = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=[exp]
        )
        with pytest.warns(UserWarning, match="no geo dimension"):
            model = m.model
        assert not any(rv.name.startswith("experiment_") for rv in model.observed_RVs)


# =============================================================================
# Period-window parsing (off-by-one and out-of-range guards)
# =============================================================================


class TestPeriodParsing:
    def test_first_period_window_resolves_to_index_zero(self, periods, model_config):
        # A window on the very first period must resolve to (0, 0), not skip.
        m = BayesianMMM(_make_panel(periods), model_config, TrendConfig())
        first = periods[0]
        assert m._period_to_indices((first, first)) == (0, 0)

    def test_full_range_window(self, periods, model_config):
        m = BayesianMMM(_make_panel(periods), model_config, TrendConfig())
        assert m._period_to_indices((periods[0], periods[-1])) == (0, len(periods) - 1)

    def test_out_of_range_before_returns_none(self, periods, model_config):
        m = BayesianMMM(_make_panel(periods), model_config, TrendConfig())
        assert m._period_to_indices(("2018-01-01", "2018-12-31")) is None

    def test_out_of_range_after_returns_none(self, periods, model_config):
        m = BayesianMMM(_make_panel(periods), model_config, TrendConfig())
        assert m._period_to_indices(("2030-01-01", "2030-12-31")) is None

    def test_reversed_window_returns_none(self, periods, model_config):
        m = BayesianMMM(_make_panel(periods), model_config, TrendConfig())
        assert m._period_to_indices((periods[20], periods[5])) is None

    def test_out_of_range_experiment_skipped_with_warning(self, periods, model_config):
        exps = [ExperimentMeasurement("TV", ("2018-01-01", "2018-12-31"), 1e4, 1e3)]
        m = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=exps
        )
        with pytest.warns(UserWarning, match="outside the panel"):
            model = m.model
        assert not any(rv.name.startswith("experiment_") for rv in model.observed_RVs)

    def test_first_period_experiment_is_added(self, periods, model_config):
        # The off-by-one previously skipped an experiment on the first period.
        first = periods[0]
        exps = [ExperimentMeasurement("TV", (first, first), 1e4, 1e3)]
        m = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=exps
        )
        model = m.model
        assert any(rv.name.startswith("experiment_TV") for rv in model.observed_RVs)

    def test_colliding_explicit_names_get_distinct_nodes(self, periods, model_config):
        # Two experiments with the same explicit name must not clobber each
        # other's likelihood node or its companion estimand Deterministic.
        exps = [
            ExperimentMeasurement("TV", PERIOD, value=2e4, se=3e3, name="geo_lift"),
            ExperimentMeasurement(
                "Digital", PERIOD, value=1e4, se=2e3, name="geo_lift"
            ),
        ]
        m = BayesianMMM(
            _make_panel(periods), model_config, TrendConfig(), experiments=exps
        )
        model = m.model  # must not raise on duplicate variable names
        observed = [rv.name for rv in model.observed_RVs if rv.name != "y_obs"]
        assert len(observed) == 2
        assert len(set(observed)) == 2


# =============================================================================
# Reusable attach helper (model-agnostic)
# =============================================================================


class TestAttachHelper:
    def test_attach_normal_node(self):
        meas = ExperimentMeasurement("TV", PERIOD, value=3.0, se=0.5)
        with pm.Model():
            theta = pm.Normal("theta", 0, 1)
            attach_experiment_likelihood("exp", theta * 2 + 1, meas)
            model = pm.modelcontext(None)
        assert "exp" in model.named_vars
        assert "exp_model_estimand" in model.named_vars


# =============================================================================
# End-to-end (slow): fit, predict, serialize
# =============================================================================


@pytest.mark.slow
class TestEndToEnd:
    def test_experiment_moves_posterior_toward_measured_value(self, periods):
        # The feature's contract: a tight-SE experiment whose value sits far from
        # the uncalibrated estimate must pull the model-implied estimand toward it.
        cfg = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=2,
            n_draws=150,
            n_tune=150,
        )
        trend = TrendConfig(type=TrendType.LINEAR)
        window = (str(periods[0].date()), str(periods[-1].date()))

        base = BayesianMMM(_make_panel(periods), cfg, trend)
        base.fit(draws=150, tune=150, chains=2, random_seed=0, progressbar=False)
        tv = base.channel_names.index("TV")
        cc = base._trace.posterior["channel_contributions"].isel(channel=tv).values
        base_contrib = float(cc.sum(axis=-1).mean()) * base.y_std
        assert base_contrib > 0

        # Experiment asserts TV contributed ~3x what the base model thinks,
        # measured very precisely -> the likelihood should dominate.
        target = 3.0 * base_contrib
        exp = ExperimentMeasurement(
            "TV",
            window,
            value=target,
            se=0.02 * target,
            estimand=ExperimentEstimand.CONTRIBUTION,
        )
        cal = BayesianMMM(_make_panel(periods), cfg, trend, experiments=[exp])
        cal.fit(draws=150, tune=150, chains=2, random_seed=0, progressbar=False)
        node = _estimand_node(cal.model, "TV", "contribution")
        cal_contrib = float(cal._trace.posterior[node].values.mean())

        # Moved toward the experimental value and got closer to it than baseline.
        assert cal_contrib > base_contrib
        assert abs(cal_contrib - target) < abs(base_contrib - target)

    def test_fit_and_predict_with_experiment(self, periods):
        cfg = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=1,
            n_draws=40,
            n_tune=40,
        )
        exps = [
            ExperimentMeasurement(
                "TV", PERIOD, value=2.5, se=0.4, estimand=ExperimentEstimand.ROAS
            )
        ]
        m = BayesianMMM(
            _make_panel(periods),
            cfg,
            TrendConfig(type=TrendType.LINEAR),
            experiments=exps,
        )
        m.fit(draws=40, tune=40, chains=1, random_seed=1, progressbar=False)
        # The experiment estimand is recorded in the posterior.
        assert any("model_estimand" in v for v in m._trace.posterior.data_vars)
        # Prediction (and counterfactual prediction) still works.
        pred = m.predict(random_seed=1)
        assert pred.y_pred_mean.shape == (len(periods),)
        xcf = m.X_media_raw.copy()
        xcf[:, 0] *= 0.5
        pred2 = m.predict(X_media=xcf, random_seed=1)
        assert pred2.y_pred_mean.shape == (len(periods),)

    def test_serializer_roundtrips_experiments(self, periods, tmp_path):
        from mmm_framework.serialization import MMMSerializer

        cfg = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=1,
            n_draws=40,
            n_tune=40,
        )
        exps = [
            ExperimentMeasurement(
                "TV",
                PERIOD,
                value=2.0,
                se=0.5,
                estimand=ExperimentEstimand.MROAS,
                spend_lift_pct=10.0,
            )
        ]
        panel = _make_panel(periods)
        m = BayesianMMM(
            panel, cfg, TrendConfig(type=TrendType.LINEAR), experiments=exps
        )
        m.fit(draws=40, tune=40, chains=1, random_seed=1, progressbar=False)

        out = tmp_path / "model"
        MMMSerializer.save(m, str(out))
        loaded = MMMSerializer.load(str(out), _make_panel(periods))
        assert len(loaded.experiments) == 1
        assert loaded.experiments[0] == exps[0]
        # Rebuilt graph includes the experiment likelihood node.
        assert any(
            rv.name.startswith("experiment_TV_mroas")
            for rv in loaded.model.observed_RVs
        )
