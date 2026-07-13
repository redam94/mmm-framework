"""Multiplicative (semi-log + saturation) specification (#144).

``ModelConfig.specification`` used to be a silent no-op — the model always fit
the additive form. Now ``MULTIPLICATIVE`` builds a genuine multiplicative model:
``log(y) = a + sum_c beta_c * sat_c(adstock(x_c)) + ...``, i.e. each channel is
the SAME saturation curve as the additive form but acts as a multiplicative
lift on sales (``exp(beta_c)`` at full saturation). Contract pinned here:

* the ADDITIVE default is unchanged (no ``log_lift_*`` RVs, ``beta_*`` free);
* MULTIPLICATIVE keeps the saturation RVs, samples the interpretable
  ``log_lift_<ch>`` (max log-lift) and derives ``beta_<ch>`` + ``max_pct_lift_<ch>``;
* a strictly-positive KPI is required (log link); zero spend is fine (sat(0)=0);
* it recovers a planted max % lift on a synthetic world;
* the component decomposition is EXACT (LMDI) — it sums to the fitted y;
* surfaces that are only correct on the additive scale stay guarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework import (
    BayesianMMM,
    ModelConfig,
    ModelConfigBuilder,
    ModelSpecification,
    TrendConfig,
    TrendType,
)
from mmm_framework.config import (
    AdstockConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    SaturationConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset

CHANNELS = ["TV", "Search", "Social"]


def _sat(x: np.ndarray) -> np.ndarray:
    xn = x / x.max()
    return 1.0 - np.exp(-3.0 * xn)  # logistic saturation, ~0.95 at max spend


def _semilog_panel(
    log_lifts: dict[str, float], *, n: int = 140, seed: int = 1, noise: float = 0.03
) -> PanelDataset:
    """Synthetic semi-log + saturation world: log(y) = a + Σ log_lift·sat(x)."""
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2021-01-04", periods=n, freq="W-MON")
    # spend spanning low->high so the saturation curve is traced to near-full
    spend = {c: np.clip(rng.gamma(2.0, 0.4, n), 0, None) for c in CHANNELS}
    spend["Search"][::9] = 0.0  # flighted channel: zero-spend weeks are fine
    logy = 6.0 + sum(log_lifts[c] * _sat(spend[c]) for c in CHANNELS)
    y = np.exp(logy + rng.normal(0, noise, n))
    coords = PanelCoordinates(
        periods=periods, geographies=None, products=None, channels=CHANNELS, controls=[]
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(
                name=c,
                dimensions=[DimensionType.PERIOD],
                adstock=AdstockConfig.none(),
                saturation=SaturationConfig.logistic(),
            )
            for c in CHANNELS
        ],
        controls=[],
    )
    return PanelDataset(
        y=pd.Series(y, name="Sales"),
        X_media=pd.DataFrame(spend),
        X_controls=None,
        coords=coords,
        index=periods,
        config=config,
    )


def _mmm(spec: ModelSpecification) -> BayesianMMM:
    return BayesianMMM(
        _semilog_panel({c: 0.3 for c in CHANNELS}),
        ModelConfig(specification=spec),
        TrendConfig(type=TrendType.NONE),
    )


def _free(m: BayesianMMM) -> set[str]:
    return {v.name for v in m.model.free_RVs}


def test_default_specification_is_additive():
    assert ModelConfig().specification == ModelSpecification.ADDITIVE
    assert ModelConfigBuilder().build().specification == ModelSpecification.ADDITIVE


def test_builder_multiplicative_sets_specification():
    cfg = ModelConfigBuilder().multiplicative().build()
    assert cfg.specification == ModelSpecification.MULTIPLICATIVE


def test_additive_has_no_lift_rvs():
    m = _mmm(ModelSpecification.ADDITIVE)
    assert not m._multiplicative
    free = _free(m)
    assert not any(n.startswith("log_lift_") for n in free)
    assert "sat_lam_TV" in free  # additive keeps its saturation RVs


def test_multiplicative_keeps_saturation_and_adds_lift():
    m = _mmm(ModelSpecification.MULTIPLICATIVE)
    assert m._multiplicative
    free = _free(m)
    named = set(m.model.named_vars)
    for c in CHANNELS:
        assert f"log_lift_{c}" in free  # interpretable max log-lift is the free RV
        assert f"sat_lam_{c}" in free  # saturation is KEPT (unlike pure log-log)
        assert f"beta_{c}" in named and f"beta_{c}" not in free  # derived Deterministic
        assert f"max_pct_lift_{c}" in named  # interpretable % lift Deterministic


def test_multiplicative_requires_positive_kpi():
    panel = _semilog_panel({c: 0.3 for c in CHANNELS})
    panel.y.iloc[0] = -1.0
    with pytest.raises(ValueError, match="strictly positive KPI"):
        BayesianMMM(panel, ModelConfig(specification=ModelSpecification.MULTIPLICATIVE))


def test_multiplicative_allows_zero_spend():
    """sat(0)=0 gives a channel no lift, so flighted (zero-spend) weeks are fine."""
    panel = _semilog_panel({c: 0.3 for c in CHANNELS})
    assert (panel.X_media["Search"].to_numpy() == 0).any()  # DGP has zeros
    m = BayesianMMM(panel, ModelConfig(specification=ModelSpecification.MULTIPLICATIVE))
    assert m.model is not None  # builds without error


def test_extension_models_reject_multiplicative():
    from mmm_framework.mmm_extensions.config import (
        MediatorConfig,
        MediatorType,
        NestedModelConfig,
    )
    from mmm_framework.mmm_extensions.models.nested import NestedMMM

    rng = np.random.default_rng(0)
    media = np.abs(rng.normal(100, 20, (60, 2)))
    y = 1000 + 2 * media[:, 0] + rng.normal(0, 40, 60)
    idx = pd.date_range("2022-01-03", periods=60, freq="W-MON")
    cfg = NestedModelConfig(
        mediators=(
            MediatorConfig(name="Awareness", mediator_type=MediatorType.FULLY_LATENT),
        )
    )
    with pytest.raises(NotImplementedError, match="[Mm]ultiplicative"):
        NestedMMM(
            media,
            y,
            ["TV", "Digital"],
            cfg,
            index=idx,
            model_config=ModelConfig(specification=ModelSpecification.MULTIPLICATIVE),
        )


@pytest.mark.slow
class TestMultiplicativeFit:
    def _fit(self, log_lifts):
        panel = _semilog_panel(log_lifts, seed=3)
        cfg = (
            ModelConfigBuilder()
            .multiplicative()
            .bayesian_numpyro()
            .with_chains(4)
            .with_draws(600)
            .with_tune(600)
            .build()
        )
        mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.NONE))
        return mmm, mmm.fit(random_seed=0)

    def test_recovers_max_pct_lift(self):
        truth = {"TV": 0.40, "Search": 0.15, "Social": 0.28}
        mmm, results = self._fit(truth)
        post = results.trace.posterior
        for c, ll in truth.items():
            true_pct = np.exp(ll) - 1.0
            est_pct = float(post[f"max_pct_lift_{c}"].mean())
            assert abs(est_pct - true_pct) < 0.15, (c, est_pct, true_pct)
        est = {c: float(post[f"max_pct_lift_{c}"].mean()) for c in truth}
        assert est["TV"] > est["Social"] > est["Search"]  # ordering recovered

    def test_predict_is_positive_original_scale(self):
        mmm, _ = self._fit({"TV": 0.40, "Search": 0.15, "Social": 0.28})
        pred = mmm.predict(return_original_scale=True)
        assert np.all(pred.y_pred_mean > 0)
        assert np.isfinite(pred.y_pred_samples).all()

    def test_decomposition_is_exact_lmdi(self):
        """The LMDI waterfall sums to the model's fitted (exp-of-log) prediction."""
        mmm, results = self._fit({"TV": 0.40, "Search": 0.15, "Social": 0.28})
        decomp = mmm.compute_component_decomposition()

        # media split is exact within the media block
        mbc = decomp.media_by_channel.to_numpy().sum(axis=1)
        np.testing.assert_allclose(mbc, decomp.media_total, rtol=1e-9)

        # full waterfall sums to exp(sum of the log-scale component means)
        post = results.trace.posterior

        def cm(v):
            return (
                post[v].mean(dim=["chain", "draw"]).values
                if v in post
                else np.zeros(mmm.n_obs)
            )

        log_pred = mmm.y_mean + mmm.y_std * (
            cm("intercept_component")
            + cm("trend_component")
            + cm("seasonality_component")
            + cm("controls_total")
            + cm("media_total")
        )
        expected = np.exp(log_pred)
        total = (
            decomp.intercept
            + decomp.trend
            + decomp.seasonality
            + decomp.controls_total
            + decomp.media_total
        )
        np.testing.assert_allclose(total, expected, rtol=1e-8)
        # a finite, positive baseline exists (the whole point of saturation)
        assert np.all(decomp.intercept > 0)

    def test_additive_only_surfaces_are_guarded(self):
        mmm, _ = self._fit({"TV": 0.40, "Search": 0.15, "Social": 0.28})
        with pytest.raises(NotImplementedError, match="multiplicative"):
            mmm.compute_marginal_contributions()

    def test_counterfactual_contributions_work(self):
        mmm, _ = self._fit({"TV": 0.40, "Search": 0.15, "Social": 0.28})
        contrib = mmm.compute_counterfactual_contributions(random_seed=1)
        assert set(contrib.total_contributions.index) == set(CHANNELS)
        assert np.isfinite(contrib.total_contributions.to_numpy()).all()
