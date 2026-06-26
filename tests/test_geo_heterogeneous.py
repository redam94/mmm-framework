"""Per-geo channel effectiveness (V3, ``HierarchicalConfig.vary_media_by_geo``).

The pooled model has one shared ``beta`` per channel and only per-geo *intercept*
offsets, so when channels work differently in different markets (the
``make_geo_heterogeneous`` trap) its per-geo attribution is structurally stuck at
a spend-weighted average. V3 partial-pools a per-geo coefficient per channel, so
the per-geo effectiveness *ordering* becomes estimable. Validated with a real
(slow) NUTS fit recovering the planted multipliers.
"""

from __future__ import annotations

import numpy as np
import pytest


def _het_model(vary: bool, n_weeks: int = 130, *, draws: int = 400, tune: int = 800):
    from mmm_framework.config import HierarchicalConfig, InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
    from mmm_framework.synth import dgp_geo

    panel = dgp_geo.make_geo_heterogeneous(n_weeks=n_weeks).panel()
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_NUMPYRO,
        n_chains=2,
        n_draws=draws,
        n_tune=tune,
        use_parametric_adstock=True,
        hierarchical=HierarchicalConfig(vary_media_by_geo=vary),
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))


# ---------------------------------------------------------------------------
# fast: graph shape (no sampling)
# ---------------------------------------------------------------------------


def test_per_geo_beta_is_a_geo_vector_when_enabled():
    """With ``vary_media_by_geo`` on (and geo data), each channel's ``beta_<ch>``
    becomes a per-geo Deterministic plus its non-centered hyperparameters."""
    mmm = _het_model(vary=True)
    assert mmm.has_geo and mmm.n_geos == 4
    m = mmm.model  # builds lazily
    names = set(m.named_vars)
    for ch in mmm.channel_names:
        assert f"beta_{ch}" in names
        assert f"beta_{ch}_logmu" in names
        assert f"beta_{ch}_logtau" in names
        assert f"beta_{ch}_z" in names
        # the realized per-geo coefficient is a length-n_geos vector
        shape = tuple(m[f"beta_{ch}"].shape.eval())
        assert shape == (mmm.n_geos,), f"{ch}: {shape}"


def test_pooled_beta_stays_scalar_by_default():
    """Default (off): one shared scalar ``beta_<ch>`` per channel, no per-geo
    hyperparameters — i.e. byte-compatible with the historical graph."""
    mmm = _het_model(vary=False)
    m = mmm.model
    names = set(m.named_vars)
    for ch in mmm.channel_names:
        assert f"beta_{ch}" in names
        assert f"beta_{ch}_logmu" not in names
        shape = tuple(m[f"beta_{ch}"].shape.eval())
        assert shape == (), f"{ch} should be scalar, got {shape}"


def test_non_geo_model_ignores_the_flag():
    """A national (no-geo) dataset can't vary by geo; the flag is a silent no-op
    and the channel betas stay scalar."""
    from mmm_framework.config import HierarchicalConfig, InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM
    from mmm_framework.synth import dgp

    panel = dgp.build("clean", seed=3, n_weeks=80).panel()
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_NUMPYRO,
        hierarchical=HierarchicalConfig(vary_media_by_geo=True),
    )
    mmm = BayesianMMM(panel, cfg)
    assert not mmm.has_geo
    m = mmm.model
    for ch in mmm.channel_names:
        assert tuple(m[f"beta_{ch}"].shape.eval()) == ()


# ---------------------------------------------------------------------------
# slow: parameter recovery against the planted multipliers
# ---------------------------------------------------------------------------


def _beta_by_geo(mmm, channel: str) -> np.ndarray:
    """Posterior-mean per-geo coefficient vector, ordered by ``mmm.geo_names``."""
    arr = mmm._trace.posterior[f"beta_{channel}"].values  # (chain, draw, geo)
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0)


@pytest.mark.slow
def test_recovers_per_geo_effectiveness_ordering():
    """V3 recovers the per-geo effectiveness ordering the pooled model can't.

    The DGP scales each channel's shared response by a known per-geo multiplier
    (``effect_multipliers`` in the scenario notes). The fitted per-geo beta should
    track that multiplier across geos *within* a channel; the pooled model assigns
    every geo the same beta, so it has zero discriminating power by construction.
    """
    from scipy.stats import pearsonr

    from mmm_framework.synth import dgp_geo

    scen = dgp_geo.make_geo_heterogeneous()
    mults = scen.notes["effect_multipliers"]  # {geo: {channel: mult}}

    mmm = _het_model(vary=True)
    mmm.fit(random_seed=11, progressbar=False)

    geo_names = mmm.geo_names
    corrs = []
    for ch in mmm.channel_names:
        fitted = _beta_by_geo(mmm, ch)  # ordered by geo_names
        truth = np.array([mults[g][ch] for g in geo_names])
        if np.std(truth) > 0 and np.std(fitted) > 0:
            corrs.append(pearsonr(fitted, truth)[0])

    mean_corr = float(np.mean(corrs))
    assert mean_corr > 0.5, (
        f"per-geo betas should track the planted multipliers; "
        f"mean corr={mean_corr:.3f}, per-channel={dict(zip(mmm.channel_names, corrs))}"
    )

    # TV has the widest planted spread (North 1.8 ... West 0.3): its ordering
    # must come out right.
    tv = dict(zip(geo_names, _beta_by_geo(mmm, "TV")))
    assert tv["North"] > tv["West"], f"TV North vs West not recovered: {tv}"


@pytest.mark.slow
def test_geo_forecast_uses_per_geo_beta():
    """The out-of-time forecaster threads the per-geo coefficient through each
    cell's forward pass (so a V3 fit still reproduces the in-graph predict)."""
    from mmm_framework.validation.backtest import PosteriorForecaster

    mmm = _het_model(vary=True, n_weeks=104, draws=250, tune=400)
    mmm.fit(random_seed=5, progressbar=False)

    f = PosteriorForecaster(mmm)
    samples = f.forecast(
        mmm.X_media_raw,
        mmm.X_controls_raw,
        np.arange(mmm.n_obs),
        include_noise=False,
        random_seed=5,
    )
    mu_numpy = samples.mean(axis=0)
    mu_graph = mmm.predict(random_seed=5).y_pred_mean
    corr = np.corrcoef(mu_numpy, mu_graph)[0, 1]
    assert corr > 0.99, f"geo per-beta forward-pass mismatch: corr={corr:.4f}"
