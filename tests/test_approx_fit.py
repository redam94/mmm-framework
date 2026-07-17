"""Tests for approximate fit methods (MAP / Laplace / ADVI / Pathfinder) and
the exact SMC method.

The approximate methods exist to fit a model in seconds for quick checking —
the test contract is that they return a drop-in :class:`MMMResults` whose
posterior works everywhere the NUTS trace does (summary, ArviZ, ``predict``),
and that they are clearly flagged as approximate. SMC is the anti-contract:
an exact sampler whose results must NOT be flagged approximate, with real
R-hat/ESS across its independent runs and a log marginal likelihood.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    ControlVariableConfig,
    DimensionType,
    FitMethod,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM
from mmm_framework.model.results import MMMResults
from mmm_framework.model.trend_config import TrendConfig, TrendType


@pytest.fixture
def simple_panel():
    periods = pd.date_range("2020-01-06", periods=52, freq="W-MON")
    n = len(periods)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=["Price"],
    )
    rng = np.random.default_rng(42)
    y = pd.Series(1000 + rng.standard_normal(n) * 100, name="Sales")
    X_media = pd.DataFrame(
        {
            "TV": np.abs(rng.standard_normal(n) * 50 + 100),
            "Digital": np.abs(rng.standard_normal(n) * 30 + 80),
        }
    )
    X_controls = pd.DataFrame({"Price": 10 + rng.standard_normal(n) * 0.5})
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    return PanelDataset(
        y=y,
        X_media=X_media,
        X_controls=X_controls,
        coords=coords,
        index=periods,
        config=cfg,
    )


@pytest.fixture
def model_config():
    return ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=100,
        n_tune=100,
        target_accept=0.8,
    )


@pytest.fixture
def trend_config():
    return TrendConfig(type=TrendType.LINEAR)


def _assert_usable_posterior(results: MMMResults, mmm: BayesianMMM, expected_draws):
    """The approximate posterior must be a drop-in for the NUTS trace."""
    assert isinstance(results, MMMResults)
    assert results.approximate is True
    post = results.trace.posterior
    assert post.sizes["chain"] == 1
    assert post.sizes["draw"] == expected_draws
    # Core parameters AND deterministics are present (so reporting/predict work).
    assert "beta_TV" in post
    # R-hat / ESS are undefined for a single-path approximation.
    assert results.diagnostics["approximate"] is True
    assert results.diagnostics["rhat_max"] is None
    assert results.diagnostics["ess_bulk_min"] is None
    # summary() and predict() must not raise.
    results.summary()
    pred = mmm.predict()
    assert np.shape(pred.y_pred_mean) == (mmm.n_obs,)


def test_map_fit(simple_panel, model_config, trend_config):
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    results = mmm.fit(method="map", random_seed=42)
    assert results.diagnostics["fit_method"] == "map"
    _assert_usable_posterior(results, mmm, expected_draws=1)


def test_advi_fit(simple_panel, model_config, trend_config):
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    # `n` (VI iterations) is kept small for test speed.
    results = mmm.fit(method="advi", draws=50, random_seed=42, n=2000)
    assert results.diagnostics["fit_method"] == "advi"
    assert results.diagnostics.get("elbo") is not None
    _assert_usable_posterior(results, mmm, expected_draws=50)


def test_fullrank_advi_fit(simple_panel, model_config, trend_config):
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    results = mmm.fit(method="fullrank_advi", draws=50, random_seed=42, n=2000)
    assert results.diagnostics["fit_method"] == "fullrank_advi"
    _assert_usable_posterior(results, mmm, expected_draws=50)


def test_fit_method_accepts_enum(simple_panel, model_config, trend_config):
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    results = mmm.fit(method=FitMethod.MAP, random_seed=42)
    assert results.approximate is True


def test_config_fit_method_default(simple_panel, trend_config):
    """`fit()` with no method falls back to model_config.fit_method."""
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=1,
        n_tune=1,
        fit_method=FitMethod.MAP,
    )
    mmm = BayesianMMM(simple_panel, cfg, trend_config)
    results = mmm.fit(random_seed=42)
    assert results.approximate is True
    assert results.diagnostics["fit_method"] == "map"


def test_pathfinder_fit(simple_panel, model_config, trend_config):
    # pymc-extras is a declared dependency (pyproject: pymc-extras>=0.6,<0.7),
    # so pathfinder works out of the box — no importorskip.
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    results = mmm.fit(method="pathfinder", draws=50, random_seed=42)
    assert results.diagnostics["fit_method"] == "pathfinder"
    _assert_usable_posterior(results, mmm, expected_draws=50)


def test_no_parameter_independent_deterministics(simple_panel):
    """Pathfinder's trace conversion (pymc_extras) cannot batch a Deterministic
    that depends on no free RV — vectorize_graph gives it no chain/draw dims
    and az.from_dict rejects the shape. The graph anchors would-be constants
    (absent structure components, observed-data y_obs_scaled) on a free RV
    with a zero-weight term (`_anchored_det`). Pin the invariant on the
    maximal-constants config: national panel, trend none, seasonality off.
    Observed RVs are blockers — the conversion substitutes their data."""
    try:
        from pytensor.graph.traversal import ancestors
    except ImportError:  # older pytensor
        from pytensor.graph.basic import ancestors

    from mmm_framework.config import SeasonalityConfig

    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=10,
        n_tune=10,
        seasonality=SeasonalityConfig(yearly=None, monthly=None, weekly=None),
    )
    mmm = BayesianMMM(simple_panel, cfg, TrendConfig(type=TrendType.NONE))
    model = mmm.model
    free = set(model.free_RVs)
    observed = list(model.observed_RVs)
    constant = [
        d.name
        for d in model.deterministics
        if not any(a in free for a in ancestors([d], blockers=observed))
    ]
    assert constant == [], f"parameter-independent deterministics: {constant}"


def test_laplace_fit(simple_panel, model_config, trend_config):
    # pymc-extras is a declared dependency, so Laplace works out of the box.
    # MAP point + Gaussian curvature draws — still approximate, but a step up
    # from bare MAP for model checking.
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    results = mmm.fit(method="laplace", draws=50, random_seed=42)
    assert results.diagnostics["fit_method"] == "laplace"
    assert results.diagnostics.get("laplace") is True
    _assert_usable_posterior(results, mmm, expected_draws=50)


def test_laplace_missing_dep_message(
    monkeypatch, simple_panel, model_config, trend_config
):
    """When pymc_extras is absent, the error names the install path."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pymc_extras":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    with pytest.raises(ImportError, match="pymc_extras"):
        mmm.fit(method="laplace", random_seed=42)


class TestSMCFit:
    """SMC is an EXACT sampler — the anti-contract of the approximate tier."""

    def test_smc_fit_contract(self, simple_panel, model_config, trend_config):
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        with warnings.catch_warnings():
            # A deliberately small particle count -> expected low-ESS warnings.
            warnings.simplefilter("ignore")
            results = mmm.fit(
                method="smc",
                draws=120,
                chains=2,
                random_seed=42,
                cores=1,
                progressbar=False,
            )
        assert isinstance(results, MMMResults)
        # NOT approximate: SMC must never trip the uncalibrated-uncertainty
        # banners / report gating.
        assert results.approximate is False
        d = results.diagnostics
        assert d["fit_method"] == "smc"
        assert d["approximate"] is False
        # R-hat / ESS across the 2 independent SMC runs are real numbers
        # (disagreeing runs = the multimodality signal).
        assert d["rhat_max"] is not None and np.isfinite(d["rhat_max"])
        assert d["ess_bulk_min"] is not None and np.isfinite(d["ess_bulk_min"])
        # Divergences do not exist for SMC — None and never flagged.
        assert d["divergences"] is None
        assert "divergences" not in (d.get("flags") or [])
        # Log marginal likelihood (mean + per-run) recorded for Bayes factors.
        assert np.isfinite(d["log_marginal_likelihood"])
        assert len(d["log_marginal_likelihood_per_run"]) == 2
        # The posterior is a normal trace: params AND deterministics, all runs.
        post = results.trace.posterior
        assert post.sizes["chain"] == 2
        assert post.sizes["draw"] == 120
        assert "beta_TV" in post
        results.summary()
        pred = mmm.predict()
        assert np.shape(pred.y_pred_mean) == (mmm.n_obs,)

    def test_smc_enum_is_exact(self):
        assert FitMethod.SMC.is_approximate is False
        assert FitMethod.LAPLACE.is_approximate is True
        assert FitMethod.NUTS.is_approximate is False


def test_pathfinder_missing_dep_message(
    monkeypatch, simple_panel, model_config, trend_config
):
    """When pymc_extras is absent, the error names the install path."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pymc_extras":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    with pytest.raises(ImportError, match="pymc_extras"):
        mmm.fit(method="pathfinder", random_seed=42)


@pytest.mark.slow
def test_nuts_still_default(simple_panel, model_config, trend_config):
    """The default path is unchanged: full NUTS, not approximate."""
    mmm = BayesianMMM(simple_panel, model_config, trend_config)
    results = mmm.fit(draws=50, tune=50, chains=1, random_seed=42)
    assert results.approximate is False
    assert results.diagnostics["fit_method"] == "nuts"
    assert results.diagnostics["rhat_max"] is not None
