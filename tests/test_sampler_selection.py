"""How ``fit()`` picks its NUTS backend and acceptance rate.

Three defects motivated this file (issues #169, #170, #171):

* ``ModelConfig.target_accept`` was written by the builder and never read at
  the only place it is consumed, so ``.with_target_accept(0.95)`` was a silent
  no-op.
* ``BayesianMMM.fit()`` had no ``nuts_sampler`` parameter, so passing the
  keyword that ``BaseExtendedMMM.fit()`` *does* accept fell into ``**kwargs``
  and collided with the explicit argument (``TypeError`` naming ``pm.sample``).
* ``nutpie`` was a declared core dependency that no code path could select.

The sampler is spied on rather than run: these assert what reaches
``pm.sample``, so no MCMC is needed.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from mmm_framework.builders.model import ModelConfigBuilder
from mmm_framework.config import (
    ControlVariableConfig,
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.mmm_extensions.models.base import BaseExtendedMMM
from mmm_framework.model import BayesianMMM
from mmm_framework.model.trend_config import TrendConfig, TrendType


@pytest.fixture
def panel() -> PanelDataset:
    periods = pd.date_range("2020-01-06", periods=52, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(42)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=["Price"],
    )
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
        y=pd.Series(1000 + rng.standard_normal(n) * 100, name="Sales"),
        X_media=pd.DataFrame(
            {
                "TV": np.abs(rng.standard_normal(n) * 50 + 100),
                "Digital": np.abs(rng.standard_normal(n) * 30 + 80),
            }
        ),
        X_controls=pd.DataFrame({"Price": 10 + rng.standard_normal(n) * 0.5}),
        coords=coords,
        index=periods,
        config=cfg,
    )


class _SampleReached(Exception):
    """Raised by the spy so the test stops before any real sampling."""


@pytest.fixture
def sample_spy(monkeypatch):
    """Capture the kwargs ``fit()`` passes to ``pm.sample`` and stop there."""
    calls: list[dict] = []

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        raise _SampleReached

    monkeypatch.setattr(pm, "sample", _spy)
    return calls


def _fit_kwargs(model: BayesianMMM, calls: list[dict], **fit_kwargs) -> dict:
    with pytest.raises(_SampleReached):
        model.fit(draws=10, tune=10, chains=1, **fit_kwargs)
    assert len(calls) == 1, f"pm.sample called {len(calls)} times, expected 1"
    return calls[0]


def _model(panel: PanelDataset, config: ModelConfig) -> BayesianMMM:
    return BayesianMMM(panel, config, TrendConfig(type=TrendType.LINEAR))


# ---------------------------------------------------------------------------
# #169 — ModelConfig.target_accept must reach the sampler
# ---------------------------------------------------------------------------


class TestTargetAccept:
    def test_config_value_reaches_the_sampler(self, panel, sample_spy):
        """`.with_target_accept(0.95)` is honored without repeating it at fit()."""
        config = (
            ModelConfigBuilder()
            .bayesian_pymc()
            .with_target_accept(0.95)
            .with_chains(1)
            .build()
        )
        assert _fit_kwargs(_model(panel, config), sample_spy)["target_accept"] == 0.95

    def test_explicit_argument_wins_over_config(self, panel, sample_spy):
        config = ModelConfigBuilder().bayesian_pymc().with_target_accept(0.95).build()
        kwargs = _fit_kwargs(_model(panel, config), sample_spy, target_accept=0.99)
        assert kwargs["target_accept"] == 0.99

    def test_untouched_config_still_defaults_to_0_9(self, panel, sample_spy):
        """Byte-identical for anyone who never touched the knob."""
        config = ModelConfigBuilder().bayesian_pymc().build()
        assert _fit_kwargs(_model(panel, config), sample_spy)["target_accept"] == 0.9


# ---------------------------------------------------------------------------
# #170 — nuts_sampler is a real parameter, and both families agree on it
# ---------------------------------------------------------------------------


class TestNutsSamplerKeyword:
    def test_keyword_reaches_the_sampler_once(self, panel, sample_spy):
        """The call that used to raise `got multiple values for 'nuts_sampler'`."""
        config = ModelConfigBuilder().bayesian_pymc().build()
        kwargs = _fit_kwargs(_model(panel, config), sample_spy, nuts_sampler="numpyro")
        assert kwargs["nuts_sampler"] == "numpyro"

    def test_explicit_argument_wins_over_config(self, panel, sample_spy):
        config = ModelConfigBuilder().bayesian_numpyro().build()
        kwargs = _fit_kwargs(_model(panel, config), sample_spy, nuts_sampler="pymc")
        assert kwargs["nuts_sampler"] == "pymc"

    @pytest.mark.parametrize(
        ("builder_method", "expected"),
        [
            ("bayesian_pymc", "pymc"),
            ("bayesian_numpyro", "numpyro"),
            ("bayesian_nutpie", "nutpie"),
        ],
    )
    def test_config_selects_the_backend(
        self, panel, sample_spy, builder_method, expected
    ):
        config = getattr(ModelConfigBuilder(), builder_method)().build()
        kwargs = _fit_kwargs(_model(panel, config), sample_spy)
        assert kwargs["nuts_sampler"] == expected

    def test_both_model_families_accept_the_keyword(self):
        """The asymmetry that made the same call work on one family and raise
        on the other."""
        for fit in (BayesianMMM.fit, BaseExtendedMMM.fit):
            params = inspect.signature(fit).parameters
            assert "nuts_sampler" in params, (
                f"{fit.__qualname__} does not declare nuts_sampler — it would be "
                "swallowed by **kwargs and collide with the explicit argument"
            )


# ---------------------------------------------------------------------------
# #171 — nutpie is selectable
# ---------------------------------------------------------------------------


class TestNutpieIsReachable:
    def test_config_exposes_every_declared_backend(self):
        assert (
            ModelConfig(inference_method=InferenceMethod.BAYESIAN_PYMC).nuts_sampler
            == "pymc"
        )
        assert (
            ModelConfig(inference_method=InferenceMethod.BAYESIAN_NUMPYRO).nuts_sampler
            == "numpyro"
        )
        assert (
            ModelConfig(inference_method=InferenceMethod.BAYESIAN_NUTPIE).nuts_sampler
            == "nutpie"
        )

    def test_nutpie_counts_as_bayesian(self):
        config = ModelConfig(inference_method=InferenceMethod.BAYESIAN_NUTPIE)
        assert config.is_bayesian
        assert not config.use_numpyro

    def test_frequentist_config_reports_the_historical_default(self):
        """No NUTS backend to report — must not raise on a non-Bayesian config."""
        config = ModelConfig(inference_method=InferenceMethod.FREQUENTIST_RIDGE)
        assert config.nuts_sampler == "pymc"

    def test_declared_nutpie_version_is_one_pymc_will_drive(self):
        """The <0.16.10 floor shipped a sampler pymc refuses to start."""
        from importlib.metadata import version

        from packaging.version import Version

        assert Version(version("nutpie")) >= Version("0.16.10")
