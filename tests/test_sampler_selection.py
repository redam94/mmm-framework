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
import warnings

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
        """No NUTS backend to report — must not raise on a non-Bayesian config.

        Reading the property stays safe so a config can be inspected and
        round-tripped. ``fit()`` branches on ``is_frequentist`` *before* reading
        it, so the value is never consumed on this config — which is what stops
        the fall-through that used to silently run NUTS (#188).
        """
        config = ModelConfig(inference_method=InferenceMethod.FREQUENTIST_RIDGE)
        assert config.nuts_sampler == "pymc"
        assert config.is_frequentist

    def test_declared_nutpie_version_is_one_pymc_will_drive(self):
        """The <0.16.10 floor shipped a sampler pymc refuses to start."""
        from importlib.metadata import version

        from packaging.version import Version

        assert Version(version("nutpie")) >= Version("0.16.10")


# ---------------------------------------------------------------------------
# #181 / #180 — declared-but-unimplemented inference methods must refuse
# ---------------------------------------------------------------------------


class TestFrequentistInferenceMethods:
    """``frequentist_ridge`` / ``frequentist_cvxpy``, from silent NUTS to real.

    The history matters for what these assert. Originally ``fit()`` dispatched
    on ``FitMethod`` and never on ``InferenceMethod``, and ``nuts_sampler`` fell
    back to ``"pymc"`` for anything outside its Bayesian map — so asking for a
    frequentist point estimate returned a full posterior, after paying for MCMC,
    with no warning. #181 turned that into a loud refusal; #188 turned the
    refusal into a dispatch. These tests pin the third state and, just as
    importantly, that the *second* one is gone: a supported path must not warn.
    """

    FREQUENTIST = [
        InferenceMethod.FREQUENTIST_RIDGE,
        InferenceMethod.FREQUENTIST_CVXPY,
    ]

    @pytest.mark.parametrize("method", FREQUENTIST)
    def test_construction_does_not_warn(self, method):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            config = ModelConfig(inference_method=method)
        assert config.is_implemented

    @pytest.mark.parametrize("method", FREQUENTIST)
    def test_is_frequentist_and_not_bayesian(self, method):
        config = ModelConfig(inference_method=method)
        assert config.is_frequentist
        assert not config.is_bayesian

    def test_estimator_is_named_per_method(self):
        assert (
            ModelConfig(
                inference_method=InferenceMethod.FREQUENTIST_RIDGE
            ).frequentist_estimator
            == "ridge"
        )
        assert (
            ModelConfig(
                inference_method=InferenceMethod.FREQUENTIST_CVXPY
            ).frequentist_estimator
            == "constrained"
        )
        assert (
            ModelConfig(
                inference_method=InferenceMethod.BAYESIAN_PYMC
            ).frequentist_estimator
            is None
        )

    def test_fit_never_reaches_pm_sample(self, panel, sample_spy):
        """The whole point: a frequentist request must not run MCMC.

        This is the original defect stated positively. It is asserted on the
        DISPATCHING path rather than on a refusal, because a refusal that never
        reaches ``pm.sample`` proves nothing once the method is implemented.
        """
        config = ModelConfig(
            inference_method=InferenceMethod.FREQUENTIST_RIDGE,
            bootstrap_samples=8,
            optim_maxiter=4,
        )
        model = _model(panel, config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.fit(
                search_kwargs={"budget": 4, "horizon": 6, "max_origins": 1}
            )

        assert not sample_spy, "pm.sample was reached for a frequentist fit"
        assert results.diagnostics["inference_family"] == "frequentist"
        assert results.diagnostics["estimator"] == "ridge"
        # Not an approximate posterior — not a posterior at all.
        assert results.approximate is False
        # R-hat/ESS describe a sampler; there is no chain here.
        assert results.converged is None

    @pytest.mark.parametrize(
        "method",
        [
            InferenceMethod.BAYESIAN_PYMC,
            InferenceMethod.BAYESIAN_NUMPYRO,
            InferenceMethod.BAYESIAN_NUTPIE,
        ],
    )
    def test_bayesian_methods_neither_warn_nor_raise(self, panel, sample_spy, method):
        """The dispatch must not touch the Bayesian path."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            config = ModelConfig(inference_method=method, n_draws=10, n_tune=10)
        assert config.is_implemented
        assert not config.is_frequentist
        _fit_kwargs(_model(panel, config), sample_spy)  # reaches pm.sample

    def test_enum_values_are_retained(self):
        """Refusing must not mean deleting — stored configs still parse.

        Removing the values would break the frozen-enum contract
        (tests/test_api_contracts.py) and is a MAJOR-version event.
        """
        assert InferenceMethod("frequentist_ridge") is InferenceMethod.FREQUENTIST_RIDGE
        assert InferenceMethod("frequentist_cvxpy") is InferenceMethod.FREQUENTIST_CVXPY

    def test_excel_template_selects_the_paradigm(self):
        """The spreadsheet path used to reject these at parse time, which was
        right while they were unimplemented and wrong the moment #188 landed —
        a gate that outlives the thing it gated is stale in the same way an
        unlabelled interval is, one surface removed."""
        from mmm_framework.excel_config.parser import _build_model_config

        for value, expected in (
            ("frequentist_ridge", InferenceMethod.FREQUENTIST_RIDGE),
            ("frequentist_cvxpy", InferenceMethod.FREQUENTIST_CVXPY),
        ):
            cfg = _build_model_config({"Inference Method": value})
            assert cfg.inference_method is expected, value
            assert cfg.is_frequentist
            # `FitMethod` has no frequentist member; a spreadsheet-built config
            # must not claim NUTS before it has been fitted either.
            assert cfg.fit_method is None, value

    def test_excel_template_still_falls_back_on_a_typo(self):
        """A typo in a spreadsheet cell has always produced a working model
        rather than a failed parse, and that is not this change's call to
        make."""
        from mmm_framework.excel_config.parser import _build_model_config

        cfg = _build_model_config({"Inference Method": "frequntist_ridge"})
        assert cfg.inference_method is InferenceMethod.BAYESIAN_NUMPYRO
