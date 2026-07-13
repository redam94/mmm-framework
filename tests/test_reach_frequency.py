"""Reach & frequency channel modeling — #141.

A channel can be declared reach/frequency-measured: its column is *reach* and its
effect is ``reach · g(frequency)`` where ``g`` is a frequency-saturation curve
(diminishing returns to added exposures). Contract:

* off by default — a channel is an ordinary volume channel, the frequency column
  (if present) is a plain linear control (no gain RVs, R0.1);
* on ⇒ the frequency column is REMOVED from the linear control block (not
  double-counted), the channel emits ``freq_gain_<ch>`` +
  ``effective_frequency_<ch>`` + the shape RV, and ``beta_<ch>`` /
  ``channel_contributions`` keep their meaning (R0.4);
* recovers a planted frequency-wearout shape on the synthetic world;
* reporting surfaces a reach-vs-frequency insight.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mmm_framework import BayesianMMM, ModelConfig, ModelConfigBuilder, TrendConfig
from mmm_framework.config import (
    AdstockConfig,
    FrequencyResponse,
    ReachFrequencyConfig,
)
from mmm_framework.model import TrendType
from mmm_framework.synth.dgp import make_reach_frequency


def _panel(*, adstock_none: bool = True):
    sc = make_reach_frequency()
    panel = sc.panel()
    if adstock_none:
        for mc in panel.config.media_channels:
            if mc.name == "TV":
                mc.adstock = AdstockConfig.none()
    return panel, sc


def _rf(response=FrequencyResponse.EXPONENTIAL):
    return ReachFrequencyConfig(
        channel="TV", frequency_column="Frequency", response=response
    )


def test_off_frequency_is_a_linear_control():
    panel, _ = _panel()
    m = BayesianMMM(panel, ModelConfig(), TrendConfig(type=TrendType.LINEAR))
    named = set(m.model.named_vars)
    assert "freq_gain_TV" not in named
    assert "effective_frequency_TV" not in named
    # Frequency + Price both remain linear controls.
    assert set(m.control_names) == {"Price", "Frequency"}


def test_on_pulls_frequency_and_adds_gain():
    panel, _ = _panel()
    m = BayesianMMM(
        panel, ModelConfig(reach_frequency=[_rf()]), TrendConfig(type=TrendType.LINEAR)
    )
    named = set(m.model.named_vars)
    free = {v.name for v in m.model.free_RVs}
    assert "freq_k_TV" in free  # exponential shape RV
    assert "freq_gain_TV" in named
    assert "effective_frequency_TV" in named
    # Frequency pulled out of the control block; Price stays.
    assert m.control_names == ["Price"]


def test_hill_response_uses_slope_and_halfsat():
    panel, _ = _panel()
    m = BayesianMMM(
        panel,
        ModelConfig(reach_frequency=[_rf(FrequencyResponse.HILL)]),
        TrendConfig(type=TrendType.LINEAR),
    )
    free = {v.name for v in m.model.free_RVs}
    assert "freq_halfsat_TV" in free
    assert "freq_slope_TV" in free
    assert "freq_k_TV" not in free  # exponential RV absent under Hill


def test_unknown_channel_warns():
    panel, _ = _panel()
    with pytest.warns(UserWarning, match="not a modeled media channel"):
        BayesianMMM(
            panel,
            ModelConfig(
                reach_frequency=[
                    ReachFrequencyConfig(channel="Nope", frequency_column="Frequency")
                ]
            ),
            TrendConfig(type=TrendType.LINEAR),
        ).model  # graph builds lazily; force it


def test_unknown_frequency_column_warns():
    panel, _ = _panel()
    with pytest.warns(UserWarning, match="not a control column"):
        BayesianMMM(
            panel,
            ModelConfig(
                reach_frequency=[
                    ReachFrequencyConfig(channel="TV", frequency_column="Nope")
                ]
            ),
            TrendConfig(type=TrendType.LINEAR),
        )


def test_gain_is_bounded_and_increasing_in_frequency():
    """g(frequency) ∈ (0, 1] and rises with frequency (a saturating curve)."""
    panel, sc = _panel()
    cfg = ModelConfigBuilder().map_fit().with_reach_frequency(_rf()).build()
    m = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(random_seed=0)
    gain = m._trace.posterior["freq_gain_TV"].mean(dim=["chain", "draw"]).values
    assert np.all(gain > 0) and np.all(gain <= 1.0 + 1e-6)
    freq = sc.notes["frequency"]
    # higher frequency ⇒ higher gain: the curve is strictly monotone (concave, so
    # sort by frequency and check the gain never decreases).
    order = np.argsort(freq)
    assert np.all(np.diff(gain[order]) >= -1e-9)


@pytest.mark.slow
class TestReachFrequencyFit:
    def _fit(self, response=FrequencyResponse.EXPONENTIAL):
        panel, sc = _panel()
        cfg = ModelConfigBuilder().map_fit().with_reach_frequency(_rf(response)).build()
        m = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(random_seed=0)
        return m, sc

    def test_recovers_positive_frequency_saturation(self):
        m, sc = self._fit()
        k = float(m._trace.posterior["freq_k_TV"].mean())
        ef = float(m._trace.posterior["effective_frequency_TV"].mean())
        assert k > 0  # a real (diminishing-returns) frequency-saturation rate
        # effective frequency lands in a plausible range around the planted value
        assert 2.0 < ef < 9.0
        assert sc.notes["true_freq_k"] > 0

    def test_frequency_model_beats_ignoring_frequency(self):
        """Modeling reach × frequency fits better than treating frequency as a
        plain linear control (the volume-substitute baseline)."""
        m_rf, _ = self._fit()
        panel, _ = _panel()
        base = ModelConfigBuilder().map_fit().build()
        m_base = BayesianMMM(panel, base, TrendConfig(type=TrendType.LINEAR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_base.fit(random_seed=0)

        def _r2(m):
            # original-scale fit: mean = y_mean + y_std * Σ(standardized components).
            post = m._trace.posterior
            comps = [
                v
                for v in post.data_vars
                if v.endswith("_component") or v.endswith("_total")
            ]
            mu_std = np.zeros(m.n_obs)
            for v in comps:
                mu_std = mu_std + post[v].mean(dim=["chain", "draw"]).values.reshape(-1)
            pred = m.y_mean + m.y_std * mu_std
            obs = np.asarray(m.panel.y).reshape(-1)
            ss_res = float(np.sum((obs - pred) ** 2))
            ss_tot = float(np.sum((obs - obs.mean()) ** 2))
            return 1.0 - ss_res / ss_tot

        assert _r2(m_rf) >= _r2(m_base) - 1e-6

    def test_reporting_surfaces_effective_frequency(self):
        m, _ = self._fit()
        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        bundle = BayesianMMMExtractor(m).extract()
        assert bundle.reach_frequency is not None
        assert "TV" in bundle.reach_frequency
        info = bundle.reach_frequency["TV"]
        assert info["response"] == "exponential"
        assert info["effective_frequency"] > 0
