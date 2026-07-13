"""Cross-channel synergy / interaction terms — #142.

Opt-in ``beta_ij * sat_i * sat_j`` terms let a channel pair do more (synergy /
halo) or less (cannibalization) than the sum of its parts. Contract:

* off by default — strictly additive, graph byte-identical (R0.1);
* on ⇒ ``beta_int_<a>_<b>`` RV + ``interaction_component`` deterministic;
* sign-aware prior (positive / negative / any), shrunk toward zero;
* recovers the planted synergy SIGN on ``synth/dgp.make_synergy``;
* the interaction is a separate line in the decomposition (waterfall closes).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mmm_framework import BayesianMMM, ModelConfig, ModelConfigBuilder, TrendConfig
from mmm_framework.config import ChannelInteraction
from mmm_framework.model import TrendType
from mmm_framework.synth.dgp import make_synergy


def _panel():
    return make_synergy(seed=8).panel()


def _free(m):
    return {v.name for v in m.model.free_RVs}


def test_channel_interaction_config():
    ci = ChannelInteraction(
        channel_a="TV", channel_b="Search", expected_sign="positive"
    )
    assert ci.name == "TV_x_Search"
    with pytest.raises(ValueError, match="itself"):
        ChannelInteraction(channel_a="TV", channel_b="TV")


def test_default_is_additive():
    assert ModelConfig().channel_interactions == []
    m = BayesianMMM(_panel(), ModelConfig(), TrendConfig(type=TrendType.LINEAR))
    assert "beta_int_TV_Search" not in _free(m)
    assert "interaction_component" not in set(m.model.named_vars)


def test_on_builds_interaction_term():
    ci = [
        ChannelInteraction(channel_a="TV", channel_b="Search", expected_sign="positive")
    ]
    m = BayesianMMM(
        _panel(),
        ModelConfig(channel_interactions=ci),
        TrendConfig(type=TrendType.LINEAR),
    )
    assert "beta_int_TV_Search" in _free(m)
    named = set(m.model.named_vars)
    assert "interaction_component" in named
    assert "interaction_contributions" in named
    assert m._interaction_names == ["TV_x_Search"]
    # additive contract preserved
    assert "channel_contributions" in named and "beta_TV" in _free(m)


def test_unknown_channel_skipped_with_warning():
    ci = [ChannelInteraction(channel_a="TV", channel_b="Nope")]
    m = BayesianMMM(
        _panel(),
        ModelConfig(channel_interactions=ci),
        TrendConfig(type=TrendType.LINEAR),
    )
    with pytest.warns(UserWarning, match="unknown channel"):
        named = set(m.model.named_vars)  # lazy build triggers the warning here
    assert "interaction_component" not in named


@pytest.mark.slow
class TestInteractionFit:
    def _fit(self, interactions):
        cfg = (
            ModelConfigBuilder()
            .map_fit()
            .with_channel_interactions(*interactions)
            .build()
        )
        m = BayesianMMM(_panel(), cfg, TrendConfig(type=TrendType.LINEAR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(random_seed=0)
        return m

    def test_recovers_positive_synergy_sign(self):
        """make_synergy plants gamma > 0 (TV primes Search)."""
        ci = ChannelInteraction(
            channel_a="TV", channel_b="Search", expected_sign="any", prior_sigma=0.5
        )
        m = self._fit([ci])
        b = float(m._trace.posterior["beta_int_TV_Search"].mean())
        assert b > 0

    def test_negative_sign_prior_is_non_positive(self):
        ci = ChannelInteraction(
            channel_a="TV", channel_b="Display", expected_sign="negative"
        )
        m = self._fit([ci])
        # a "negative" interaction is the reflection of a HalfNormal -> <= 0
        b = m._trace.posterior["beta_int_TV_Display"].values
        assert np.all(b <= 0)

    def test_decomposition_has_synergy_and_closes(self):
        ci = ChannelInteraction(
            channel_a="TV", channel_b="Search", expected_sign="any", prior_sigma=0.5
        )
        m = self._fit([ci])
        dec = m.compute_component_decomposition()
        assert dec.total_interactions is not None
        total = (
            dec.intercept
            + dec.trend
            + dec.seasonality
            + dec.media_total
            + dec.controls_total
            + dec.interactions
        )
        post = m._trace.posterior

        def cm(v):
            return (
                post[v].mean(dim=["chain", "draw"]).values
                if v in post
                else np.zeros(m.n_obs)
            )

        expected = m.y_mean + m.y_std * (
            cm("intercept_component")
            + cm("trend_component")
            + cm("seasonality_component")
            + cm("media_total")
            + cm("controls_total")
            + cm("interaction_component")
        )
        np.testing.assert_allclose(total, expected, rtol=1e-8)
        assert "Synergy / Interactions" in dec.summary()["Component"].tolist()

        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        assert "Synergy" in BayesianMMMExtractor(m)._get_component_totals()
