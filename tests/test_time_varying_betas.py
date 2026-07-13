"""Time-varying media coefficients — TVP (#137).

Opt-in ``MediaChannelConfig.time_varying`` models ``log(beta_{c,t})`` as a smooth
random walk so a channel's effectiveness can drift (creative fatigue, secular
decay). Contract pinned here:

* off by default — a model with no time-varying channel has NO TVP RVs (R0.1);
* on ⇒ ``beta_tv_<ch>`` trajectory + the per-step ``_level`` / ``_tvpsigma`` /
  ``_tvpz`` RVs; ``beta_<ch>`` is still emitted as the time-AVERAGE summary (R0.4);
* recovers a planted drift on ``synth/dgp.make_time_varying_beta`` (sign / shape);
* the trajectory is surfaced for reporting (``bundle.time_varying_betas``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mmm_framework import BayesianMMM, ModelConfig, ModelConfigBuilder, TrendConfig
from mmm_framework.model import TrendType
from mmm_framework.synth.dgp import make_time_varying_beta


def _scenario_panel(time_varying: set[str] | None = None):
    panel = make_time_varying_beta(seed=6).panel()
    for mc in panel.config.media_channels:
        if time_varying and mc.name in time_varying:
            mc.time_varying = True
    return panel


def _free(m: BayesianMMM) -> set[str]:
    return {v.name for v in m.model.free_RVs}


def test_default_channel_is_not_time_varying():
    from mmm_framework.config import MediaChannelConfig

    assert MediaChannelConfig(name="TV").time_varying is False


def test_off_has_no_tvp_rvs():
    m = BayesianMMM(
        _scenario_panel(), ModelConfig(), TrendConfig(type=TrendType.LINEAR)
    )
    free = _free(m)
    assert not any("_tvp" in n or n.endswith("_level") for n in free)
    assert not any(n.startswith("beta_tv_") for n in m.model.named_vars)
    assert "beta_TV" in free  # scalar coefficient, as today


def test_on_builds_random_walk_and_keeps_contract():
    m = BayesianMMM(
        _scenario_panel({"TV", "Search"}),
        ModelConfig(),
        TrendConfig(type=TrendType.LINEAR),
    )
    free = _free(m)
    named = set(m.model.named_vars)
    for ch in ("TV", "Search"):
        assert f"beta_{ch}_level" in free
        assert f"beta_{ch}_tvpsigma" in free
        assert f"beta_{ch}_tvpz" in free
        assert f"beta_tv_{ch}" in named  # per-period trajectory
        # R0.4: beta_<ch> still emitted, now the time-average Deterministic
        assert f"beta_{ch}" in named and f"beta_{ch}" not in free
    # an untouched channel keeps its scalar coefficient
    assert "beta_Display" in free
    # channel_contributions contract preserved
    assert "channel_contributions" in named


def test_time_varying_channel_excluded_from_grouping():
    from mmm_framework.config import ModelConfig as MC

    panel = _scenario_panel({"TV"})
    for mc in panel.config.media_channels:
        if mc.name in ("TV", "Search"):
            mc.parent_channel = "grp"
    m = BayesianMMM(
        panel, MC(use_grouped_media_priors=True), TrendConfig(type=TrendType.LINEAR)
    )
    assert "TV" not in m._pooled_channels  # TVP takes precedence


@pytest.mark.slow
class TestTVPFit:
    def _fit(self, panel, method="map"):
        cfg = ModelConfigBuilder()
        if method == "nuts":
            cfg = cfg.bayesian_numpyro().with_chains(4).with_draws(600).with_tune(700)
        else:
            cfg = cfg.map_fit()
        m = BayesianMMM(panel, cfg.build(), TrendConfig(type=TrendType.LINEAR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(random_seed=0)
        return m

    def _traj(self, m, ch):
        return m._trace.posterior[f"beta_tv_{ch}"].mean(dim=["chain", "draw"]).values

    def test_recovers_planted_drift(self):
        """TV fatigues (1.4→0.6) and Search steps UP at the midpoint (0.6→1.5)."""
        m = self._fit(_scenario_panel({"TV", "Search"}), method="nuts")
        tv = self._traj(m, "TV")
        sr = self._traj(m, "Search")
        n = len(tv)
        q = n // 4
        assert tv[:q].mean() > tv[-q:].mean()  # TV declines
        assert sr[:q].mean() < sr[-q:].mean()  # Search rises

    def test_reporting_surfaces_trajectory(self):
        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        m = self._fit(_scenario_panel({"TV"}))
        bundle = BayesianMMMExtractor(m).extract()
        assert bundle.time_varying_betas is not None
        assert "TV" in bundle.time_varying_betas
        tv = bundle.time_varying_betas["TV"]
        assert len(tv["median"]) == m.n_periods
        assert len(tv["lower"]) == len(tv["upper"]) == m.n_periods
