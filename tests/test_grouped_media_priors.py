"""Grouped / hierarchical priors for collinear channels — DF-2 (#140).

Partial-pool the coefficients of channels sharing a ``parent_channel`` group
toward a shared log-normal mean, so genuinely EXCHANGEABLE collinear channels
borrow strength. Off by default. Contract pinned here (the DF-2 acceptance
gate in technical-docs/deferred-causal-features.md):

* **A1 / R0.2** — flag off ⇒ the graph and posterior are byte-identical to
  today (configuring ``parent_channel`` alone changes nothing).
* structure — flag on ⇒ shared ``beta_mu_/beta_tau_/beta_z_`` group RVs,
  ``beta_<ch>`` stays a Deterministic (name contract R0.4), pooled channels
  recorded for disclosure (R0.6).
* **A3 / DF2.4** — a calibrated (``roi_prior``) or explicitly-priored channel
  is EXCLUDED from the pool; its prior wins.
* **A2** — with genuinely different, independently-identified channels the
  adaptive ``tau_group`` lets them escape the pool (no silent over-shrinkage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework import (
    BayesianMMM,
    ModelConfig,
    ModelConfigBuilder,
    TrendConfig,
    TrendType,
)
from mmm_framework.config import (
    AdstockConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    PriorConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset


def _panel(
    channels: list[str],
    parents: dict[str, str] | None = None,
    calibrated: set[str] | None = None,
    *,
    betas: dict[str, float] | None = None,
    n: int = 90,
    seed: int = 0,
) -> PanelDataset:
    parents = parents or {}
    calibrated = calibrated or set()
    betas = betas or {c: 1.5 for c in channels}
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2021-01-04", periods=n, freq="W-MON")
    X = pd.DataFrame({c: np.abs(rng.normal(100, 30, n)) for c in channels})
    y = 1000.0 + sum(betas[c] * X[c] for c in channels) + rng.normal(0, 40, n)
    media = []
    for c in channels:
        kw = dict(
            name=c, dimensions=[DimensionType.PERIOD], adstock=AdstockConfig.none()
        )
        if c in parents:
            kw["parent_channel"] = parents[c]
        if c in calibrated:
            kw["roi_prior"] = PriorConfig.gamma(alpha=4, beta=2)
        media.append(MediaChannelConfig(**kw))
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=media,
        controls=[],
    )
    coords = PanelCoordinates(
        periods=periods, geographies=None, products=None, channels=channels, controls=[]
    )
    return PanelDataset(
        y=pd.Series(y, name="Sales"),
        X_media=X,
        X_controls=None,
        coords=coords,
        index=periods,
        config=config,
    )


def _mmm(panel: PanelDataset, grouped: bool) -> BayesianMMM:
    return BayesianMMM(
        panel,
        ModelConfig(use_grouped_media_priors=grouped),
        TrendConfig(type=TrendType.NONE),
    )


def _free(m: BayesianMMM) -> set[str]:
    return {v.name for v in m.model.free_RVs}


CH = ["TV", "Facebook", "Instagram"]
SOCIAL = {"Facebook": "social", "Instagram": "social"}


def test_default_flag_is_off():
    assert ModelConfig().use_grouped_media_priors is False
    assert ModelConfigBuilder().build().use_grouped_media_priors is False
    assert (
        ModelConfigBuilder()
        .with_grouped_media_priors()
        .build()
        .use_grouped_media_priors
        is True
    )


def test_off_is_fully_gated():
    """A1 structure: with the flag off, parent_channel changes NOTHING."""
    with_groups = _mmm(_panel(CH, parents=SOCIAL), grouped=False)
    no_groups = _mmm(_panel(CH), grouped=False)
    assert with_groups._pooled_channels == set()
    assert _free(with_groups) == _free(no_groups)
    assert not any(
        n.startswith(("beta_mu_", "beta_tau_", "beta_z_")) for n in _free(with_groups)
    )


def test_on_builds_group_prior_and_keeps_contract():
    m = _mmm(_panel(CH, parents=SOCIAL), grouped=True)
    free = _free(m)
    named = set(m.model.named_vars)
    assert m._pooled_channels == {"Facebook", "Instagram"}
    assert {"beta_mu_social", "beta_tau_social", "beta_z_social"} <= free
    # R0.4: beta_<ch> still emitted, now as a derived Deterministic for pooled ch
    for ch in ("Facebook", "Instagram"):
        assert f"beta_{ch}" in named and f"beta_{ch}" not in free
    assert "beta_TV" in free  # ungrouped channel keeps its independent prior


def test_calibrated_channel_excluded_from_pool():
    """A3 / DF2.4: a channel with a calibrated roi_prior is not pooled."""
    m = _mmm(_panel(CH, parents=SOCIAL, calibrated={"Facebook"}), grouped=True)
    # Facebook excluded -> only Instagram left in the group -> < 2 members -> no pool
    assert "Facebook" not in m._pooled_channels
    assert m._pooled_channels == set()
    # Facebook keeps its own (calibrated) coefficient RV
    assert "beta_Facebook" in {v.name for v in m.model.free_RVs}


def test_explicit_coefficient_prior_excluded_from_pool():
    panel = _panel(CH, parents=SOCIAL)
    # give Instagram an explicit coefficient prior -> excluded
    for mc in panel.config.media_channels:
        if mc.name == "Instagram":
            mc.coefficient_prior = PriorConfig.half_normal(sigma=1.0)
    m = _mmm(panel, grouped=True)
    assert "Instagram" not in m._pooled_channels


def test_singleton_group_not_pooled():
    m = _mmm(_panel(["TV", "Facebook"], parents={"Facebook": "social"}), grouped=True)
    assert m._pooled_channels == set()  # one member -> nothing to pool


@pytest.mark.slow
class TestGroupedFit:
    def _fit(self, panel, grouped, method="map"):
        cfg = ModelConfigBuilder().with_grouped_media_priors(grouped)
        if method == "nuts":
            cfg = (
                cfg.bayesian_numpyro()
                .with_chains(4)
                .with_draws(600)
                .with_tune(800)
                .with_target_accept(0.95)
            )
        else:
            cfg = cfg.map_fit()
        mmm = BayesianMMM(panel, cfg.build(), TrendConfig(type=TrendType.NONE))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mmm.fit(random_seed=0)
        return mmm

    def test_off_posterior_bit_identical(self):
        """A1 / R0.2: same-seed MAP fit, flag off, is identical with/without groups."""
        a = self._fit(_panel(CH, parents=SOCIAL, seed=1), grouped=False)
        b = self._fit(_panel(CH, seed=1), grouped=False)
        for ch in CH:
            ba = float(a._trace.posterior[f"beta_{ch}"].mean())
            bb = float(b._trace.posterior[f"beta_{ch}"].mean())
            np.testing.assert_allclose(ba, bb, rtol=1e-10, atol=1e-10)

    def test_calibration_precedence_posterior(self):
        """A3: a calibrated channel's estimate is the same whether grouping is on."""
        panel = _panel(CH, parents=SOCIAL, calibrated={"Facebook"}, seed=2)
        off = self._fit(panel, grouped=False)
        on = self._fit(panel, grouped=True)
        fb_off = float(off._trace.posterior["beta_Facebook"].mean())
        fb_on = float(on._trace.posterior["beta_Facebook"].mean())
        np.testing.assert_allclose(fb_off, fb_on, rtol=1e-6, atol=1e-6)

    def test_no_over_shrinkage_of_distinct_channels(self):
        """A2: independently-identified channels of DIFFERENT strength escape the
        pool — the adaptive tau_group preserves their between-channel ratio rather
        than collapsing them to a shared value.

        Uses scale-invariant ratios: the raw ``beta_<ch>`` values are standardized
        coefficients (and grouped betas are log-normal, so their absolute means are
        not directly comparable to the independent Gamma), but the RATIO of a
        strong to a weak channel is a fair, scale-free measure of over-shrinkage.
        """
        betas = {"S1": 1.0, "S2": 2.0, "S3": 3.0}
        chans = list(betas)
        parents = {c: "social" for c in chans}
        panel = _panel(chans, parents=parents, betas=betas, n=160, seed=5)

        off = self._fit(panel, grouped=False, method="nuts")
        on = self._fit(panel, grouped=True, method="nuts")

        def ratio(m):
            b = {c: float(m._trace.posterior[f"beta_{c}"].mean()) for c in chans}
            assert b["S3"] > b["S2"] > b["S1"], b  # ordering preserved
            return b["S3"] / b["S1"]

        off_ratio, on_ratio = ratio(off), ratio(on)
        # the strong/weak ratio is NOT collapsed: partial pooling shrinks it a
        # little, but an over-shrinking pool would flatten it toward 1.
        assert on_ratio > 1.6
        assert on_ratio > 0.6 * off_ratio

    def test_reporting_discloses_pooled_channels(self):
        """R0.6 / DF2.6: the extractor records pooled channels for disclosure."""
        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        m = self._fit(_panel(CH, parents=SOCIAL, seed=3), grouped=True)
        bundle = BayesianMMMExtractor(m).extract()
        assert bundle.pooled_channels == ["Facebook", "Instagram"]
