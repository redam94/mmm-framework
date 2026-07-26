"""The bootstrap must produce intervals that actually cover.

#186 is the highest-risk issue in epic #180: a plausible bootstrap that silently
under-covers is worse than no interval at all, because a narrow interval reads as
confidence. So the acceptance evidence here is **empirical coverage on simulated
truth**, graded by ``diagnostics/coverage.py`` — the same machinery that grades
the Bayesian path — and not a plausibility argument about the algorithm.

The load-bearing test is :class:`TestCoverageBlockVsIid`. It plants an AR(1)
error process, which is the thing MMM residuals actually do and the thing an iid
residual bootstrap gets wrong, and measures both. Measured at ρ = 0.6 over 60
simulations of the ``make_clean`` world (``scripts`` note in
``technical-docs/frequentist-estimation.md`` §5): the iid bootstrap's 90%
intervals cover well below nominal, and the block version restores them. The
test asserts the *direction and size* of that gap at a cheaper budget.

Note what is deliberately **not** claimed: coverage for the true parameter under
a working penalty. Ridge is biased, percentile intervals cover the estimator's
sampling distribution, and no interval method fixes that — the penalty here is
small enough (effective dof ≈ column count) that the bias is not what is being
measured.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.config.enums import SaturationType
from mmm_framework.diagnostics.coverage import build_recovery_result
from mmm_framework.frequentist.bootstrap import (
    _resample_residuals,
    _rows_by_cell,
    bc_interval,
    bca_interval,
    bootstrap_fit,
    estimate_block_length,
    moving_block_indices,
    residual_autocorrelation,
)
from mmm_framework.frequentist.design import UnsupportedModelError, build_design_matrix
from mmm_framework.frequentist.ridge import fit_ridge
from mmm_framework.model.trend_config import TrendConfig, TrendType

from test_design_equivalence import CHANNELS, _configure, _panel

ALPHA = {c: {"alpha": 0.55} for c in CHANNELS}
LAM = {c: {"sat_lam": 2.7} for c in CHANNELS}
MC = ModelConfig()
TC = TrendConfig(type=TrendType.LINEAR)


def _ready(geos=None, n_periods=104):
    panel = _panel(geos=geos, n_periods=n_periods)
    _configure(panel, "geometric", SaturationType.LOGISTIC)
    return panel


def _boot(panel, **kw):
    kw.setdefault("model_config", MC)
    kw.setdefault("trend_config", TC)
    kw.setdefault("alpha", ALPHA)
    kw.setdefault("lam", LAM)
    kw.setdefault("penalty", 0.1)
    kw.setdefault("n_boot", 30)
    kw.setdefault("seed", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bootstrap_fit(panel, **kw)


def _ar1(n, rho, sd, rng):
    """AR(1) errors with marginal standard deviation ``sd``."""
    if rho <= 0:
        return rng.normal(0, sd, n)
    innov = sd * np.sqrt(1 - rho**2)
    out = np.empty(n)
    out[0] = rng.normal(0, sd)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + rng.normal(0, innov)
    return out


def _with_y(panel, y_raw):
    return dataclasses.replace(
        panel,
        y=pd.Series(np.asarray(y_raw, float), index=panel.y.index, name=panel.y.name),
    )


# --------------------------------------------------------------------------- #
# block length
# --------------------------------------------------------------------------- #


class TestBlockLength:
    def test_white_noise_gives_the_iid_bootstrap(self):
        """ρ̂ = 0 must reduce to block length 1, not to a magic constant."""
        assert estimate_block_length(0.0, 156) == 1

    def test_block_grows_with_dependence(self):
        lengths = [estimate_block_length(r, 156) for r in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert lengths == sorted(lengths)
        assert lengths[0] < lengths[-1]

    def test_block_is_clipped_below_a_quarter_of_the_series(self):
        """Unclipped, ρ̂ → 1 asks for blocks longer than the series — which
        resamples one block and produces no variability at all."""
        assert estimate_block_length(0.99, 100) <= 25
        assert estimate_block_length(0.99, 100) >= 1

    def test_short_series_falls_back_to_iid(self):
        assert estimate_block_length(0.8, 3) == 1

    def test_negative_autocorrelation_reads_as_zero(self):
        rng = np.random.default_rng(0)
        e = np.empty(200)
        e[0] = rng.normal()
        for t in range(1, 200):  # ρ = −0.6
            e[t] = -0.6 * e[t - 1] + rng.normal(0, 0.8)
        rho = residual_autocorrelation(e, np.arange(200))
        assert rho == 0.0

    def test_autocorrelation_recovers_a_planted_rho(self):
        rng = np.random.default_rng(3)
        e = _ar1(4000, 0.6, 1.0, rng)
        rho = residual_autocorrelation(e, np.arange(4000))
        assert rho == pytest.approx(0.6, abs=0.05)

    def test_panel_autocorrelation_does_not_span_cell_boundaries(self):
        """Stacking cells into one flat series manufactures a lag-1 pair at every
        geo boundary; the pooled within-cell estimator must not."""
        rng = np.random.default_rng(5)
        per_cell = 300
        n_cells = 4
        resid = np.concatenate([_ar1(per_cell, 0.5, 1.0, rng) for _ in range(n_cells)])
        time_idx = np.tile(np.arange(per_cell), n_cells)
        cell_idx = np.repeat(np.arange(n_cells), per_cell)
        rho = residual_autocorrelation(resid, time_idx, cell_idx, n_cells)
        assert rho == pytest.approx(0.5, abs=0.06)


class TestMovingBlocks:
    def test_returns_exactly_n_indices_in_range(self):
        rng = np.random.default_rng(0)
        for b in (1, 3, 8, 40):
            idx = moving_block_indices(52, b, rng)
            assert idx.shape == (52,)
            assert idx.min() >= 0 and idx.max() < 52

    def test_blocks_are_contiguous_runs(self):
        """The point of a block is that consecutive draws stay consecutive."""
        idx = moving_block_indices(60, 5, np.random.default_rng(1))
        runs = idx.reshape(-1, 5)[:-1]  # last block may be truncated
        assert np.all(np.diff(runs, axis=1) == 1)

    def test_block_length_one_is_the_iid_bootstrap(self):
        idx = moving_block_indices(200, 1, np.random.default_rng(2))
        # Consecutive-pair fraction under iid is ~1/n, nowhere near a block draw.
        assert float(np.mean(np.diff(idx) == 1)) < 0.1

    def test_panel_cells_share_one_resample(self):
        """A resampled week must carry every geography's residual together."""
        per_cell, n_cells = 50, 3
        resid = np.arange(per_cell * n_cells, dtype=float)
        time_idx = np.tile(np.arange(per_cell), n_cells)
        cell_idx = np.repeat(np.arange(n_cells), per_cell)
        rows = _rows_by_cell(time_idx, cell_idx, n_cells)
        out = _resample_residuals(resid, rows, 5, np.random.default_rng(0))
        # resid[g*per_cell + t] == g*per_cell + t, so a synchronized resample
        # leaves the same OFFSET pattern in every cell.
        grid = out.reshape(n_cells, per_cell) - (np.arange(n_cells)[:, None] * per_cell)
        assert np.array_equal(grid[0], grid[1])
        assert np.array_equal(grid[0], grid[2])


# --------------------------------------------------------------------------- #
# bias-corrected intervals
# --------------------------------------------------------------------------- #


class TestBiasCorrectedIntervals:
    def test_bc_matches_percentile_for_a_median_unbiased_distribution(self):
        draws = np.linspace(-3, 3, 2000)  # even count, symmetric about 0
        point = 0.0  # exactly half the replicates below ⇒ z0 = 0
        lo, hi = bc_interval(draws, point, 0.9)
        assert lo == pytest.approx(np.quantile(draws, 0.05), abs=1e-6)
        assert hi == pytest.approx(np.quantile(draws, 0.95), abs=1e-6)

    def test_bc_shifts_toward_the_point_estimate_when_replicates_are_biased(self):
        rng = np.random.default_rng(0)
        draws = rng.normal(1.0, 1.0, 20000)  # replicates sit ABOVE the point
        plain = (np.quantile(draws, 0.05), np.quantile(draws, 0.95))
        lo, hi = bc_interval(draws, 0.0, 0.9)
        assert lo < plain[0] and hi < plain[1]

    def test_degenerate_replicates_fall_back_to_percentile(self):
        """A coefficient pinned at a non-negativity boundary has every replicate
        on one side; BC must not return an infinite endpoint."""
        draws = np.zeros(100)
        lo, hi = bc_interval(draws, 0.0, 0.9)
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_bca_falls_back_to_bc_without_a_usable_jackknife(self):
        draws = np.linspace(-3, 3, 501)
        assert bca_interval(draws, 0.0, np.ones(50), 0.9) == bc_interval(
            draws, 0.0, 0.9
        )

    def test_bca_acceleration_moves_the_interval_on_a_skewed_statistic(self):
        rng = np.random.default_rng(1)
        draws = rng.gamma(2.0, 1.0, 20000)
        jack = rng.gamma(2.0, 1.0, 200)
        assert bca_interval(draws, 2.0, jack, 0.9) != bc_interval(draws, 2.0, 0.9)


# --------------------------------------------------------------------------- #
# the container contract
# --------------------------------------------------------------------------- #


class TestContainerContract:
    @pytest.fixture(scope="class")
    def fit(self):
        return _boot(_ready(), n_boot=40)

    def test_shape_is_one_chain_by_n_boot(self, fit):
        post = fit[0].posterior
        assert post.sizes["chain"] == 1
        assert post.sizes["draw"] == 40

    def test_carries_the_deterministics_downstream_code_reads(self, fit):
        names = set(fit[0].posterior.data_vars)
        assert {
            "channel_contributions",
            "media_total",
            "controls_total",
            "y_obs_scaled",
            "intercept_component",
            "trend_component",
            "seasonality_component",
            *(f"beta_{c}" for c in CHANNELS),
        } <= names

    def test_indexes_like_a_nuts_trace(self, fit):
        """Generic ``*_dim_0`` axes would break every ``.sel(channel=...)``."""
        cc = fit[0].posterior["channel_contributions"]
        assert cc.dims == ("chain", "draw", "obs", "channel")
        assert list(cc.coords["channel"].values) == CHANNELS

    def test_contributions_equal_beta_times_the_design_column(self, fit):
        """The trace is evaluated out of the model's own graph, so it cannot
        drift from the Bayesian definition of a contribution."""
        idata, _ = fit
        design = build_design_matrix(
            _ready(), ALPHA, LAM, model_config=MC, trend_config=TC
        )
        media = design.blocks["media"]
        post = idata.posterior
        for j, ch in enumerate(CHANNELS):
            col = design.X[:, media.start + j]
            theta = np.asarray(post[f"beta_{ch}"].values[0]) / design.roi_scale.get(
                ch, 1.0
            )
            got = np.asarray(post["channel_contributions"].values[0])[:, :, j]
            assert np.allclose(got, theta[:, None] * col[None, :], atol=1e-9)

    def test_media_total_closes_over_the_channels(self, fit):
        post = fit[0].posterior
        assert np.allclose(
            np.asarray(post["media_total"].values),
            np.asarray(post["channel_contributions"].values).sum(axis=-1),
            atol=1e-9,
        )

    def test_replicates_centre_on_the_point_estimate(self, fit):
        idata, diag = fit
        for ch in CHANNELS:
            point = diag["point_estimate"][f"media_{ch}"]
            draws = np.asarray(idata.posterior[f"beta_{ch}"].values).ravel()
            spread = float(np.std(draws))
            assert abs(float(np.mean(draws)) - point) < 0.7 * spread

    def test_geo_panel_emits_the_identified_composite(self):
        """``geo_sigma`` and ``geo_offset`` are an arbitrary factorization of one
        identified quantity; publishing them as estimates would be a fiction."""
        idata, _ = _boot(_ready(geos=["N", "S", "W"]), n_boot=12)
        names = set(idata.posterior.data_vars)
        assert "geo_effect" in names
        assert not ({"geo_sigma", "geo_offset"} & names)

    def test_spline_trend_emits_the_composite_coefficient(self):
        panel = _ready()
        idata, _ = _boot(
            panel, trend_config=TrendConfig(type=TrendType.SPLINE), n_boot=8
        )
        names = set(idata.posterior.data_vars)
        assert "spline_coef" in names
        assert not ({"spline_scale", "spline_coef_raw"} & names)

    def test_roi_mode_emits_roi_draws(self):
        idata, _ = _boot(
            _ready(), model_config=ModelConfig(media_prior_mode="roi"), n_boot=8
        )
        assert {f"roi_{c}" for c in CHANNELS} <= set(idata.posterior.data_vars)


# --------------------------------------------------------------------------- #
# provenance — what #188 renders
# --------------------------------------------------------------------------- #


class TestProvenance:
    def test_stamps_the_frequentist_contract(self):
        _, diag = _boot(_ready(), n_boot=8)
        assert diag["inference_family"] == "frequentist"
        assert diag["estimator"] == "ridge"
        assert diag["interval_kind"] == "bootstrap_percentile"
        # `approximate` is the WRONG flag: an approximate fit is a bad posterior,
        # a ridge fit is not a posterior at all (spec §8).
        assert diag["approximate"] is False
        # FitMethod has no frequentist member; defaulting it is what makes the
        # interactive report print "NUTS".
        assert diag["fit_method"] is None

    def test_cheap_path_is_labelled_conditional_on_selection(self):
        _, diag = _boot(_ready(), n_boot=8)
        assert diag["interval_semantics"] == "conditional_on_selection"
        assert any("OMITS selection uncertainty" in c for c in diag["caveats"])

    def test_every_rendered_caveat_says_confidence_not_credible(self):
        _, diag = _boot(_ready(), n_boot=8)
        joined = " ".join(diag["caveats"]).lower()
        assert "confidence" in joined
        assert "not credible intervals" in joined
        assert "ridge is biased" in joined

    def test_diagnostics_are_json_safe(self):
        """They ride through the LangGraph checkpointer, which msgpack-serializes."""
        import json

        _, diag = _boot(_ready(), n_boot=8)
        json.loads(json.dumps(diag))

    def test_block_length_and_rho_are_reported(self):
        _, diag = _boot(_ready(), n_boot=8)
        assert diag["block_length"] >= 1
        assert 0.0 <= diag["residual_rho"] <= 0.99
        assert diag["block_length_source"] == "estimated"
        assert diag["effective_dof"] <= diag["n_params"]

    def test_explicit_block_length_is_recorded_as_explicit(self):
        _, diag = _boot(_ready(), n_boot=8, block_length=6)
        assert diag["block_length"] == 6
        assert diag["block_length_source"] == "explicit"


class TestRefusals:
    def test_alpha_without_lam(self):
        with pytest.raises(ValueError, match="together"):
            _boot(_ready(), lam=None)

    def test_too_few_replicates(self):
        with pytest.raises(ValueError, match="at least 2"):
            _boot(_ready(), n_boot=1)

    def test_refit_search_refuses_a_non_extrapolating_trend(self):
        with pytest.raises(UnsupportedModelError, match="refit_search"):
            _boot(
                _ready(),
                trend_config=TrendConfig(type=TrendType.SPLINE),
                refit_search=True,
                n_boot=4,
            )

    def test_unsupported_model_still_refuses_through_the_bootstrap(self):
        with pytest.raises(UnsupportedModelError, match="Gaussian-process"):
            _boot(_ready(), trend_config=TrendConfig(type=TrendType.GP))


class TestSelectionResampling:
    """``refit_search=True`` is the honest interval; it must be reachable and
    must widen the result, not merely relabel it."""

    @pytest.mark.slow
    def test_refit_search_is_labelled_and_wider(self):
        panel = _ready(n_periods=60)
        search_kwargs = {"budget": 12, "horizon": 10, "max_origins": 1}
        cheap = _boot(
            panel,
            alpha=None,
            lam=None,
            penalty=None,
            n_boot=25,
            search_kwargs=search_kwargs,
            seed=4,
        )
        honest = _boot(
            panel,
            alpha=None,
            lam=None,
            penalty=None,
            n_boot=25,
            refit_search=True,
            search_kwargs=search_kwargs,
            seed=4,
        )
        assert cheap[1]["interval_semantics"] == "conditional_on_selection"
        assert honest[1]["interval_semantics"] == "selection_resampled"
        assert any("includes selection uncertainty" in c for c in honest[1]["caveats"])

        def width(fit):
            cc = np.asarray(fit[0].posterior["channel_contributions"].values[0])
            totals = cc.sum(axis=1)  # (draw, channel)
            lo, hi = np.percentile(totals, [5, 95], axis=0)
            return float(np.mean(hi - lo))

        assert width(honest) > width(cheap)


# --------------------------------------------------------------------------- #
# THE acceptance evidence
# --------------------------------------------------------------------------- #


def _coverage_at(panel, rho, block_length, n_sims, n_boot, level=0.90):
    """Empirical coverage of per-channel contribution intervals at planted truth."""
    design = build_design_matrix(panel, ALPHA, LAM, model_config=MC, trend_config=TC)
    base = fit_ridge(design, penalty=0.1)
    theta = base.theta.copy()
    media = design.blocks["media"]
    y_mean, y_std = design.scaling["y_mean"], design.scaling["y_std"]
    fitted = design.X @ theta

    truths, draws = {}, {}
    for j, ch in enumerate(CHANNELS):
        col = media.start + j
        truths[f"contribution_{ch}"] = float(design.X[:, col].sum() * theta[col])
        draws[f"contribution_{ch}"] = []

    for i in range(n_sims):
        e = _ar1(design.n_obs, rho, base.residual_sd, np.random.default_rng(700 + i))
        sim = _with_y(panel, (fitted + e) * y_std + y_mean)
        idata, _ = _boot(sim, n_boot=n_boot, block_length=block_length, seed=900 + i)
        cc = np.asarray(idata.posterior["channel_contributions"].values[0])
        for j, ch in enumerate(CHANNELS):
            draws[f"contribution_{ch}"].append(cc[:, :, j].sum(axis=1))

    result = build_recovery_result(
        truths,
        draws,
        levels=(level,),
        truth_source=f"planted (AR1 rho={rho})",
        sampler=f"ridge+block{block_length}",
        n_sims_requested=n_sims,
    )
    stats = [t.coverage_at(level) for t in result.targets]
    return float(np.mean([s.coverage for s in stats if s is not None])), result


@pytest.mark.slow
class TestCoverageBlockVsIid:
    """The number that decides whether the bootstrap story is real.

    Measured at n_sims=60 / n_boot=300 on the ``make_clean`` world (recorded in
    ``technical-docs/frequentist-estimation.md`` §5); asserted here at a cheaper
    budget, so the tolerances are deliberately loose — a tight threshold on 24
    simulations would be a flaky test asserting Monte-Carlo noise.
    """

    N_SIMS = 24
    N_BOOT = 200

    def test_iid_undercovers_and_blocks_fix_it_under_autocorrelation(self):
        panel = _ready(n_periods=104)
        iid, _ = _coverage_at(panel, 0.6, 1, self.N_SIMS, self.N_BOOT)
        block, res = _coverage_at(panel, 0.6, None, self.N_SIMS, self.N_BOOT)
        assert iid < 0.85, (
            f"iid bootstrap covered {iid:.0%} of 90% intervals — expected it to "
            "under-cover under AR(1) errors; if it did not, the test world lost "
            "its autocorrelation"
        )
        assert block > iid + 0.05, (
            f"block bootstrap covered {block:.0%} vs iid {iid:.0%} — the block "
            "correction is not doing its job"
        )
        assert block >= 0.80, f"block bootstrap still under-covers at {block:.0%}"

    def test_white_noise_needs_no_correction(self):
        """The block length is estimated, so an uncorrelated world must not be
        paying a width penalty for a dependence that is not there."""
        panel = _ready(n_periods=104)
        iid, _ = _coverage_at(panel, 0.0, 1, self.N_SIMS, self.N_BOOT)
        block, _ = _coverage_at(panel, 0.0, None, self.N_SIMS, self.N_BOOT)
        assert iid >= 0.80
        assert abs(block - iid) < 0.15
