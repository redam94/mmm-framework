"""Tests for EVOI and the EIG/EVOI priority grid (planning/evoi.py +
planning/priority.py), on synthetic response curves with known structure: a
channel whose ROI uncertainty straddles the allocation decision must carry the
EVOI; a certain channel must not."""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning import (
    ResponseCurves,
    compute_evoi_for_channel,
    compute_evpi,
    compute_experiment_priorities,
    fit_evoi_surrogate,
    optimize_budget,
    recommend_experiments,
    surrogate_evoi,
)

MULTS = np.linspace(0.0, 2.0, 21)


def _two_channel_curves(
    coef0_draws: np.ndarray, coef1_draws: np.ndarray, base: float = 100.0
) -> ResponseCurves:
    """sqrt response curves with per-draw coefficients: contribution_c(s) =
    coef_c[d] * sqrt(s). Decision-relevant uncertainty lives in the coefs."""
    base_spend = np.array([base, base])
    spend = base_spend[:, None] * MULTS[None, :]  # (C, G)
    contributions = np.stack(
        [
            np.stack([c0 * np.sqrt(spend[0]), c1 * np.sqrt(spend[1])])
            for c0, c1 in zip(coef0_draws, coef1_draws)
        ]
    )  # (D, C, G)
    return ResponseCurves(
        channel_names=["uncertain", "certain"],
        multipliers=MULTS,
        base_spend=base_spend,
        contributions=contributions,
    )


@pytest.fixture()
def decision_straddling_curves() -> ResponseCurves:
    """ch0's coefficient is bimodal (0.5 or 2.0): which channel deserves the
    budget flips across draws. ch1 is known exactly (coef 1.0)."""
    rng = np.random.default_rng(3)
    D = 120
    coef0 = np.where(rng.random(D) < 0.5, 0.5, 2.0)
    coef1 = np.full(D, 1.0)
    return _two_channel_curves(coef0, coef1)


class TestEvpi:
    def test_evpi_positive_when_decision_uncertain(self, decision_straddling_curves):
        res = compute_evpi(decision_straddling_curves)
        assert res.evpi > 0
        assert res.v_current > 0

    def test_evpi_zero_when_posterior_degenerate(self):
        curves = _two_channel_curves(np.full(50, 2.0), np.full(50, 1.0))
        res = compute_evpi(curves)
        assert res.evpi == pytest.approx(0.0, abs=1e-9)

    def test_reuses_precomputed_allocations(self, decision_straddling_curves):
        opt = optimize_budget(curves=decision_straddling_curves, random_seed=0)
        res = compute_evpi(
            decision_straddling_curves,
            total_budget=opt.total_budget,
            per_draw_alloc=opt.per_draw_alloc,
            optimal_alloc=opt.optimal_alloc,
        )
        fresh = compute_evpi(decision_straddling_curves)
        assert res.evpi == pytest.approx(fresh.evpi, rel=0.05)


class TestEvoi:
    def test_uncertain_channel_carries_evoi(self, decision_straddling_curves):
        curves = decision_straddling_curves
        g1 = int(np.argmin(np.abs(curves.multipliers - 1.0)))
        port = compute_evpi(curves)
        spend = curves.base_spend

        roi0 = curves.contributions[:, 0, g1] / spend[0]
        roi1 = curves.contributions[:, 1, g1] / spend[1]
        # a precise experiment on the channel that decides the allocation
        evoi0 = compute_evoi_for_channel(curves, 0, roi0, sigma_exp=0.1)
        # the certain channel: an experiment teaches nothing, moves nothing
        evoi1 = compute_evoi_for_channel(curves, 1, roi1, sigma_exp=0.1)
        assert evoi0 > 0
        assert evoi0 <= port.evpi * 1.05  # EVPI bounds EVOI (MC tolerance)
        assert evoi1 == pytest.approx(0.0, abs=port.evpi * 0.02)
        assert evoi0 > 10 * max(evoi1, 1e-9)

    def test_more_precise_experiment_is_worth_more(self, decision_straddling_curves):
        """The paired estimator must be monotone in experiment precision (the
        unpaired version could clip a precise experiment to zero on MC noise)."""
        curves = decision_straddling_curves
        g1 = int(np.argmin(np.abs(curves.multipliers - 1.0)))
        roi0 = curves.contributions[:, 0, g1] / curves.base_spend[0]
        precise = compute_evoi_for_channel(curves, 0, roi0, sigma_exp=0.0125)
        vague = compute_evoi_for_channel(curves, 0, roi0, sigma_exp=0.5)
        assert precise > vague
        # a near-perfect experiment on the deciding channel recovers ~all of EVPI
        port = compute_evpi(curves)
        assert precise == pytest.approx(port.evpi, rel=0.25)

    def test_common_random_numbers_reproducible(self, decision_straddling_curves):
        curves = decision_straddling_curves
        g1 = int(np.argmin(np.abs(curves.multipliers - 1.0)))
        roi0 = curves.contributions[:, 0, g1] / curves.base_spend[0]
        rng = np.random.default_rng(11)
        d_idx = rng.integers(0, len(roi0), size=32)
        z = rng.standard_normal(32)
        a = compute_evoi_for_channel(curves, 0, roi0, 0.1, outcome_draws=(d_idx, z))
        b = compute_evoi_for_channel(curves, 0, roi0, 0.1, outcome_draws=(d_idx, z))
        assert a == b


class TestPriorityGrid:
    def test_known_winner_ranks_first_with_quadrants(self, decision_straddling_curves):
        grid, portfolio = compute_experiment_priorities(
            curves=decision_straddling_curves,
            design_type="geo_holdout",
            random_seed=0,
        )
        assert [g.channel for g in grid][0] == "uncertain"
        by_name = {g.channel: g for g in grid}
        assert by_name["uncertain"].quadrant == "test_now"
        assert by_name["certain"].quadrant == "deprioritize"
        assert by_name["uncertain"].priority > by_name["certain"].priority
        assert portfolio["evpi"] > 0
        # JSON-safety of the row payload
        row = grid[0].to_dict()
        assert isinstance(row["eig"], float) and isinstance(row["quadrant"], str)

    def test_decay_and_retest_from_evidence(self, decision_straddling_curves):
        evidence = {
            "uncertain": {"end_date": "2024-01-01"},  # ancient evidence
            "certain": {"end_date": "2026-06-01"},  # fresh evidence
        }
        grid, _ = compute_experiment_priorities(
            curves=decision_straddling_curves,
            design_type="geo_holdout",
            evidence=evidence,
            as_of="2026-06-10",
            random_seed=0,
        )
        by_name = {g.channel: g for g in grid}
        old = by_name["uncertain"]
        assert old.weeks_since_evidence > 100
        assert old.eig_decayed > old.eig  # decay inflates effective uncertainty
        assert old.retest_due
        fresh = by_name["certain"]
        assert fresh.weeks_since_evidence < 2
        # certain channel: tiny posterior sd -> fresh experiment teaches ~nothing
        assert not fresh.retest_due


class TestRecommendExperimentsMethods:
    def test_eig_evoi_table_and_designs(self, decision_straddling_curves):
        table, designs = recommend_experiments(
            None,
            curves=decision_straddling_curves,
            top_k=1,
            random_seed=0,
            method="eig_evoi",
        )
        assert {"eig", "evoi", "quadrant", "priority"} <= set(table.columns)
        assert table.iloc[0]["channel"] == "uncertain"
        d = designs[0]
        assert d["priority_method"] == "eig_evoi"
        assert d["design_key"] == "national_pulse"  # mmm=None -> no geo
        assert d["sigma_exp"] > 0 and "EIG" in d["why"]

    def test_heuristic_fallback_still_works(self, decision_straddling_curves):
        table, designs = recommend_experiments(
            None,
            curves=decision_straddling_curves,
            top_k=1,
            random_seed=0,
            method="heuristic",
        )
        assert "eig" not in table.columns
        assert designs[0]["priority_method"] == "heuristic"

    def test_unknown_method_rejected(self, decision_straddling_curves):
        with pytest.raises(ValueError, match="Unknown method"):
            recommend_experiments(
                None, curves=decision_straddling_curves, method="vibes"
            )


class TestEvoiSurrogate:
    """The Gaussian (Raiffa–Schlaifer) surrogate that prices a GRID of design
    precisions from at most two anchored preposterior MCs — the cheap EVOI the
    experiment optimizer's net-value Pareto axis runs on."""

    @staticmethod
    def _roi_and_evpi(curves):
        g1 = int(np.argmin(np.abs(curves.multipliers - 1.0)))
        roi0 = curves.contributions[:, 0, g1] / curves.base_spend[0]
        return roi0, compute_evpi(curves)

    def test_two_anchor_fit_tracks_mc_inside_bracket(self, decision_straddling_curves):
        """Fitted at the sigma extremes, the surrogate reproduces the MC EVOI
        at an interior sigma to ~±20% — even on this bimodal posterior."""
        curves = decision_straddling_curves
        roi0, port = self._roi_and_evpi(curves)
        tau = float(roi0.std())
        sig_lo, sig_hi = tau / 2, 2 * tau
        rng = np.random.default_rng(5)
        od = (rng.integers(0, len(roi0), size=48), rng.standard_normal(48))
        anchors = [
            (s, compute_evoi_for_channel(curves, 0, roi0, float(s), outcome_draws=od))
            for s in (sig_lo, sig_hi)
        ]
        sur = fit_evoi_surrogate(tau, anchors)
        assert sur is not None
        # exact at the anchors by construction
        assert sur(sig_lo) == pytest.approx(anchors[0][1], rel=1e-6)
        assert sur(sig_hi) == pytest.approx(anchors[1][1], rel=1e-6)
        # interior point tracks a fresh MC evaluation
        mid = tau
        mc_mid = compute_evoi_for_channel(curves, 0, roi0, float(mid), outcome_draws=od)
        assert sur(mid) == pytest.approx(mc_mid, rel=0.25)

    def test_surrogate_monotone_in_precision(self, decision_straddling_curves):
        curves = decision_straddling_curves
        roi0, port = self._roi_and_evpi(curves)
        tau = float(roi0.std())
        anchors = [
            (s, compute_evoi_for_channel(curves, 0, roi0, float(s)))
            for s in (tau / 2, 2 * tau)
        ]
        sur = fit_evoi_surrogate(tau, anchors)
        vals = [sur(s) for s in (tau / 4, tau / 2, tau, 2 * tau, 4 * tau)]
        assert vals == sorted(vals, reverse=True)  # sharper design worth more
        assert all(v >= 0 for v in vals)

    def test_evpi_cap_applies(self):
        sur = fit_evoi_surrogate(1.0, [(0.5, 100.0), (2.0, 10.0)])
        assert sur(0.01, evpi=5.0) <= 5.0

    def test_degenerate_inputs_refuse_or_zero(self):
        assert fit_evoi_surrogate(0.0, [(0.5, 1.0)]) is None
        assert fit_evoi_surrogate(1.0, [(0.5, 0.0), (2.0, -1.0)]) is None
        assert fit_evoi_surrogate(1.0, []) is None
        # single anchor degrades to the delta=0 (sd-ratio) scaling
        sur = fit_evoi_surrogate(1.0, [(1.0, 3.0)])
        assert sur is not None and sur.delta == 0.0
        assert sur(1.0) == pytest.approx(3.0, rel=1e-9)
        one = surrogate_evoi(3.0, 1.0, 0.5, 1.0)
        assert sur(0.5) == pytest.approx(one, rel=1e-9)

    def test_single_anchor_helper_scales_by_sd_ratio(self):
        # at sigma_ref it returns the anchor; sharper > anchor; weaker < anchor
        assert surrogate_evoi(10.0, 0.2, 0.2, 0.5) == pytest.approx(10.0)
        assert surrogate_evoi(10.0, 0.2, 0.05, 0.5) > 10.0
        assert surrogate_evoi(10.0, 0.2, 0.8, 0.5) < 10.0
        assert surrogate_evoi(10.0, 0.2, 0.0, 0.5, evpi=11.0) <= 11.0
        assert surrogate_evoi(0.0, 0.2, 0.1, 0.5) == 0.0
