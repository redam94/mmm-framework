"""Tests for extrapolation flags + decision-confidence headline (issue #105).

The optimizer must (a) flag a recommendation that scales spend past the observed
range, (b) widen that channel's recommended-spend CI, and (c) expose an expected
regret so the report headline can lead with decision confidence. Report sections
must render the flag + confidence. Curves/optimizer are exercised with a
hand-built ``ResponseCurves`` (no MCMC).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from mmm_framework.planning.budget import (
    ResponseCurves,
    _result_to_report_dict,
    optimize_budget,
)


def _curves(obs_max=None, n_obs=10, seed=0):
    """Two channels on concave sqrt curves; A constant-spend, B spiky."""
    mults = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    base = np.array([100.0, 100.0])
    grid = base[:, None] * mults[None, :]
    betas = np.array([2.0, 2.2])
    rng = np.random.default_rng(seed)
    D, C = 60, 2
    contrib = np.zeros((D, C, len(mults)))
    for d in range(D):
        b = betas * (1 + 0.08 * rng.standard_normal(C))
        contrib[d] = b[:, None] * np.sqrt(grid)
    return ResponseCurves(
        ["A", "B"],
        mults,
        base,
        contrib,
        obs_max_spend=obs_max,
        n_obs=n_obs if obs_max is not None else None,
    )


class TestMaxObsMultiplier:
    def test_constant_channel_is_one(self):
        # A: constant per-obs spend (10 each) → max==mean → m_obs=1.
        rc = _curves(obs_max=np.array([10.0, 30.0]), n_obs=10)
        m = rc.max_obs_multiplier
        assert m is not None
        assert np.isclose(m[0], 1.0)
        assert np.isclose(m[1], 3.0)  # B: max 30, mean 10 → 3×

    def test_none_when_range_absent(self):
        rc = _curves(obs_max=None)
        assert rc.max_obs_multiplier is None


class TestOptimizerFlags:
    def _run(self, obs_max):
        rc = _curves(obs_max=np.array(obs_max), n_obs=10)
        # Force scale-up (budget 400 vs current 200) so both hit 2×.
        return optimize_budget(
            curves=rc, total_budget=400.0, max_multiplier=2.5, random_seed=0
        )

    def test_flags_and_ci_and_regret(self):
        res = self._run([10.0, 30.0])  # A m_obs=1 (extrapolates at 2×), B m_obs=3 (ok)
        t = res.table.set_index("channel")
        assert (
            t.loc["A", "within_observed_range"] is False
            or not t.loc["A", "within_observed_range"]
        )
        assert bool(t.loc["B", "within_observed_range"]) is True
        assert res.n_extrapolated == 1
        # Expected regret is a non-negative number.
        assert res.expected_regret >= 0.0
        # Recommended-spend CI present.
        assert "optimal_spend_p5" in res.table.columns
        assert t.loc["A", "optimal_spend_p5"] <= t.loc["A", "optimal_spend"]

    def test_extrapolated_ci_is_wider(self):
        # The extrapolated channel's CI is inflated relative to the same channel
        # when it stays within range.
        res_extrap = self._run([10.0, 30.0])
        res_within = self._run([60.0, 60.0])  # both m_obs=6 → within at 2×
        a_ex = res_extrap.table.set_index("channel").loc["A"]
        a_in = res_within.table.set_index("channel").loc["A"]
        width_ex = a_ex["optimal_spend_p95"] - a_ex["optimal_spend_p5"]
        width_in = a_in["optimal_spend_p95"] - a_in["optimal_spend_p5"]
        assert width_ex > width_in

    def test_all_within_range_when_ample_headroom(self):
        res = self._run([100.0, 100.0])  # m_obs=10 → nothing extrapolates
        assert res.n_extrapolated == 0
        assert all(res.table["within_observed_range"])

    def test_backcompat_no_range_all_within(self):
        rc = _curves(obs_max=None)
        res = optimize_budget(
            curves=rc, total_budget=400.0, max_multiplier=2.5, random_seed=0
        )
        # No observed-range info → every channel counted within-range, no crash.
        assert res.n_extrapolated == 0
        assert all(res.table["within_observed_range"])


class TestReportDict:
    def test_propagates_flags(self):
        rc = _curves(obs_max=np.array([10.0, 30.0]), n_obs=10)
        res = optimize_budget(
            curves=rc, total_budget=400.0, max_multiplier=2.5, random_seed=0
        )
        plan = _result_to_report_dict(res)
        assert "expected_regret" in plan and "n_extrapolated" in plan
        a = next(r for r in plan["allocation"] if r["channel"] == "A")
        assert "within_observed_range" in a
        assert "optimal_spend_p5" in a and "recommended_multiplier" in a


def _plan_payload(within_a=False):
    return {
        "total_budget": 400.0,
        "current_total": 200.0,
        "expected_uplift": 17.4,
        "uplift_hdi": [5.0, 30.0],
        "prob_positive_uplift": 0.82,
        "expected_regret": 0.11,
        "n_extrapolated": 0 if within_a else 1,
        "n_draws": 60,
        "allocation": [
            {
                "channel": "TV",
                "current_spend": 100,
                "optimal_spend": 200,
                "change_pct": 100.0,
                "within_observed_range": within_a,
                "recommended_multiplier": 2.0,
                "max_obs_multiplier": 1.0,
                "optimal_spend_p5": 120,
                "optimal_spend_p95": 200,
            },
            {
                "channel": "Search",
                "current_spend": 100,
                "optimal_spend": 150,
                "change_pct": 50.0,
                "within_observed_range": True,
                "recommended_multiplier": 1.5,
                "max_obs_multiplier": 3.0,
                "optimal_spend_p5": 140,
                "optimal_spend_p95": 165,
            },
        ],
        "notes": [],
    }


class TestClassicAllocationRender:
    def _html(self, plan):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        return MMMReportGenerator(
            data=MMMDataBundle(channel_names=["TV", "Search"]),
            config=ReportConfig(),
            allocation=plan,
        ).render()

    def test_confidence_headline_leads(self):
        html = self._html(_plan_payload())
        assert "chance this plan beats the current allocation" in html
        assert "Expected regret" in html

    def test_extrapolation_flag_in_table(self):
        html = self._html(_plan_payload(within_a=False))
        assert "Extrapolated" in html
        assert "In tested range" in html  # Search
        assert "Range" in html  # column header

    def test_recommended_spend_ci_rendered(self):
        html = self._html(_plan_payload())
        # The recommended-spend interval appears in the table.
        assert "120" in html and "200" in html


class TestAugurAllocationRender:
    def test_regret_and_extrapolation(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        cfg = dataclasses.replace(ReportConfig(), shell="augur")
        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["TV", "Search"]),
            config=cfg,
            allocation=_plan_payload(),
        ).render()
        assert "Chance it beats today" in html
        assert "Expected regret" in html
        assert "extrapolated" in html
