"""Tests for the Augur readout's data layer: the evidence-tier classifier and
the CMO/planner insights engine (deterministic, offline)."""

from __future__ import annotations

import pytest

from mmm_framework.reporting.extractors.bundle import MMMDataBundle
from mmm_framework.reporting.helpers.reallocation import (
    channel_rows,
    classify_tier,
    reallocation_groups,
    illustrative_flighting_totals,
    TIER_META,
)

# aliased so pytest does not collect the helper as a test case
from mmm_framework.reporting.helpers.reallocation import (
    test_candidates as rank_test_candidates,
)
from mmm_framework.reporting.insights import build_report_insights, report_facts

# Augur reference numbers (the scorecard the template ships with).
_ROI = {
    "Video": (1.52, 1.08, 2.08),
    "Print": (1.66, 0.46, 3.22),
    "Radio": (1.08, 0.31, 2.07),
    "Social": (0.83, 0.50, 1.23),
    "Display": (0.71, 0.36, 1.13),
    "Search": (0.68, 0.41, 1.00),
    "TV": (0.45, 0.35, 0.57),
}
_SPEND = {
    "Video": 4300,
    "Print": 2100,
    "Radio": 3200,
    "Social": 6100,
    "Display": 5000,
    "Search": 7600,
    "TV": 11300,
}
_MROAS = {
    "Video": 1.49,
    "Print": 0.70,
    "Radio": 0.48,
    "Social": 0.53,
    "Display": 0.54,
    "Search": 0.42,
    "TV": 0.57,
}
_EXPECTED_TIER = {
    "Video": "scale",
    "Print": "test",
    "Radio": "test",
    "Social": "hold",
    "Display": "hold",
    "Search": "reduce",
    "TV": "reduce",
}


def _bundle() -> MMMDataBundle:
    return MMMDataBundle(
        channel_roi={
            ch: {"mean": m, "lower": lo, "upper": hi}
            for ch, (m, lo, hi) in _ROI.items()
        },
        channel_spend=dict(_SPEND),
        channel_contribution={
            ch: {"mean": 5000.0, "lower": 3000.0, "upper": 8000.0} for ch in _ROI
        },
        current_spend=dict(_SPEND),
        blended_roi={"mean": 0.81, "lower": 0.44, "upper": 1.26},
        marketing_attributed_revenue={"mean": 32100, "lower": 17500, "upper": 49700},
        marketing_contribution_pct={"mean": 0.253, "lower": 0.138, "upper": 0.392},
        total_revenue=127000,
        fit_statistics={"r2": 0.91, "mape": 0.06},
        posterior_predictive={
            "r2": 0.91,
            "ci_level": 0.8,
            "coverage": [{"nominal": 0.8, "empirical": 0.82}],
        },
        estimands={
            f"marginal_roas:{ch}": {"mean": v, "status": "ok", "kind": "roi"}
            for ch, v in _MROAS.items()
        },
        channel_names=list(_ROI),
    )


# ── classifier ───────────────────────────────────────────────────────────────
class TestClassifier:
    def test_classify_tier_boundaries(self):
        assert classify_tier(1.5, 1.1, 2.0) == "scale"  # whole CI clears
        assert classify_tier(0.45, 0.35, 0.57) == "reduce"  # whole CI below
        assert classify_tier(1.66, 0.46, 3.22) == "test"  # high central, straddles
        assert classify_tier(0.83, 0.50, 1.23) == "hold"  # near break-even
        # break-even touch: upper exactly 1.0 -> reduce (not above)
        assert classify_tier(0.68, 0.41, 1.00) == "reduce"

    def test_rows_reproduce_augur_scorecard(self):
        rows = channel_rows(_bundle())
        got = {r["name"]: r["tier"] for r in rows}
        assert got == _EXPECTED_TIER

    def test_rows_ordered_by_priority(self):
        rows = channel_rows(_bundle())
        order = [r["name"] for r in rows]
        # scale first, then test, hold, reduce; within tier by ROI desc.
        assert order == ["Video", "Print", "Radio", "Social", "Display", "Search", "TV"]

    def test_groups_and_candidates(self):
        rows = channel_rows(_bundle())
        groups = reallocation_groups(rows)
        assert [r["name"] for r in groups["scale"]] == ["Video"]
        assert {r["name"] for r in groups["reduce"]} == {"Search", "TV"}
        cands = [r["name"] for r in rank_test_candidates(rows)]
        # widest relative CI first; Print (0.46-3.22) is the widest
        assert cands[0] == "Print"

    def test_marginal_roas_from_estimands(self):
        rows = {r["name"]: r for r in channel_rows(_bundle())}
        assert rows["Video"]["mroas"] == pytest.approx(1.49)
        assert rows["Search"]["mroas"] == pytest.approx(0.42)

    def test_spend_share(self):
        rows = {r["name"]: r for r in channel_rows(_bundle())}
        assert rows["TV"]["spend_share"] == pytest.approx(11300 / sum(_SPEND.values()))

    def test_tier_meta_complete(self):
        for tier in ("scale", "test", "hold", "reduce"):
            assert {"action", "read", "css", "color"} <= set(TIER_META[tier])

    def test_no_roi_rows_empty(self):
        assert channel_rows(MMMDataBundle()) == []

    def test_illustrative_flighting_preserves_budget(self):
        rows = channel_rows(_bundle())
        fl = illustrative_flighting_totals(rows, n_weeks=52)
        assert fl is not None
        total = sum(_SPEND.values())
        # current and recommended both sum to the modelled annual spend
        assert sum(fl["current_total"]) == pytest.approx(total, rel=1e-6)
        assert sum(fl["recommended_total"]) == pytest.approx(total, rel=1e-6)
        assert len(fl["weeks"]) == 52

    def test_flighting_deterministic(self):
        rows = channel_rows(_bundle())
        a = illustrative_flighting_totals(rows)
        b = illustrative_flighting_totals(rows)
        assert a["recommended_total"] == b["recommended_total"]

    def test_flighting_preserves_budget_without_scale_channel(self):
        # reduce + hold only (no scale): freed budget must redeploy to hold so
        # the recommended plan still sums to the same annual budget.
        b = MMMDataBundle(
            channel_roi={
                "Lose": {"mean": 0.5, "lower": 0.3, "upper": 0.7},  # reduce
                "Keep": {"mean": 0.9, "lower": 0.6, "upper": 1.2},  # hold
            },
            channel_spend={"Lose": 1000.0, "Keep": 500.0},
            channel_names=["Lose", "Keep"],
        )
        rows = channel_rows(b)
        fl = illustrative_flighting_totals(rows)
        assert sum(fl["recommended_total"]) == pytest.approx(1500.0, rel=1e-6)
        assert sum(fl["current_total"]) == pytest.approx(1500.0, rel=1e-6)

    def test_flighting_preserves_budget_all_reduce(self):
        # no recipients at all -> reduce channels are not trimmed (budget held)
        b = MMMDataBundle(
            channel_roi={
                "A": {"mean": 0.5, "lower": 0.3, "upper": 0.7},
                "B": {"mean": 0.4, "lower": 0.25, "upper": 0.6},
            },
            channel_spend={"A": 1000.0, "B": 800.0},
            channel_names=["A", "B"],
        )
        rows = channel_rows(b)
        fl = illustrative_flighting_totals(rows)
        assert sum(fl["recommended_total"]) == pytest.approx(1800.0, rel=1e-6)

    def test_spend_share_uses_current_spend_fallback(self):
        # a channel only present in current_spend must still count toward the
        # denominator, so shares sum to ~1.
        b = MMMDataBundle(
            channel_roi={
                "A": {"mean": 1.2, "lower": 1.05, "upper": 1.4},
                "B": {"mean": 0.8, "lower": 0.5, "upper": 1.1},
            },
            channel_spend={"A": 100.0},  # B only in current_spend
            current_spend={"B": 100.0},
            channel_names=["A", "B"],
        )
        rows = channel_rows(b)
        shares = [r["spend_share"] for r in rows]
        assert all(s is not None for s in shares)
        assert sum(shares) == pytest.approx(1.0, rel=1e-6)


# ── insights ─────────────────────────────────────────────────────────────────
class TestInsights:
    def test_fallback_complete_and_offline(self):
        ins = build_report_insights(_bundle(), llm=None)
        # every fixed slot present
        for slot in (
            "headline",
            "standfirst",
            "kpi_gloss",
            "fit_gloss",
            "tests_intro",
            "next_steps",
        ):
            assert ins.get(slot), slot
        # one "what to do" per channel
        for ch in _ROI:
            assert ins.get(f"channel:{ch}"), ch

    def test_grounded_numbers(self):
        ins = build_report_insights(_bundle(), llm=None)
        # blended ROI is a ratio -> $0.81, NOT rounded to $1
        assert "$0.81" in ins["standfirst"]
        assert "$1 " not in ins["standfirst"].replace("per $1", "")
        # marketing share grounded
        assert "25" in ins["standfirst"]
        # the scale winner and a reduce loser named in the headline
        assert "Video" in ins["headline"]

    def test_channel_prose_matches_tier(self):
        ins = build_report_insights(_bundle(), llm=None)
        assert "clears break-even" in ins["channel:Video"]  # scale
        assert "below break-even" in ins["channel:TV"]  # reduce
        assert "too wide to fund" in ins["channel:Print"]  # test

    def test_fit_gloss_uses_r2_and_coverage(self):
        ins = build_report_insights(_bundle(), llm=None)
        assert "0.91" in ins["fit_gloss"]
        assert "82%" in ins["fit_gloss"]

    def test_facts_structure(self):
        f = report_facts(_bundle())
        assert f["top"]["name"] == "Video"
        assert {r["name"] for r in f["reduce"]} == {"Search", "TV"}
        assert f["r2"] == pytest.approx(0.91)

    def test_minimal_bundle_does_not_crash(self):
        # only channel_roi present — fallback still returns a complete dict
        b = MMMDataBundle(
            channel_roi={"A": {"mean": 1.2, "lower": 1.05, "upper": 1.4}},
            channel_names=["A"],
        )
        ins = build_report_insights(b, llm=None)
        assert ins["headline"]
        assert ins["channel:A"]

    def test_empty_bundle_returns_dict(self):
        ins = build_report_insights(MMMDataBundle(), llm=None)
        assert isinstance(ins, dict)
        assert "headline" in ins  # generic fallback headline
