"""Tests for the in-flight pacing / actual-vs-plan loop (issue #107)."""

from __future__ import annotations

import numpy as np

from mmm_framework.planning.budget import ResponseCurves
from mmm_framework.planning.pacing import (
    compute_pacing,
    expected_outcome_delta,
    pacing_report,
)


class TestComputePacing:
    def test_schedule_divergence_and_flags(self):
        planned = {
            "schedule": [
                {"period": "W1", "TV": 100, "Search": 50},
                {"period": "W2", "TV": 100, "Search": 50},
                {"period": "W3", "TV": 100, "Search": 50},
            ]
        }
        actual = {
            "schedule": [
                {"period": "W1", "TV": 130, "Search": 48},
                {"period": "W2", "TV": 125, "Search": 52},
            ]
        }
        res = compute_pacing(planned, actual, threshold=0.1)
        # Only the 2 elapsed periods are compared (plan truncated).
        assert res.periods == ["W1", "W2"]
        by = {c.channel: c for c in res.channels}
        assert by["TV"].planned == 200 and by["TV"].actual == 255
        assert by["TV"].status == "over-pacing"
        assert by["Search"].status == "on-track"
        assert res.flagged == ["TV"]

    def test_under_pacing(self):
        res = compute_pacing({"TV": 100}, {"TV": 60}, threshold=0.1)
        assert res.channels[0].status == "under-pacing"
        assert res.channels[0].divergence_pct < 0

    def test_not_started_when_both_zero(self):
        # Not started = nothing planned in this window AND nothing spent.
        res = compute_pacing(
            {"TV": 0, "Radio": 100}, {"TV": 0, "Radio": 100}, threshold=0.1
        )
        by = {c.channel: c for c in res.channels}
        assert by["TV"].status == "not-started"
        assert "TV" not in res.flagged

    def test_zero_delivery_against_plan_is_under_pacing(self):
        res = compute_pacing({"TV": 100}, {"TV": 0}, threshold=0.1)
        assert res.channels[0].status == "under-pacing"
        assert "TV" in res.flagged

    def test_totals_input(self):
        res = compute_pacing(
            {"TV": 100, "Search": 50}, {"TV": 120, "Search": 45}, threshold=0.1
        )
        assert res.planned_total == 150
        assert res.actual_total == 165
        assert res.flagged == ["TV"]  # +20% over; Search -10% within

    def test_list_of_rows_input(self):
        res = compute_pacing(
            [{"period": "W1", "TV": 100}],
            [{"period": "W1", "TV": 150}],
            threshold=0.1,
        )
        assert res.channels[0].status == "over-pacing"

    def test_per_channel_arrays(self):
        res = compute_pacing({"TV": [50, 50]}, {"TV": [70, 40]}, threshold=0.05)
        assert res.channels[0].planned == 100
        assert res.channels[0].actual == 110

    def test_portfolio_divergence(self):
        res = compute_pacing({"A": 100, "B": 100}, {"A": 120, "B": 90}, threshold=0.5)
        assert abs(res.divergence_pct - 0.05) < 1e-9  # (210-200)/200


def _curves():
    mults = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    base = np.array([300.0, 150.0])
    grid = base[:, None] * mults[None, :]
    betas = np.array([2.0, 2.2])
    rng = np.random.default_rng(0)
    D = 60
    contrib = np.zeros((D, 2, len(mults)))
    for d in range(D):
        contrib[d] = (betas * (1 + 0.08 * rng.standard_normal(2)))[:, None] * np.sqrt(
            grid
        )
    return ResponseCurves(["TV", "Search"], mults, base, contrib)


class TestOutcomeDelta:
    def test_over_delivery_positive_delta(self):
        rc = _curves()
        od = expected_outcome_delta(
            rc, {"TV": 300, "Search": 150}, {"TV": 360, "Search": 150}
        )
        assert od is not None
        assert od["mean"] > 0  # spending more on a saturating curve still adds
        assert od["lower"] < od["upper"]

    def test_under_delivery_negative_delta(self):
        rc = _curves()
        od = expected_outcome_delta(
            rc, {"TV": 300, "Search": 150}, {"TV": 200, "Search": 150}
        )
        assert od["mean"] < 0

    def test_none_without_curves(self):
        assert expected_outcome_delta(object(), {}, {}) is None


class TestPacingReport:
    def test_combines_pacing_and_delta(self):
        rc = _curves()
        planned = {"schedule": [{"period": "W1", "TV": 300, "Search": 150}]}
        actual = {"schedule": [{"period": "W1", "TV": 360, "Search": 150}]}
        res = pacing_report(planned, actual, curves=rc, threshold=0.1)
        assert res.flagged == ["TV"]
        assert res.outcome_delta is not None
        assert res.outcome_delta["mean"] > 0

    def test_no_curves_no_delta(self):
        res = pacing_report({"TV": 100}, {"TV": 120}, threshold=0.1)
        assert res.outcome_delta is None


class TestReportSection:
    def _bundle(self):
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        res = compute_pacing(
            {
                "schedule": [
                    {"period": "W1", "TV": 100, "Search": 50},
                    {"period": "W2", "TV": 100, "Search": 50},
                ]
            },
            {"schedule": [{"period": "W1", "TV": 130, "Search": 48}]},
            threshold=0.1,
        )
        res.outcome_delta = {"mean": 3.7, "lower": 3.3, "upper": 4.2}
        return MMMDataBundle(channel_names=["TV", "Search"], pacing=res.to_dict())

    def test_section_renders(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig

        html = MMMReportGenerator(data=self._bundle(), config=ReportConfig()).render()
        assert "In-flight pacing" in html
        assert "Off-pace" in html
        assert "Over-pacing" in html
        assert "Pacing by channel" in html
        assert "Expected KPI impact" in html

    def test_section_via_generator_param(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        res = compute_pacing({"TV": 100}, {"TV": 120}, threshold=0.1)
        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["TV"]),
            config=ReportConfig(),
            pacing=res.to_dict(),
        ).render()
        assert "In-flight pacing" in html

    def test_section_absent_without_data(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["TV"]), config=ReportConfig()
        ).render()
        assert "In-flight pacing" not in html


class TestModelOp:
    def test_op_payload_and_guard(self):
        from mmm_framework.agents.model_ops import OPS

        op = OPS["check_pacing"]
        planned = {"schedule": [{"period": "W1", "TV": 100, "Search": 50}]}
        actual = {"schedule": [{"period": "W1", "TV": 130, "Search": 48}]}
        res = op(None, planned=planned, actual=actual, threshold=0.1)
        assert res["error"] is None
        assert res["dashboard"]["pacing"]["flagged"] == ["TV"]
        assert "tables" in res
        # Missing inputs → clear error, no crash.
        assert op(None, planned=None, actual=actual)["error"]
