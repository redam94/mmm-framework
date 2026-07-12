"""CFO one-pager — P&L rollup + spend-cut revenue/profit-at-risk (issue #108)."""

from __future__ import annotations

import dataclasses

import numpy as np

from mmm_framework.agents import model_ops as M
from mmm_framework.reporting.helpers.cfo import cfo_facts


class FakeMMM:
    """Additive model: contribution ∝ sqrt(spend) with small posterior noise."""

    channel_names = ["TV", "Search"]
    X_media_raw = np.full((10, 2), 100.0)
    y_raw = np.full(10, 500.0)  # total KPI = 5000

    def sample_channel_contributions(
        self, X_media=None, max_draws=None, random_seed=None
    ):
        X = self.X_media_raw if X_media is None else X_media
        base = np.sqrt(np.clip(X, 0, None))  # (obs, C)
        rng = np.random.RandomState(random_seed or 0)
        return base[None, :, :] * (1 + 0.03 * rng.randn(50, 1, 1))


class TestCfoFacts:
    def test_rollup_and_cuts(self):
        f = cfo_facts(FakeMMM(), margin=0.4, cut_levels=(0.1, 0.25, 0.5))
        assert f["kpi_total"] == 5000.0
        mc = f["marketing_contribution"]
        assert mc["lower"] <= mc["mean"] <= mc["upper"]
        # base = total - marketing
        assert abs(f["base_contribution"] - (f["kpi_total"] - mc["mean"])) < 1e-6
        # deeper cuts put more at risk; profit = revenue * margin
        cuts = f["spend_cuts"]
        assert [c["cut_pct"] for c in cuts] == [0.1, 0.25, 0.5]
        assert cuts[0]["revenue_at_risk"] < cuts[2]["revenue_at_risk"]
        assert abs(cuts[0]["profit_at_risk"] - cuts[0]["revenue_at_risk"] * 0.4) < 1e-6
        # at-risk CI is ordered
        assert (
            cuts[0]["revenue_lower"]
            <= cuts[0]["revenue_at_risk"]
            <= cuts[0]["revenue_upper"]
        )

    def test_no_margin_omits_profit(self):
        f = cfo_facts(FakeMMM(), margin=None, cut_levels=(0.25,))
        assert "profit_at_risk" not in f["spend_cuts"][0]
        assert f["margin"] is None


class TestCfoOp:
    def test_op_payload_and_table(self):
        res = M.OPS["cfo_summary"](FakeMMM(), margin=0.4, kpi_name="revenue")
        assert res["error"] is None
        assert "cfo" in res["dashboard"]
        assert res["tables"] and len(res["tables"][0]["rows"]) == 3
        assert "Profit at risk" in res["content"]

    def test_op_in_registry(self):
        assert "cfo_summary" in M.OPS

    def test_op_error_as_data_on_bad_model(self):
        res = M.OPS["cfo_summary"](object())
        assert res["error"] and res["content"] is None and res["dashboard"] == {}


_CFO = {
    "kpi_total": 5000.0,
    "marketing_contribution": {"mean": 1200.0, "lower": 1000.0, "upper": 1400.0},
    "base_contribution": 3800.0,
    "marketing_pct": 0.24,
    "margin": 0.4,
    "hdi_prob": 0.9,
    "spend_cuts": [
        {
            "cut_pct": 0.1,
            "revenue_at_risk": 120.0,
            "revenue_lower": 100.0,
            "revenue_upper": 140.0,
            "pct_of_kpi": 0.024,
            "profit_at_risk": 48.0,
        },
        {
            "cut_pct": 0.5,
            "revenue_at_risk": 560.0,
            "revenue_lower": 500.0,
            "revenue_upper": 620.0,
            "pct_of_kpi": 0.112,
            "profit_at_risk": 224.0,
        },
    ],
}


def _render(cfo=None, shell="classic"):
    from mmm_framework.reporting import MMMReportGenerator, ReportConfig
    from mmm_framework.reporting.extractors.bundle import MMMDataBundle

    cfg = dataclasses.replace(ReportConfig(), shell=shell)
    return MMMReportGenerator(
        data=MMMDataBundle(channel_names=["TV"], model_kind="mmm"),
        config=cfg,
        cfo=cfo,
    ).render()


class TestCfoSections:
    def test_classic_renders(self):
        html = _render(_CFO)
        assert "CFO one-pager" in html
        assert "Revenue at risk" in html
        assert "Profit at risk" in html  # margin present

    def test_classic_absent_without_data(self):
        assert "CFO one-pager" not in _render(None)

    def test_augur_renders(self):
        html = _render(_CFO, shell="augur")
        assert "P&amp;L view" in html or "P&L view" in html
        assert "Revenue at risk" in html

    def test_augur_absent_without_data(self):
        html = _render(None, shell="augur")
        assert "P&amp;L view" not in html and "P&L view" not in html


def test_extractor_fills_cfo_for_mmm():
    """The bayesian extractor mixin auto-fills bundle.cfo for a model exposing the
    channel response surface (best-effort)."""
    from mmm_framework.reporting.extractors.bundle import MMMDataBundle
    from mmm_framework.reporting.extractors.mixins import EstimandPPCMixin

    class _X(EstimandPPCMixin):
        def _estimand_model(self):
            return FakeMMM()

    bundle = MMMDataBundle(channel_names=["TV", "Search"])
    _X()._extract_cfo(bundle)
    assert bundle.cfo is not None
    assert bundle.cfo["kpi_total"] == 5000.0
