"""One break-even across every artifact (issue #221).

The v1.3 analogue was #188: a number rendered with the wrong semantics across a
reporting stack. Here two artifacts from ONE fitted model gave opposite
instructions — `deck/engine.py` computed `1/margin` while the Augur HTML took
`channel_rows`' default of 1.0, so at margin 0.4 a channel was tiered Scale in
the report and Reduce in the deck, with no cross-reference.
"""

from __future__ import annotations

import pytest

from mmm_framework.reporting.helpers.measurement import (
    BreakEven,
    resolve_break_even,
)
from mmm_framework.reporting.helpers.reallocation import classify_tier


class TestResolveBreakEven:
    def test_no_margin_is_revenue_basis_at_one(self):
        be = resolve_break_even(None)
        assert (be.value, be.basis) == (1.0, "revenue")
        assert not be.is_profit_basis
        assert be.disclosure() == ""  # nothing to disclose

    def test_margin_moves_the_reference_not_the_number(self):
        be = resolve_break_even(0.4)
        assert be.value == pytest.approx(2.5)
        assert be.basis == "profit"
        assert be.margin == pytest.approx(0.4)

    def test_disclosure_names_the_margin_and_its_source(self):
        d = resolve_break_even(0.4, value_source="project economics").disclosure()
        assert "2.50" in d and "40%" in d
        assert "project economics" in d
        assert "constant gross margin" in d  # the assumption, stated

    def test_a_percentage_is_refused_not_silently_used(self):
        """margin=40 would give a break-even of 0.025 and tier everything
        Scale."""
        with pytest.raises(ValueError, match="fraction"):
            resolve_break_even(40)
        with pytest.raises(ValueError):
            resolve_break_even(0.0)


class TestDeckAndHtmlAgree:
    """The issue's own acceptance criterion: a channel whose tier flips at
    margin 0.4 must be tiered the SAME by both artifacts."""

    # mean 1.8 clears a 1.0 bar and fails a 2.5 one
    ROI = (1.8, 1.2, 2.4)

    def _augur_break_even(self, margin):
        from mmm_framework.reporting.augur_sections import AugurSection
        from mmm_framework.reporting.config import ReportConfig

        bundle = type("B", (), {"cfo": {"margin": margin}})()
        return AugurSection(data=bundle, config=ReportConfig()).break_even().value

    def _deck_break_even(self, margin):
        # mirrors deck/engine.py's resolution
        return resolve_break_even(margin, default=1.0).value

    @pytest.mark.parametrize("margin", [None, 0.4, 0.65, 1.0])
    def test_both_surfaces_resolve_the_same_break_even(self, margin):
        assert self._augur_break_even(margin) == pytest.approx(
            self._deck_break_even(margin)
        )

    def test_the_tier_that_used_to_disagree_now_agrees(self):
        margin = 0.4
        html_tier = classify_tier(*self.ROI, self._augur_break_even(margin))
        deck_tier = classify_tier(*self.ROI, self._deck_break_even(margin))
        assert html_tier == deck_tier == "reduce"

        # and the pre-fix behaviour is what this guards against
        stale_html_tier = classify_tier(*self.ROI, 1.0)
        assert stale_html_tier == "scale" and stale_html_tier != deck_tier

    def test_a_malformed_margin_does_not_desync_the_two(self):
        """A bad margin must not leave one surface on profit basis and the
        other on revenue."""
        assert self._augur_break_even(40) == pytest.approx(1.0)


class TestProfitBasisIsDisclosed:
    def test_banner_present_only_on_a_profit_basis(self):
        from mmm_framework.reporting.augur_sections import AugurHeadlineSection
        from mmm_framework.reporting.config import ReportConfig

        def banner(margin):
            bundle = type("B", (), {"cfo": {"margin": margin}})()
            sec = AugurHeadlineSection(data=bundle, config=ReportConfig())
            return sec._basis_banner()

        assert banner(None) == ""
        out = banner(0.4)
        assert "40%" in out and "2.50" in out


class TestMetricMetaBasis:
    def test_basis_defaults_to_revenue_so_numbers_are_unchanged(self):
        from mmm_framework.reporting.helpers.measurement import MetricMeta
        from mmm_framework.config.enums import MeasurementUnit

        m = MetricMeta(
            unit=MeasurementUnit.SPEND,
            is_monetary=True,
            cost_basis=None,
            roi_label="ROI",
            marginal_label="Marginal ROAS",
            value_units="ROI",
            divisor_units="$",
            reference=1.0,
        )
        assert m.basis == "revenue"
        assert m.value_per_kpi is None and m.value_source is None
        d = m.to_dict()
        assert d["basis"] == "revenue"
        # the definition travels with the number
        assert {"reference", "basis", "value_per_kpi", "value_source"} <= set(d)


def test_break_even_is_carried_not_a_bare_float():
    """The assumption must travel with the value, so any surface rendering a
    non-1.0 reference can name where it came from."""
    be = resolve_break_even(0.5, value_source="explicit")
    assert isinstance(be, BreakEven)
    assert (be.value, be.margin, be.value_source) == (2.0, 0.5, "explicit")


class TestMaskedSumRefusesInsteadOfWideningTheWindow:
    """`_masked_sum` returned the FULL-series sum on a dtype/length mismatch.

    That is not a tolerant fallback, it is a wrong divisor: a windowed ROI
    silently divided by every period's spend instead of the window's,
    understating the metric with no error. Every masked ROI rides it —
    windowed marginal ROAS, the interactive per-period divisor, `analysis.py`.
    """

    def _fn(self):
        from mmm_framework.reporting.helpers.measurement import _masked_sum

        return _masked_sum

    def test_correct_mask_still_windows(self):
        import numpy as np

        s = np.array([1.0, 2.0, 3.0, 4.0])
        m = np.array([True, False, True, False])
        assert self._fn()(s, m) == pytest.approx(4.0)

    def test_none_mask_sums_everything(self):
        import numpy as np

        assert self._fn()(np.array([1.0, 2.0]), None) == pytest.approx(3.0)

    def test_length_mismatch_raises_naming_both_shapes(self):
        import numpy as np

        with pytest.raises(ValueError) as exc:
            self._fn()(np.array([1.0, 2.0, 3.0]), np.array([True, False]), channel="TV")
        msg = str(exc.value)
        assert "2" in msg and "3" in msg and "TV" in msg

    def test_non_boolean_mask_raises(self):
        import numpy as np

        with pytest.raises(ValueError, match="boolean"):
            self._fn()(np.array([1.0, 2.0]), np.array([1, 0]), channel="TV")

    def test_the_old_behaviour_would_have_been_wrong(self):
        """Pins WHY this raises: the fallback returned 10.0 (full series) where
        the window is 3.0 — a 3.3x understatement of any ROI using it."""
        import numpy as np

        s = np.array([1.0, 2.0, 3.0, 4.0])
        full = float(s.sum())
        assert full == 10.0
        with pytest.raises(ValueError):
            self._fn()(s, np.array([True, True, False]), channel="TV")


class TestClassicReportDisclosesItsMargin:
    """An artifact carrying a profit-basis number must name its assumption.

    The classic CFO note was inverted: empty exactly when a margin WAS given,
    so a rendered "Profit at risk" column never said what margin produced it.
    """

    def _html(self, margin):
        from mmm_framework.reporting.config import ReportConfig
        from mmm_framework.reporting.sections import CFOSection

        cfo = {
            "kpi_total": 5000.0,
            "marketing_contribution": {
                "mean": 1200.0,
                "lower": 1000.0,
                "upper": 1400.0,
            },
            "base_contribution": 3800.0,
            "marketing_pct": 0.24,
            "margin": margin,
            "hdi_prob": 0.9,
            "spend_cuts": [
                {
                    "cut_pct": 0.1,
                    "revenue_at_risk": 100.0,
                    "revenue_lower": 80.0,
                    "revenue_upper": 120.0,
                    "pct_of_kpi": 0.02,
                    **({"profit_at_risk": 40.0} if margin else {}),
                },
            ],
        }
        bundle = type("B", (), {"cfo": cfo})()
        return CFOSection(data=bundle, config=ReportConfig()).render()

    def test_names_the_margin_and_the_assumption(self):
        html = self._html(0.4)
        assert "Profit at risk" in html  # the profit number is rendered
        assert "40%" in html  # ...and its margin is named
        assert "constant gross margin" in html  # ...and the assumption stated

    def test_without_a_margin_it_asks_for_one(self):
        html = self._html(None)
        assert "Provide a gross margin" in html
        assert "constant gross margin" not in html
