"""One KPI valuation, resolved once, with provenance (#215).

The behaviour under test is a refusal, not a calculation. Before v1.4 a missing
valuation resolved to ``1.0`` and the Planner's "Fund to breakeven" control
turned that into a budget recommendation ~1000x too large on a KPI denominated
in thousands — rendered with credible intervals. So the load-bearing assertions
here are that absence stays absent and that no code path invents a number.
"""

from __future__ import annotations

import pytest

from mmm_framework.finance import (
    KpiKind,
    KpiValuation,
    UnresolvedValueError,
    kpi_to_dollars,
)


class TestKpiValuation:
    def test_revenue_uses_margin_directly(self):
        assert KpiValuation(gross_margin=0.4).value_per_kpi() == pytest.approx(0.4)

    def test_units_multiply_margin_by_price(self):
        v = KpiValuation(kind=KpiKind.UNITS, gross_margin=0.6, price=10.0)
        assert v.value_per_kpi() == 6.0

    def test_units_without_price_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="price"):
            KpiValuation(kind=KpiKind.UNITS, gross_margin=0.6)

    def test_other_kind_is_never_convertible(self):
        assert (
            KpiValuation(kind=KpiKind.OTHER, gross_margin=0.4).value_per_kpi() is None
        )

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 40.0])
    def test_margin_is_a_fraction_not_a_percentage(self, bad):
        """`gross_margin=40` used to be accepted and multiply profit by 40.

        The pre-1.4 resolver guarded only `m <= 0`, and `save_preference`
        validated nothing but branding, so an agent could persist it.
        """
        with pytest.raises(ValueError):
            KpiValuation(gross_margin=bad)

    def test_scale_carries_the_kpi_denomination(self):
        """The ~1000x case: a KPI column denominated in thousands."""
        v = KpiValuation(gross_margin=0.4, scale=1000.0)
        assert v.value_per_kpi() == pytest.approx(400.0)

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValueError):
            KpiValuation(gross_margin=0.4, margin=0.4)


class TestPrecedence:
    def test_param_beats_everything(self):
        r = kpi_to_dollars(
            override={"gross_margin": 0.5},
            spec={"valuation": {"gross_margin": 0.4}},
            preferences={"economics": {"gross_margin": 0.3}},
        )
        assert (r.value_per_kpi, r.source) == (0.5, "param")

    def test_spec_beats_preferences(self):
        r = kpi_to_dollars(
            spec={"valuation": {"gross_margin": 0.4}},
            preferences={"economics": {"gross_margin": 0.3}},
        )
        assert (r.value_per_kpi, r.source) == (0.4, "spec")

    def test_preferences_beat_branding(self):
        r = kpi_to_dollars(
            preferences={"economics": {"gross_margin": 0.3}},
            branding={"economics": {"gross_margin": 0.2}},
        )
        assert (r.value_per_kpi, r.source) == (0.3, "preferences")

    def test_branding_is_the_last_resort(self):
        r = kpi_to_dollars(branding={"economics": {"gross_margin": 0.2}})
        assert (r.value_per_kpi, r.source) == (0.2, "branding")

    def test_units_resolve_through_the_chain(self):
        r = kpi_to_dollars(
            preferences={
                "economics": {"kind": "units", "gross_margin": 0.6, "price": 10.0}
            }
        )
        assert r.value_per_kpi == 6.0
        assert r.kind is KpiKind.UNITS


class TestUnresolved:
    def test_nothing_set_is_unresolved_not_one(self):
        """The whole point. Absence must not become 1.0."""
        r = kpi_to_dollars()
        assert r.value_per_kpi is None
        assert r.is_dollar is False
        assert r.source == "none"

    def test_empty_blobs_are_unresolved(self):
        r = kpi_to_dollars(spec={}, preferences={}, branding={"economics": {}})
        assert not r.is_dollar

    def test_require_raises_naming_the_decision(self):
        with pytest.raises(UnresolvedValueError, match="Fund-to-breakeven"):
            kpi_to_dollars().require("Fund-to-breakeven allocation")

    def test_require_returns_the_value_when_resolved(self):
        assert kpi_to_dollars(override={"gross_margin": 0.4}).require("x") == 0.4

    def test_explicit_other_stops_the_chain(self):
        """A declared non-monetary KPI is an ANSWER, not a gap.

        Falling through to a lower-precedence margin would contradict the
        explicit choice and silently produce dollars for a KPI the user said is
        not money.
        """
        r = kpi_to_dollars(
            spec={"valuation": {"kind": "other"}},
            preferences={"economics": {"gross_margin": 0.3}},
        )
        assert not r.is_dollar
        assert r.source == "spec"

    def test_invalid_stored_preference_is_skipped_with_a_warning(self):
        """One bad saved preference must not break every plan."""
        r = kpi_to_dollars(
            preferences={"economics": {"gross_margin": 40}},
            branding={"economics": {"gross_margin": 0.2}},
        )
        assert (r.value_per_kpi, r.source) == (0.2, "branding")
        assert r.warnings and "gross_margin is a fraction" in r.warnings[0]

    def test_invalid_everything_is_unresolved_not_a_crash(self):
        r = kpi_to_dollars(preferences={"economics": {"gross_margin": 40}})
        assert not r.is_dollar
        assert r.warnings


class TestResolvedValuePayload:
    def test_describe_states_the_source(self):
        r = kpi_to_dollars(override={"gross_margin": 0.4})
        assert "source=param" in r.describe()

    def test_describe_says_why_when_unresolved(self):
        assert "Not dollar-denominated" in kpi_to_dollars().describe()
        r = kpi_to_dollars(override={"kind": "other"})
        assert "not denominated in money" in r.describe()

    def test_to_dict_carries_provenance(self):
        d = kpi_to_dollars(override={"gross_margin": 0.4}).to_dict()
        assert d["value_per_kpi"] == 0.4
        assert d["source"] == "param" and d["is_dollar"] is True
        assert d["kind"] == "revenue" and d["currency"] == "USD"


def test_module_is_lean_core():
    """Planning, agents and the server all import this; keep it dependency-free."""
    import sys

    before = set(sys.modules)
    import importlib

    importlib.reload(importlib.import_module("mmm_framework.finance.valuation"))
    newly = set(sys.modules) - before
    banned = {"fastapi", "langchain", "langgraph", "httpx", "numpy", "pandas"}
    assert not (banned & {m.split(".")[0] for m in newly})
