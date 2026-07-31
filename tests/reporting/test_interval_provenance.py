"""One estimand, two renders, each stating its interval (#277).

The classic report renders `contribution_roi` twice and both sections are
default-on: `ChannelROISection` at an **80% equal-tailed** interval,
`EstimandsSection` at a **94% true HDI**. Neither said so, and a reader had no
way to tell they were the same quantity — or that the visibly narrower one is
not the more precise estimate but the same posterior at a lower mass under a
different definition.

The decision taken: keep both renders and label each with its mass AND its
definition, sourced from provenance that travels with the number rather than
from a literal at the render site. The two defaults, previously set 1,500 lines
apart in different packages, are now stated together in `estimands.spec`.
"""

from __future__ import annotations

import re

import pytest

from mmm_framework.estimands.spec import (
    DASHBOARD_INTERVAL_MASS,
    ESTIMAND_INTERVAL_MASS,
    INTERVAL_KIND_BY_HDI_METHOD,
    INTERVAL_KIND_ETI,
    INTERVAL_KIND_HDI,
    EstimandResult,
    Realization,
    interval_label,
)
from mmm_framework.reporting.config import ReportConfig, SectionConfig
from mmm_framework.reporting.extractors.bundle import MMMDataBundle
from mmm_framework.reporting.sections import ChannelROISection, EstimandsSection


def _headers(html: str) -> list[str]:
    return [h[4:-5] for h in re.findall(r"<th>[^<]*</th>", html)]


class TestTheDefaultsAreStatedTogether:
    def test_both_masses_live_in_one_module(self):
        assert ESTIMAND_INTERVAL_MASS == 0.94
        assert DASHBOARD_INTERVAL_MASS == 0.80

    def test_the_section_default_reads_the_shared_constant(self):
        assert SectionConfig().credible_interval == DASHBOARD_INTERVAL_MASS

    def test_the_estimand_default_reads_the_shared_constant(self):
        assert EstimandResult(name="x").hdi_prob == ESTIMAND_INTERVAL_MASS

    def test_the_divergence_is_documented_where_it_is_defined(self):
        import inspect

        from mmm_framework.estimands import spec

        src = inspect.getsource(spec)
        i = src.index("ESTIMAND_INTERVAL_MASS: float")
        preamble = src[max(0, i - 1200) : i]
        assert "DASHBOARD_INTERVAL_MASS" in preamble, (
            "the two defaults must be documented together, not cross-referenced"
        )


class TestIntervalLabel:
    @pytest.mark.parametrize(
        "mass,kind,want",
        [
            (0.94, INTERVAL_KIND_HDI, "94% HDI"),
            (0.8, INTERVAL_KIND_ETI, "80% ETI"),
            (0.9, INTERVAL_KIND_ETI, "90% ETI"),
        ],
    )
    def test_mass_and_definition(self, mass, kind, want):
        assert interval_label(mass, kind) == want

    @pytest.mark.parametrize("kind", [None, ""])
    def test_unknown_definition_yields_no_label(self, kind):
        """A mass with no definition is exactly what this issue is about."""
        assert interval_label(0.94, kind) == ""

    def test_unknown_mass_yields_no_label(self):
        assert interval_label(None, INTERVAL_KIND_HDI) == ""

    def test_only_az_hdi_is_a_true_hdi(self):
        """`compute_hdi_bounds` is percentile-based despite its name."""
        assert INTERVAL_KIND_BY_HDI_METHOD["az_hdi"] == INTERVAL_KIND_HDI
        assert INTERVAL_KIND_BY_HDI_METHOD["percentile"] == INTERVAL_KIND_ETI
        assert INTERVAL_KIND_BY_HDI_METHOD["finite_percentile"] == INTERVAL_KIND_ETI

    def test_every_realization_method_is_mapped(self):
        methods = Realization.model_fields["hdi_method"].annotation.__args__
        assert set(methods) <= set(INTERVAL_KIND_BY_HDI_METHOD)


class TestProvenanceTravelsWithTheNumber:
    def test_the_result_carries_its_definition(self):
        from mmm_framework.estimands.registry import get as get_estimand

        assert get_estimand("contribution_roi").realization.hdi_method == "az_hdi"
        assert get_estimand("marginal_roas").realization.hdi_method == (
            "finite_percentile"
        )

    def test_default_result_kind_is_equal_tailed(self):
        assert EstimandResult(name="x").interval_kind == INTERVAL_KIND_ETI


class TestBothRendersAreLabelled:
    @staticmethod
    def _roi_bundle(mass, kind):
        b = MMMDataBundle()
        b.channel_names = ["TV"]
        b.channel_roi = {
            "TV": {
                "mean": 1.93,
                "lower": 1.71,
                "upper": 2.14,
                "reference": 1.0,
                "is_monetary": True,
                "value_units": "ROI",
                "interval_mass": mass,
                "interval_kind": kind,
            }
        }
        return b

    @staticmethod
    def _estimand_bundle(mass, kind):
        b = MMMDataBundle()
        b.channel_names = ["TV"]
        b.estimands = {
            "contribution_roi:TV": {
                "mean": 1.93,
                "lower": 1.62,
                "upper": 2.25,
                "kind": "roi",
                "units": "",
                "hdi_prob": mass,
                "interval_mass": mass,
                "interval_kind": kind,
            }
        }
        return b

    def test_channel_roi_states_eti_and_its_mass(self):
        html = ChannelROISection(
            data=self._roi_bundle(DASHBOARD_INTERVAL_MASS, INTERVAL_KIND_ETI),
            config=ReportConfig(),
            section_config=SectionConfig(enabled=True),
        ).render()
        assert "80% ETI" in _headers(html)

    def test_estimands_states_hdi_and_its_mass(self):
        html = EstimandsSection(
            data=self._estimand_bundle(ESTIMAND_INTERVAL_MASS, INTERVAL_KIND_HDI),
            config=ReportConfig(),
            section_config=SectionConfig(enabled=True),
        ).render()
        assert "94% HDI" in _headers(html)

    def test_a_reader_can_tell_the_two_apart(self):
        """The point of the change: both are present and distinguishable."""
        roi = ChannelROISection(
            data=self._roi_bundle(DASHBOARD_INTERVAL_MASS, INTERVAL_KIND_ETI),
            config=ReportConfig(),
            section_config=SectionConfig(enabled=True),
        ).render()
        est = EstimandsSection(
            data=self._estimand_bundle(ESTIMAND_INTERVAL_MASS, INTERVAL_KIND_HDI),
            config=ReportConfig(),
            section_config=SectionConfig(enabled=True),
        ).render()

        assert "80% ETI" in roi and "94% HDI" not in roi
        assert "94% HDI" in est and "80% ETI" not in est
        # The same point estimate in both, which is the reader's real problem.
        assert "1.93" in roi and "1.93" in est

    def test_the_label_follows_the_data_not_the_section_config(self):
        """A section configured at 80% but handed 90% data says 90%."""
        html = ChannelROISection(
            data=self._roi_bundle(0.90, INTERVAL_KIND_ETI),
            config=ReportConfig(),
            section_config=SectionConfig(enabled=True, credible_interval=0.80),
        ).render()
        assert "90% ETI" in _headers(html)

    @pytest.mark.parametrize("section", ["roi", "estimands"])
    def test_a_bundle_without_provenance_stays_neutral(self, section):
        """No definition in the data means none is asserted in the render."""
        if section == "roi":
            b = self._roi_bundle(DASHBOARD_INTERVAL_MASS, INTERVAL_KIND_ETI)
            b.channel_roi["TV"].pop("interval_kind")
            b.channel_roi["TV"].pop("interval_mass")
            html = ChannelROISection(
                data=b,
                config=ReportConfig(),
                section_config=SectionConfig(enabled=True),
            ).render()
            want = "80% CI"
        else:
            b = self._estimand_bundle(ESTIMAND_INTERVAL_MASS, INTERVAL_KIND_HDI)
            b.estimands["contribution_roi:TV"].pop("interval_kind")
            b.estimands["contribution_roi:TV"].pop("interval_mass")
            html = EstimandsSection(
                data=b,
                config=ReportConfig(),
                section_config=SectionConfig(enabled=True),
            ).render()
            want = "94% CI"

        headers = _headers(html)
        assert want in headers
        assert not any(h.endswith(("ETI", "HDI")) for h in headers)
