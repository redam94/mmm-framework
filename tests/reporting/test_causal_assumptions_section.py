"""Tests for the Causal Assumptions report section (P0-2 reporting half)."""

from dataclasses import replace

from mmm_framework.reporting import MMMReportGenerator, ReportConfig
from mmm_framework.reporting.config import SectionConfig
from mmm_framework.reporting.extractors.bundle import MMMDataBundle


def _render(bundle, config=None):
    return MMMReportGenerator(data=bundle, config=config or ReportConfig()).render()


class TestCausalAssumptionsSection:
    def test_caveat_always_renders(self):
        # Even with no causal metadata, the honest-framing caveat must appear.
        html = _render(MMMDataBundle(channel_names=["TV"]))
        assert "Causal Assumptions" in html
        assert "no unobserved confounding" in html.lower()
        assert "unobserved demand" in html.lower()
        assert "SUTVA" in html

    def test_robustness_table_renders_when_present(self):
        bundle = MMMDataBundle(
            channel_names=["TV", "Digital"],
            causal_assumptions={
                "identification_strategy": "Backdoor adjustment on Seasonality, Price.",
                "assumed_confounders": ["Seasonality", "Price"],
                "robustness": {
                    "channels": [
                        {
                            "channel": "TV",
                            "robustness_value": 0.42,
                            "partial_r2": 0.30,
                            "is_fragile": False,
                        },
                        {
                            "channel": "Digital",
                            "robustness_value": 0.05,
                            "partial_r2": 0.02,
                            "is_fragile": True,
                        },
                    ],
                    "caveat": "OLS-analogy robustness value.",
                },
            },
        )
        html = _render(bundle)
        assert "Robustness to Unobserved Confounding" in html
        assert "Backdoor adjustment" in html
        assert "Seasonality" in html
        assert "Fragile" in html and "Robust" in html

    def test_section_can_be_disabled(self):
        cfg = replace(ReportConfig(), causal_assumptions=SectionConfig(enabled=False))
        html = _render(MMMDataBundle(channel_names=["TV"]), cfg)
        assert "Identification rests on assumptions" not in html
