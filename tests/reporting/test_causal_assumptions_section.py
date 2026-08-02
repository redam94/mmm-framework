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

    def test_non_finite_robustness_value_is_not_reported_as_robust(self):
        """A NaN RV must never render as a green "Robust" verdict.

        `is_fragile` is `isfinite(rv) and rv < threshold`, so a NaN is *not
        fragile* and used to fall through to the Robust branch while the value
        column printed the literal "nan". Approximate (MAP/ADVI) fits produce
        exactly this, so the report asserted a passed sensitivity check that
        had never been computed.
        """
        bundle = MMMDataBundle(
            channel_names=["TV"],
            causal_assumptions={
                "robustness": {
                    "channels": [
                        {
                            "channel": "TV",
                            "robustness_value": float("nan"),
                            "partial_r2": float("nan"),
                            "is_fragile": False,
                        },
                    ],
                    "caveat": "OLS-analogy robustness value.",
                },
            },
        )
        html = _render(bundle)
        assert "Robustness to Unobserved Confounding" in html
        assert ">nan<" not in html.lower()  # no literal NaN in a cell
        assert "Not assessable" in html
        # the row must not be labelled Robust
        assert ">Robust<" not in html
        # and the reader is told what to do about it
        assert "re-fit with nuts" in html.lower()

    def test_finite_robustness_value_still_reads_robust(self):
        """The fix must not make every channel unassessable."""
        bundle = MMMDataBundle(
            channel_names=["TV"],
            causal_assumptions={
                "robustness": {
                    "channels": [
                        {
                            "channel": "TV",
                            "robustness_value": 0.42,
                            "partial_r2": 0.30,
                            "is_fragile": False,
                        },
                    ],
                    "caveat": "OLS-analogy robustness value.",
                },
            },
        )
        html = _render(bundle)
        assert ">Robust<" in html
        assert "Not assessable" not in html

    def test_section_can_be_disabled(self):
        cfg = replace(ReportConfig(), causal_assumptions=SectionConfig(enabled=False))
        html = _render(MMMDataBundle(channel_names=["TV"]), cfg)
        assert "Identification rests on assumptions" not in html
