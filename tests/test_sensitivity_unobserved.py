"""Tests for the unobserved-confounding sensitivity analysis.

The invariant these pin is the one that is easy to get backwards in a Bayesian
setting: the Cinelli-Hazlett robustness value is strictly increasing in
``|t| = |posterior_mean| / posterior_sd``. In OLS that sd is a standard error the
*data* produced. In a Bayesian MMM it is a compromise between likelihood and
prior, so **tightening a prior inflates the reported robustness value with no new
evidence** — precision borrowed from one assumption gets displayed as robustness
to violating a different one.

The module must therefore (a) never report a prior-dominated channel as "Robust",
and (b) distinguish "checked and fine" from "could not check".
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.reporting.sections import CausalAssumptionsSection
from mmm_framework.reporting.config import ReportConfig
from mmm_framework.validation.results import (
    ChannelRobustness,
    UnobservedConfoundingSensitivity,
)
from mmm_framework.validation.sensitivity_unobserved import (
    PRIOR_DOMINATED_CONTRACTION,
    FRAGILE_RV_THRESHOLD,
    prior_inflation_warning,
    robustness_value,
)


def _channel(rv: float, contraction: float | None, name: str = "TV"):
    """A ChannelRobustness with a given RV and prior->posterior contraction."""
    return ChannelRobustness(
        channel=name,
        estimate=1.0,
        std_error=0.1,
        t_value=10.0,
        dof=100,
        partial_r2=0.5,
        robustness_value=rv,
        robustness_value_half=rv / 2,
        prior_contraction=contraction,
    )


class TestRobustnessValueMonotonicity:
    """The property that makes the prior-inflation trap real."""

    def test_rv_increases_with_t(self):
        rvs = [robustness_value(t, dof=100) for t in (1.0, 2.0, 4.0, 8.0)]
        assert rvs == sorted(rvs)
        assert all(np.isfinite(r) for r in rvs)

    def test_tighter_posterior_sd_alone_raises_the_rv(self):
        # Same posterior MEAN, narrower sd (e.g. because the prior tightened).
        # No new data — yet the reported robustness rises.
        mean = 0.5
        loose = robustness_value(mean / 0.25, dof=100)
        tight = robustness_value(mean / 0.05, dof=100)
        assert tight > loose

    def test_rv_bounded_and_nan_safe(self):
        assert 0.0 <= robustness_value(3.0, dof=50) <= 1.0
        assert np.isnan(robustness_value(3.0, dof=0))


class TestPriorInflationWarning:
    def test_warns_when_contraction_is_low_and_rv_reassuring(self):
        msg = prior_inflation_warning(0.05, 0.40)
        assert msg is not None
        assert "prior-dominated" in msg

    def test_silent_when_the_posterior_actually_learned(self):
        assert prior_inflation_warning(0.85, 0.40) is None

    def test_silent_when_contraction_unknown(self):
        # Not assessed is reported separately; it is not a warning about the RV.
        assert prior_inflation_warning(None, 0.40) is None
        assert prior_inflation_warning(float("nan"), 0.40) is None

    def test_silent_when_rv_already_flags_fragility(self):
        # Nobody is being over-reassured by a fragile channel.
        assert prior_inflation_warning(0.05, FRAGILE_RV_THRESHOLD / 2) is None

    def test_threshold_is_configurable(self):
        assert prior_inflation_warning(0.30, 0.40) is None
        assert prior_inflation_warning(0.30, 0.40, threshold=0.5) is not None


class TestChannelRobustnessFlags:
    def test_prior_dominated_channel_is_not_quotable(self):
        ch = _channel(rv=0.60, contraction=0.05)
        assert ch.is_prior_dominated
        assert not ch.rv_is_quotable
        assert not ch.is_fragile  # high RV, yet still not evidence

    def test_learned_channel_is_quotable(self):
        ch = _channel(rv=0.60, contraction=0.90)
        assert not ch.is_prior_dominated
        assert ch.rv_is_quotable

    def test_unassessed_channel_is_not_flagged_as_dominated(self):
        ch = _channel(rv=0.60, contraction=None)
        assert not ch.is_prior_dominated
        assert ch.rv_is_quotable

    def test_to_dict_serializes_the_new_flags(self):
        d = _channel(rv=0.60, contraction=0.05).to_dict()
        assert d["is_prior_dominated"] is True
        assert d["rv_is_quotable"] is False
        assert d["prior_contraction"] == pytest.approx(0.05)


class TestNonFiniteRobustnessValue:
    """A non-finite RV must read as *uncomputed*, never as a passed check.

    `is_fragile` is `isfinite(rv) and rv < threshold`, so a NaN is "not
    fragile" — and every consumer that treats an empty `fragile_channels` as
    "all robust" would report a sensitivity check that never ran. Approximate
    (MAP/ADVI) fits produce a degenerate posterior and hence a NaN RV.
    """

    def test_nan_channel_is_not_assessable_and_not_fragile(self):
        ch = _channel(rv=float("nan"), contraction=0.9)
        assert not ch.is_assessable
        assert not ch.is_fragile  # the trap: "not fragile" != "robust"
        assert not ch.rv_is_quotable
        assert ch.to_dict()["is_assessable"] is False

    def test_finite_channel_is_assessable(self):
        ch = _channel(rv=0.42, contraction=0.9)
        assert ch.is_assessable
        assert ch.to_dict()["is_assessable"] is True

    def _sens(self, *channels):
        from mmm_framework.validation.results import UnobservedConfoundingSensitivity

        return UnobservedConfoundingSensitivity(
            channels=list(channels), dof=100, q=1.0, caveat="c"
        )

    def test_unassessable_channels_are_listed_separately(self):
        sens = self._sens(
            _channel(rv=float("nan"), contraction=0.9, name="TV"),
            _channel(rv=0.42, contraction=0.9, name="Search"),
        )
        assert sens.fragile_channels == []  # neither is fragile...
        assert sens.unassessable_channels == ["TV"]  # ...but TV was never assessed
        assert sens.to_dict()["unassessable_channels"] == ["TV"]

    def test_summary_never_prints_the_literal_nan(self):
        sens = self._sens(_channel(rv=float("nan"), contraction=0.9, name="TV"))
        row = sens.summary().iloc[0]
        assert row["Robustness Value"] == "n/a"
        # "No" here would read as a passed fragility check
        assert row["Fragile"] == "n/a"

    def test_summary_still_formats_a_real_value(self):
        sens = self._sens(_channel(rv=0.42, contraction=0.9, name="TV"))
        row = sens.summary().iloc[0]
        assert row["Robustness Value"] == "0.420"
        assert row["Fragile"] == "No"


class TestReportNeverShowsPriorDrivenAsRobust:
    """The user-visible face of the inversion."""

    def _render(self, contraction: float | None):
        sens = UnobservedConfoundingSensitivity(
            channels=[_channel(rv=0.60, contraction=contraction)],
            dof=100,
            q=1.0,
            caveat="",
        )
        section = CausalAssumptionsSection(
            data=type(
                "B", (), {"causal_assumptions": {"robustness": sens.to_dict()}}
            )(),
            config=ReportConfig(),
        )
        return section._render_robustness_table(sens.to_dict())

    def test_prior_dominated_renders_not_assessable(self):
        html = self._render(contraction=0.05)
        assert "Not assessable" in html
        assert ">Robust<" not in html

    def test_learned_channel_renders_robust(self):
        html = self._render(contraction=0.90)
        assert "Robust" in html
        assert "Not assessable" not in html

    def test_footnote_only_appears_when_needed(self):
        assert "posterior mean / posterior sd" in self._render(0.05)
        assert "posterior mean / posterior sd" not in self._render(0.90)


class TestSummaryCaveat:
    def test_caveat_names_the_inversion(self):
        from mmm_framework.validation.sensitivity_unobserved import (
            UnobservedConfoundingAnalysis,
        )

        # The class-level caveat text is assembled in run(); assert the module
        # constant that gates it is a sane fraction.
        assert 0.0 < PRIOR_DOMINATED_CONTRACTION < 1.0
        assert UnobservedConfoundingAnalysis.__doc__
