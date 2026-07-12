"""Estimated long-term (brand) split across every report surface (issues
#106 / #122).

The within-window carryover split and the assumption-driven multiplier scenario
are covered by ``test_long_term_section.py``. This covers the deeper wiring: when
a model actually *fits* a slow brand-equity stock (e.g. ``LongTermBrandMMM``), the
classic report, the augur client deck, and the interactive results report all
surface the genuine ESTIMATE (mean long-term share + credible interval + the
per-channel brand/activation split) rather than only the caveat.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.reporting.helpers.longterm import (
    build_long_term_facts,
    estimated_long_term_split,
)


# ---------------------------------------------------------------------------
# a fake fitted model exposing the LongTermBrandMMM trace deterministics
# ---------------------------------------------------------------------------
def _fake_model(*, with_channels=True, frac_mean=0.7):
    """A duck-typed model whose ``_trace.posterior`` carries a scale-free
    ``long_term_fraction`` and per-channel brand/activation contributions."""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(3)
    chain, draw, obs = 2, 40, 12
    channels = ["TV", "Search"]

    frac = np.clip(rng.normal(frac_mean, 0.03, size=(chain, draw)), 0, 1)
    data_vars = {
        "long_term_fraction": (("chain", "draw"), frac),
    }
    coords = {"chain": range(chain), "draw": range(draw)}
    if with_channels:
        # TV is brand-heavy (long-term dominates), Search is activation-heavy.
        brand = np.stack(
            [
                np.abs(rng.normal(80, 5, size=(chain, draw, obs))),  # TV brand
                np.abs(rng.normal(10, 2, size=(chain, draw, obs))),  # Search brand
            ],
            axis=-1,
        )
        act = np.stack(
            [
                np.abs(rng.normal(20, 4, size=(chain, draw, obs))),  # TV activation
                np.abs(rng.normal(90, 6, size=(chain, draw, obs))),  # Search activation
            ],
            axis=-1,
        )
        data_vars["brand_contributions"] = (("chain", "draw", "obs", "channel"), brand)
        data_vars["activation_contributions"] = (
            ("chain", "draw", "obs", "channel"),
            act,
        )
        coords["channel"] = channels
        coords["obs"] = range(obs)
    post = xr.Dataset(data_vars, coords=coords)

    class _Trace:
        posterior = post

    class _M:
        channel_names = channels
        _trace = _Trace()

    return _M()


# ---------------------------------------------------------------------------
# estimated_long_term_split (reads the trace)
# ---------------------------------------------------------------------------
class TestEstimatedSplit:
    def test_reads_fraction_and_channels(self):
        est = estimated_long_term_split(_fake_model(frac_mean=0.7))
        assert est is not None
        f = est["long_term_fraction"]
        assert 0.6 < f["mean"] < 0.8
        assert f["lower"] <= f["mean"] <= f["upper"]
        chans = {c["channel"]: c for c in est["channels"]}
        assert set(chans) == {"TV", "Search"}
        # TV is brand-heavy → high long-term share; Search is activation-heavy.
        assert chans["TV"]["long_term_pct"] > 0.6
        assert chans["Search"]["long_term_pct"] < 0.4
        # short + long fractions complement
        for c in est["channels"]:
            assert abs(c["long_term_pct"] + c["short_term_pct"] - 1.0) < 1e-6

    def test_fraction_only_when_no_channel_deterministics(self):
        est = estimated_long_term_split(_fake_model(with_channels=False))
        assert est is not None
        assert est["long_term_fraction"]["mean"] > 0
        assert est["channels"] == []

    def test_none_for_plain_model(self):
        class _Plain:
            channel_names = ["TV"]
            _trace = None

        assert estimated_long_term_split(_Plain()) is None
        assert estimated_long_term_split(object()) is None


# ---------------------------------------------------------------------------
# build_long_term_facts folds in the estimate + it outranks carryover/funnel
# ---------------------------------------------------------------------------
class TestBuildFactsWithEstimate:
    def _est(self):
        return estimated_long_term_split(_fake_model())

    def test_estimated_measured_flag_and_payload(self):
        facts = build_long_term_facts(
            ["TV", "Search"],
            {"TV": [0.5, 0.3, 0.2], "Search": [0.9, 0.1]},
            estimated=self._est(),
        )
        assert facts["measured"] == "estimated"  # outranks "carryover"
        assert facts["estimated"]["long_term_fraction"]["mean"] > 0
        # the within-window carryover rows are still present alongside
        assert facts["channels"]

    def test_estimate_survives_even_without_adstock(self):
        # no adstock → normally "none"; the estimate keeps the section alive.
        facts = build_long_term_facts(["TV"], None, estimated=self._est())
        assert facts["measured"] == "estimated"
        assert facts["estimated"]["channels"]

    def test_no_estimate_is_unchanged(self):
        facts = build_long_term_facts(["TV"], {"TV": [0.5, 0.3, 0.2]}, estimated=None)
        assert facts["measured"] == "carryover"
        assert "estimated" not in facts


def _bundle_with_estimate():
    from mmm_framework.reporting.extractors.bundle import MMMDataBundle

    est = estimated_long_term_split(_fake_model())
    lt = build_long_term_facts(
        ["TV", "Search"],
        {"TV": [0.5, 0.25, 0.13, 0.06], "Search": [0.9, 0.08, 0.02]},
        contribution={"TV": 100000.0, "Search": 60000.0},
        estimated=est,
    )
    return MMMDataBundle(channel_names=["TV", "Search"], long_term=lt)


# ---------------------------------------------------------------------------
# classic report section
# ---------------------------------------------------------------------------
class TestClassicSection:
    def _html(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig

        return MMMReportGenerator(
            data=_bundle_with_estimate(), config=ReportConfig()
        ).render()

    def test_renders_estimated_callout(self):
        html = self._html()
        assert "Estimated long-term (brand) contribution" in html
        assert "credible interval" in html
        # the estimate-aware caveat (not the "only short-term measured" one)
        assert "How to read this estimate" in html

    def test_estimate_leads_the_section(self):
        html = self._html()
        i_est = html.find("Estimated long-term (brand) contribution")
        i_caveat = html.find("How to read this estimate")
        assert i_est != -1 and i_caveat != -1
        assert i_est < i_caveat  # the measurement leads, the caveat follows


# ---------------------------------------------------------------------------
# augur client-deck section
# ---------------------------------------------------------------------------
class TestAugurSection:
    def _section(self, bundle):
        from mmm_framework.reporting.augur_sections import AugurLongTermSection
        from mmm_framework.reporting.config import ReportConfig

        return AugurLongTermSection(data=bundle, config=ReportConfig())

    def test_renders_estimate(self):
        html = self._section(_bundle_with_estimate()).render()
        assert "estimates" in html.lower()
        assert "long-term (brand)" in html
        assert "%" in html

    def test_gates_off_without_data(self):
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        section = self._section(MMMDataBundle(channel_names=["TV"]))
        assert section.is_enabled is False
        assert section.render() == ""

    def test_caveat_when_no_estimate(self):
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        lt = build_long_term_facts(["TV"], {"TV": [0.6, 0.4]})  # carryover only
        section = self._section(MMMDataBundle(channel_names=["TV"], long_term=lt))
        html = section.render()
        assert "under-credits" in html.lower() or "short-term" in html.lower()
        assert "estimates" not in html.lower()


# ---------------------------------------------------------------------------
# interactive results report
# ---------------------------------------------------------------------------
class TestInteractiveSection:
    def _est_facts(self):
        return estimated_long_term_split(_fake_model())

    def test_section_renders_when_estimate_present(self):
        from mmm_framework.reporting.interactive.generator import (
            InteractiveReportGenerator,
        )

        # a minimal facts dict is enough — the generator gates each section on
        # its own key; only "long_term" needs to be populated here.
        facts = {"meta": {"channels": ["TV"]}, "long_term": self._est_facts()}
        html = InteractiveReportGenerator(facts=facts).generate_report()
        assert 'id="longTermPanel"' in html
        assert 'href="#long-term"' in html
        assert "renderLongTerm" in html  # the JS engine is embedded

    def test_section_gates_off_without_estimate(self):
        from mmm_framework.reporting.interactive.generator import (
            InteractiveReportGenerator,
        )

        facts = {"meta": {"channels": ["TV"]}, "long_term": None}
        html = InteractiveReportGenerator(facts=facts).generate_report()
        assert 'id="longTermPanel"' not in html
        assert 'href="#long-term"' not in html
