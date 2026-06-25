"""Tests for the convergence gate (Phase 1 / I1).

A non-converged MCMC fit must never be returned without a signal. These tests
pin the centralized verdict logic, the ``converged`` properties on the core and
extended results, and the executive-summary non-convergence banner.
"""

from __future__ import annotations

import warnings

import pytest

from mmm_framework.diagnostics import convergence as conv
from mmm_framework.diagnostics.convergence import ConvergenceWarning


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
GOOD = {"divergences": 0, "rhat_max": 1.005, "ess_bulk_min": 800, "approximate": False}
BAD_RHAT = {"divergences": 0, "rhat_max": 1.2, "ess_bulk_min": 800, "approximate": False}
BAD_DIV = {"divergences": 7, "rhat_max": 1.0, "ess_bulk_min": 800, "approximate": False}
BAD_ESS = {"divergences": 0, "rhat_max": 1.0, "ess_bulk_min": 50, "approximate": False}
APPROX = {"approximate": True, "rhat_max": None, "ess_bulk_min": None}


def test_is_converged_true_for_good_fit():
    assert conv.is_converged(GOOD) is True
    assert conv.convergence_flags(GOOD) == []


@pytest.mark.parametrize(
    "diag,flag",
    [(BAD_RHAT, "rhat"), (BAD_DIV, "divergences"), (BAD_ESS, "ess")],
)
def test_is_converged_false_for_each_failure(diag, flag):
    assert conv.is_converged(diag) is False
    assert flag in conv.convergence_flags(diag)


def test_approximate_fit_is_not_assessable():
    # None is NOT "converged" -- it means "not assessable" (no warning either).
    assert conv.is_converged(APPROX) is None
    assert conv.convergence_flags(APPROX) == []


def test_empty_diagnostics_not_assessable():
    assert conv.is_converged({}) is None


def test_warn_if_not_converged_emits_on_bad():
    with pytest.warns(ConvergenceWarning, match="NOT converged"):
        emitted = conv.warn_if_not_converged(BAD_DIV, label="TestModel")
    assert emitted is True


def test_warn_if_not_converged_silent_on_good_and_approx():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        assert conv.warn_if_not_converged(GOOD) is False
        assert conv.warn_if_not_converged(APPROX) is False


def test_annotate_fills_keys():
    d = dict(BAD_RHAT)
    conv.annotate(d)
    assert d["converged"] is False
    assert "rhat" in d["flags"]


# ---------------------------------------------------------------------------
# Results.converged properties (no fitting -- construct dataclasses directly)
# ---------------------------------------------------------------------------
def test_mmmresults_converged_property():
    from mmm_framework.model.results import MMMResults

    good = MMMResults(trace=None, model=None, panel=None, diagnostics=dict(GOOD))
    bad = MMMResults(trace=None, model=None, panel=None, diagnostics=dict(BAD_DIV))
    approx = MMMResults(
        trace=None, model=None, panel=None, diagnostics=dict(APPROX), approximate=True
    )
    assert good.converged is True
    assert bad.converged is False and "divergences" in bad.convergence_flags
    assert approx.converged is None


def test_extended_modelresults_converged_property():
    from mmm_framework.mmm_extensions.results import ModelResults

    bad = ModelResults(trace=None, model=None, config=None, diagnostics=dict(BAD_ESS))
    good = ModelResults(trace=None, model=None, config=None, diagnostics=dict(GOOD))
    assert bad.converged is False and "ess" in bad.convergence_flags
    assert good.converged is True


# ---------------------------------------------------------------------------
# Executive-summary banner
# ---------------------------------------------------------------------------
def _exec_section(diagnostics: dict):
    from mmm_framework.reporting.config import ReportConfig, SectionConfig
    from mmm_framework.reporting.data_extractors import MMMDataBundle
    from mmm_framework.reporting.sections import ExecutiveSummarySection

    bundle = MMMDataBundle()
    bundle.total_revenue = 1_000_000
    bundle.blended_roi = {"mean": 1.7, "lower": 1.4, "upper": 2.0}
    bundle.diagnostics = diagnostics
    return ExecutiveSummarySection(
        data=bundle,
        config=ReportConfig(),
        section_config=SectionConfig(enabled=True),
    )


def test_executive_summary_banner_on_nonconverged():
    html = _exec_section(dict(BAD_RHAT)).render()
    assert "NOT converged" in html
    assert "do not act on these numbers" in html.lower()


def test_executive_summary_no_banner_on_converged():
    html = _exec_section(dict(GOOD)).render()
    assert "NOT converged" not in html


def test_executive_summary_no_banner_on_approximate():
    html = _exec_section(dict(APPROX)).render()
    assert "NOT converged" not in html
