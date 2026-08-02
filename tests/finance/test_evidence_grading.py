"""Which way is better, and against what (#221).

The defect this module fixes: a cost-per-outcome estimand was graded by the
ROI rule — reference 0.0, higher-is-better — so its interval cleared the bar by
construction and every channel read "Strong". Executed on the shipped grader
before the fix, a $45 CPA and a $2 CPA got the same verdict.

The load-bearing assertions are therefore about *direction* and about absence:
a cost runs the other way, and zero is not a break-even cost.
"""

from __future__ import annotations

import pytest

from mmm_framework.finance.evidence import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    classify_evidence,
    is_cost_kind,
    is_ratio_kind,
    resolve_reference,
    verdict_label,
)

# --------------------------------------------------------------------------
# metric classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,units",
    [
        ("cost_per_outcome", "$/conversion"),
        ("cost_per_outcome", ""),
        ("cpa", ""),
        ("", "$/conversion"),
        ("", "cost / acquisition"),
    ],
)
def test_cost_kinds_recognized(kind, units):
    assert is_cost_kind(kind, units) is True
    assert is_ratio_kind(kind, units) is False, "cost wins over the ratio test"


@pytest.mark.parametrize(
    "kind,units",
    [
        ("roi", "ratio"),
        ("marginal_roas", "x"),
        ("contribution", "$"),
        ("awareness_lift", "KPI"),
        # An ROI's DIVISOR is dollars; that must not make it a cost metric.
        ("roi", "$"),
    ],
)
def test_non_cost_kinds_are_not_swept_up(kind, units):
    assert is_cost_kind(kind, units) is False


def test_ratio_and_contribution_references_unchanged():
    """Every pre-existing grading is byte-identical."""
    assert resolve_reference("roi", "ratio").value == 1.0
    assert resolve_reference("marginal_roas", "").value == 1.0
    assert resolve_reference("contribution", "$").value == 0.0
    assert resolve_reference("awareness_lift", "KPI").value == 0.0
    for kind, units in [("roi", "ratio"), ("contribution", "$")]:
        assert resolve_reference(kind, units).direction == HIGHER_IS_BETTER


# --------------------------------------------------------------------------
# the cost rule
# --------------------------------------------------------------------------


def test_a_cost_has_no_free_reference():
    ref = resolve_reference("cost_per_outcome", "$/conversion")
    assert ref.value is None, "zero is not a break-even — nothing beats free"
    assert ref.resolved is False
    assert ref.direction == LOWER_IS_BETTER
    assert ref.basis == "unresolved"


def test_unresolved_reference_grades_na_not_strong():
    ref = resolve_reference("cost_per_outcome", "$/conversion")
    assert (
        classify_evidence(status="ok", mean=46.0, lower=30.0, upper=62.0, reference=ref)
        == "na"
    )
    # The pre-fix rule, pinned: this is what "graded against 0" produced, and
    # it produced it for EVERY cost, however ruinous.
    assert (
        classify_evidence(status="ok", mean=46.0, lower=30.0, upper=62.0, reference=0.0)
        == "strong"
    )


@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        (30.0, 62.0, "below"),  # credibly costs more than a conversion is worth
        (1.0, 3.0, "strong"),  # credibly cheaper
        (12.0, 31.0, "uncertain"),  # straddles
    ],
)
def test_cost_graded_against_the_value_of_one_outcome(lower, upper, expected):
    ref = resolve_reference("cost_per_outcome", "$/conversion", explicit=20.0)
    assert ref.direction == LOWER_IS_BETTER
    assert (
        classify_evidence(
            status="ok",
            mean=(lower + upper) / 2,
            lower=lower,
            upper=upper,
            reference=ref,
        )
        == expected
    )


def test_the_same_interval_grades_opposite_ways_by_direction():
    """The direction is the whole fix: identical numbers, opposite verdicts."""
    kwargs = dict(status="ok", mean=46.0, lower=30.0, upper=62.0, reference=20.0)
    assert classify_evidence(**kwargs, direction=HIGHER_IS_BETTER) == "strong"
    assert classify_evidence(**kwargs, direction=LOWER_IS_BETTER) == "below"


def test_verdict_label_follows_the_direction():
    """ "Below reference" on a cost that is ABOVE its bar reads backwards."""
    assert verdict_label("below", HIGHER_IS_BETTER) == "Below reference"
    assert verdict_label("below", LOWER_IS_BETTER) == "Above reference"
    assert verdict_label("strong", LOWER_IS_BETTER) == "Strong"
    assert verdict_label("na") == "Not assessable"


# --------------------------------------------------------------------------
# the hint (what the UI renders)
# --------------------------------------------------------------------------


def test_hint_names_the_bar_actually_used():
    """The dashboard derived its hint from `is_ratio`, so it printed
    "vs 0 (no effect)" next to a profit break-even of 2.5 it was grading
    against. The hint now comes from the same call that sets the bar."""
    assert resolve_reference("roi", "ratio").hint == "vs 1.0 (break-even)"
    assert resolve_reference("contribution", "$").hint == "vs 0 (no effect)"
    assert "2.5" in resolve_reference("roi", "ratio", explicit=2.5).hint
    assert "0" in resolve_reference("roi", "impressions", explicit=0.0).hint
    assert (
        resolve_reference("cost_per_outcome", "$/conversion", explicit=20.0).hint
        == "vs 20.00 (break-even cost)"
    )


def test_explicit_reference_beats_every_heuristic():
    """An efficiency metric carries reference 0 though its kind is "roi"."""
    ref = resolve_reference("roi", "ratio", explicit=0.0)
    assert ref.value == 0.0 and ref.is_ratio is False
    assert (
        classify_evidence(status="ok", mean=5.0, lower=2.0, upper=9.0, reference=ref)
        == "strong"
    )


def test_missing_interval_or_bad_status_is_na():
    ref = resolve_reference("roi", "ratio")
    assert (
        classify_evidence(
            status="unsupported", mean=None, lower=None, upper=None, reference=ref
        )
        == "na"
    )
    assert (
        classify_evidence(status="ok", mean=1.0, lower=None, upper=2.0, reference=ref)
        == "na"
    )
