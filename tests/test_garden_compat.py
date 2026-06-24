"""Tests for the Model Garden compatibility suite (mmm_framework.garden.compat).

The suite fits candidate models on synthetic worlds, so the end-to-end tiers are
marked ``slow``. A reference-compatible model must pass all blocking tiers; a
model that silently breaks original-scale prediction must FAIL the scaling tier
(this is the "confidently-wrong" guard the duck-typed contract needs)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.garden import run_compatibility_check
from mmm_framework.garden.compat import BLOCKING_TIERS


def _tier(report, name):
    return next(t for t in report["tiers"] if t["name"] == name)


@pytest.mark.slow
def test_reference_model_passes_all_blocking_tiers():
    from mmm_framework.garden import CustomMMM

    report = run_compatibility_check(
        CustomMMM, scenarios=("clean",), n_weeks=52, check_carryover=False
    )
    assert report["blocking_passed"] is True, report["summary"]
    assert report["is_bayesian_mmm_subclass"] is True
    # every blocking tier that ran actually passed
    for t in report["tiers"]:
        if t["blocking"] and not t["skipped"]:
            assert t["passed"], f"blocking tier {t['name']} failed: {t['detail']}"
    # advisory accuracy score computed against the answer key
    assert report["score"] is not None and 0.0 <= report["score"] <= 1.0


@pytest.mark.slow
def test_broken_original_scale_fails_scaling_tier():
    """A model whose predict() returns standardized/zeroed values (not original
    KPI scale) builds + fits fine but must fail the (blocking) scaling tier."""
    from mmm_framework.garden import CustomMMM
    from mmm_framework.model.results import PredictionResults

    class BrokenScaling(CustomMMM):
        """Deliberately broken: predict ignores the original-scale contract."""

        def predict(self, *args, **kwargs):  # noqa: D401
            n = int(self.n_obs)
            z = np.zeros(n)
            return PredictionResults(
                posterior_predictive=None,
                y_pred_mean=z,
                y_pred_std=z,
                y_pred_hdi_low=z,
                y_pred_hdi_high=z,
                y_pred_samples=np.zeros((1, n)),
            )

    report = run_compatibility_check(
        BrokenScaling, scenarios=("clean",), n_weeks=52, check_carryover=False
    )
    assert report["blocking_passed"] is False
    assert _tier(report, "scaling")["passed"] is False
    assert "scaling" in BLOCKING_TIERS


@pytest.mark.slow
def test_incompatible_class_fails_static_tier_without_fitting():
    """A class that doesn't satisfy the contract is rejected at the static tier,
    so no build/fit is attempted."""

    class NotAModel:
        def fit(self, **kw): ...  # missing predict/sample_channel_contributions

    report = run_compatibility_check(NotAModel, scenarios=("clean",))
    assert report["blocking_passed"] is False
    assert _tier(report, "static")["passed"] is False
    # the suite short-circuits: only the static tier ran
    assert [t["name"] for t in report["tiers"]] == ["static"]
