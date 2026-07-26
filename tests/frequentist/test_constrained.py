"""Hard constraints — the one capability with no Bayesian equivalent.

A `HalfNormal` prior makes a negative coefficient unlikely; it cannot make it
impossible, cannot say "these channels sum to the booked number", and cannot
enforce an ordering exactly. A convex program does all three, and with the
transforms fixed the problem is a QP, so it solves to global optimality.

Two properties matter more than the arithmetic, and both are asserted here:

* the constraint machinery must **not perturb the unconstrained case** — a
  no-constraint solve has to reproduce `fit_ridge` to solver tolerance, or the
  estimator has two answers for the same question;
* an infeasible set must **raise**, never silently return the unconstrained
  solution. The caller asked for a guarantee; quietly violating it is the worst
  available behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.config.enums import SaturationType
from mmm_framework.frequentist.constrained import (
    InfeasibleConstraints,
    fit_constrained,
    nonneg,
    ordering,
    sum_at_most,
    sum_equals,
)
from mmm_framework.frequentist.design import build_design_matrix
from mmm_framework.frequentist.ridge import fit_ridge
from mmm_framework.model.trend_config import TrendConfig, TrendType

from test_design_equivalence import CHANNELS, _configure, _panel

ALPHA = {c: {"alpha": 0.55} for c in CHANNELS}
LAM = {c: {"sat_lam": 2.7} for c in CHANNELS}


@pytest.fixture(scope="module")
def design():
    panel = _panel()
    _configure(panel, "geometric", SaturationType.LOGISTIC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_design_matrix(
            panel,
            alpha=ALPHA,
            lam=LAM,
            model_config=ModelConfig(),
            trend_config=TrendConfig(type=TrendType.LINEAR),
        )


def _planted(design, **values) -> np.ndarray:
    """A noiseless outcome from a chosen coefficient vector."""
    theta = np.zeros(design.n_params)
    theta[design.columns.index("intercept")] = 1.0
    for name, v in values.items():
        theta[design.columns.index(name)] = v
    return design.X @ theta


class TestBaseCaseIsUnperturbed:
    def test_no_constraints_reproduces_the_ridge_solution(self, design):
        """Constraint machinery that moves the base case is worse than none."""
        for penalty in (0.0, 1.0, 25.0):
            got = fit_constrained(design, penalty=penalty).theta
            expected = fit_ridge(design, penalty=penalty).theta
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_an_inactive_constraint_changes_nothing(self, design):
        """A constraint the solution already satisfies must not bind."""
        free = fit_ridge(design, penalty=1.0)
        j = design.columns.index("media_TV")
        loose = sum_at_most(design, ["media_TV"], total=1e6)
        got = fit_constrained(design, constraints=[loose], penalty=1.0)
        np.testing.assert_allclose(got.theta, free.theta, rtol=1e-5, atol=1e-6)
        assert got.active == []
        assert not got.at_boundary[j]


class TestNonNegativity:
    def test_a_planted_negative_coefficient_is_pinned_at_exactly_zero(self, design):
        """Not 'a small negative' — exactly the boundary, and flagged as such."""
        y = _planted(design, media_TV=-4.0, media_Digital=2.0)
        fit = fit_constrained(design, y, constraints=nonneg(design), penalty=0.0)
        j = design.columns.index("media_TV")
        assert fit.theta[j] == pytest.approx(0.0, abs=1e-7)
        assert fit.at_boundary[j]
        assert any("media_TV >= 0" in label for label in fit.active)

    def test_a_boundary_coefficient_is_reported_not_inferred(self, design):
        """A pinned coefficient has no meaningful two-sided interval, so it is
        surfaced rather than left to be guessed from a suspiciously small number."""
        y = _planted(design, media_TV=-4.0)
        fit = fit_constrained(design, y, constraints=nonneg(design), penalty=0.0)
        assert fit.at_boundary.any()
        assert fit.diagnostics["n_active"] >= 1
        assert not fit.at_boundary[design.columns.index("intercept")]

    def test_agrees_with_the_scipy_nnls_path_on_the_shared_case(self, design):
        """`fit_ridge(nonneg=True)` handles this without cvxpy; the two must agree,
        or the optional extra silently changes the answer."""
        y = _planted(design, media_TV=-4.0, media_Digital=2.0)
        via_scipy = fit_ridge(design, y, penalty=0.0, nonneg=True)
        via_cvxpy = fit_constrained(design, y, constraints=nonneg(design), penalty=0.0)
        media = [design.columns.index(f"media_{c}") for c in CHANNELS]
        np.testing.assert_allclose(
            via_cvxpy.theta[media], via_scipy.theta[media], rtol=1e-4, atol=1e-6
        )


class TestOrderingAndSums:
    def test_ordering_is_enforced_exactly(self, design):
        """A planted ordering violation must come out as equality, not 'close'."""
        y = _planted(design, media_TV=0.5, media_Digital=3.0)
        fit = fit_constrained(
            design,
            y,
            constraints=[ordering(design, "media_TV", "media_Digital")],
            penalty=0.0,
        )
        tv = fit.theta[design.columns.index("media_TV")]
        dig = fit.theta[design.columns.index("media_Digital")]
        assert tv >= dig - 1e-7
        assert tv == pytest.approx(dig, abs=1e-5)  # the constraint binds

    def test_a_sum_equality_hits_the_booked_number(self, design):
        """The 'match what finance already booked' case."""
        target = 30.0
        fit = fit_constrained(
            design,
            constraints=[sum_equals(design, ["media_TV", "media_Digital"], target)],
            penalty=0.0,
        )
        realized = sum(
            design.X[:, design.columns.index(f"media_{c}")].sum()
            * fit.theta[design.columns.index(f"media_{c}")]
            for c in CHANNELS
        )
        assert realized == pytest.approx(target, rel=1e-5)
        assert fit.active, "an equality constraint is always active"

    def test_a_binding_cap_is_respected_and_reported_active(self, design):
        y = _planted(design, media_TV=3.0, media_Digital=3.0)
        cap = 5.0
        fit = fit_constrained(
            design,
            y,
            constraints=[sum_at_most(design, ["media_TV", "media_Digital"], cap)],
            penalty=0.0,
        )
        realized = sum(
            design.X[:, design.columns.index(f"media_{c}")].sum()
            * fit.theta[design.columns.index(f"media_{c}")]
            for c in CHANNELS
        )
        assert realized <= cap + 1e-5
        assert fit.active


class TestFailureModes:
    def test_infeasible_constraints_raise_and_name_what_failed(self, design):
        """Never silently return the unconstrained solution."""
        with pytest.raises(InfeasibleConstraints) as exc:
            fit_constrained(
                design,
                constraints=[
                    sum_equals(design, ["media_TV"], 100.0),
                    sum_at_most(design, ["media_TV"], 1.0),
                ],
                penalty=0.0,
            )
        message = str(exc.value)
        assert "media_TV" in message
        assert "NOT returned" in message

    def test_an_unknown_column_raises_rather_than_misplacing_the_constraint(
        self, design
    ):
        with pytest.raises(KeyError, match="media_Radio"):
            ordering(design, "media_Radio", "media_TV")

    def test_a_wrong_width_constraint_is_rejected(self, design):
        from mmm_framework.frequentist.constrained import Constraint

        bad = Constraint(a=np.ones(3), op=">=", b=0.0, label="wrong width")
        with pytest.raises(ValueError, match="coefficients but the design has"):
            fit_constrained(design, constraints=[bad])

    def test_negative_penalty_is_rejected(self, design):
        with pytest.raises(ValueError, match="non-negative"):
            fit_constrained(design, penalty=-1.0)


def test_the_extra_is_named_when_cvxpy_is_absent(design, monkeypatch):
    """The ImportError must be actionable, and must point at the scipy path for
    the non-negativity case that does not need the extra at all."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "cvxpy" or name.startswith("cvxpy."):
            raise ImportError("No module named 'cvxpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError) as exc:
        fit_constrained(design)
    message = str(exc.value)
    assert "mmm-framework[frequentist]" in message
    assert "nonneg=True" in message
