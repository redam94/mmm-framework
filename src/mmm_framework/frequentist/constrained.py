"""Hard constraints on a fixed-transform design — the convex-program estimator.

This is the one capability in the frequentist path with **no Bayesian
equivalent**. A prior can make a negative coefficient *unlikely* — a `HalfNormal`
puts zero mass below zero, which is as close as a prior gets — but it cannot
express "these three channels' contributions sum to the number finance already
booked", and it cannot make an ordering hold exactly. A convex program states
both as constraints and the solver either satisfies them or reports the problem
infeasible.

With per-channel transforms held fixed the objective is a **convex QP**,

.. math::

    \\min_\\theta \\; \\lVert y - X\\theta \\rVert^2 + \\lambda \\lVert P^{1/2}\\theta \\rVert^2
    \\quad\\text{subject to linear constraints,}

so it solves to global optimality. There is no local-minimum caveat and no
initialization to worry about — which is exactly why the constrained case is
worth a dependency while the unconstrained case is not.

What a constraint costs
-----------------------
A hard constraint is **an assumption carrying no uncertainty**. Every interval
computed from a constrained fit is conditional on the constraint being true, and
nothing in the bootstrap will tell you it was wrong — a mis-specified equality
constraint moves the estimate and narrows the interval around the wrong place.

Two specific traps this module surfaces rather than hides:

* **Boundary solutions.** A coefficient pinned at its constraint has no
  meaningful two-sided interval. Bootstrap replicates pile up against the same
  boundary, the percentile interval collapses toward a point, and the result
  reads as a precisely-estimated small number rather than "the constraint is
  doing the work". :attr:`ConstrainedFit.at_boundary` marks them, so #186 and the
  reporting layer can render "at constraint" instead of an estimate.
* **Infeasibility.** An over-specified constraint set has no solution. Returning
  the unconstrained fit in that case would be the worst possible behaviour — the
  caller asked for a guarantee and would get a silent violation — so this raises
  :class:`InfeasibleConstraints` naming what failed.

`cvxpy` is an optional extra (``mmm-framework[frequentist]``), imported lazily
inside :func:`fit_constrained`. Non-negativity alone does **not** need it —
:func:`~mmm_framework.frequentist.ridge.fit_ridge` handles that case through
``scipy.optimize.nnls`` — so the extra is justified only by the richer families
below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from .design import DesignMatrix

__all__ = [
    "Constraint",
    "ConstrainedFit",
    "InfeasibleConstraints",
    "fit_constrained",
    "nonneg",
    "ordering",
    "sum_equals",
    "sum_at_most",
]

_EXTRA_HINT = (
    "Constrained estimation needs cvxpy, which ships in an optional extra.\n"
    "    uv sync --extra frequentist        # or:  pip install 'mmm-framework[frequentist]'\n"
    "For a non-negativity constraint alone you do not need it — "
    "`fit_ridge(design, nonneg=True)` solves that with scipy."
)


class InfeasibleConstraints(ValueError):
    """The constraint set admits no solution.

    Raised instead of silently returning the unconstrained fit: the caller asked
    for a guarantee, and quietly handing back a solution that violates it is
    worse than failing.
    """


@dataclass(frozen=True)
class Constraint:
    """One linear constraint on the coefficient vector.

    Expressed as ``a @ theta {<=,==,>=} b`` over the design's columns. Build
    these with the helpers below rather than by hand where possible — they map
    column *names* to indices, so a design change surfaces as a `KeyError`
    rather than a silently misplaced constraint.

    Attributes:
        a: ``(k,)`` coefficient row over the design's columns.
        op: ``"<="``, ``"=="`` or ``">="``.
        b: Right-hand side.
        label: Human-readable name, used in the infeasibility message and in
            reporting.
    """

    a: "NDArray[np.floating]"
    op: Literal["<=", "==", ">="]
    b: float
    label: str


@dataclass(frozen=True)
class ConstrainedFit:
    """The result of a constrained solve.

    Attributes:
        theta: ``(k,)`` solved coefficients, in the design's parameterization.
        penalty: The ridge penalty applied.
        active: Labels of constraints active at the solution (satisfied with
            equality). These are the assumptions doing work.
        at_boundary: ``(k,)`` boolean — coefficients pinned by an active
            constraint. **They have no meaningful two-sided interval.**
        objective: Optimal objective value.
        status: The solver's status string.
        solver: Which solver ran.
        columns: The design's column names.
        diagnostics: Extras (constraint labels, solve time, ``n_obs``).
    """

    theta: "NDArray[np.floating]"
    penalty: float
    active: list[str]
    at_boundary: "NDArray[np.bool_]"
    objective: float
    status: str
    solver: str
    columns: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        """Coefficients keyed by column name."""
        return dict(zip(self.columns, (float(v) for v in self.theta), strict=False))


# --------------------------------------------------------------------------- #
# constraint builders
# --------------------------------------------------------------------------- #


def _index(design: "DesignMatrix", column: str) -> int:
    try:
        return design.columns.index(column)
    except ValueError as exc:
        raise KeyError(
            f"{column!r} is not a design column. Available: {design.columns}"
        ) from exc


def nonneg(
    design: "DesignMatrix", columns: "list[str] | None" = None
) -> list[Constraint]:
    """``theta_j >= 0`` for the named columns (default: the whole media block).

    Note this alone does not justify the `cvxpy` dependency —
    ``fit_ridge(design, nonneg=True)`` solves it with scipy. It is here so a
    non-negativity requirement can be *combined* with the richer families.
    """
    if columns is None:
        block = design.blocks.get("media")
        columns = design.columns[block] if block is not None else []
    out = []
    for name in columns:
        a = np.zeros(len(design.columns))
        a[_index(design, name)] = 1.0
        out.append(Constraint(a=a, op=">=", b=0.0, label=f"{name} >= 0"))
    return out


def ordering(design: "DesignMatrix", higher: str, lower: str) -> Constraint:
    """``theta_higher >= theta_lower`` — a known effectiveness ordering."""
    a = np.zeros(len(design.columns))
    a[_index(design, higher)] = 1.0
    a[_index(design, lower)] = -1.0
    return Constraint(a=a, op=">=", b=0.0, label=f"{higher} >= {lower}")


def _contribution_row(
    design: "DesignMatrix", columns: list[str]
) -> "NDArray[np.floating]":
    """Row mapping theta to the summed contribution of ``columns``.

    A column's total contribution over the window is ``sum(X[:, j]) * theta_j``,
    so a constraint on contributions is linear in theta — which is what keeps
    the problem a QP.
    """
    a = np.zeros(len(design.columns))
    for name in columns:
        j = _index(design, name)
        a[j] = float(design.X[:, j].sum())
    return a


def sum_equals(design: "DesignMatrix", columns: list[str], total: float) -> Constraint:
    """Total contribution of ``columns`` equals ``total`` (standardized scale).

    The "match the number finance already booked" case. ``total`` is on the
    design's standardized outcome scale; divide a KPI-unit figure by
    ``design.scaling["y_std"]`` first.
    """
    return Constraint(
        a=_contribution_row(design, columns),
        op="==",
        b=float(total),
        label=f"sum({', '.join(columns)}) == {total:g}",
    )


def sum_at_most(design: "DesignMatrix", columns: list[str], total: float) -> Constraint:
    """Total contribution of ``columns`` is at most ``total`` (standardized scale)."""
    return Constraint(
        a=_contribution_row(design, columns),
        op="<=",
        b=float(total),
        label=f"sum({', '.join(columns)}) <= {total:g}",
    )


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #


def fit_constrained(
    design: "DesignMatrix",
    y: "NDArray[np.floating] | None" = None,
    *,
    constraints: "list[Constraint] | None" = None,
    penalty: float = 0.0,
    penalize: "NDArray[np.bool_] | NDArray[np.floating] | None" = None,
    solver: str | None = None,
    boundary_tol: float = 1e-7,
) -> ConstrainedFit:
    """Solve the penalized least-squares problem subject to linear constraints.

    Args:
        design: The design from
            :func:`~mmm_framework.frequentist.design.build_design_matrix`.
        y: Outcome. Defaults to ``design.y``.
        constraints: Linear constraints, built with the helpers in this module.
            ``None`` or empty solves the unconstrained problem, which must
            reproduce :func:`~mmm_framework.frequentist.ridge.fit_ridge` — a test
            pins that, because constraint machinery that perturbs the base case
            is worse than no constraint machinery.
        penalty: Ridge strength. Same meaning as in ``fit_ridge``.
        penalize: Per-column penalty weights (boolean mask or float vector).
            Defaults to the design's mask.
        solver: cvxpy solver name. ``None`` lets cvxpy choose.
        boundary_tol: Slack below which a constraint counts as active.

    Returns:
        The :class:`ConstrainedFit`.

    Raises:
        ImportError: If ``cvxpy`` is not installed, with the extra named.
        InfeasibleConstraints: If the constraint set admits no solution, naming
            the constraints involved.
        ValueError: On shape mismatches or a negative penalty.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - exercised by the lean gate
        raise ImportError(_EXTRA_HINT) from exc

    if penalty < 0:
        raise ValueError(f"penalty must be non-negative, got {penalty}")

    X = np.asarray(design.X, dtype=float)
    y_vec = np.asarray(design.y if y is None else y, dtype=float)
    if y_vec.shape[0] != X.shape[0]:
        raise ValueError(f"y has {y_vec.shape[0]} rows but the design has {X.shape[0]}")

    raw = design.penalize if penalize is None else np.asarray(penalize)
    weights = raw.astype(float)
    if np.any(weights < 0):
        raise ValueError("penalize weights must be non-negative")

    constraints = list(constraints or [])
    theta = cp.Variable(X.shape[1])
    objective = cp.sum_squares(X @ theta - y_vec)
    if penalty > 0 and np.any(weights > 0):
        objective = objective + penalty * cp.sum_squares(
            cp.multiply(np.sqrt(weights), theta)
        )

    cp_constraints = []
    for c in constraints:
        a = np.asarray(c.a, dtype=float)
        if a.shape[0] != X.shape[1]:
            raise ValueError(
                f"constraint {c.label!r} has {a.shape[0]} coefficients but the "
                f"design has {X.shape[1]} columns"
            )
        expr = a @ theta
        if c.op == "<=":
            cp_constraints.append(expr <= c.b)
        elif c.op == ">=":
            cp_constraints.append(expr >= c.b)
        elif c.op == "==":
            cp_constraints.append(expr == c.b)
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown constraint op {c.op!r}")

    problem = cp.Problem(cp.Minimize(objective), cp_constraints)
    problem.solve(solver=solver) if solver else problem.solve()

    if problem.status in ("infeasible", "infeasible_inaccurate"):
        raise InfeasibleConstraints(
            "No coefficient vector satisfies every constraint:\n  "
            + "\n  ".join(c.label for c in constraints)
            + "\nRelax or remove one. The unconstrained solution is NOT returned "
            "— you asked for a guarantee, and silently violating it would be worse "
            "than failing."
        )
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise InfeasibleConstraints(
            f"Solver returned status {problem.status!r}; no trustworthy solution."
        )

    solved = np.asarray(theta.value, dtype=float)

    # Active set: a constraint satisfied with equality is doing work, and any
    # coefficient it touches is pinned rather than estimated.
    active: list[str] = []
    at_boundary = np.zeros(X.shape[1], dtype=bool)
    for c in constraints:
        a = np.asarray(c.a, dtype=float)
        slack = abs(float(a @ solved) - c.b)
        if slack <= boundary_tol:
            active.append(c.label)
            at_boundary |= a != 0

    return ConstrainedFit(
        theta=solved,
        penalty=float(penalty),
        active=active,
        at_boundary=at_boundary,
        objective=float(problem.value),
        status=str(problem.status),
        solver=str(
            problem.solver_stats.solver_name if problem.solver_stats else solver
        ),
        columns=list(design.columns),
        diagnostics={
            "constraints": [c.label for c in constraints],
            "n_obs": int(X.shape[0]),
            "n_params": int(X.shape[1]),
            "n_active": len(active),
        },
    )
