"""Weak-identification diagnostics for gradient-based point fits (#203).

Gradient-based point fits (``find_MAP``, and the optimize step inside
``fit_laplace``) optimize in PyMC's **unconstrained** space. A parameter bounded
to ``[0, 1]`` — every ``Beta``-prior parameter in an MMM: ``adstock_alpha_*``,
``sat_half_*``, ``sat_exponent_*`` — reaches the optimizer as a logit, and one
bounded to ``(0, inf)`` reaches it as a log. Both maps saturate in float64:

    >>> import numpy as np
    >>> 1.0 / (1.0 + np.exp(-37.0)) == 1.0     # sigmoid hits *exactly* 1.0
    True
    >>> np.exp(-746.0) == 0.0                  # exp underflows to *exactly* 0.0
    True

Past those points ``1 - alpha`` (or ``alpha``) is exactly zero, and the density's
gradient terms — ``(a-1)/alpha`` and ``-(b-1)/(1-alpha)`` for a Beta — divide by
zero *inside the compiled PyTensor function*. The user-visible failure is a bare
``ZeroDivisionError`` naming a ``Composite`` op.

Why not clip the gradient
-------------------------
Clipping the returned gradient cannot help: the division happens **inside** the
compiled function, so the exception is raised before any gradient value exists to
clip. The two things that *would* work are inserting a ``clip`` into the model
graph (which changes the model, and is the caller's decision) or keeping the
optimizer inside the region where the transform is float64-representable. This
module does the latter — a box constraint on the unconstrained coordinates,
handed to a bounds-aware optimizer (L-BFGS-B).

The box is a numerical guardrail on the **search path**, not a constraint on the
answer. In the #203 repro no coordinate ends *at* a limit: the guardrail's job is
to stop the line search from evaluating the singular region, after which the
optimizer settles well inside. So "did the guardrail bind?" is a weak signal on
its own, and :func:`weak_identification_report` carries the real one.

What actually flags a non-identifiable direction
------------------------------------------------
Curvature, at the returned point, of the *negative* log posterior in the
unconstrained space — but **correlation-scaled** before it is eigendecomposed.
Raw Hessian eigenvalues are not invariant to rescaling a parameter, so on a real
MMM they mostly rank parameters by the units they happen to be measured in.
Scaling by the diagonal (``D^-1/2 H D^-1/2``) removes that, and separates four
findings that are otherwise easy to confuse:

* a **ridge** — a small scaled eigenvalue, i.e. a parameter *combination* that
  trades off. This is the real MMM non-identifiability, and it is usually the
  saturation elbow against the channel coefficient: measured against planted
  truth, a linear-response panel puts ``sat_lam`` and ``beta`` in a single
  direction at condition index ~104, while the same panel with genuine
  saturation drops to ~26. Jin et al. (2017) call the Hill parameters
  "essentially unidentifiable"; this is what that looks like numerically.
* an **uninformed parameter** — a near-zero *diagonal* entry. Nothing is trading
  off; the density is simply flat in that one coordinate, which is what sends
  the optimizer to the boundary and triggers the guardrail above.
* a **prior-determined parameter** — curvature that is entirely the prior's. The
  posterior is not flat and the optimizer is nowhere near a boundary, so neither
  test above sees it, but the data contributed nothing. See below.
* a **negative** eigenvalue — the point is not even a local maximum. A positive
  diagonal scaling is a congruence, so Sylvester's law of inertia means signs
  survive the scaling and this test stays exact.

A ridge, a flat parameter or a saddle is a reason to distrust a Laplace
approximation, which *inverts this matrix* to get its covariance.
``pymc_extras`` projects a non-PSD Hessian to the nearest PSD matrix rather than
failing, so a Laplace fit can return confident-looking draws built on a
direction the data never constrained. That silent case is the one this report
exists for.

Who is doing the constraining
-----------------------------
``model.logp() == model.varlogp + model.datalogp`` — prior plus data — and
Hessians add, so one extra compiled Hessian splits the posterior curvature into
the part the likelihood paid for and the part the prior did. (The identity holds
to machine precision, which the tests pin.) Two readings come out of it:

* ``IdentificationReport.effective_parameters`` = ``tr(H_post^-1 H_lik)``, the
  Bayesian effective number of parameters: how many of the model's parameters
  the DATA determines, against how many it has. It is 2.0 for two well-measured
  coefficients, 1.0 when two coefficients share one regressor, and ~0 under a
  prior tight enough to swamp the likelihood. Computed as the sum of the
  per-direction shares below, so no inversion is needed — which matters
  precisely because ``H_post`` is singular in the interesting cases.
* an ``informed_fraction`` per direction and per parameter, which turns "this is
  unconstrained" into "and here is who is constraining it". A parameter with a
  small fraction is :class:`PriorDetermined`: it will look confident in the
  posterior, and that confidence is the prior's.

For a complementary view that works from posterior *draws* rather than curvature
at a point, see :func:`mmm_framework.diagnostics.learning.parameter_learning`
(prior-to-posterior contraction, overlap and shift).

Convention: the Hessian is taken with ``jacobian=True`` — the log density in the
unconstrained space, which is the quantity Laplace approximates. ``find_MAP``
locates its mode with ``jacobian=False`` (the mode of the untransformed
posterior), so the point is not the exact mode of the jacobian-adjusted density.
Flatness survives the distinction; the condition index is an order of magnitude,
not a precise quantity.

Everything here is local — curvature at one point. It answers "what does this
fit resolve, here" and cannot see a multimodal posterior, a distant better mode,
or a likelihood that is flat somewhere else. For global questions use NUTS, or
:mod:`mmm_framework.diagnostics.sbc` and
:mod:`mmm_framework.diagnostics.coverage`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pymc as pm

__all__ = [
    "SAFE_UNCONSTRAINED_LIMITS",
    "BoundaryHit",
    "FlatDirection",
    "IdentificationReport",
    "PriorDetermined",
    "UninformedParameter",
    "bounded_find_MAP",
    "boundary_failure_message",
    "guarded_find_MAP",
    "guardrail_warning",
    "has_guardable_parameters",
    "unconstrained_box",
    "weak_identification_report",
]

#: Per-transform limits on the unconstrained coordinate, chosen so the forward
#: map stays strictly inside its support in float64 with room to spare.
#:
#: * ``logodds``/``interval`` — ``sigmoid`` returns exactly ``1.0`` past ~36.7.
#:   At 30, ``1 - sigmoid(30) == 9.3e-14``, so the reciprocal in the density's
#:   gradient is ~1.1e13: large, finite, and never binding for a parameter the
#:   likelihood actually informs (``logit(0.9999) == 9.2``).
#: * ``log`` — ``exp`` underflows to exactly ``0.0`` at −746 and overflows at
#:   710. At ±200 the reciprocal is ~7e86: finite, and ``exp(-200) == 1.4e-87``
#:   is far below any scale parameter with meaning.
SAFE_UNCONSTRAINED_LIMITS: dict[str, float] = {
    "logodds": 30.0,
    "interval": 30.0,
    "log": 200.0,
}

#: scipy ``minimize`` methods that accept ``bounds=``. A guarded retry forces one
#: of these; ``pymc_extras.fit_laplace`` defaults to ``BFGS``, which does not.
_BOUNDS_CAPABLE = frozenset(
    {
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "Powell",
        "trust-constr",
        "COBYLA",
        "Nelder-Mead",
    }
)

_FALLBACK_METHOD = "L-BFGS-B"


@dataclass(frozen=True)
class BoundaryHit:
    """An unconstrained coordinate that finished at its guardrail."""

    parameter: str
    unconstrained_value: float
    limit: float
    transform: str

    def describe(self) -> str:
        side = "upper" if self.unconstrained_value > 0 else "lower"
        return (
            f"{self.parameter} ran to its {side} guardrail "
            f"({self.unconstrained_value:+.1f} in {self.transform} space, "
            f"limit ±{self.limit:g}) — the likelihood never pulled it back, so "
            "it is pinned at the edge of its support rather than estimated"
        )


@dataclass(frozen=True)
class FlatDirection:
    """A direction the fit cannot resolve, from the scale-invariant curvature.

    ``eigenvalue`` is an eigenvalue of the *correlation-scaled* Hessian
    ``D^-1/2 H D^-1/2`` (``D = |diag(H)|``), not of ``H`` itself. Raw Hessian
    eigenvalues change when a parameter is rescaled, so they conflate "measured
    in small units" with "not identified"; the scaled ones do not. ``D`` is a
    positive diagonal, so the scaling is a congruence and Sylvester's law of
    inertia preserves eigenvalue *signs* — a negative eigenvalue is still an
    unambiguous saddle.

    ``condition_index`` is Belsley's ``sqrt(lambda_max / lambda_i)``; the
    conventional collinearity threshold is 30.
    """

    eigenvalue: float
    condition_index: float
    loadings: tuple[tuple[str, float], ...]
    kind: str  # "ridge" | "nonconvex"
    #: Share of this direction's curvature contributed by the likelihood rather
    #: than the prior; ``None`` when the split was not computed. See
    #: :class:`IdentificationReport.effective_parameters`.
    informed_fraction: float | None = None

    def describe(self) -> str:
        combo = ", ".join(f"{name} ({load:+.2f})" for name, load in self.loadings)
        if self.kind == "nonconvex":
            return (
                f"negative curvature ({self.eigenvalue:.2e}) along [{combo}] — "
                "this point is a saddle, not a maximum"
            )
        head = (
            f"[{combo}] trade off almost exactly (condition index "
            f"{self.condition_index:.0f}) — the fit can move a long way along "
            "this combination for almost no change in fit, so the individual "
            "values are far less determined than they look"
        )
        if self.informed_fraction is None:
            return head
        # What little determination exists is either the prior's or the data's,
        # and the two call for different remedies. Saying "the prior decides
        # this" when the data supplies 64% of the curvature would be wrong.
        if self.informed_fraction < 0.25:
            return (
                f"{head}. The prior supplies "
                f"{1.0 - self.informed_fraction:.0%} of what curvature there "
                "is, so more of the same data will not separate them — that "
                "needs an experiment, or a prior you are willing to defend"
            )
        return (
            f"{head}. The data does supply "
            f"{self.informed_fraction:.0%} of the curvature here, so this is "
            "weak evidence rather than none: more variation along this "
            "combination would sharpen it"
        )


@dataclass(frozen=True)
class UninformedParameter:
    """A parameter with negligible curvature of its own at the fitted point.

    Distinct from a ridge: nothing is trading off, the log density is simply
    flat in this one coordinate. This is what drives an optimizer out to the
    support boundary in the first place, so it is usually the companion finding
    to a :class:`BoundaryHit`.
    """

    parameter: str
    curvature: float
    relative_curvature: float

    def describe(self) -> str:
        return (
            f"{self.parameter} has essentially no curvature of its own "
            f"({self.curvature:.2e}, {self.relative_curvature:.1e} of the "
            "best-determined parameter) — neither the data nor the prior pins "
            "it down at this point"
        )


@dataclass(frozen=True)
class PriorDetermined:
    """A parameter whose posterior curvature comes from the prior, not the data.

    The third case, and the one neither of the others catches: the parameter is
    perfectly well determined — the posterior is not flat and the optimizer is
    nowhere near a boundary — but every bit of that determination came from the
    prior. Reporting it as "fine" would be reporting the prior back to the user
    as a finding.

    ``informed_fraction`` is ``H_lik[i,i] / H_post[i,i]``, the share of the
    parameter's own curvature contributed by the likelihood. Read the DIAGONAL
    version only in this direction — a small value is trustworthy (for a PSD
    likelihood curvature, a zero diagonal entry forces its whole row to zero, so
    the likelihood really is flat in that coordinate), but a *large* value is
    not evidence of identification, because the diagonal cannot see a ridge: in
    a two-coefficients-on-one-regressor model both coefficients score 1.00 while
    only their sum is identified. For ridges use
    :attr:`FlatDirection.informed_fraction`, which is computed along the
    eigenvector and does see the cancellation.
    """

    parameter: str
    informed_fraction: float

    def describe(self) -> str:
        return (
            f"{self.parameter} is pinned by its prior, not by the data — the "
            f"likelihood contributes {self.informed_fraction:.1%} of its "
            "curvature. Its posterior will look confident; that confidence is "
            "the prior's, so the value is only as good as that choice"
        )


@dataclass
class IdentificationReport:
    """Curvature and guardrail findings for one point fit."""

    verdict: str  # "ok" | "weak" | "non-identified" | "unknown"
    flat_directions: tuple[FlatDirection, ...] = ()
    uninformed: tuple[UninformedParameter, ...] = ()
    prior_determined: tuple[PriorDetermined, ...] = ()
    boundary_hits: tuple[BoundaryHit, ...] = ()
    #: ``tr(H_post^-1 H_lik)``: how many of the model's parameters the DATA
    #: actually determines, as opposed to how many it has. The Bayesian
    #: effective number of parameters — 2.0 for two well-measured coefficients,
    #: 1.0 when two coefficients share one regressor, ~0 under a prior tight
    #: enough to swamp the likelihood. ``None`` when the split was not computed.
    effective_parameters: float | None = None
    condition_index: float | None = None
    min_scaled_eigenvalue: float | None = None
    n_parameters: int = 0
    guardrail_engaged: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def nonconvex(self) -> bool:
        """Whether the point is a saddle rather than a maximum."""
        return any(d.kind == "nonconvex" for d in self.flat_directions)

    @property
    def laplace_usable(self) -> bool:
        """Whether a Laplace covariance from this curvature would mean anything.

        Laplace inverts this matrix. A negative eigenvalue means there is no
        covariance to invert to (``pymc_extras`` projects to the nearest PSD
        matrix and reports draws anyway); a ridge means the inverse is dominated
        by a direction the data never constrained.
        """
        if self.min_scaled_eigenvalue is None:
            return False
        # An uninformed parameter is as fatal as a ridge here even though it
        # leaves the *scaled* spectrum clean: its raw curvature is ~1e-13, so
        # the covariance Laplace reports for it is ~1e13 wide, or whatever the
        # nearest-PSD projection substitutes.
        return (
            self.min_scaled_eigenvalue > 0
            and not self.flat_directions
            and not self.uninformed
            and not self.boundary_hits
        )

    def to_dict(self) -> dict[str, Any]:
        """A checkpointer-safe payload for ``results.diagnostics``.

        Every value is a builtin ``str``/``float``/``bool``/``list`` — no numpy
        scalars, which the LangGraph msgpack checkpointer cannot serialize.
        """
        return {
            "verdict": self.verdict,
            "guardrail_engaged": self.guardrail_engaged,
            "effective_parameters": self.effective_parameters,
            "condition_index": self.condition_index,
            "min_scaled_eigenvalue": self.min_scaled_eigenvalue,
            "n_parameters": self.n_parameters,
            "laplace_usable": self.laplace_usable,
            "nonconvex": self.nonconvex,
            "flat_directions": [
                {
                    "eigenvalue": d.eigenvalue,
                    "condition_index": d.condition_index,
                    "kind": d.kind,
                    "informed_fraction": d.informed_fraction,
                    "loadings": [[n, v] for n, v in d.loadings],
                }
                for d in self.flat_directions
            ],
            "prior_determined": [
                {
                    "parameter": p.parameter,
                    "informed_fraction": p.informed_fraction,
                }
                for p in self.prior_determined
            ],
            "uninformed": [
                {
                    "parameter": u.parameter,
                    "curvature": u.curvature,
                    "relative_curvature": u.relative_curvature,
                }
                for u in self.uninformed
            ],
            "boundary_hits": [
                {
                    "parameter": h.parameter,
                    "unconstrained_value": h.unconstrained_value,
                    "limit": h.limit,
                    "transform": h.transform,
                }
                for h in self.boundary_hits
            ],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """Plain-language summary, safe to embed in an exception or agent reply."""
        lines: list[str] = []
        if self.boundary_hits:
            lines.append("Parameters that ran to the numerical guardrail:")
            lines.extend(f"  * {h.describe()}" for h in self.boundary_hits)
        if self.uninformed:
            lines.append("Parameters the fit does not inform:")
            lines.extend(f"  * {u.describe()}" for u in self.uninformed)
        if self.prior_determined:
            lines.append("Parameters answered by the prior rather than the data:")
            lines.extend(f"  * {p.describe()}" for p in self.prior_determined)
        if self.flat_directions:
            lines.append("Directions the fit cannot separate:")
            lines.extend(f"  * {d.describe()}" for d in self.flat_directions)
        if self.effective_parameters is not None:
            lines.append(
                f"The data determines about {self.effective_parameters:.1f} of "
                f"this model's {self.n_parameters} parameters; the prior "
                "supplies the rest."
            )
        if self.condition_index is not None:
            lines.append(
                f"Scale-invariant condition index {self.condition_index:.0f} "
                f"over {self.n_parameters} parameters (>30 is the conventional "
                "collinearity threshold)."
            )
        if self.min_scaled_eigenvalue is not None and not self.laplace_usable:
            lines.append(
                "A Laplace approximation inverts this curvature, so its "
                "intervals on the directions above are not meaningful; NUTS "
                "explores them instead of assuming them away."
            )
        lines.extend(f"  ! {n}" for n in self.notes)
        return "\n".join(lines) if lines else "No identification problems found."


def _value_var_rvs(model: "pm.Model") -> list[tuple[Any, Any, str | None]]:
    """``(value_var, rv, transform_name)`` for each continuous value variable."""
    out = []
    for value_var in model.continuous_value_vars:
        rv = model.values_to_rvs.get(value_var)
        transform = model.rvs_to_transforms.get(rv) if rv is not None else None
        out.append((value_var, rv, getattr(transform, "name", None)))
    return out


def _labels_and_limits(model: "pm.Model") -> tuple[list[str], list[float | None]]:
    """Per-raveled-coordinate parameter labels and guardrail limits.

    Ordering follows ``model.continuous_value_vars``, which is also the ordering
    :func:`weak_identification_report` pins the Hessian to, so eigenvector
    loadings and labels cannot drift apart.
    """
    initial = model.initial_point()
    labels: list[str] = []
    limits: list[float | None] = []
    for value_var, rv, transform_name in _value_var_rvs(model):
        name = getattr(rv, "name", value_var.name)
        size = int(np.asarray(initial[value_var.name]).size)
        limit = SAFE_UNCONSTRAINED_LIMITS.get(transform_name or "")
        for i in range(size):
            labels.append(name if size == 1 else f"{name}[{i}]")
            limits.append(limit)
    return labels, limits


def unconstrained_box(
    model: "pm.Model",
) -> tuple[list[tuple[float | None, float | None]], list[str], list[float | None]]:
    """Box constraints keeping the optimizer inside float64-representable space.

    Returns ``(bounds, labels, limits)`` aligned to the raveled unconstrained
    vector, in ``model.continuous_value_vars`` order — the ordering
    ``pm.find_MAP`` ravels into and hands to ``scipy.optimize.minimize``.
    Untransformed coordinates get ``(None, None)`` and are left free.
    """
    labels, limits = _labels_and_limits(model)
    bounds = [(-lim, lim) if lim is not None else (None, None) for lim in limits]
    return bounds, labels, limits


def _boundary_hits(
    model: "pm.Model",
    point: dict[str, Any],
    labels: list[str],
    limits: list[float | None],
    *,
    rel_tol: float = 1e-6,
) -> tuple[BoundaryHit, ...]:
    """Coordinates that finished on (or within ``rel_tol`` of) their guardrail."""
    transforms = {
        getattr(rv, "name", vv.name): tname for vv, rv, tname in _value_var_rvs(model)
    }
    flat: list[float] = []
    for value_var, _rv, _t in _value_var_rvs(model):
        if value_var.name not in point:
            return ()
        flat.extend(np.asarray(point[value_var.name], dtype=float).ravel().tolist())
    if len(flat) != len(labels):  # pragma: no cover - shape drift guard
        return ()

    hits = []
    for label, limit, value in zip(labels, limits, flat):
        if limit is None or not np.isfinite(value):
            continue
        if abs(abs(value) - limit) <= rel_tol * limit:
            base = label.split("[")[0]
            hits.append(
                BoundaryHit(
                    parameter=label,
                    unconstrained_value=float(value),
                    limit=float(limit),
                    transform=transforms.get(base) or "unconstrained",
                )
            )
    return tuple(hits)


def _likelihood_curvature(
    model: "pm.Model",
    point: dict[str, Any],
    value_vars: list[Any],
) -> np.ndarray | None:
    """Hessian of the negative log **likelihood** alone, or ``None``.

    ``model.logp() == model.varlogp + model.datalogp`` — prior plus data — and
    Hessians are additive, so this one matrix splits the posterior curvature
    into the part the data paid for and the part the prior did. The identity
    holds to machine precision (``max|H_post - (H_prior + H_lik)| == 0`` on
    every model in ``tests/test_identification_diagnostics.py``), so the prior
    side is taken by subtraction rather than compiled a third time.

    ``datalogp`` is ``observedlogp + potentiallogp``, which puts every
    ``pm.Potential`` on the likelihood side of the split. That is right for a
    potential carrying an observation (the LCA garden model's ``logsumexp``
    mixture likelihood) and wrong for one encoding a soft prior constraint;
    such a model will read as better-informed than it is.
    """
    try:
        from pymc.pytensorf import hessian, rewrite_pregrad

        with warnings.catch_warnings():
            # A parameter absent from the likelihood graph makes PyTensor warn
            # that it was asked to differentiate a disconnected variable. Here
            # that disconnection IS the measurement — zero likelihood curvature
            # is what `PriorDetermined` reports — so the warning is noise, and
            # letting it out would mean the diagnostic gets loudest exactly when
            # it succeeds. The resulting zero row is correct and is used as-is.
            warnings.filterwarnings(
                "ignore",
                message=".*not part of the computational graph.*",
                category=UserWarning,
            )
            graph = hessian(
                rewrite_pregrad(model.datalogp), value_vars, negate_output=False
            )
            compiled = model.compile_fn(
                inputs=model.value_vars, outs=graph, on_unused_input="ignore"
            )
            matrix = -np.asarray(
                compiled({v.name: point[v.name] for v in model.value_vars}),
                dtype=float,
            )
    except Exception:  # noqa: BLE001 - attribution is an enrichment, never a gate
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.isfinite(matrix).all():
        return None
    return (matrix + matrix.T) / 2.0


def weak_identification_report(
    model: "pm.Model",
    point: dict[str, Any],
    *,
    condition_index_tol: float = 30.0,
    uninformed_tol: float = 1e-8,
    attribute: bool = True,
    prior_determined_tol: float = 0.05,
    max_directions: int = 4,
    loading_tol: float = 0.15,
    max_loadings: int = 4,
    guardrail_engaged: bool = False,
) -> IdentificationReport:
    """Report the directions a point fit could not resolve.

    Three distinct findings, which are routinely confused:

    * **ridges** (:class:`FlatDirection`) — parameter *combinations* that trade
      off. Detected on the correlation-scaled Hessian so the answer does not
      change when a parameter is rescaled. The classic MMM ridge is the
      saturation elbow against the channel coefficient.
    * **uninformed parameters** (:class:`UninformedParameter`) — coordinates
      with no curvature of their own. Nothing is trading off; the density is
      simply flat. This is what sends an optimizer to the support boundary.
    * **prior-determined parameters** (:class:`PriorDetermined`) — coordinates
      with plenty of curvature, all of it the prior's. Needs ``attribute=True``.
    * **boundary hits** (:class:`BoundaryHit`) — coordinates that finished on
      the numerical guardrail.

    plus ``effective_parameters``, how many parameters the data determines.

    Parameters
    ----------
    model
        The PyMC model. Only its log density is used, so this works for any
        model — core MMM, extension, or garden.
    point
        A point dict as returned by ``pm.find_MAP`` (must contain the
        *transformed* value-variable names, which ``find_MAP`` includes by
        default via ``include_transformed=True``).
    condition_index_tol
        Belsley condition index above which a scaled eigenvalue is reported as
        a ridge. 30 is the conventional collinearity threshold.
    uninformed_tol
        A parameter is "uninformed" when its own curvature is below this
        fraction of the best-determined parameter's.
    attribute
        Split the curvature into prior and likelihood contributions. Costs one
        extra compiled Hessian; degrades silently to ``None`` fields if the
        model's ``datalogp`` cannot be differentiated twice.
    prior_determined_tol
        A parameter is "prior-determined" when the likelihood supplies less than
        this share of its curvature.
    max_directions
        Cap on reported ridges, flattest first.
    loading_tol, max_loadings
        Only eigenvector components at or above ``loading_tol`` in absolute
        value are named, up to ``max_loadings`` of them.
    guardrail_engaged
        Recorded on the report; set by :func:`guarded_find_MAP` when the box was
        needed to complete the fit.

    Returns
    -------
    IdentificationReport
        ``verdict`` is ``"non-identified"`` on negative curvature, a boundary
        hit, or an uninformed parameter — the fit is standing somewhere it
        should not be. It is ``"weak"`` when only ridges or prior-determined
        parameters are found: the fit is sound and some of its numbers are the
        prior's, which in an MMM is by design (saturation is not identified from
        the sales likelihood alone). ``"ok"`` otherwise.

    Notes
    -----
    Best-effort: if the Hessian cannot be compiled or evaluated (a graph with no
    second derivative, a non-finite point), the report comes back with a note
    rather than raising — a diagnostic must never be the reason a fit fails.
    """
    value_vars = list(model.continuous_value_vars)
    labels, limits = _labels_and_limits(model)
    hits = _boundary_hits(model, point, labels, limits)

    try:
        # `compile_d2logp` resolves `vars` against the model's RVs, not its value
        # variables — passing value vars raises "Requested variable ... not found
        # among the model variables". Passing the RVs *in continuous_value_vars
        # order* is what pins the Hessian's row order to `labels`; verified equal
        # to the `vars=None` default, which is the ordering `find_MAP` ravels in.
        rvs = [model.values_to_rvs[v] for v in value_vars]
        # `negate_output=False` and negate here, rather than letting PyMC do it.
        # PyMC warns that `hessian` will stop negating in a future version, and
        # emits that FutureWarning whenever negate_output is True — explicitly
        # passed or not. Doing it ourselves is silent AND survives the default
        # flip, and the sign is load-bearing: this must be the Hessian of the
        # NEGATIVE log density, which is PSD at a true maximum, so that a
        # negative eigenvalue means a saddle.
        d2 = model.compile_d2logp(vars=rvs, jacobian=True, negate_output=False)
        hessian = -np.asarray(
            d2({v.name: point[v.name] for v in value_vars}), dtype=float
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics never block a fit
        return IdentificationReport(
            verdict="unknown",
            boundary_hits=hits,
            guardrail_engaged=guardrail_engaged,
            notes=(f"curvature unavailable ({type(exc).__name__}: {exc})",),
        )

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        return IdentificationReport(
            verdict="unknown",
            boundary_hits=hits,
            guardrail_engaged=guardrail_engaged,
            notes=("curvature unavailable (unexpected Hessian shape)",),
        )
    if not np.isfinite(hessian).all():
        return IdentificationReport(
            verdict="non-identified",
            boundary_hits=hits,
            guardrail_engaged=guardrail_engaged,
            notes=(
                "the curvature matrix contains non-finite entries — the fit sits "
                "where the log density is not twice differentiable",
            ),
        )

    symmetric = (hessian + hessian.T) / 2.0
    if len(labels) != symmetric.shape[0]:  # pragma: no cover - ordering drift
        labels = [f"param[{i}]" for i in range(symmetric.shape[0])]

    diagonal = np.abs(np.diag(symmetric))
    largest_diagonal = float(diagonal.max())
    if largest_diagonal == 0.0:  # pragma: no cover - a completely flat posterior
        return IdentificationReport(
            verdict="non-identified",
            boundary_hits=hits,
            guardrail_engaged=guardrail_engaged,
            n_parameters=len(labels),
            notes=("the log density is flat in every direction",),
        )

    uninformed = tuple(
        UninformedParameter(
            parameter=labels[i],
            curvature=float(diagonal[i]),
            relative_curvature=float(diagonal[i] / largest_diagonal),
        )
        for i in np.argsort(diagonal)
        if diagonal[i] / largest_diagonal < uninformed_tol
    )

    # Correlation-scale the curvature so eigenvalues do not depend on the units
    # each parameter happens to be measured in. A positive diagonal scaling is a
    # congruence, so eigenvalue SIGNS survive it (Sylvester) and the saddle test
    # below stays valid.
    floor = np.finfo(float).tiny
    scaling = 1.0 / np.sqrt(np.maximum(diagonal, floor))
    scaled = symmetric * scaling[:, None] * scaling[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(scaled)

    largest = float(eigenvalues.max())
    smallest = float(eigenvalues.min())
    negative_tol = -abs(largest) * 1e-10

    # Split the curvature into what the data paid for and what the prior did.
    # Per DIRECTION: f_j = (v_j' H_lik v_j) / lambda_j, computed in the scaled
    # space (the trace is invariant to the diagonal scaling, so this is the same
    # quantity as in the raw space). Their sum is tr(H_post^-1 H_lik) — the
    # Bayesian effective number of parameters — because for H = V L V',
    # tr(H^-1 H_lik) = sum_j (v_j' H_lik v_j) / lambda_j. No inversion, which
    # matters precisely because H_post is singular in the interesting cases.
    likelihood = _likelihood_curvature(model, point, value_vars) if attribute else None
    direction_fractions: dict[int, float] = {}
    prior_determined: tuple[PriorDetermined, ...] = ()
    effective_parameters: float | None = None
    if likelihood is not None and likelihood.shape == symmetric.shape:
        scaled_likelihood = likelihood * scaling[:, None] * scaling[None, :]
        total = 0.0
        for j in range(len(eigenvalues)):
            vector = eigenvectors[:, j]
            denominator = float(eigenvalues[j])
            if abs(denominator) <= floor:
                continue
            fraction = float(vector @ scaled_likelihood @ vector) / denominator
            direction_fractions[j] = fraction
            total += fraction
        effective_parameters = total

        likelihood_diagonal = np.abs(np.diag(likelihood))
        prior_determined = tuple(
            PriorDetermined(
                parameter=labels[i],
                informed_fraction=float(likelihood_diagonal[i] / diagonal[i]),
            )
            for i in np.argsort(likelihood_diagonal / np.maximum(diagonal, floor))
            # Skip coordinates already reported as uninformed: there the ratio
            # is 0/0 and says nothing. A prior-determined parameter is one the
            # prior genuinely holds up, not one nothing holds up.
            if diagonal[i] / largest_diagonal >= uninformed_tol
            and likelihood_diagonal[i] / diagonal[i] < prior_determined_tol
        )

    def _loadings(index: int) -> tuple[tuple[str, float], ...]:
        vector = eigenvectors[:, index]
        order = np.argsort(-np.abs(vector))
        named = tuple(
            (labels[k], float(vector[k]))
            for k in order[:max_loadings]
            if abs(vector[k]) >= loading_tol
        )
        top = int(order[0])
        return named or ((labels[top], float(vector[top])),)

    directions: list[FlatDirection] = []
    for index in np.argsort(eigenvalues):
        value = float(eigenvalues[index])
        if len(directions) >= max_directions:
            break
        if value < negative_tol:
            directions.append(
                FlatDirection(
                    eigenvalue=value,
                    condition_index=float("inf"),
                    loadings=_loadings(int(index)),
                    kind="nonconvex",
                    informed_fraction=direction_fractions.get(int(index)),
                )
            )
            continue
        index_value = float(np.sqrt(largest / max(value, floor)))
        if index_value <= condition_index_tol:
            break
        directions.append(
            FlatDirection(
                eigenvalue=value,
                condition_index=index_value,
                loadings=_loadings(int(index)),
                kind="ridge",
                informed_fraction=direction_fractions.get(int(index)),
            )
        )

    overall_index = float(np.sqrt(largest / max(abs(smallest), floor)))
    if any(d.kind == "nonconvex" for d in directions) or uninformed or hits:
        # Degenerate: the fit is standing somewhere it should not be.
        verdict = "non-identified"
    elif directions or prior_determined:
        # The fit is fine; some of its numbers are the prior's. Expected in an
        # MMM by design — saturation is not identified from the sales
        # likelihood — so this is a "know what you are reading", not a fault.
        verdict = "weak"
    else:
        verdict = "ok"

    return IdentificationReport(
        verdict=verdict,
        flat_directions=tuple(directions),
        uninformed=uninformed,
        prior_determined=prior_determined,
        boundary_hits=hits,
        effective_parameters=effective_parameters,
        condition_index=overall_index,
        min_scaled_eigenvalue=smallest,
        n_parameters=len(labels),
        guardrail_engaged=guardrail_engaged,
    )


def has_guardable_parameters(model: "pm.Model") -> bool:
    """Whether any coordinate has a transform the guardrail knows how to bound."""
    return any(
        tname in SAFE_UNCONSTRAINED_LIMITS for _vv, _rv, tname in _value_var_rvs(model)
    )


def bounded_find_MAP(
    model: "pm.Model",
    **kwargs: Any,
) -> tuple[dict[str, Any], IdentificationReport]:
    """``pm.find_MAP`` inside the guardrail box, plus the identification report.

    Always passes bounds, so it always takes scipy's bounded code path. Use
    :func:`guarded_find_MAP` unless you specifically want the box unconditionally
    — this is the recovery routine, not the default one.

    A method that cannot accept ``bounds=`` is replaced with ``L-BFGS-B``, and
    the substitution is recorded in the report's ``notes``.
    """
    import pymc as pm

    bounds, _labels, _limits = unconstrained_box(model)
    call = dict(kwargs)
    notes: tuple[str, ...] = ()
    method = call.get("method", "L-BFGS-B")
    if method not in _BOUNDS_CAPABLE:
        call["method"] = _FALLBACK_METHOD
        notes = (
            f"optimizer switched from {method} to {_FALLBACK_METHOD}, which "
            "accepts bounds",
        )
    call["bounds"] = bounds

    point = pm.find_MAP(model=model, **call)
    found = weak_identification_report(model, point, guardrail_engaged=True)
    if notes:
        found.notes = tuple(found.notes) + notes
    return point, found


def guarded_find_MAP(
    model: "pm.Model",
    *,
    report: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, Any], IdentificationReport | None]:
    """``pm.find_MAP`` that survives a transform saturating on its boundary.

    Runs the optimization exactly as ``pm.find_MAP`` would. If — and only if —
    that raises ``ZeroDivisionError`` (the #203 signature: a bounded parameter
    driven far enough out in unconstrained space that its forward map lands
    exactly on the support boundary), it retries inside the box from
    :func:`unconstrained_box` and returns an :class:`IdentificationReport`
    explaining which directions the fit could not pin down.

    A fit that succeeds unguarded is byte-identical to ``pm.find_MAP`` — no
    bounds are ever passed, so scipy takes the same code path it takes today.

    Parameters
    ----------
    report
        Compute the identification report even when the guardrail was not
        needed. Off by default: it costs a second-derivative compile, and a fit
        that never approached a boundary usually does not need it.
    **kwargs
        Passed through to ``pm.find_MAP``.

    Returns
    -------
    (point, report)
        ``report`` is ``None`` only when the fit succeeded unguarded and
        ``report=False``.

    Raises
    ------
    ZeroDivisionError
        Re-raised, with the mechanism and the ways out named, when the model has
        no bounded parameters to guard or the guarded retry also fails.
    """
    import pymc as pm

    try:
        point = pm.find_MAP(model=model, **kwargs)
    except ZeroDivisionError as exc:
        message = boundary_failure_message(model)
        if not has_guardable_parameters(model):
            raise ZeroDivisionError(message) from exc
        try:
            point, found = bounded_find_MAP(model, **kwargs)
        except Exception as retry_exc:  # noqa: BLE001
            raise ZeroDivisionError(
                f"{message}\n\nA bounded retry was attempted and also failed "
                f"({type(retry_exc).__name__}: {retry_exc})."
            ) from exc
        warnings.warn(guardrail_warning(found), UserWarning, stacklevel=2)
        return point, found

    return (
        (point, weak_identification_report(model, point)) if report else (point, None)
    )


def guardrail_warning(found: IdentificationReport) -> str:
    """The warning text for a fit that only completed inside the guardrail."""
    return (
        "The optimizer drove a bounded parameter onto its support boundary, "
        "where the density's gradient is undefined. The fit was completed "
        "inside a numerical guardrail, but the estimate is NOT trustworthy on "
        "the directions below — re-fit with NUTS before using it.\n"
        f"{found.summary()}"
    )


def boundary_failure_message(model: "pm.Model") -> str:
    """The #203 explanation, naming this model's bounded parameters."""
    bounded = sorted(
        getattr(rv, "name", vv.name)
        for vv, rv, tname in _value_var_rvs(model)
        if tname in SAFE_UNCONSTRAINED_LIMITS
    )
    return (
        "MAP optimization drove a bounded parameter onto its support boundary, "
        "where its prior's gradient is undefined.\n\n"
        "find_MAP optimizes in the unconstrained space: float64 sigmoid "
        "saturates to exactly 1.0 past ~37 and exp underflows to exactly 0.0 "
        "past -746, so `1 - alpha` (or `alpha`) becomes exactly zero and the "
        "density divides by it. The transformed parameters in this model are: "
        f"{', '.join(bounded) or '(none found)'}.\n\n"
        "This is a property of gradient-based point estimation on constrained "
        "parameters, not a wrong model. Ways out, cheapest first:\n"
        "  * fit(method='advi') or NUTS — neither walks the boundary the same "
        "way. NUTS is the right call if you need intervals;\n"
        "  * give the offending parameter a prior with less mass at the edge "
        "(e.g. AdstockConfig.geometric(alpha_prior=PriorConfig(distribution="
        "'Beta', params={'alpha': 2.0, 'beta': 5.0})));\n"
        "  * check the channel actually has enough spend variation to inform "
        "its carryover — a flat likelihood is what lets the optimizer wander "
        "this far.\n\n"
        "Tracking: https://github.com/redam94/mmm-framework/issues/203"
    )
