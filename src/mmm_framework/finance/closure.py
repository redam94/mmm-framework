"""Does the decomposition add up, and what is the gap called?

The model's own identity is::

    mu = intercept + trend + seasonality + geo + product + media + controls
         (+ events + synergy + levers)

Components sum to the **fitted** mean. They do not sum to the observed KPI, and
the difference is a real quantity. Several surfaces used to define a baseline as
``observed - modelled media`` instead, which makes that difference vanish into a
bar labelled "base demand" (issue #220). This module computes the honest version
once, so the CFO one-pager, the YoY waterfall and the deck stop each deriving
their own.

**The residual is not evidence the baseline is right.** This is the trap worth
stating twice. Measured on a clean synthetic world where the truth is known, the
disclosed residual was ``-51.6`` while the modelled baseline's actual error
against planted truth was ``+712`` — the residual understated the baseline's
real error by roughly 14x. Under a Gaussian likelihood with a free intercept, a
near-zero residual is an accounting property of the fit, not a validation of it.
So :class:`ClosureFacts` carries an interval on the modelled baseline next to
the residual, and :meth:`ClosureFacts.residual_reading` writes the caveat that
every renderer must show.

Where ``fitted_total`` comes from, in order:

* ``compute_component_decomposition()`` — the core :class:`BayesianMMM` path,
  and the only one that is the model's stated identity rather than a proxy for
  it. Under a multiplicative specification this is the exact LMDI
  reconstruction, and the gap to ``predict()`` is carried separately as
  :attr:`ClosureFacts.jensen_gap` rather than folded in.
* the ``mu`` Deterministic — the extension families register it in original
  units; core ``BayesianMMM`` does not register one at all.
* the posterior-predictive mean, which is a proxy and is labelled as one.

Needs numpy and a fitted model, unlike the rest of this package. See
:mod:`mmm_framework.finance.lines` for the vocabulary the bridge lines use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lines import ABSORBING, MODELLED, RESIDUAL, BridgeLine

__all__ = [
    "FITTED_COMPONENT_FIELDS",
    "MATERIAL_UNEXPLAINED_PCT",
    "ClosureFacts",
    "MediaReconciliation",
    "decomposition_closure",
    "fitted_total",
]

#: Component totals that sum to the model's fitted outcome. Kept explicit (not a
#: wildcard over the dataclass) so a new component added upstream shows up as a
#: widening residual rather than silently changing what "fitted" means.
FITTED_COMPONENT_FIELDS: tuple[str, ...] = (
    "total_intercept",
    "total_trend",
    "total_seasonality",
    "total_media",
    "total_controls",
    "total_geo",
    "total_product",
    "total_events",
    "total_interactions",
    "total_levers",
)

#: Above this share of the observed total, the residual gets its own card. Set
#: from measured fits rather than picked: Student-t worlds land at -0.86% and
#: +1.50%, well-specified Normal MAP fits at ~0.14%.
MATERIAL_UNEXPLAINED_PCT = 0.005

#: Beyond this relative divergence between the two media totals a model can
#: report, they are treated as disagreeing rather than as rounding. Measured on
#: a NestedMMM the two differ by ~10x, so the threshold is nowhere near tight.
_RECONCILE_TOL = 0.05


@dataclass(frozen=True)
class MediaReconciliation:
    """The two media totals a fitted model can report, and whether they agree.

    A decomposition can close to the fitted total perfectly while the media line
    inside it is badly wrong; closing a bridge around a wrong number is the
    failure mode this exists to catch. Measured on a ``NestedMMM``, the
    decomposition media total was ``2108.8`` while the same model's
    ``sample_channel_contributions`` gave ``22634.2`` against a planted truth of
    ``19591.7`` — the bridge closed, and the media number was off by ~10x.
    """

    decomposition_media: float | None
    contribution_media: float | None
    ratio: float | None
    agrees: bool
    reason: str = ""

    def describe(self) -> str:
        if self.agrees:
            return "The decomposition and contribution media totals agree."
        if self.reason:
            return self.reason
        return (
            f"Media total disagrees between sources: decomposition "
            f"{self.decomposition_media:,.4g} vs contributions "
            f"{self.contribution_media:,.4g} (ratio {self.ratio:,.3g}). The "
            "bridge may close while the media line inside it is wrong."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decomposition_media": self.decomposition_media,
            "contribution_media": self.contribution_media,
            "ratio": self.ratio,
            "agrees": self.agrees,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClosureFacts:
    """Whether a fit's decomposition closes, and what the gap is called.

    ``fitted_total`` is the model's own sum of components. ``unexplained`` is
    ``observed_total - fitted_total`` and is a **disclosed** number, never
    folded into a component. The identity every consumer may rely on::

        sum(components) + unexplained == observed_total

    When ``closure_available`` is ``False`` no fitted total could be
    established; ``unexplained`` is then ``None`` and any baseline a caller
    renders necessarily absorbs the residual, which it must say.
    """

    observed_total: float
    fitted_total: float | None
    unexplained: float | None
    unexplained_pct: float | None
    basis: str
    """How ``fitted_total`` was obtained: ``"components"``, ``"mu"``,
    ``"predictive_mean"`` or ``"unavailable"``."""
    closure_available: bool
    specification: str
    """``"additive"`` or ``"multiplicative"``."""

    media_total: float | None = None
    baseline_total: float | None = None
    baseline_lower: float | None = None
    baseline_upper: float | None = None
    interval_mass: float | None = None
    baseline_interval_basis: str = ""
    """What the baseline interval does and does not include — ``"media_only"``
    when it propagates media uncertainty against a point fitted total, which is
    narrower than the truth."""

    jensen_gap: float | None = None
    """Multiplicative fits only: LMDI reconstruction minus the posterior-
    predictive mean. The additive reconstruction of a log-scale model and the
    mean of its exponentiated predictions are different quantities, and the
    difference belongs on the page rather than inside a component."""

    reconciliation: MediaReconciliation | None = None
    lines: list[BridgeLine] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def closes(self) -> bool:
        """``True`` when the lines add to the observed total within 1e-6."""
        if not self.closure_available or self.fitted_total is None:
            return False
        gap = self.observed_total - (self.fitted_total + (self.unexplained or 0.0))
        return abs(gap) <= max(abs(self.observed_total) * 1e-6, 1e-9)

    @property
    def is_material(self) -> bool:
        """``True`` when the residual is large enough to deserve its own card."""
        if self.unexplained_pct is None:
            return False
        return abs(self.unexplained_pct) >= MATERIAL_UNEXPLAINED_PCT

    @property
    def baseline_interval_width(self) -> float | None:
        if self.baseline_lower is None or self.baseline_upper is None:
            return None
        return float(self.baseline_upper - self.baseline_lower)

    def residual_reading(self) -> str:
        """The sentence a renderer must show beside the residual.

        Written here rather than in each template because the number invites
        exactly the wrong conclusion. A residual near zero says the fit
        reproduces its own training total; it says nothing about whether the
        baseline inside that total is the right size, and the two can differ by
        an order of magnitude.
        """
        if not self.closure_available:
            return (
                "No fitted total could be established for this model, so the "
                "residual is unknown and any baseline shown here is a leftover "
                "that absorbs it."
            )
        pct = self.unexplained_pct
        head = (
            f"The model's components sum to {self.fitted_total:,.4g} against an "
            f"observed {self.observed_total:,.4g}, leaving "
            f"{self.unexplained:,.4g}"
        )
        head += f" ({pct:+.2%}) unexplained." if pct is not None else " unexplained."
        tail = (
            " A small residual is an accounting property of a fit with a free "
            "intercept, not a validation of the baseline: read it next to the "
            "interval on the modelled baseline, which is the quantity that "
            "actually carries the uncertainty."
        )
        return head + tail

    def to_dict(self) -> dict[str, Any]:
        """Payload shape for report facts and REST responses."""
        return {
            "observed_total": self.observed_total,
            "fitted_total": self.fitted_total,
            "unexplained": self.unexplained,
            "unexplained_pct": self.unexplained_pct,
            "basis": self.basis,
            "closure_available": self.closure_available,
            "closes": self.closes,
            "is_material": self.is_material,
            "specification": self.specification,
            "media_total": self.media_total,
            "baseline_total": self.baseline_total,
            "baseline_lower": self.baseline_lower,
            "baseline_upper": self.baseline_upper,
            "interval_mass": self.interval_mass,
            "baseline_interval_basis": self.baseline_interval_basis,
            "jensen_gap": self.jensen_gap,
            "reconciliation": (
                self.reconciliation.to_dict() if self.reconciliation else None
            ),
            "lines": [ln.to_dict() for ln in self.lines],
            "warnings": list(self.warnings),
            "residual_reading": self.residual_reading(),
        }


def _observed_total(model: Any) -> float:
    """The observed KPI total. ``y_raw`` on the core model, ``y`` on extensions."""
    for attr in ("y_raw", "y"):
        y = getattr(model, attr, None)
        if y is None:
            continue
        total = float(np.nansum(np.asarray(y, dtype=float)))
        if np.isfinite(total):
            return total
    raise ValueError(
        "Model exposes neither `y_raw` nor `y`; the observed total a closure "
        "reconciles against cannot be established."
    )


def _component_sum(decomp: Any) -> tuple[float | None, float | None]:
    """``(fitted_total, media_total)`` from a ``ComponentDecomposition``."""
    total = 0.0
    seen = False
    for name in FITTED_COMPONENT_FIELDS:
        v = getattr(decomp, name, None)
        if v is None:
            continue
        fv = float(np.sum(np.asarray(v, dtype=float)))
        if np.isfinite(fv):
            total += fv
            seen = True
    if not seen or not np.isfinite(total):
        return None, None
    media = getattr(decomp, "total_media", None)
    media_f = None
    if media is not None:
        m = float(np.sum(np.asarray(media, dtype=float)))
        media_f = m if np.isfinite(m) else None
    return float(total), media_f


def _mu_draws(model: Any) -> np.ndarray | None:
    """Per-draw fitted mean ``(n_draws, n_obs)`` from a ``mu`` Deterministic.

    The extension families register ``mu`` in original units; core
    ``BayesianMMM`` registers no such variable, so this returns ``None`` there
    and the caller falls through.
    """
    trace = getattr(model, "_trace", None)
    posterior = getattr(trace, "posterior", None) if trace is not None else None
    if posterior is None or "mu" not in posterior:
        return None
    try:
        vals = np.asarray(posterior["mu"].values, dtype=float)
    except Exception:  # noqa: BLE001 — a malformed trace is not a closure error
        return None
    if vals.ndim < 2:
        return None
    return vals.reshape(-1, vals.shape[-1])


def fitted_total(model: Any) -> tuple[float | None, str, float | None]:
    """``(total, basis, media_total)`` for the model's fitted outcome.

    Exported because the CFO rollup needs the same number without paying for a
    second round of contribution sampling. ``basis`` is one of ``"components"``,
    ``"mu"``, ``"predictive_mean"``, ``"unavailable"``.
    """
    try:
        decomp = model.compute_component_decomposition()
    except Exception:  # noqa: BLE001 — fall through to the next source
        decomp = None
    if decomp is not None:
        total, media = _component_sum(decomp)
        if total is not None:
            return total, "components", media

    mu = _mu_draws(model)
    if mu is not None:
        total = float(np.nansum(mu.mean(axis=0)))
        if np.isfinite(total):
            return total, "mu", None

    try:
        pred = model.predict(return_original_scale=True, random_seed=0)
        total = float(np.nansum(np.asarray(pred.y_pred_mean, dtype=float)))
        if np.isfinite(total):
            return total, "predictive_mean", None
    except Exception:  # noqa: BLE001
        pass
    return None, "unavailable", None


def _predictive_total(model: Any) -> float | None:
    try:
        pred = model.predict(return_original_scale=True, random_seed=0)
        total = float(np.nansum(np.asarray(pred.y_pred_mean, dtype=float)))
        return total if np.isfinite(total) else None
    except Exception:  # noqa: BLE001
        return None


def _media_draws(model: Any, *, max_draws: int, random_seed: int) -> np.ndarray | None:
    """Per-draw total media contribution, or ``None`` when unavailable.

    Returns ``None`` rather than raising on a multiplicative spec:
    ``sample_channel_contributions`` refuses there by design (it works on the
    additive log scale), and the LMDI media total from the decomposition is the
    correct answer in that case.
    """
    X = getattr(model, "X_media_raw", None)
    if X is None:
        X = getattr(model, "X_media", None)
    if X is None:
        return None
    try:
        draws = model.sample_channel_contributions(
            X_media=np.asarray(X, dtype=float),
            max_draws=max_draws,
            random_seed=random_seed,
        )
    except Exception:  # noqa: BLE001 — includes the multiplicative refusal
        return None
    arr = np.asarray(draws, dtype=float)
    if arr.ndim != 3:
        return None
    return arr.sum(axis=(1, 2))


def _eti(draws: np.ndarray, mass: float) -> tuple[float, float]:
    """Equal-tailed interval, matching the convention used across reporting."""
    lo = float(np.percentile(draws, 100.0 * (1.0 - mass) / 2.0))
    hi = float(np.percentile(draws, 100.0 * (1.0 + mass) / 2.0))
    return lo, hi


#: Relative width below which an interval is no interval. Deliberately loose
#: enough to catch a tiny-but-nonzero span that renders as an identical pair of
#: bounds, which misleads exactly as much as an exactly-zero one (#249).
_COLLAPSE_RATIO = 1e-6


def _is_collapsed(lo: float, hi: float, draws: np.ndarray) -> bool:
    """``True`` when this interval has no width worth reporting.

    A single-draw posterior is the usual cause and is tested directly rather
    than inferred from the width, because it is the reason rather than a
    symptom: an approximate fit produces one draw, so every interval computed
    from it lands on its own point estimate.
    """
    if np.asarray(draws).size < 2:
        return True
    scale = max(abs(lo), abs(hi), 1.0)
    return (hi - lo) <= scale * _COLLAPSE_RATIO


def decomposition_closure(
    model: Any,
    results: Any = None,
    *,
    hdi_prob: float = 0.90,
    max_draws: int = 300,
    random_seed: int = 0,
    decomposition_media: float | None = None,
) -> ClosureFacts:
    """Reconcile a fitted model's components against the observed KPI total.

    Parameters
    ----------
    model:
        A fitted :class:`~mmm_framework.model.base.BayesianMMM` or extension
        model.
    results:
        Accepted for call-site symmetry with the reporting extractors. The
        closure reads the model, which is the only object that carries the
        component identity; passing results does not change the answer.
    hdi_prob:
        Mass of the interval carried on the modelled baseline. Stated on the
        line rather than assumed, so a renderer never has to guess.
    max_draws, random_seed:
        Posterior thinning and the seed for the contribution draws behind the
        baseline interval.
    decomposition_media:
        A second, independently-derived media total to reconcile against the
        contribution draws. The core model supplies its own (from
        ``compute_component_decomposition()``) and this argument is unnecessary
        there. **Extension models need it**: their second source is
        ``ExtendedMMMExtractor.component_totals``, which lives in the reporting
        layer, and only the caller knows which of its keys are media. Without
        it the extension branch reports one media total and cannot tell you
        whether the other agrees — see :class:`MediaReconciliation` for why that
        matters more than the closure itself.

    Returns
    -------
    ClosureFacts
        With ``lines`` ready to render as a bridge. The lines are the *addends*
        and sum to ``observed_total``; the observed total is the target, not a
        line. The bridge closes **because the residual is one of the lines**,
        not because a component was redefined to absorb it.
    """
    observed = _observed_total(model)
    multiplicative = bool(getattr(model, "_multiplicative", False))
    spec = "multiplicative" if multiplicative else "additive"
    warnings: list[str] = []

    total, basis, own_media = fitted_total(model)
    # An explicitly-supplied total wins: the caller reached for it because the
    # model could not produce a second source on its own.
    decomp_media = (
        own_media if decomposition_media is None else float(decomposition_media)
    )

    media_draws = _media_draws(model, max_draws=max_draws, random_seed=random_seed)
    contrib_media = float(np.mean(media_draws)) if media_draws is not None else None

    # Which media number to PUBLISH. The model's own component total when it has
    # one — under a multiplicative spec the LMDI total is the only total correct
    # on the original scale, and the contribution path refuses there rather than
    # returning a wrong number — otherwise the contribution draws.
    #
    # An injected `decomposition_media` deliberately does NOT get published. It
    # arrives from a different layer precisely because the model could not
    # produce it, which makes it the number under suspicion rather than the
    # number to trust: on a NestedMMM the extractor's media total was 2108.8
    # against contributions of 22634.2 and a planted truth of 19591.7. Its job
    # here is to make that disagreement visible, not to win it.
    media_total = own_media if own_media is not None else contrib_media

    reconciliation: MediaReconciliation | None = None
    if decomp_media is not None and contrib_media is not None:
        denom = max(abs(decomp_media), abs(contrib_media))
        ratio = (contrib_media / decomp_media) if abs(decomp_media) > 1e-12 else None
        agrees = denom <= 1e-12 or abs(contrib_media - decomp_media) / denom <= (
            _RECONCILE_TOL
        )
        reconciliation = MediaReconciliation(
            decomposition_media=decomp_media,
            contribution_media=contrib_media,
            ratio=ratio,
            agrees=agrees,
        )
        if not agrees:
            warnings.append(reconciliation.describe())

    jensen_gap: float | None = None
    if multiplicative and total is not None and basis == "components":
        pred_total = _predictive_total(model)
        if pred_total is not None:
            jensen_gap = total - pred_total

    if total is None:
        # Nothing to reconcile against. Say so, and hand back a single absorbing
        # line so a caller that renders anyway renders the caveat with it.
        line = BridgeLine(
            name="Base (residual absorbed)",
            value=observed - (media_total or 0.0),
            provenance=ABSORBING,
            note=(
                "No fitted total could be established, so this line is a "
                "leftover and carries the model residual."
            ),
        )
        lines = [line]
        if media_total is not None:
            lines.insert(
                0,
                BridgeLine(
                    name="Media", value=media_total, provenance=MODELLED, basis=basis
                ),
            )
        warnings.append(
            "No fitted total could be established; the decomposition cannot be "
            "reconciled against the observed KPI."
        )
        return ClosureFacts(
            observed_total=observed,
            fitted_total=None,
            unexplained=None,
            unexplained_pct=None,
            basis=basis,
            closure_available=False,
            specification=spec,
            media_total=media_total,
            reconciliation=reconciliation,
            lines=lines,
            warnings=warnings,
        )

    unexplained = observed - total
    unexplained_pct = unexplained / observed if abs(observed) > 1e-12 else None

    baseline_total = total - media_total if media_total is not None else None
    baseline_lower = baseline_upper = None
    interval_mass: float | None = None
    interval_basis = ""
    if baseline_total is not None and media_draws is not None:
        # Per-draw baseline against a point fitted total. This propagates media
        # uncertainty only, so it is NARROWER than the truth — named, because a
        # width that omits a source of uncertainty and does not say so is worse
        # than no width at all.
        baseline_draws = float(total) - media_draws
        lo, hi = _eti(baseline_draws, hdi_prob)
        if _is_collapsed(lo, hi, baseline_draws):
            # An approximate (MAP/ADVI) fit has a single-draw posterior, so the
            # interval lands on the point estimate. A zero-width interval is the
            # visual language of an extremely precise estimate, which is the
            # opposite of what an approximate fit means (#249). Report it absent.
            interval_basis = "collapsed"
            warnings.append(
                "The interval on the modelled baseline collapsed onto the point "
                "estimate — this fit produced no spread to summarise, which an "
                "approximate (MAP / ADVI) fit does by construction. Read the "
                "residual on its own: there is no width here to weigh it "
                "against, and a residual alone does not validate a baseline."
            )
        else:
            baseline_lower, baseline_upper = lo, hi
            interval_mass = hdi_prob
            interval_basis = "media_only"
    elif baseline_total is not None:
        interval_basis = "unavailable"
        warnings.append(
            "No contribution draws are available for this fit, so the modelled "
            "baseline is a point estimate. Read the residual with that in mind: "
            "there is no interval here to weigh it against."
        )

    lines: list[BridgeLine] = []
    if media_total is not None:
        lines.append(
            BridgeLine(
                name="Media",
                value=media_total,
                provenance=MODELLED,
                basis=("LMDI decomposition" if multiplicative else basis),
            )
        )
    if baseline_total is not None:
        lines.append(
            BridgeLine(
                name="Base (non-marketing)",
                value=baseline_total,
                provenance=MODELLED,
                lower=baseline_lower,
                upper=baseline_upper,
                interval_mass=interval_mass,
                basis=basis,
                note=(
                    "Interval propagates media uncertainty only."
                    if interval_basis == "media_only"
                    else ""
                ),
            )
        )
    if not lines:
        lines.append(
            BridgeLine(
                name="Fitted total", value=total, provenance=MODELLED, basis=basis
            )
        )
    lines.append(
        BridgeLine(
            name="Unexplained",
            value=unexplained,
            provenance=RESIDUAL,
            note="Observed minus fitted. Not a measure of whether the baseline is right.",
        )
    )

    if multiplicative and jensen_gap is not None:
        warnings.append(
            f"Multiplicative fit: the fitted total is the exact LMDI "
            f"reconstruction on the original scale. It differs from the "
            f"posterior-predictive mean by {jensen_gap:,.4g}, which is the "
            "Jensen gap between an exponentiated mean and the mean of "
            "exponentials, not an error in either."
        )

    return ClosureFacts(
        observed_total=observed,
        fitted_total=total,
        unexplained=unexplained,
        unexplained_pct=unexplained_pct,
        basis=basis,
        closure_available=True,
        specification=spec,
        media_total=media_total,
        baseline_total=baseline_total,
        baseline_lower=baseline_lower,
        baseline_upper=baseline_upper,
        interval_mass=interval_mass,
        baseline_interval_basis=interval_basis,
        jensen_gap=jensen_gap,
        reconciliation=reconciliation,
        lines=lines,
        warnings=warnings,
    )
