"""Decision arms: promo depth competes with media for the same money (#226).

The optimizer's decision vector was dollars of media spend — `Σs = B` — which
has no slot for a decision whose cost is not its own value: a promotion's cost
is margin given away (depth × price × units), not a spend line. The epic's
original headline, "trade a price cut against media", is deliberately NOT
shipped as a recommendation: the repo's own published measurement
(``docs/blog-modelled-one-p.html``) recovers 39% of a planted price elasticity
confidently, and 9% with zero designed variation. **Promo depth is the shipping
headline; price is a labelled what-if evaluator that refuses to recommend.**

The governing reduction — and why the shipped diff is small: every arm is
re-parameterized by its **realized cost**. A media arm's cost is its spend
(identity), a promo arm's cost is ``cost_fn(level)``; each arm's response curve
is expressed as contribution-vs-cost on the shared multiplier grid. The
existing allocator then works untouched — the budget constraint is already
``Σ cost = B``, ``mode='free'`` already maximizes ``value·KPI − Σ cost``
(profit), and the KKT water level is again a single number **because the
decision space is homogeneous dollars**: the per-group shadow-price correction
the issue anticipated for a level-space decision vector is made unnecessary by
construction, and the per-arm ``marginal_roas`` labels carry
"per dollar of realized cost" instead.

What refuses, and why:

* :func:`promo_roi` on a **flag-valued** promo column — a 0/1 event flag has no
  depth and therefore no ΔP×Q cost; dividing by a normalized flag prints a
  ratio with no units.
* :func:`promo_roi` / :func:`build_arm_curves` without a **unit cost** or a
  **valuation** — a silent ``value_per_kpi=1.0`` is the exact defect
  :mod:`mmm_framework.finance` exists to prevent.
* :func:`price_whatif` always refuses to emit a recommendation — it evaluates a
  stated scenario and labels the elasticity with its measured attenuation.
* ``compute_response_curves`` (in :mod:`.budget`) now refuses a channel whose
  divisor is not monetary: an impressions column summed into a dollar budget
  was allocated as if impressions were dollars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .budget import (
    BudgetOptimizationResult,
    ResponseCurves,
    check_concavity,
    compute_response_curves,
    optimize_budget,
)

__all__ = [
    "ArmCurves",
    "DecisionArm",
    "PromoArmResult",
    "build_arm_curves",
    "check_concavity",
    "optimize_arms",
    "price_whatif",
    "promo_roi",
]


@dataclass(frozen=True)
class DecisionArm:
    """One decision the allocator may move, with its own axis and cost basis.

    ``levels`` is the arm's decision grid in its OWN units (dollars for media,
    depth fraction for promo); ``cost_fn`` maps a level to realized dollars.
    For media ``cost_fn`` is the identity and everything downstream is
    bit-identical to the pre-arm path.
    """

    name: str
    kind: str  # "media" | "promo"
    levels: np.ndarray
    level_units: str
    base_level: float
    obs_min: float
    obs_max: float
    cost_fn: Callable[[float], float] | None = None

    def cost(self, level: float) -> float:
        return float(level) if self.cost_fn is None else float(self.cost_fn(level))


class ArmCurves(ResponseCurves):
    """A :class:`ResponseCurves` whose "channels" are decision arms in COST
    space, plus the metadata that maps a recommended cost back to a level.

    Subclasses rather than wraps so every existing consumer
    (``optimize_budget``, ``budget_frontier``, the per-draw loop,
    ``within_observed_range``) runs unchanged; the arm metadata rides along
    for the callers that need to re-label.
    """

    def __init__(
        self,
        *,
        arms: list[DecisionArm],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.arms = list(arms)

    @property
    def arm_kinds(self) -> list[str]:
        return [a.kind for a in self.arms]

    @property
    def is_mixed(self) -> bool:
        """True when the portfolio mixes cost bases (media + promo)."""
        return len(set(self.arm_kinds)) > 1

    def level_for_cost(self, arm_name: str, cost: float) -> float:
        """Map a recommended realized cost back to the arm's own level."""
        arm = next(a for a in self.arms if a.name == arm_name)
        if arm.cost_fn is None:
            return float(cost)
        costs = np.array([arm.cost(lv) for lv in arm.levels], dtype=float)
        return float(np.interp(cost, costs, arm.levels))


def _promo_depth_series(model: Any, promo_var: str) -> np.ndarray:
    """The promo lever's RAW observed series off a fitted model."""
    names = list(getattr(model, "lever_names", []) or [])
    if promo_var not in names:
        raise ValueError(
            f"'{promo_var}' is not a lever on this model (levers: {names}). "
            "Configure it with ModelConfigBuilder().with_promotions(...)."
        )
    X = np.asarray(model.X_levers_raw, dtype=float)
    return X[:, names.index(promo_var)]


def _refuse_flag_or_unknown_units(depth: np.ndarray, promo_var: str) -> None:
    """The promo cost basis is refusable (#226).

    A 0/1 event flag has no depth and therefore no ΔP×Q cost; a column outside
    [0, 1] is not a discount fraction and dividing by its max prints a ratio of
    unknown units. Refuse both by name rather than normalize and pretend.
    """
    vals = np.unique(depth[np.isfinite(depth)])
    nonzero = vals[vals != 0]
    if nonzero.size == 0:
        raise ValueError(
            f"promo lever '{promo_var}' is all zero in the training window — "
            "there is no observed depth to build a response curve from."
        )
    if np.all(np.isin(vals, (0.0, 1.0))):
        raise ValueError(
            f"promo lever '{promo_var}' is a 0/1 event flag. A flag has no "
            "depth, so no discount cost (depth x price x units) exists and a "
            "promo ROI computed from it would be a ratio with no units. "
            "Provide the actual discount-depth column instead."
        )
    if float(np.nanmax(depth)) > 1.0 + 1e-9 or float(np.nanmin(depth)) < -1e-9:
        raise ValueError(
            f"promo lever '{promo_var}' takes values outside [0, 1], so it is "
            "not a discount fraction and its cost basis is unknown. The model "
            "normalizes it by its max internally, but a COST cannot be priced "
            "on a normalized column of unknown units — supply the column as a "
            "depth fraction (0.25 = 25% off)."
        )


@dataclass(frozen=True)
class PromoArmResult:
    """A promo ROI with everything it cannot travel without."""

    promo_var: str
    roi_mean: float
    roi_lower: float | None
    roi_upper: float | None
    interval_mass: float
    lift_kpi_mean: float
    realized_cost: float
    unit_cost: float
    value_per_kpi: float
    value_source: str | None
    avg_depth: float
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def promo_roi(
    model: Any,
    promo_var: str,
    *,
    unit_cost: float,
    value_per_kpi: float | None = None,
    value_source: str | None = None,
    hdi_prob: float = 0.90,
    max_draws: int = 300,
    random_seed: int = 0,
) -> PromoArmResult:
    """Margin dollars returned per margin dollar given away, per draw.

    ``lift`` is the incremental KPI of the observed promo calendar — the
    difference of two paired ``sample_lever_contributions`` passes (observed
    depth vs. zero depth) — valued at ``value_per_kpi``; ``cost`` is
    ``avg_depth × unit_cost``, the frozen cost basis. Refuses on a flag-valued
    or unknown-unit promo column and (via
    :class:`~mmm_framework.finance.UnresolvedValueError`) without a valuation —
    never a silent ``1.0``.
    """
    from ..finance import UnresolvedValueError

    if value_per_kpi is None:
        raise UnresolvedValueError(f"Promo ROI for '{promo_var}'")
    depth = _promo_depth_series(model, promo_var)
    _refuse_flag_or_unknown_units(depth, promo_var)

    names = list(model.lever_names)
    X = np.asarray(model.X_levers_raw, dtype=float)
    X0 = X.copy()
    X0[:, names.index(promo_var)] = 0.0
    on = model.sample_lever_contributions(
        max_draws=max_draws, random_seed=random_seed, components=("promo_component",)
    )["promo_component"]
    off = model.sample_lever_contributions(
        X_levers=X0,
        max_draws=max_draws,
        random_seed=random_seed,
        components=("promo_component",),
    )["promo_component"]
    lift_draws = (on - off).sum(axis=1)  # (D,) incremental KPI units

    avg_depth = float(np.mean(depth))
    realized_cost = avg_depth * float(unit_cost)
    if realized_cost <= 0:
        raise ValueError(
            f"promo lever '{promo_var}': realized cost is {realized_cost:.4g} "
            "(avg depth x unit_cost) — nothing was given away, so no ROI exists."
        )
    roi_draws = (lift_draws * float(value_per_kpi)) / realized_cost

    caveats: list[str] = []
    n = roi_draws.shape[0]
    mean = float(roi_draws.mean())
    if n < 2:
        lo = hi = None
        caveats.append(
            "The interval collapsed onto the point estimate — this fit "
            "produced no spread to summarise (approximate MAP/ADVI fits do "
            "this by construction)."
        )
    else:
        lo = float(np.percentile(roi_draws, 100.0 * (1.0 - hdi_prob) / 2.0))
        hi = float(np.percentile(roi_draws, 100.0 * (1.0 + hdi_prob) / 2.0))
        if f"{lo:.3f}" == f"{hi:.3f}":
            lo = hi = None
            caveats.append("The interval collapsed onto the point estimate.")
    caveats.append(
        "ROI is conditional on the stated economics: the unit cost "
        f"({unit_cost:,.4g} per unit average depth) and the valuation "
        f"({value_per_kpi:,.4g}/KPI unit, source: {value_source or 'caller'}) "
        "are inputs, not measurements."
    )
    return PromoArmResult(
        promo_var=promo_var,
        roi_mean=mean,
        roi_lower=lo,
        roi_upper=hi,
        interval_mass=float(hdi_prob),
        lift_kpi_mean=float(lift_draws.mean()),
        realized_cost=realized_cost,
        unit_cost=float(unit_cost),
        value_per_kpi=float(value_per_kpi),
        value_source=value_source,
        avg_depth=avg_depth,
        caveats=caveats,
    )


def build_arm_curves(
    model: Any,
    *,
    promo_var: str,
    unit_cost: float,
    multipliers: tuple[float, ...] | None = None,
    max_draws: int = 200,
    random_seed: int | None = 42,
) -> ArmCurves:
    """Media arms + a promo-depth arm, on one shared realized-cost grid.

    Media curves come from :func:`~.budget.compute_response_curves` unchanged
    (bit-identity pinned by test). The promo arm's decision is **average
    weekly depth**; its response at level ``d`` is one paired
    ``sample_lever_contributions`` pass with the promo column set to
    ``full(n, d)``, and its realized cost is ``d × unit_cost`` — linear, so the
    arm shares the media multiplier grid exactly (``cost = base_cost × m``).

    Support semantics carry over: the promo arm's ``obs_max_spend`` is the cost
    of the deepest single observed week, so ``within_observed_range`` flags an
    out-of-support depth recommendation exactly as it flags spend.
    """
    depth = _promo_depth_series(model, promo_var)
    _refuse_flag_or_unknown_units(depth, promo_var)
    if float(unit_cost) <= 0:
        raise ValueError("unit_cost must be positive — it prices the giveaway.")

    media = compute_response_curves(
        model,
        multipliers=multipliers,
        max_draws=max_draws,
        random_seed=random_seed,
    )
    mults = media.multipliers
    n_obs = int(np.asarray(model.X_levers_raw).shape[0])
    names = list(model.lever_names)
    p_idx = names.index(promo_var)

    base_depth = float(np.mean(depth))
    if base_depth <= 0:
        raise ValueError(
            f"promo lever '{promo_var}' has zero average observed depth; "
            "there is no base level to scale."
        )

    # One paired lever pass per grid point, shared seed so the promo curve's
    # draws difference cleanly. The zero-depth pass anchors the curve at 0.
    X = np.asarray(model.X_levers_raw, dtype=float)
    Xz = X.copy()
    Xz[:, p_idx] = 0.0
    off = model.sample_lever_contributions(
        X_levers=Xz,
        max_draws=max_draws,
        random_seed=random_seed,
        components=("promo_component",),
    )["promo_component"].sum(
        axis=1
    )  # (D,)

    contribs = []
    for m in mults:
        d_level = base_depth * float(m)
        Xm = X.copy()
        Xm[:, p_idx] = d_level
        on = model.sample_lever_contributions(
            X_levers=Xm,
            max_draws=max_draws,
            random_seed=random_seed,
            components=("promo_component",),
        )["promo_component"].sum(axis=1)
        contribs.append(on - off)  # incremental KPI at this depth, per draw
    promo_curve = np.stack(contribs, axis=0).T  # (D, G)

    D = min(media.contributions.shape[0], promo_curve.shape[0])
    contributions = np.concatenate(
        [media.contributions[:D], promo_curve[:D, None, :]], axis=1
    )  # (D, C+1, G)

    base_cost = base_depth * float(unit_cost)
    arm_name = f"Promo depth ({promo_var})"
    arms = [
        DecisionArm(
            name=str(ch),
            kind="media",
            levels=media.base_spend[i] * mults,
            level_units="$",
            base_level=float(media.base_spend[i]),
            obs_min=0.0,
            obs_max=(
                float(media.obs_max_spend[i])
                if media.obs_max_spend is not None
                else float("nan")
            ),
            cost_fn=None,
        )
        for i, ch in enumerate(media.channel_names)
    ] + [
        DecisionArm(
            name=arm_name,
            kind="promo",
            levels=base_depth * mults,
            level_units="avg weekly depth (fraction)",
            base_level=base_depth,
            obs_min=0.0,
            obs_max=float(np.max(depth)),
            cost_fn=lambda lv, _u=float(unit_cost): float(lv) * _u,
        )
    ]

    # The promo arm's per-observation "spend" is that week's realized cost
    # (depth_t x unit_cost / n_obs on the average-year basis), so the
    # max-observed multiplier works out to max(depth)/mean(depth) — exactly
    # the support rule the issue asks for.
    obs_max = (
        np.append(media.obs_max_spend, np.max(depth) * float(unit_cost) / n_obs)
        if media.obs_max_spend is not None
        else None
    )

    return ArmCurves(
        arms=arms,
        channel_names=list(media.channel_names) + [arm_name],
        multipliers=mults,
        base_spend=np.append(media.base_spend, base_cost),
        contributions=contributions,
        obs_max_spend=obs_max,
        n_obs=media.n_obs,
    )


def optimize_arms(
    model: Any = None,
    *,
    curves: ArmCurves | None = None,
    promo_var: str | None = None,
    unit_cost: float | None = None,
    value_per_kpi: float | None = None,
    value_source: str | None = None,
    total_budget: float | None = None,
    objective: str = "mean",
    mode: str = "fixed",
    max_draws: int = 200,
    random_seed: int | None = 42,
    **kwargs: Any,
) -> BudgetOptimizationResult:
    """Joint media + promo-depth allocation on the shared realized-cost grid.

    A thin orchestration over :func:`~.budget.optimize_budget` (which carries
    the risk objectives, constraints, per-draw uncertainty and the concavity
    gate): build the arm curves, run the allocator in cost space, then re-label
    the result rows with each arm's kind, level units and the recommended
    LEVEL (a depth, not a dollar figure, for the promo arm).

    The profit objective is ``mode='free'`` — it already maximizes
    ``value_per_kpi·KPI − Σcost`` and refuses without a valuation. Under
    ``mode='fixed'`` the valuation is optional (scale-free argmax) and the
    result reallocates the stated budget across media dollars AND promo
    margin-giveaway dollars.
    """
    if curves is None:
        if model is None or promo_var is None or unit_cost is None:
            raise ValueError(
                "optimize_arms needs either prebuilt `curves` or "
                "(model, promo_var, unit_cost)."
            )
        curves = build_arm_curves(
            model,
            promo_var=promo_var,
            unit_cost=unit_cost,
            max_draws=max_draws,
            random_seed=random_seed,
        )

    res = optimize_budget(
        curves=curves,
        total_budget=total_budget,
        objective=objective,
        mode=mode,
        value_per_kpi=value_per_kpi,
        value_source=value_source,
        max_draws=max_draws,
        random_seed=random_seed,
        **kwargs,
    )

    # Re-label rows with arm metadata; recommended cost -> recommended level.
    by_name = {a.name: a for a in curves.arms}
    table = res.table.copy()
    table["arm_kind"] = [by_name[ch].kind for ch in table["channel"]]
    table["level_units"] = [by_name[ch].level_units for ch in table["channel"]]
    table["optimal_level"] = [
        curves.level_for_cost(ch, float(spend))
        for ch, spend in zip(table["channel"], table["optimal_spend"])
    ]
    res.table = table
    res.notes.append(
        "Mixed cost bases: media rows are dollars of spend; promo rows are "
        "dollars of margin given away (depth x unit cost), with the "
        "recommended DEPTH in `optimal_level`. Marginal figures are per "
        "dollar of realized cost, which is what makes them comparable."
    )
    return res


def price_whatif(
    model: Any,
    factor: float,
    *,
    hdi_prob: float = 0.90,
    max_draws: int = 300,
    random_seed: int = 0,
) -> dict[str, Any]:
    """A labelled price scenario that REFUSES to emit a recommendation.

    Evaluates "price × factor, everything else observed" through the fitted
    price lever and returns the KPI delta with an interval — and a
    ``recommendation`` field that is always ``None`` with the reason stated:
    the repo's own published measurement recovers 39% of a planted elasticity
    under the shipped mechanism (confidently, with the right sign), and 9%
    with no designed variation. An optimizer pointed at a coefficient measured
    at 39% of truth would recommend price moves whose real P&L consequence is
    ~2.5x what the model believes. The default posture is refusal regardless
    of the endogeneity screen: a Granger-style lead/lag flag not firing is
    weak evidence of exogeneity, not a licence.
    """
    if getattr(model, "_price_lever", None) is None:
        raise ValueError(
            "This model has no price lever. Configure one with "
            "ModelConfigBuilder().with_price(PriceConfig(...))."
        )
    if factor <= 0:
        raise ValueError("factor must be positive (0.95 = a 5% price cut).")

    names = list(model.lever_names)
    price_var = model._price_lever[0].variable
    X = np.asarray(model.X_levers_raw, dtype=float)
    Xs = X.copy()
    Xs[:, names.index(price_var)] = Xs[:, names.index(price_var)] * float(factor)

    on = model.sample_lever_contributions(
        max_draws=max_draws, random_seed=random_seed, components=("price_component",)
    )["price_component"]
    scen = model.sample_lever_contributions(
        X_levers=Xs,
        max_draws=max_draws,
        random_seed=random_seed,
        components=("price_component",),
    )["price_component"]
    delta = (scen - on).sum(axis=1)  # (D,) KPI units

    mean = float(delta.mean())
    if delta.shape[0] < 2:
        lo = hi = None
    else:
        lo = float(np.percentile(delta, 100.0 * (1.0 - hdi_prob) / 2.0))
        hi = float(np.percentile(delta, 100.0 * (1.0 + hdi_prob) / 2.0))
        if f"{lo:.3f}" == f"{hi:.3f}":
            lo = hi = None

    return {
        "price_var": price_var,
        "factor": float(factor),
        "kpi_delta_mean": mean,
        "kpi_delta_lower": lo,
        "kpi_delta_upper": hi,
        "interval_mass": float(hdi_prob),
        "recommendation": None,
        "refusal_reason": (
            "Price recommendations are refused by design: the shipped "
            "sign-guarded elasticity recovers ~39% of a planted truth in the "
            "repo's own measurement (9% with no designed price variation), so "
            "the P&L consequence of a recommended move would be ~2.5x what "
            "the model believes. This scenario is an evaluation of a stated "
            "hypothetical, conditional on the fitted (attenuated) elasticity "
            "— run a pricing experiment before acting."
        ),
    }
