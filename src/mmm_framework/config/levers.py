"""Price and promotion levers (#138).

Price and promotion are among the largest sales drivers for CPG / retail / DTC
and are *decisions the client controls*, so planners want an elasticity and a
promo ROI — not a nuisance control coefficient. These configs promote a control
column to a first-class lever with its own transform and prior:

- **Price** enters as ``beta_price · log(price / reference)`` (a log-price /
  discount-depth term) with a sign guard (elasticity ≤ 0).
- **Promotion** enters as ``beta_promo · adstock(promo)`` — a lift with its own
  carryover (pull-forward / decay), distinct from an instantaneous linear bump.

A variable named here is removed from the linear control block, so it is not
double-counted.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PriceConfig(BaseModel):
    """A price lever (log-price elasticity with an optional reference / gap)."""

    #: Name of the price column (a control variable in the data).
    variable: str
    #: Reference / regular price the model responds to the gap from. A float is
    #: an absolute reference; ``"mean"`` / ``"median"`` / ``"max"`` derive it from
    #: the data (median ≈ the usual "regular" price). ``None`` uses ``log(price)``
    #: with the constant absorbed by the intercept (still a valid elasticity).
    reference: float | Literal["mean", "median", "max"] | None = "median"
    #: Prior sigma of the (sign-guarded, ≤ 0) elasticity coefficient.
    elasticity_prior_sigma: float = Field(default=0.5, gt=0)

    model_config = {"extra": "forbid"}


class PromoConfig(BaseModel):
    """A promotion lever (lift with its own carryover)."""

    #: Name of the promo column (a control variable): a discount % or an event
    #: flag. Normalized by its max before adstock.
    variable: str
    #: Max carryover lag (weeks) for the promo's own geometric adstock. ``<= 1``
    #: ⇒ no carryover (an instantaneous lift).
    adstock_lmax: int = Field(default=4, ge=1, le=52)
    #: Prior sigma of the (non-negative) promo lift coefficient.
    lift_prior_sigma: float = Field(default=0.5, gt=0)
    #: Allow a negative promo effect (e.g. a pull-forward that later depresses
    #: sales). Default False ⇒ a HalfNormal lift ≥ 0.
    allow_negative: bool = False

    model_config = {"extra": "forbid"}
