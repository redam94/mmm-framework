"""Cross-channel interaction / synergy configuration (#142).

The base model is strictly additive across channels, so it cannot express that
two channels together do more (synergy / halo — TV priming Search) or less
(cannibalization) than the sum of their parts. A :class:`ChannelInteraction`
adds a ``beta_ij · sat(x_i) · sat(x_j)`` term for a named pair, with a
sign-aware prior shrunk toward zero.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChannelInteraction(BaseModel):
    """One named channel-pair interaction term.

    ``expected_sign`` encodes the belief: ``"positive"`` (synergy) uses a
    HalfNormal so the effect can only lift; ``"negative"`` (cannibalization) its
    reflection; ``"any"`` a Normal centered at 0. Either way the prior shrinks
    toward zero, since interactions are weakly identified without designed
    variation.
    """

    channel_a: str
    channel_b: str
    prior_sigma: float = Field(default=0.3, gt=0)
    expected_sign: Literal["positive", "negative", "any"] = "any"

    model_config = {"extra": "forbid"}

    @field_validator("channel_b")
    @classmethod
    def _distinct(cls, v: str, info) -> str:
        if v == info.data.get("channel_a"):
            raise ValueError("A channel cannot interact with itself.")
        return v

    @property
    def name(self) -> str:
        return f"{self.channel_a}_x_{self.channel_b}"
