"""Reach & frequency channel modeling (#141).

A media channel can be declared *reach/frequency-measured*: its column is
**reach** (distinct people reached per period) and its effect is
``reach · g(frequency)`` where ``g`` is a **frequency-saturation** curve —
diminishing returns to added exposures (the "3+ frequency wearout"). This lets
the model answer the core planning question *"buy more reach or more
frequency?"*, which raw impressions (reach × frequency collapsed to a volume)
cannot.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FrequencyResponse(str, Enum):
    """Shape of the frequency-saturation curve ``g(f)`` (``f`` = avg frequency,
    normalized by its mean so the shape prior is scale-free)."""

    #: ``g(f) = 1 - exp(-k f)`` — diminishing returns from the first exposure,
    #: monotone, asymptotes to 1. The safe planner default.
    EXPONENTIAL = "exponential"
    #: ``g(f) = f^s / (f^s + h^s)`` — S-shaped, models a *minimum effective
    #: frequency* threshold (low frequency is nearly wasted, then it kicks in).
    HILL = "hill"


class ReachFrequencyConfig(BaseModel):
    """Declare a channel as reach modulated by a frequency-saturation curve.

    The named ``channel``'s media column is treated as **reach**. The
    ``frequency_column`` is a per-period average-frequency series supplied in the
    control block — it is pulled out (not double-counted as a linear control) and
    used only to build the frequency gain. The channel's effect becomes
    ``beta · sat(adstock(reach · g(frequency)))`` — the standard media pipeline
    with the input re-mixed to *effective reach*.

    Off by default (``ModelConfig.reach_frequency = []``): a channel is an
    ordinary volume channel, byte-identical to today (R0.1/R0.2).
    """

    #: Media channel whose column is reach (must be a modeled channel).
    channel: str
    #: Per-period average-frequency series (must be a control column).
    frequency_column: str
    #: Frequency-saturation shape.
    response: FrequencyResponse = FrequencyResponse.EXPONENTIAL
    #: Prior scale on the shape parameter (``k`` for exponential; the slope /
    #: half-saturation priors for Hill). Larger ⇒ faster frequency saturation
    #: a priori.
    frequency_prior_scale: float = Field(default=1.0, gt=0)

    model_config = {"extra": "forbid"}
