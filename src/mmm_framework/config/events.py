"""Holiday / event effect configuration (#137... #143).

A first-class way to declare sharp, date-specific effects — Black Friday, Prime
Day, a product launch, a one-off PR event — that the smooth Fourier seasonality
cannot represent. Each event becomes a regressor column (built by
:func:`mmm_framework.transforms.events.build_event_regressors`) with its own
coefficient, so its contribution is estimated and reported *separately* from
seasonality.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EventSpec(BaseModel):
    """One named event (or recurring holiday).

    Exactly one of ``dates`` (explicit) or ``holiday`` (a named holiday looked up
    from the country calendar) identifies when the event occurs.
    """

    name: str
    #: Explicit event dates (any pandas-parseable date strings), e.g. a launch.
    dates: list[str] | None = None
    #: A named holiday from the config's ``country`` calendar (e.g. "Christmas
    #: Day", "Thanksgiving"). Case-insensitive substring match on the calendar's
    #: holiday names. Recurs every year in the data window.
    holiday: str | None = None
    #: Weeks of "shoulder" before / after the event that also carry an effect
    #: (0 = the event period only).
    pre_weeks: int = Field(default=0, ge=0)
    post_weeks: int = Field(default=0, ge=0)
    #: Geometric decay applied across the shoulder periods (0 = flat 1.0 over the
    #: whole window; 0.5 = each step away from the peak is half as strong).
    decay: float = Field(default=0.0, ge=0.0, lt=1.0)
    #: Prior sigma for this event's coefficient (overrides the group default).
    prior_sigma: float | None = None

    model_config = {"extra": "forbid"}


class EventsConfig(BaseModel):
    """A set of holiday / event regressors to add to the model.

    Off by default (``ModelConfig.events is None``). Effects enter as an additive
    ``event_component`` distinct from the Fourier ``seasonality_component``, so
    they do not double-count.
    """

    #: Country code for named-holiday lookups (e.g. "US", "GB"). Requires the
    #: optional ``holidays`` package. ``None`` ⇒ only ``custom_events`` are used.
    country: str | None = None
    #: Which of the country calendar's holidays to include as regressors. Empty
    #: ⇒ every holiday in the calendar (one regressor per distinct holiday name).
    holidays: list[str] = Field(default_factory=list)
    #: User-defined events (launches, PR moments, or overrides of a named holiday
    #: with custom windows/decay).
    custom_events: list[EventSpec] = Field(default_factory=list)
    #: Default coefficient prior sigma for events without their own.
    prior_sigma: float = Field(default=0.5, gt=0)
    #: Shoulder weeks / decay applied to the country-holiday regressors.
    holiday_pre_weeks: int = Field(default=0, ge=0)
    holiday_post_weeks: int = Field(default=1, ge=0)
    holiday_decay: float = Field(default=0.5, ge=0.0, lt=1.0)

    model_config = {"extra": "forbid"}

    def is_empty(self) -> bool:
        return not self.custom_events and not (self.country or self.holidays)
