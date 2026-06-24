"""Observation-model (likelihood) configuration.

A :class:`LikelihoodConfig` declares the KPI's observation family + link + family
parameters. It is a **declared** field on :class:`~mmm_framework.config.model.ModelConfig`
(so it does not violate ``extra:"forbid"``), and defaults to ``normal``/``identity``
— byte-identical to the historical hard-coded ``pm.Normal`` likelihood.

The built-in additive model fits only the **Gaussian** families
(``normal``/``student_t``) directly, because its component priors are calibrated
for standardized ``y`` on an identity link. The non-Gaussian families
(``binomial``/``poisson``/``beta``/…) change the observation scale and need a
non-identity link, so they are read by models that define their own observation
block — e.g. a binomial **awareness** model that overrides ``_build_model`` and
writes ``pm.Binomial(n=number_of_trials, p=sigmoid(mu), …)`` itself. The family
declares the *scale/observation type*; a bespoke count like ``number_of_trials``
can live either here in ``params`` or in the model's own ``CONFIG_SCHEMA``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .enums import LikelihoodFamily, LinkFunction

# Canonical link per family. ``link=None`` resolves to these; an explicit link
# must be one of the allowed links for the family.
_CANONICAL_LINK: dict[LikelihoodFamily, LinkFunction] = {
    LikelihoodFamily.NORMAL: LinkFunction.IDENTITY,
    LikelihoodFamily.STUDENT_T: LinkFunction.IDENTITY,
    LikelihoodFamily.LOGNORMAL: LinkFunction.IDENTITY,
    LikelihoodFamily.BINOMIAL: LinkFunction.LOGIT,
    LikelihoodFamily.BETA_BINOMIAL: LinkFunction.LOGIT,
    LikelihoodFamily.BETA: LinkFunction.LOGIT,
    LikelihoodFamily.POISSON: LinkFunction.LOG,
    LikelihoodFamily.NEGATIVE_BINOMIAL: LinkFunction.LOG,
}

# Links each family will accept (canonical first). Kept permissive but coherent:
# a Gaussian family must stay on the identity link; a bounded/count family may
# not use the identity link.
_ALLOWED_LINKS: dict[LikelihoodFamily, tuple[LinkFunction, ...]] = {
    LikelihoodFamily.NORMAL: (LinkFunction.IDENTITY,),
    LikelihoodFamily.STUDENT_T: (LinkFunction.IDENTITY,),
    LikelihoodFamily.LOGNORMAL: (LinkFunction.IDENTITY,),
    LikelihoodFamily.BINOMIAL: (LinkFunction.LOGIT,),
    LikelihoodFamily.BETA_BINOMIAL: (LinkFunction.LOGIT,),
    LikelihoodFamily.BETA: (LinkFunction.LOGIT,),
    LikelihoodFamily.POISSON: (LinkFunction.LOG,),
    LikelihoodFamily.NEGATIVE_BINOMIAL: (LinkFunction.LOG,),
}


class LikelihoodConfig(BaseModel):
    """Observation family + link + family parameters for the KPI likelihood.

    Parameters
    ----------
    family:
        Observation family. Default ``NORMAL`` (historical behavior).
    link:
        Link mapping the linear predictor ``mu`` to the family's natural
        parameter. ``None`` (default) resolves to the family's canonical link.
    params:
        Family parameters. ``student_t`` accepts ``nu`` (degrees of freedom);
        ``negative_binomial`` accepts ``alpha``. ``binomial``/``beta_binomial``
        may carry ``n_trials`` (a positive int, or a per-observation column name)
        here, but a custom model is free to source it from its own
        ``CONFIG_SCHEMA`` instead (e.g. an awareness model's ``number_of_trials``)
        — the family declares the *scale/observation type* (which drives whether
        ``y`` is standardized), while bespoke counts can live in ``model_params``.
        Unknown keys are passed through for a model to interpret.
    """

    family: LikelihoodFamily = LikelihoodFamily.NORMAL
    link: LinkFunction | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> "LikelihoodConfig":
        if self.link is None:
            self.link = _CANONICAL_LINK[self.family]
        elif self.link not in _ALLOWED_LINKS[self.family]:
            allowed = ", ".join(link.value for link in _ALLOWED_LINKS[self.family])
            raise ValueError(
                f"link {self.link.value!r} is not valid for the "
                f"{self.family.value!r} family (allowed: {allowed})"
            )
        n = self.params.get("n_trials")
        if n is not None and not isinstance(n, str):
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ValueError(
                    f"params['n_trials'] must be a positive integer or a column "
                    f"name, got {n!r}"
                )
        return self

    # -- ergonomic factories ---------------------------------------------

    @classmethod
    def normal(cls) -> "LikelihoodConfig":
        """The default Gaussian likelihood (identity link, standardized ``y``)."""
        return cls(family=LikelihoodFamily.NORMAL)

    @classmethod
    def student_t(cls, nu: float = 4.0) -> "LikelihoodConfig":
        """Heavier-tailed Gaussian-scale likelihood (robust to outliers)."""
        return cls(family=LikelihoodFamily.STUDENT_T, params={"nu": nu})

    @classmethod
    def binomial(cls, n_trials: int | str) -> "LikelihoodConfig":
        """Binomial likelihood (logit link). ``n_trials`` is the per-observation
        denominator — a positive int or a trials-column name. For models that
        define their own observation block (e.g. an awareness model)."""
        return cls(family=LikelihoodFamily.BINOMIAL, params={"n_trials": n_trials})

    @property
    def is_gaussian(self) -> bool:
        return self.family.is_gaussian

    @property
    def standardizes_y(self) -> bool:
        return self.family.standardizes_y


__all__ = ["LikelihoodConfig"]
