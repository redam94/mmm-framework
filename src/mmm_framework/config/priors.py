"""Prior distribution configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import PriorType


class PriorConfig(BaseModel):
    """Configuration for a prior distribution."""

    distribution: PriorType
    params: dict[str, float] = Field(default_factory=dict)
    dims: str | list[str] | None = None

    model_config = {"extra": "forbid"}

    @classmethod
    def half_normal(
        cls, sigma: float = 1.0, dims: str | list[str] | None = None
    ) -> PriorConfig:
        return cls(
            distribution=PriorType.HALF_NORMAL, params={"sigma": sigma}, dims=dims
        )

    @classmethod
    def gamma(
        cls, alpha: float = 2.0, beta: float = 1.0, dims: str | list[str] | None = None
    ) -> PriorConfig:
        return cls(
            distribution=PriorType.GAMMA,
            params={"alpha": alpha, "beta": beta},
            dims=dims,
        )

    @classmethod
    def beta(
        cls, alpha: float = 2.0, beta: float = 2.0, dims: str | list[str] | None = None
    ) -> PriorConfig:
        return cls(
            distribution=PriorType.BETA,
            params={"alpha": alpha, "beta": beta},
            dims=dims,
        )
