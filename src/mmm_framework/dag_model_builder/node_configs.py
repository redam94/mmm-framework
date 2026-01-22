"""
Node-Specific Configuration Classes

Provides typed configuration classes for each node type in the DAG.
These configs are used to specify priors, transformations, and other
node-specific settings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mmm_framework.config import AdstockType, SaturationType


class MediaNodeConfig(BaseModel):
    """
    Configuration for a media node.

    Attributes
    ----------
    adstock_type : AdstockType
        Type of adstock transformation.
    adstock_lmax : int
        Maximum lag for adstock.
    adstock_normalize : bool
        Whether to normalize adstock weights.
    adstock_alpha_prior_alpha : float
        Alpha parameter for adstock decay prior (Beta distribution).
    adstock_alpha_prior_beta : float
        Beta parameter for adstock decay prior (Beta distribution).
    saturation_type : SaturationType
        Type of saturation transformation.
    coefficient_prior_sigma : float
        Sigma for the coefficient prior (HalfNormal).
    parent_channel : str | None
        Parent channel for hierarchical media grouping.
    """

    adstock_type: AdstockType = AdstockType.GEOMETRIC
    adstock_lmax: int = Field(default=8, ge=1, le=52)
    adstock_normalize: bool = True
    adstock_alpha_prior_alpha: float = Field(default=1.0, gt=0)
    adstock_alpha_prior_beta: float = Field(default=3.0, gt=0)
    saturation_type: SaturationType = SaturationType.HILL
    saturation_kappa_prior_alpha: float = Field(default=2.0, gt=0)
    saturation_kappa_prior_beta: float = Field(default=2.0, gt=0)
    saturation_slope_prior_sigma: float = Field(default=1.5, gt=0)
    saturation_beta_prior_sigma: float = Field(default=1.5, gt=0)
    coefficient_prior_sigma: float = Field(default=2.0, gt=0)
    parent_channel: str | None = None

    model_config = {"extra": "forbid"}


class ControlNodeConfig(BaseModel):
    """
    Configuration for a control node.

    Attributes
    ----------
    allow_negative : bool
        Whether the coefficient can be negative.
    coefficient_prior_mu : float
        Mean of the coefficient prior (Normal distribution).
    coefficient_prior_sigma : float
        Sigma of the coefficient prior (Normal distribution).
    use_shrinkage : bool
        Whether to apply shrinkage (horseshoe-like) prior.
    """

    allow_negative: bool = True
    coefficient_prior_mu: float = 0.0
    coefficient_prior_sigma: float = Field(default=1.0, gt=0)
    use_shrinkage: bool = False

    model_config = {"extra": "forbid"}


class KPINodeConfig(BaseModel):
    """
    Configuration for a KPI (target) node.

    Attributes
    ----------
    log_transform : bool
        Whether to log-transform the KPI (for multiplicative models).
    floor_value : float
        Minimum value for log safety.
    """

    log_transform: bool = False
    floor_value: float = Field(default=1e-6, gt=0)

    model_config = {"extra": "forbid"}


class MediatorNodeConfig(BaseModel):
    """
    Configuration for a mediator node.

    Attributes
    ----------
    mediator_type : str
        Type of mediator observation model.
        Options: "fully_observed", "partially_observed", "aggregated_survey", "fully_latent"
    observation_noise_sigma : float
        Observation noise sigma for observed mediators.
    allow_direct_effect : bool
        Whether to allow direct media -> outcome effects (bypassing mediator).
    direct_effect_sigma : float
        Prior sigma for direct effect.
    media_effect_constraint : str
        Constraint on media -> mediator effect. Options: "none", "positive", "negative".
    media_effect_sigma : float
        Prior sigma for media -> mediator effect.
    outcome_effect_sigma : float
        Prior sigma for mediator -> outcome effect.
    apply_adstock : bool
        Whether to apply adstock to media -> mediator pathway.
    apply_saturation : bool
        Whether to apply saturation to media -> mediator pathway.
    """

    mediator_type: str = "partially_observed"
    observation_noise_sigma: float = Field(default=0.1, gt=0)
    allow_direct_effect: bool = True
    direct_effect_sigma: float = Field(default=0.5, gt=0)
    media_effect_constraint: str = "positive"
    media_effect_sigma: float = Field(default=1.0, gt=0)
    outcome_effect_sigma: float = Field(default=1.0, gt=0)
    apply_adstock: bool = True
    apply_saturation: bool = True

    model_config = {"extra": "forbid"}


class OutcomeNodeConfig(BaseModel):
    """
    Configuration for an outcome node (non-primary KPI).

    Attributes
    ----------
    include_trend : bool
        Whether to include trend component.
    include_seasonality : bool
        Whether to include seasonality component.
    intercept_prior_sigma : float
        Prior sigma for intercept.
    media_effect_sigma : float
        Prior sigma for media effects.
    log_transform : bool
        Whether to log-transform the outcome.
    """

    include_trend: bool = True
    include_seasonality: bool = True
    intercept_prior_sigma: float = Field(default=2.0, gt=0)
    media_effect_sigma: float = Field(default=0.5, gt=0)
    log_transform: bool = False

    model_config = {"extra": "forbid"}


# Type alias for node configs
NodeConfig = (
    MediaNodeConfig
    | ControlNodeConfig
    | KPINodeConfig
    | MediatorNodeConfig
    | OutcomeNodeConfig
)


def parse_node_config(node_type: str, config_dict: dict) -> NodeConfig:
    """
    Parse a config dict into the appropriate NodeConfig type.

    Parameters
    ----------
    node_type : str
        Type of node ("media", "control", "kpi", "mediator", "outcome").
    config_dict : dict
        Dictionary of configuration values.

    Returns
    -------
    NodeConfig
        Parsed configuration object.

    Raises
    ------
    ValueError
        If node_type is unknown.
    """
    config_classes = {
        "media": MediaNodeConfig,
        "control": ControlNodeConfig,
        "kpi": KPINodeConfig,
        "mediator": MediatorNodeConfig,
        "outcome": OutcomeNodeConfig,
    }

    if node_type not in config_classes:
        raise ValueError(f"Unknown node type: {node_type}")

    return config_classes[node_type](**config_dict)
