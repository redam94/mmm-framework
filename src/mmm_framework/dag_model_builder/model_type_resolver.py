"""
Model Type Resolver

Determines which model class to use based on DAG structure.
"""

from __future__ import annotations

from enum import Enum

from .dag_spec import DAGSpec, EdgeType, NodeType


class ModelType(str, Enum):
    """Type of model to build based on DAG structure."""

    BAYESIAN_MMM = "bayesian_mmm"
    NESTED_MMM = "nested_mmm"
    MULTIVARIATE_MMM = "multivariate_mmm"
    COMBINED_MMM = "combined_mmm"


def resolve_model_type(dag: DAGSpec) -> ModelType:
    """
    Determine the appropriate model class based on DAG structure.

    Decision logic:
    1. Has mediators + multiple outcomes → CombinedMMM
    2. Has mediators only → NestedMMM
    3. Multiple outcomes or cross-effects only → MultivariateMMM
    4. Otherwise → BayesianMMM

    Parameters
    ----------
    dag : DAGSpec
        The DAG specification.

    Returns
    -------
    ModelType
        The resolved model type.

    Examples
    --------
    >>> dag = DAGSpec(
    ...     nodes=[
    ...         DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
    ...         DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
    ...     ],
    ...     edges=[DAGEdge(source="tv", target="sales")]
    ... )
    >>> resolve_model_type(dag)
    <ModelType.BAYESIAN_MMM: 'bayesian_mmm'>
    """
    has_mediators = dag.has_mediators
    has_multiple_outcomes = dag.has_multiple_outcomes
    has_cross_effects = dag.has_cross_effects

    # Combined: mediators + (multiple outcomes OR cross-effects)
    if has_mediators and (has_multiple_outcomes or has_cross_effects):
        return ModelType.COMBINED_MMM

    # Nested: has mediators only
    if has_mediators:
        return ModelType.NESTED_MMM

    # Multivariate: multiple outcomes or cross-effects
    if has_multiple_outcomes or has_cross_effects:
        return ModelType.MULTIVARIATE_MMM

    # Default: basic BayesianMMM
    return ModelType.BAYESIAN_MMM


def get_model_class(model_type: ModelType):
    """
    Get the model class for a given model type.

    Uses lazy imports to avoid loading PyMC unless needed.

    Parameters
    ----------
    model_type : ModelType
        The model type.

    Returns
    -------
    type
        The model class.

    Raises
    ------
    ValueError
        If model_type is unknown.
    """
    if model_type == ModelType.BAYESIAN_MMM:
        from mmm_framework.model import BayesianMMM

        return BayesianMMM

    if model_type == ModelType.NESTED_MMM:
        from mmm_framework.mmm_extensions.models import NestedMMM

        return NestedMMM

    if model_type == ModelType.MULTIVARIATE_MMM:
        from mmm_framework.mmm_extensions.models import MultivariateMMM

        return MultivariateMMM

    if model_type == ModelType.COMBINED_MMM:
        from mmm_framework.mmm_extensions.models import CombinedMMM

        return CombinedMMM

    raise ValueError(f"Unknown model type: {model_type}")


def describe_model_type(dag: DAGSpec) -> str:
    """
    Get a human-readable description of the model type for a DAG.

    Parameters
    ----------
    dag : DAGSpec
        The DAG specification.

    Returns
    -------
    str
        Description of the model type and why it was selected.
    """
    model_type = resolve_model_type(dag)

    n_media = len(dag.media_nodes)
    n_controls = len(dag.control_nodes)
    n_mediators = len(dag.mediator_nodes)
    n_outcomes = len(dag.outcome_nodes)
    n_cross = sum(1 for e in dag.edges if e.edge_type == EdgeType.CROSS_EFFECT)

    base_info = (
        f"DAG has {n_media} media, {n_controls} controls, "
        f"{n_mediators} mediators, {n_outcomes} outcomes, "
        f"{n_cross} cross-effects."
    )

    descriptions = {
        ModelType.BAYESIAN_MMM: (
            f"{base_info}\n"
            "Selected: BayesianMMM (standard model with media, controls, "
            "trend, and seasonality)."
        ),
        ModelType.NESTED_MMM: (
            f"{base_info}\n"
            "Selected: NestedMMM (mediation model with mediating variables "
            "between media and outcomes)."
        ),
        ModelType.MULTIVARIATE_MMM: (
            f"{base_info}\n"
            "Selected: MultivariateMMM (multiple outcomes with potential "
            "cross-effects between them)."
        ),
        ModelType.COMBINED_MMM: (
            f"{base_info}\n"
            "Selected: CombinedMMM (mediation + multiple outcomes, "
            "the most complex model structure)."
        ),
    }

    return descriptions[model_type]
