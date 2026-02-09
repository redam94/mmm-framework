"""
DAG Model Builder

Build MMM models from DAG specifications with automatic model type selection.

This module provides a fluent builder API for constructing Marketing Mix Models
from directed acyclic graphs (DAGs) that define variable relationships.

Key Features:
- Automatic model type selection (BayesianMMM, NestedMMM, MultivariateMMM, CombinedMMM)
- DAG validation (acyclicity, connectivity, type compatibility)
- Translation from DAG to framework configuration objects
- Frontend adapter for React Flow integration

Examples
--------
Basic usage with a simple DAG:

>>> from mmm_framework.dag_model_builder import (
...     DAGModelBuilder, DAGSpec, DAGNode, DAGEdge, NodeType
... )
>>> dag = DAGSpec(
...     nodes=[
...         DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
...         DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
...         DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL),
...     ],
...     edges=[
...         DAGEdge(source="tv", target="sales"),
...         DAGEdge(source="price", target="sales"),
...     ]
... )
>>> model = (
...     DAGModelBuilder()
...     .with_dag(dag)
...     .with_mff_data("data.csv")
...     .bayesian_numpyro()
...     .build()
... )

Using convenience functions:

>>> from mmm_framework.dag_model_builder import create_simple_dag, DAGModelBuilder
>>> dag = create_simple_dag(
...     kpi_name="Sales",
...     media_names=["TV", "Digital", "Radio"],
...     control_names=["Price", "Distribution"],
... )
>>> model = DAGModelBuilder().with_dag(dag).with_mff_data(df).build()

With mediation:

>>> from mmm_framework.dag_model_builder import create_mediation_dag
>>> dag = create_mediation_dag(
...     kpi_name="Sales",
...     media_names=["TV", "Digital"],
...     mediator_name="Awareness",
...     include_direct_effects=True,
... )
>>> # Automatically uses NestedMMM
>>> model = DAGModelBuilder().with_dag(dag).with_mff_data(df).build()
"""

from .dag_spec import DAGEdge, DAGNode, DAGSpec, EdgeType, NodeType
from .node_configs import (
    ControlNodeConfig,
    KPINodeConfig,
    MediaNodeConfig,
    MediatorNodeConfig,
    NodeConfig,
    OutcomeNodeConfig,
    parse_node_config,
)
from .validation import (
    DAGValidationError,
    ValidationResult,
    is_acyclic,
    validate_complete,
    validate_dag,
    validate_dag_against_data,
)
from .model_type_resolver import (
    ModelType,
    describe_model_type,
    get_model_class,
    resolve_model_type,
)
from .config_translator import (
    dag_to_combined_config,
    dag_to_mff_config,
    dag_to_multivariate_config,
    dag_to_nested_config,
)
from .builder import DAGBuildError, DAGModelBuilder
from .frontend_adapter import (
    create_mediation_dag,
    create_simple_dag,
    dag_spec_to_react_flow,
    react_flow_to_dag_spec,
)

__all__ = [
    # DAG Specification
    "DAGSpec",
    "DAGNode",
    "DAGEdge",
    "NodeType",
    "EdgeType",
    # Node Configs
    "MediaNodeConfig",
    "ControlNodeConfig",
    "KPINodeConfig",
    "MediatorNodeConfig",
    "OutcomeNodeConfig",
    "NodeConfig",
    "parse_node_config",
    # Validation
    "ValidationResult",
    "DAGValidationError",
    "validate_dag",
    "validate_dag_against_data",
    "validate_complete",
    "is_acyclic",
    # Model Type Resolution
    "ModelType",
    "resolve_model_type",
    "get_model_class",
    "describe_model_type",
    # Config Translation
    "dag_to_mff_config",
    "dag_to_nested_config",
    "dag_to_multivariate_config",
    "dag_to_combined_config",
    # Builder
    "DAGModelBuilder",
    "DAGBuildError",
    # Frontend Adapter
    "react_flow_to_dag_spec",
    "dag_spec_to_react_flow",
    "create_simple_dag",
    "create_mediation_dag",
]
