"""
DAG Specification Classes

Defines the core data structures for representing model DAGs:
- DAGNode: A single node (variable) in the graph
- DAGEdge: A directed edge (relationship) between nodes
- DAGSpec: The complete DAG specification
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Type of node in the DAG."""

    KPI = "kpi"
    MEDIA = "media"
    CONTROL = "control"
    MEDIATOR = "mediator"
    OUTCOME = "outcome"


class EdgeType(str, Enum):
    """Type of edge in the DAG."""

    DIRECT = "direct"  # Standard direct effect
    MEDIATED = "mediated"  # Effect through a mediator
    CROSS_EFFECT = "cross_effect"  # Cross-outcome effect (halo/cannibalization)


class DAGNode(BaseModel):
    """
    A node in the DAG representing a variable.

    Attributes
    ----------
    id : str
        Unique identifier for the node.
    variable_name : str
        Name of the variable in the MFF dataset.
    node_type : NodeType
        Type of node (KPI, MEDIA, CONTROL, MEDIATOR, OUTCOME).
    label : str | None
        Display label for the node (defaults to variable_name).
    dimensions : list[str]
        Dimensions this variable is defined over (e.g., ["Period", "Geography"]).
    config : dict[str, Any]
        Node-specific configuration (adstock, saturation, priors, etc.).
    """

    id: str
    variable_name: str
    node_type: NodeType
    label: str | None = None
    dimensions: list[str] = Field(default_factory=lambda: ["Period"])
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @property
    def display_label(self) -> str:
        """Get display label, defaulting to variable_name."""
        return self.label or self.variable_name

    @property
    def is_target(self) -> bool:
        """Check if this node is a target/outcome variable."""
        return self.node_type in (NodeType.KPI, NodeType.OUTCOME)

    @property
    def is_input(self) -> bool:
        """Check if this node is an input variable."""
        return self.node_type in (NodeType.MEDIA, NodeType.CONTROL)


class DAGEdge(BaseModel):
    """
    A directed edge in the DAG representing a relationship.

    Attributes
    ----------
    source : str
        ID of the source node.
    target : str
        ID of the target node.
    edge_type : EdgeType
        Type of edge (DIRECT, MEDIATED, CROSS_EFFECT).
    """

    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECT

    model_config = {"extra": "forbid"}


class DAGSpec(BaseModel):
    """
    Complete DAG specification for an MMM model.

    Attributes
    ----------
    nodes : list[DAGNode]
        All nodes in the DAG.
    edges : list[DAGEdge]
        All edges in the DAG.
    metadata : dict[str, Any]
        Optional metadata (e.g., frontend layout info).

    Examples
    --------
    >>> dag = DAGSpec(
    ...     nodes=[
    ...         DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
    ...         DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
    ...     ],
    ...     edges=[
    ...         DAGEdge(source="tv", target="sales"),
    ...     ]
    ... )
    """

    nodes: list[DAGNode]
    edges: list[DAGEdge]
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def get_node(self, node_id: str) -> DAGNode | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_by_variable(self, variable_name: str) -> DAGNode | None:
        """Get a node by variable name."""
        for node in self.nodes:
            if node.variable_name == variable_name:
                return node
        return None

    def get_nodes_by_type(self, node_type: NodeType) -> list[DAGNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.node_type == node_type]

    def get_incoming_edges(self, node_id: str) -> list[DAGEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target == node_id]

    def get_outgoing_edges(self, node_id: str) -> list[DAGEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_parents(self, node_id: str) -> list[DAGNode]:
        """Get all parent nodes of a given node."""
        incoming = self.get_incoming_edges(node_id)
        parents = []
        for edge in incoming:
            node = self.get_node(edge.source)
            if node:
                parents.append(node)
        return parents

    def get_children(self, node_id: str) -> list[DAGNode]:
        """Get all child nodes of a given node."""
        outgoing = self.get_outgoing_edges(node_id)
        children = []
        for edge in outgoing:
            node = self.get_node(edge.target)
            if node:
                children.append(node)
        return children

    @property
    def kpi_nodes(self) -> list[DAGNode]:
        """Get all KPI nodes."""
        return self.get_nodes_by_type(NodeType.KPI)

    @property
    def media_nodes(self) -> list[DAGNode]:
        """Get all media nodes."""
        return self.get_nodes_by_type(NodeType.MEDIA)

    @property
    def control_nodes(self) -> list[DAGNode]:
        """Get all control nodes."""
        return self.get_nodes_by_type(NodeType.CONTROL)

    @property
    def mediator_nodes(self) -> list[DAGNode]:
        """Get all mediator nodes."""
        return self.get_nodes_by_type(NodeType.MEDIATOR)

    @property
    def outcome_nodes(self) -> list[DAGNode]:
        """Get all outcome nodes (including KPI)."""
        return self.get_nodes_by_type(NodeType.KPI) + self.get_nodes_by_type(
            NodeType.OUTCOME
        )

    @property
    def has_mediators(self) -> bool:
        """Check if DAG has any mediator nodes."""
        return len(self.mediator_nodes) > 0

    @property
    def has_multiple_outcomes(self) -> bool:
        """Check if DAG has multiple outcome/KPI nodes."""
        return len(self.outcome_nodes) > 1

    @property
    def has_cross_effects(self) -> bool:
        """Check if DAG has any cross-effect edges."""
        return any(e.edge_type == EdgeType.CROSS_EFFECT for e in self.edges)

    @property
    def node_ids(self) -> list[str]:
        """Get all node IDs."""
        return [n.id for n in self.nodes]

    @property
    def variable_names(self) -> list[str]:
        """Get all variable names."""
        return [n.variable_name for n in self.nodes]

    def to_adjacency_list(self) -> dict[str, list[str]]:
        """Convert DAG to adjacency list representation."""
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            adj[edge.source].append(edge.target)
        return adj
