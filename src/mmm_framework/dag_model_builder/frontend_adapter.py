"""
Frontend Adapter

Converts between React Flow frontend JSON format and DAGSpec Python objects.
"""

from __future__ import annotations

from typing import Any

from .dag_spec import DAGEdge, DAGNode, DAGSpec, EdgeType, NodeType


def react_flow_to_dag_spec(
    nodes: list[dict],
    edges: list[dict],
) -> DAGSpec:
    """
    Convert React Flow node/edge format to DAGSpec.

    React Flow format (from frontend)::

        {
            "nodes": [
                {
                    "id": "node_abc123",
                    "type": "default",
                    "position": {"x": 100, "y": 200},
                    "data": {
                        "label": "TV Spend",
                        "type": "media",
                        "variableName": "tv_spend",
                        "config": {"adstockType": "geometric", ...}
                    }
                }
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "node_abc",
                    "target": "node_xyz",
                    "data": {"edgeType": "direct"}
                }
            ]
        }

    Args:
        nodes: List of React Flow node objects.
        edges: List of React Flow edge objects.

    Returns:
        Converted DAG specification.
    """
    dag_nodes = []
    dag_edges = []

    # Node type mapping (frontend -> backend)
    node_type_map = {
        "kpi": NodeType.KPI,
        "media": NodeType.MEDIA,
        "control": NodeType.CONTROL,
        "mediator": NodeType.MEDIATOR,
        "outcome": NodeType.OUTCOME,
        # Additional frontend variations
        "target": NodeType.KPI,
        "channel": NodeType.MEDIA,
        "factor": NodeType.CONTROL,
    }

    # Edge type mapping
    edge_type_map = {
        "direct": EdgeType.DIRECT,
        "mediated": EdgeType.MEDIATED,
        "cross_effect": EdgeType.CROSS_EFFECT,
        "crossEffect": EdgeType.CROSS_EFFECT,
        # Default
        "default": EdgeType.DIRECT,
    }

    # Convert nodes
    for node in nodes:
        node_id = node.get("id", "")
        data = node.get("data", {})

        # Get node type from data
        raw_type = data.get("type", "").lower()
        node_type = node_type_map.get(raw_type, NodeType.MEDIA)

        # Get variable name (try multiple field names)
        variable_name = (
            data.get("variableName")
            or data.get("variable_name")
            or data.get("name")
            or data.get("label")
            or node_id
        )

        # Get label
        label = data.get("label") or variable_name

        # Get dimensions
        dimensions = data.get("dimensions", ["Period"])
        if isinstance(dimensions, str):
            dimensions = [dimensions]

        # Get config (convert camelCase to snake_case)
        raw_config = data.get("config", {})
        config = _convert_config_keys(raw_config)

        dag_node = DAGNode(
            id=node_id,
            variable_name=variable_name,
            node_type=node_type,
            label=label,
            dimensions=dimensions,
            config=config,
        )
        dag_nodes.append(dag_node)

    # Convert edges
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")

        # Get edge type from data
        data = edge.get("data", {})
        raw_edge_type = data.get("edgeType", data.get("edge_type", "direct")).lower()
        edge_type = edge_type_map.get(raw_edge_type, EdgeType.DIRECT)

        dag_edge = DAGEdge(
            source=source,
            target=target,
            edge_type=edge_type,
        )
        dag_edges.append(dag_edge)

    # Preserve frontend metadata (positions, etc.)
    metadata = {
        "frontend_positions": {
            n.get("id"): n.get("position", {}) for n in nodes if "position" in n
        }
    }

    return DAGSpec(nodes=dag_nodes, edges=dag_edges, metadata=metadata)


def dag_spec_to_react_flow(dag: DAGSpec) -> dict:
    """
    Convert DAGSpec back to React Flow format for frontend.

    Parameters
    ----------
    dag : DAGSpec
        The DAG specification.

    Returns
    -------
    dict
        React Flow compatible JSON structure.
    """
    # Node type mapping (backend -> frontend)
    node_type_map = {
        NodeType.KPI: "kpi",
        NodeType.MEDIA: "media",
        NodeType.CONTROL: "control",
        NodeType.MEDIATOR: "mediator",
        NodeType.OUTCOME: "outcome",
    }

    # Edge type mapping
    edge_type_map = {
        EdgeType.DIRECT: "direct",
        EdgeType.MEDIATED: "mediated",
        EdgeType.CROSS_EFFECT: "crossEffect",
    }

    # Get frontend positions if available
    positions = dag.metadata.get("frontend_positions", {})

    # Convert nodes
    react_nodes = []
    for i, node in enumerate(dag.nodes):
        # Calculate position if not available
        default_position = {"x": 100 + (i % 4) * 200, "y": 100 + (i // 4) * 150}
        position = positions.get(node.id, default_position)

        react_node = {
            "id": node.id,
            "type": "default",
            "position": position,
            "data": {
                "label": node.display_label,
                "type": node_type_map.get(node.node_type, "media"),
                "variableName": node.variable_name,
                "dimensions": node.dimensions,
                "config": _convert_config_keys_to_camel(node.config),
            },
        }
        react_nodes.append(react_node)

    # Convert edges
    react_edges = []
    for i, edge in enumerate(dag.edges):
        react_edge = {
            "id": f"e{i}",
            "source": edge.source,
            "target": edge.target,
            "data": {
                "edgeType": edge_type_map.get(edge.edge_type, "direct"),
            },
        }
        react_edges.append(react_edge)

    return {
        "nodes": react_nodes,
        "edges": react_edges,
    }


def _convert_config_keys(config: dict) -> dict:
    """Convert camelCase config keys to snake_case."""
    result = {}
    for key, value in config.items():
        # Convert camelCase to snake_case
        snake_key = _camel_to_snake(key)
        result[snake_key] = value
    return result


def _convert_config_keys_to_camel(config: dict) -> dict:
    """Convert snake_case config keys to camelCase."""
    result = {}
    for key, value in config.items():
        # Convert snake_case to camelCase
        camel_key = _snake_to_camel(key)
        result[camel_key] = value
    return result


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def create_simple_dag(
    kpi_name: str,
    media_names: list[str],
    control_names: list[str] | None = None,
    dimensions: list[str] | None = None,
) -> DAGSpec:
    """
    Create a simple DAG with all media and controls pointing to a single KPI.

    This is a convenience function for creating basic model structures.

    Parameters
    ----------
    kpi_name : str
        Name of the KPI variable.
    media_names : list[str]
        Names of media variables.
    control_names : list[str] | None
        Names of control variables.
    dimensions : list[str] | None
        Dimensions for all variables.

    Returns
    -------
    DAGSpec
        The created DAG specification.

    Examples
    --------
    >>> dag = create_simple_dag(
    ...     kpi_name="Sales",
    ...     media_names=["TV", "Digital", "Radio"],
    ...     control_names=["Price", "Distribution"],
    ... )
    """
    control_names = control_names or []
    dimensions = dimensions or ["Period"]

    nodes = []
    edges = []

    # Create KPI node
    kpi_node = DAGNode(
        id="kpi",
        variable_name=kpi_name,
        node_type=NodeType.KPI,
        dimensions=dimensions,
    )
    nodes.append(kpi_node)

    # Create media nodes and edges
    for i, media_name in enumerate(media_names):
        media_node = DAGNode(
            id=f"media_{i}",
            variable_name=media_name,
            node_type=NodeType.MEDIA,
            dimensions=dimensions,
        )
        nodes.append(media_node)

        edge = DAGEdge(source=f"media_{i}", target="kpi")
        edges.append(edge)

    # Create control nodes and edges
    for i, control_name in enumerate(control_names):
        control_node = DAGNode(
            id=f"control_{i}",
            variable_name=control_name,
            node_type=NodeType.CONTROL,
            dimensions=dimensions,
        )
        nodes.append(control_node)

        edge = DAGEdge(source=f"control_{i}", target="kpi")
        edges.append(edge)

    return DAGSpec(nodes=nodes, edges=edges)


def create_mediation_dag(
    kpi_name: str,
    media_names: list[str],
    mediator_name: str,
    control_names: list[str] | None = None,
    include_direct_effects: bool = True,
    dimensions: list[str] | None = None,
) -> DAGSpec:
    """
    Create a DAG with mediation structure.

    All media → mediator → KPI, with optional direct media → KPI effects.

    Parameters
    ----------
    kpi_name : str
        Name of the KPI variable.
    media_names : list[str]
        Names of media variables.
    mediator_name : str
        Name of the mediator variable.
    control_names : list[str] | None
        Names of control variables.
    include_direct_effects : bool
        Whether to include direct media → KPI effects.
    dimensions : list[str] | None
        Dimensions for all variables.

    Returns
    -------
    DAGSpec
        The created DAG specification.
    """
    control_names = control_names or []
    dimensions = dimensions or ["Period"]

    nodes = []
    edges = []

    # Create KPI node
    kpi_node = DAGNode(
        id="kpi",
        variable_name=kpi_name,
        node_type=NodeType.KPI,
        dimensions=dimensions,
    )
    nodes.append(kpi_node)

    # Create mediator node
    mediator_node = DAGNode(
        id="mediator",
        variable_name=mediator_name,
        node_type=NodeType.MEDIATOR,
        dimensions=dimensions,
    )
    nodes.append(mediator_node)

    # Mediator → KPI edge
    edges.append(DAGEdge(source="mediator", target="kpi"))

    # Create media nodes and edges
    for i, media_name in enumerate(media_names):
        media_node = DAGNode(
            id=f"media_{i}",
            variable_name=media_name,
            node_type=NodeType.MEDIA,
            dimensions=dimensions,
        )
        nodes.append(media_node)

        # Media → Mediator edge
        edges.append(DAGEdge(source=f"media_{i}", target="mediator"))

        # Optional direct effect
        if include_direct_effects:
            edges.append(DAGEdge(source=f"media_{i}", target="kpi"))

    # Create control nodes and edges
    for i, control_name in enumerate(control_names):
        control_node = DAGNode(
            id=f"control_{i}",
            variable_name=control_name,
            node_type=NodeType.CONTROL,
            dimensions=dimensions,
        )
        nodes.append(control_node)

        edge = DAGEdge(source=f"control_{i}", target="kpi")
        edges.append(edge)

    return DAGSpec(nodes=nodes, edges=edges)
