"""
DAG Validation

Validates DAG structure and compatibility with data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .dag_spec import DAGSpec, EdgeType, NodeType

if TYPE_CHECKING:
    from mmm_framework.data_loader import PanelDataset


@dataclass
class ValidationResult:
    """
    Result of DAG validation.

    Attributes
    ----------
    valid : bool
        Whether the DAG passed all validation checks.
    errors : list[str]
        List of validation errors (fatal).
    warnings : list[str]
        List of validation warnings (non-fatal).
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def raise_if_invalid(self) -> None:
        """Raise DAGValidationError if not valid."""
        if not self.valid:
            raise DAGValidationError(self.errors, self.warnings)


class DAGValidationError(Exception):
    """Exception raised when DAG validation fails."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        self.errors = errors
        self.warnings = warnings or []
        message = "DAG validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        if self.warnings:
            message += "\nWarnings:\n" + "\n".join(f"  - {w}" for w in self.warnings)
        super().__init__(message)


def is_acyclic(dag: DAGSpec) -> bool:
    """
    Check if the DAG is acyclic using topological sort (Kahn's algorithm).

    Parameters
    ----------
    dag : DAGSpec
        The DAG to check.

    Returns
    -------
    bool
        True if the DAG is acyclic, False otherwise.
    """
    # Build in-degree map
    in_degree = {n.id: 0 for n in dag.nodes}
    adj = dag.to_adjacency_list()

    for edges in adj.values():
        for target in edges:
            if target in in_degree:
                in_degree[target] += 1

    # Start with nodes that have no incoming edges
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    visited = 0

    while queue:
        node = queue.pop(0)
        visited += 1

        for neighbor in adj.get(node, []):
            if neighbor in in_degree:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    return visited == len(dag.nodes)


def validate_dag(dag: DAGSpec) -> ValidationResult:
    """
    Validate DAG structure.

    Checks:
    - DAG is acyclic
    - Has at least one KPI/outcome node
    - Has at least one media node
    - All edge source/target IDs exist as nodes
    - No duplicate node IDs
    - No duplicate variable names
    - Edge types are valid for node types

    Parameters
    ----------
    dag : DAGSpec
        The DAG to validate.

    Returns
    -------
    ValidationResult
        Validation result with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check for duplicate node IDs
    node_ids = [n.id for n in dag.nodes]
    if len(node_ids) != len(set(node_ids)):
        duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
        errors.append(f"Duplicate node IDs: {set(duplicates)}")

    # Check for duplicate variable names
    var_names = [n.variable_name for n in dag.nodes]
    if len(var_names) != len(set(var_names)):
        duplicates = [v for v in var_names if var_names.count(v) > 1]
        errors.append(f"Duplicate variable names: {set(duplicates)}")

    # Check for at least one KPI/outcome
    outcomes = dag.outcome_nodes
    if not outcomes:
        errors.append("DAG must have at least one KPI or OUTCOME node")

    # Check for at least one media node
    media = dag.media_nodes
    if not media:
        errors.append("DAG must have at least one MEDIA node")

    # Validate edges
    valid_node_ids = set(node_ids)
    for edge in dag.edges:
        if edge.source not in valid_node_ids:
            errors.append(f"Edge source '{edge.source}' is not a valid node ID")
        if edge.target not in valid_node_ids:
            errors.append(f"Edge target '{edge.target}' is not a valid node ID")

    # Check edge type validity
    for edge in dag.edges:
        source_node = dag.get_node(edge.source)
        target_node = dag.get_node(edge.target)

        if source_node and target_node:
            # Media can point to: KPI, OUTCOME, MEDIATOR
            if source_node.node_type == NodeType.MEDIA:
                valid_targets = {NodeType.KPI, NodeType.OUTCOME, NodeType.MEDIATOR}
                if target_node.node_type not in valid_targets:
                    errors.append(
                        f"MEDIA node '{source_node.id}' cannot point to "
                        f"{target_node.node_type.value} node '{target_node.id}'"
                    )

            # Control can point to: KPI, OUTCOME
            if source_node.node_type == NodeType.CONTROL:
                valid_targets = {NodeType.KPI, NodeType.OUTCOME}
                if target_node.node_type not in valid_targets:
                    errors.append(
                        f"CONTROL node '{source_node.id}' cannot point to "
                        f"{target_node.node_type.value} node '{target_node.id}'"
                    )

            # Mediator can point to: KPI, OUTCOME
            if source_node.node_type == NodeType.MEDIATOR:
                valid_targets = {NodeType.KPI, NodeType.OUTCOME}
                if target_node.node_type not in valid_targets:
                    errors.append(
                        f"MEDIATOR node '{source_node.id}' cannot point to "
                        f"{target_node.node_type.value} node '{target_node.id}'"
                    )

            # Cross-effect edges must be between outcomes
            if edge.edge_type == EdgeType.CROSS_EFFECT:
                if not (source_node.is_target and target_node.is_target):
                    errors.append(
                        f"CROSS_EFFECT edge from '{source_node.id}' to "
                        f"'{target_node.id}' must be between outcome nodes"
                    )

    # Check for acyclicity
    if not is_acyclic(dag):
        errors.append("DAG contains cycles")

    # Check for orphan media/control nodes (no outgoing edges to outcomes)
    for media_node in media:
        has_path_to_outcome = _has_path_to_outcome(dag, media_node.id)
        if not has_path_to_outcome:
            warnings.append(f"MEDIA node '{media_node.id}' has no path to any outcome")

    for control_node in dag.control_nodes:
        has_path_to_outcome = _has_path_to_outcome(dag, control_node.id)
        if not has_path_to_outcome:
            warnings.append(
                f"CONTROL node '{control_node.id}' has no path to any outcome"
            )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _has_path_to_outcome(dag: DAGSpec, start_id: str) -> bool:
    """Check if there's a path from start_id to any outcome node."""
    visited = set()
    queue = [start_id]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        node = dag.get_node(current)
        if node and node.is_target:
            return True

        for child in dag.get_children(current):
            if child.id not in visited:
                queue.append(child.id)

    return False


def validate_dag_against_data(
    dag: DAGSpec,
    panel: "PanelDataset",
) -> ValidationResult:
    """
    Validate DAG against available data.

    Checks:
    - All variable names in DAG exist in the panel data
    - Dimension compatibility

    Parameters
    ----------
    dag : DAGSpec
        The DAG to validate.
    panel : PanelDataset
        The panel dataset to validate against.

    Returns
    -------
    ValidationResult
        Validation result with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Get available variable names from panel
    available_vars = set()

    # KPI variable
    if hasattr(panel, "y") and hasattr(panel.y, "name"):
        available_vars.add(panel.y.name)

    # Media variables
    if hasattr(panel, "X_media") and panel.X_media is not None:
        available_vars.update(panel.X_media.columns.tolist())

    # Control variables
    if hasattr(panel, "X_controls") and panel.X_controls is not None:
        available_vars.update(panel.X_controls.columns.tolist())

    # Also check coords
    if hasattr(panel, "coords"):
        if hasattr(panel.coords, "channels"):
            available_vars.update(panel.coords.channels)
        if hasattr(panel.coords, "controls"):
            available_vars.update(panel.coords.controls)

    # Check each node's variable name exists
    for node in dag.nodes:
        # Skip mediators - they may be latent
        if node.node_type == NodeType.MEDIATOR:
            continue

        if node.variable_name not in available_vars:
            # Try case-insensitive match
            lower_available = {v.lower(): v for v in available_vars}
            if node.variable_name.lower() in lower_available:
                warnings.append(
                    f"Variable '{node.variable_name}' not found, "
                    f"but '{lower_available[node.variable_name.lower()]}' exists "
                    "(case mismatch)"
                )
            else:
                errors.append(
                    f"Variable '{node.variable_name}' for node '{node.id}' "
                    "not found in panel data"
                )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_complete(
    dag: DAGSpec,
    panel: "PanelDataset | None" = None,
) -> ValidationResult:
    """
    Perform complete validation of DAG structure and data compatibility.

    Parameters
    ----------
    dag : DAGSpec
        The DAG to validate.
    panel : PanelDataset | None
        Optional panel dataset for data validation.

    Returns
    -------
    ValidationResult
        Combined validation result.
    """
    # Structural validation
    struct_result = validate_dag(dag)

    if panel is None:
        return struct_result

    # Data validation
    data_result = validate_dag_against_data(dag, panel)

    # Combine results
    return ValidationResult(
        valid=struct_result.valid and data_result.valid,
        errors=struct_result.errors + data_result.errors,
        warnings=struct_result.warnings + data_result.warnings,
    )
