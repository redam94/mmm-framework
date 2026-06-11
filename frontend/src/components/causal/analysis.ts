// Pure causal-analysis engine for the DAG planner.
// Ported from pages/Planning/index.tsx — NO React imports here so the logic is
// unit-testable and reusable. Operates on React Flow node/edge shapes.

import { MarkerType, type Edge, type Node } from '@xyflow/react';

// ── Colors (identical to the retired Planning page) ────────────────────────

// Path colors for causal analysis - based on causal role
export const PATH_COLORS = {
  // Edge colors
  direct: '#3B82F6',           // Blue - direct causal effect (treatment -> outcome)
  mediated: '#10B981',         // Green - open indirect/mediated paths
  mediatedBlocked: '#F97316',  // Orange - closed/blocked indirect paths
  backdoor: '#EF4444',         // Red - unblocked anti-causal/backdoor paths
  backdoorBlocked: '#000000',  // Black - blocked backdoor paths (dashed)
  outcomeOnly: '#F97316',      // Orange - paths to outcome not from treatment
  default: '#6B7280',          // Gray - edges not on any causal path
};

// Node colors for causal analysis - based on position in causal structure
export const NODE_COLORS = {
  treatment: { bg: '#DBEAFE', border: '#3B82F6', text: '#1E40AF' },     // Blue
  outcome: { bg: '#D1FAE5', border: '#10B981', text: '#065F46' },       // Green
  mediator: { bg: '#F3E8FF', border: '#A855F7', text: '#6B21A8' },      // Purple
  confounder: { bg: '#FEE2E2', border: '#EF4444', text: '#991B1B' },    // Red (unblocked confounder)
  collider: { bg: '#FEE2E2', border: '#EF4444', text: '#991B1B' },      // Red (controlled collider - bad!)
  controlled: { bg: '#E5E7EB', border: '#6B7280', text: '#374151' },    // Gray (observed/controlled)
  outcomeInfluencer: { bg: '#FED7AA', border: '#F97316', text: '#C2410C' }, // Orange (impacts outcome only)
  irrelevant: { bg: '#F3F4F6', border: '#D1D5DB', text: '#6B7280' },    // Light gray (no causal role)
};

// Causal role type for nodes (user-assigned, ephemeral UI state)
export type CausalRole = 'treatment' | 'outcome' | 'controlled' | null;

// Computed causal position based on graph structure
export type CausalPosition =
  | 'treatment'           // Blue - user-designated treatment
  | 'outcome'             // Green - user-designated outcome
  | 'mediator'            // Purple - on causal path from treatment to outcome
  | 'confounder'          // Red - unblocked backdoor path source
  | 'collider'            // Red - controlled collider (introduces bias!)
  | 'controlled'          // Gray - user-controlled (blocks paths)
  | 'outcomeInfluencer'   // Orange - affects outcome but not treatment
  | 'irrelevant';         // Light gray - not on any relevant path

// Minimal structural shapes — React Flow Node/Edge satisfy these.
export interface GraphNode { id: string }
export interface GraphEdge { id: string; source: string; target: string }

// ── Graph helpers ───────────────────────────────────────────────────────────

// Helper: Build adjacency list from edges
export function buildAdjacencyList(edges: readonly GraphEdge[]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>();
  for (const edge of edges) {
    if (!adj.has(edge.source)) adj.set(edge.source, new Set());
    adj.get(edge.source)!.add(edge.target);
  }
  return adj;
}

// Helper: Build reverse adjacency list (for finding parents)
export function buildReverseAdjacencyList(edges: readonly GraphEdge[]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>();
  for (const edge of edges) {
    if (!adj.has(edge.target)) adj.set(edge.target, new Set());
    adj.get(edge.target)!.add(edge.source);
  }
  return adj;
}

// Find all directed paths from source to target (for total effect)
export function findAllDirectedPaths(
  source: string,
  target: string,
  adj: Map<string, Set<string>>,
  visited: Set<string> = new Set()
): string[][] {
  if (source === target) return [[target]];
  if (visited.has(source)) return [];

  visited.add(source);
  const paths: string[][] = [];
  const neighbors = adj.get(source) || new Set();

  for (const neighbor of neighbors) {
    const subPaths = findAllDirectedPaths(neighbor, target, adj, new Set(visited));
    for (const subPath of subPaths) {
      paths.push([source, ...subPath]);
    }
  }

  return paths;
}

// Find direct effect edges (treatment -> outcome directly)
export function findDirectEffectEdges(
  treatment: string,
  outcome: string,
  edges: readonly GraphEdge[],
): Set<string> {
  const directEdges = new Set<string>();
  for (const edge of edges) {
    if (edge.source === treatment && edge.target === outcome) {
      directEdges.add(edge.id);
    }
  }
  return directEdges;
}

// Find all edges on directed paths (for total effect)
// Also returns which edges are on blocked paths (due to controlling mediators)
export function findTotalEffectEdges(
  treatment: string,
  outcome: string,
  edges: readonly GraphEdge[],
  adj: Map<string, Set<string>>,
  controlledNodes: Set<string> = new Set()
): { totalEdges: Set<string>; blockedMediatedEdges: Set<string>; unblockedMediatedPaths: number } {
  const paths = findAllDirectedPaths(treatment, outcome, adj);
  const totalEdges = new Set<string>();
  const blockedMediatedEdges = new Set<string>();
  let unblockedMediatedPaths = 0;

  for (const path of paths) {
    const isDirect = path.length === 2; // Just treatment -> outcome
    // Check if this path is blocked by a controlled node (only mediators, not treatment/outcome)
    const mediatorNodes = path.slice(1, -1); // Nodes between treatment and outcome
    const isBlocked = mediatorNodes.some((nodeId) => controlledNodes.has(nodeId));

    for (let i = 0; i < path.length - 1; i++) {
      const edge = edges.find((e) => e.source === path[i] && e.target === path[i + 1]);
      if (edge) {
        totalEdges.add(edge.id);
        if (!isDirect && isBlocked) {
          blockedMediatedEdges.add(edge.id);
        }
      }
    }

    // Count unblocked mediated paths
    if (!isDirect && !isBlocked) {
      unblockedMediatedPaths++;
    }
  }
  return { totalEdges, blockedMediatedEdges, unblockedMediatedPaths };
}

// Find confounders (common causes of treatment and outcome)
// A confounder is a node that has a directed path to both treatment and outcome
// but is not on the causal path from treatment to outcome
// controlledNodes: nodes that are already being controlled for (block paths)
export function findConfounders(
  treatment: string,
  outcome: string,
  edges: readonly GraphEdge[],
  adj: Map<string, Set<string>>,
  controlledNodes: Set<string> = new Set()
): {
  confounders: Set<string>;
  backdoorEdges: Set<string>;
  blockedBackdoorEdges: Set<string>;
  unblockedConfounders: Set<string>;
} {
  const confounders = new Set<string>();
  const backdoorEdges = new Set<string>(); // Unblocked backdoor edges
  const blockedBackdoorEdges = new Set<string>(); // Blocked backdoor edges
  const unblockedConfounders = new Set<string>();

  // Get all nodes that are ancestors of treatment
  const treatmentAncestors = new Set<string>();
  const reverseAdj = buildReverseAdjacencyList(edges);

  function findAncestors(nodeId: string, ancestors: Set<string>, visited: Set<string>) {
    if (visited.has(nodeId)) return;
    visited.add(nodeId);
    const parents = reverseAdj.get(nodeId) || new Set();
    for (const parent of parents) {
      ancestors.add(parent);
      findAncestors(parent, ancestors, visited);
    }
  }

  findAncestors(treatment, treatmentAncestors, new Set());

  // For each ancestor of treatment, check if it also has a path to outcome
  for (const ancestor of treatmentAncestors) {
    const pathsToOutcome = findAllDirectedPaths(ancestor, outcome, adj);
    if (pathsToOutcome.length > 0) {
      // Check if any path to outcome doesn't go through treatment
      for (const path of pathsToOutcome) {
        if (!path.includes(treatment)) {
          confounders.add(ancestor);

          // Check if this backdoor path is blocked by controlled nodes
          const pathToOutcomeBlocked = path.some((nodeId) => controlledNodes.has(nodeId)) || controlledNodes.has(ancestor);
          const pathsToTreatment = findAllDirectedPaths(ancestor, treatment, adj);
          const pathToTreatmentBlocked = pathsToTreatment.every((treatPath) =>
            treatPath.some((nodeId) => controlledNodes.has(nodeId))
          ) || controlledNodes.has(ancestor);

          const isBlocked = pathToOutcomeBlocked || pathToTreatmentBlocked;

          // If path is not blocked, this is an unblocked confounder
          if (!isBlocked) {
            unblockedConfounders.add(ancestor);
          }

          // Mark the edges on the backdoor path
          for (let i = 0; i < path.length - 1; i++) {
            const edge = edges.find((e) => e.source === path[i] && e.target === path[i + 1]);
            if (edge) {
              if (isBlocked) {
                blockedBackdoorEdges.add(edge.id);
              } else {
                backdoorEdges.add(edge.id);
              }
            }
          }
          // Also mark edges from ancestor to treatment
          for (const treatPath of pathsToTreatment) {
            for (let i = 0; i < treatPath.length - 1; i++) {
              const edge = edges.find((e) => e.source === treatPath[i] && e.target === treatPath[i + 1]);
              if (edge) {
                if (isBlocked) {
                  blockedBackdoorEdges.add(edge.id);
                } else {
                  backdoorEdges.add(edge.id);
                }
              }
            }
          }
        }
      }
    }
  }

  return { confounders, backdoorEdges, blockedBackdoorEdges, unblockedConfounders };
}

// Identify which nodes need to be controlled for to block backdoor paths
export function findRequiredControls(
  treatment: string,
  outcome: string,
  edges: readonly GraphEdge[],
  alreadyControlled: Set<string> = new Set()
): string[] {
  const adj = buildAdjacencyList(edges);
  const { unblockedConfounders } = findConfounders(treatment, outcome, edges, adj, alreadyControlled);

  // Return only the confounders that are not already controlled
  return Array.from(unblockedConfounders);
}

// ── Full analysis (ported from PlanningPage's causalAnalysis useMemo) ───────

export interface PerTreatmentResult {
  directEdges: Set<string>;
  totalEdges: Set<string>;
  blockedMediatedEdges: Set<string>;
  unblockedMediatedPaths: number;
  confounders: Set<string>;
  unblockedConfounders: Set<string>;
  requiredControls: string[];
  hasDirectEffect: boolean;
  hasMediatedPaths: boolean;
  hasUnblockedMediatedPaths: boolean;
}

export interface CausalAnalysis {
  treatmentNodes: string[];
  outcomeNode: string;
  directEdges: Set<string>;
  totalEdges: Set<string>;
  blockedMediatedEdges: Set<string>;
  backdoorEdges: Set<string>;
  blockedBackdoorEdges: Set<string>;
  outcomeOnlyEdges: Set<string>;
  confounders: Set<string>;
  unblockedConfounders: Set<string>;
  controlledColliders: Set<string>;
  mediatorNodes: Set<string>;
  outcomeInfluencers: Set<string>;
  nodePositions: Map<string, CausalPosition>;
  requiredControls: string[];
  controlledNodes: Set<string>;
  allBackdoorsBlocked: boolean;
  hasColliderBias: boolean;
  hasDirectEffect: boolean;
  hasMediatedPaths: boolean;
  hasUnblockedMediatedPaths: boolean;
  effectType: 'total' | 'direct' | 'none';
  perTreatmentResults: Map<string, PerTreatmentResult>;
}

/**
 * Run the full causal analysis over a DAG given user-assigned roles.
 * Returns null when there is no treatment or no outcome (nothing to analyze) —
 * the same guard PlanningPage applied before computing.
 */
export function computeCausalAnalysis(
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
  roles: Map<string, CausalRole>,
): CausalAnalysis | null {
  const treatmentNodes = Array.from(roles.entries()).filter(([, r]) => r === 'treatment').map(([id]) => id);
  const outcomeNodes = Array.from(roles.entries()).filter(([, r]) => r === 'outcome').map(([id]) => id);
  const controlledList = Array.from(roles.entries()).filter(([, r]) => r === 'controlled').map(([id]) => id);

  if (treatmentNodes.length === 0 || outcomeNodes.length === 0) {
    return null;
  }

  // For now, use the first outcome (can extend to multiple later)
  // But support multiple treatments
  const outcomeNode = outcomeNodes[0];
  const treatmentSet = new Set(treatmentNodes);
  const controlledSet = new Set(controlledList);

  const adj = buildAdjacencyList(edges);
  const reverseAdj = buildReverseAdjacencyList(edges);

  // Aggregate results across all treatments
  const directEdges = new Set<string>();
  const totalEdges = new Set<string>();
  const blockedMediatedEdges = new Set<string>();
  const backdoorEdges = new Set<string>();
  const blockedBackdoorEdges = new Set<string>();
  const confounders = new Set<string>();
  const unblockedConfounders = new Set<string>();
  const mediatorNodes = new Set<string>();
  const requiredControls: string[] = [];
  let totalUnblockedMediatedPaths = 0;

  // Per-treatment analysis results for detailed reporting
  const perTreatmentResults = new Map<string, PerTreatmentResult>();

  // Analyze each treatment -> outcome pair
  for (const treatmentNode of treatmentNodes) {
    const treatmentDirectEdges = findDirectEffectEdges(treatmentNode, outcomeNode, edges);
    const { totalEdges: treatmentTotalEdges, blockedMediatedEdges: treatmentBlockedMediatedEdges, unblockedMediatedPaths } = findTotalEffectEdges(
      treatmentNode,
      outcomeNode,
      edges,
      adj,
      controlledSet
    );
    const { confounders: treatmentConfounders, backdoorEdges: treatmentBackdoorEdges, blockedBackdoorEdges: treatmentBlockedBackdoorEdges, unblockedConfounders: treatmentUnblockedConfounders } = findConfounders(
      treatmentNode,
      outcomeNode,
      edges,
      adj,
      controlledSet
    );
    const treatmentRequiredControls = findRequiredControls(treatmentNode, outcomeNode, edges, controlledSet);

    // Merge into aggregated sets
    treatmentDirectEdges.forEach((e) => directEdges.add(e));
    treatmentTotalEdges.forEach((e) => totalEdges.add(e));
    treatmentBlockedMediatedEdges.forEach((e) => blockedMediatedEdges.add(e));
    treatmentBackdoorEdges.forEach((e) => backdoorEdges.add(e));
    treatmentBlockedBackdoorEdges.forEach((e) => blockedBackdoorEdges.add(e));
    treatmentConfounders.forEach((c) => confounders.add(c));
    treatmentUnblockedConfounders.forEach((c) => unblockedConfounders.add(c));
    totalUnblockedMediatedPaths += unblockedMediatedPaths;

    // Add required controls (avoid duplicates)
    for (const ctrl of treatmentRequiredControls) {
      if (!requiredControls.includes(ctrl)) {
        requiredControls.push(ctrl);
      }
    }

    // Find mediators for this treatment
    const causalPaths = findAllDirectedPaths(treatmentNode, outcomeNode, adj);
    for (const path of causalPaths) {
      // Mediators are nodes between treatment and outcome (not including them)
      // Also exclude other treatment nodes from being marked as mediators
      for (let i = 1; i < path.length - 1; i++) {
        if (!treatmentSet.has(path[i])) {
          mediatorNodes.add(path[i]);
        }
      }
    }

    // Store per-treatment results
    perTreatmentResults.set(treatmentNode, {
      directEdges: treatmentDirectEdges,
      totalEdges: treatmentTotalEdges,
      blockedMediatedEdges: treatmentBlockedMediatedEdges,
      unblockedMediatedPaths,
      confounders: treatmentConfounders,
      unblockedConfounders: treatmentUnblockedConfounders,
      requiredControls: treatmentRequiredControls,
      hasDirectEffect: treatmentDirectEdges.size > 0,
      hasMediatedPaths: treatmentTotalEdges.size > treatmentDirectEdges.size,
      hasUnblockedMediatedPaths: unblockedMediatedPaths > 0,
    });
  }

  // Find nodes that influence outcome but are NOT on any treatment's causal path
  // These are nodes with a path to outcome that aren't treatment, mediators, or confounders
  const outcomeInfluencers = new Set<string>();
  const outcomeOnlyEdges = new Set<string>(); // Edges from non-treatment nodes to outcome

  for (const node of nodes) {
    const nodeId = node.id;
    if (treatmentSet.has(nodeId) || nodeId === outcomeNode) continue;
    if (mediatorNodes.has(nodeId)) continue;
    if (confounders.has(nodeId)) continue;
    if (controlledSet.has(nodeId)) continue;

    // Check if this node has a path to the outcome
    const pathsToOutcome = findAllDirectedPaths(nodeId, outcomeNode, adj);
    if (pathsToOutcome.length > 0) {
      // Check if this node influences any treatment (would make it a confounder)
      let influencesTreatment = false;
      for (const treatmentNode of treatmentNodes) {
        const pathsToTreatment = findAllDirectedPaths(nodeId, treatmentNode, adj);
        if (pathsToTreatment.length > 0) {
          influencesTreatment = true;
          break;
        }
      }
      if (!influencesTreatment) {
        // This node affects outcome but not any treatment - it's an outcome influencer
        outcomeInfluencers.add(nodeId);
        // Mark edges on paths from this node to outcome
        for (const path of pathsToOutcome) {
          for (let i = 0; i < path.length - 1; i++) {
            const edge = edges.find((e) => e.source === path[i] && e.target === path[i + 1]);
            if (edge) outcomeOnlyEdges.add(edge.id);
          }
        }
      }
    }
  }

  // Find colliders that are being controlled for inappropriately
  // A collider is a node with multiple parents (arrows pointing into it)
  // Controlling for a collider that's NOT a confounder can introduce bias (collider bias)
  // We should warn when a controlled node is a collider but not a confounder
  const controlledColliders = new Set<string>();
  for (const controlledId of controlledSet) {
    // Skip treatments and outcome
    if (treatmentSet.has(controlledId) || controlledId === outcomeNode) continue;
    // Skip if it's a legitimate confounder (controlling is correct)
    if (confounders.has(controlledId)) continue;

    // Check if this node has multiple parents (is a collider structure)
    const parents = reverseAdj.get(controlledId) || new Set();
    if (parents.size >= 2) {
      // This is a collider - check if controlling for it might open a spurious path
      // A problematic collider is one where:
      // 1. It has an ancestor of any treatment as one parent, AND
      // 2. It has an ancestor of outcome (or outcome itself) as another parent
      // OR: One parent is related to treatment and another to outcome

      // Check if any parent is a treatment or has path from any treatment
      let hasParentFromTreatmentSide = false;
      let hasParentFromOutcomeSide = false;

      for (const parent of parents) {
        // Check treatment side: parent is a treatment, or any treatment has path to parent
        if (treatmentSet.has(parent)) {
          hasParentFromTreatmentSide = true;
        } else {
          for (const treatmentNode of treatmentNodes) {
            const pathsFromTreatment = findAllDirectedPaths(treatmentNode, parent, adj);
            if (pathsFromTreatment.length > 0) {
              hasParentFromTreatmentSide = true;
              break;
            }
          }
        }

        // Check outcome side: parent is outcome (unlikely), or parent has path to outcome not through this node
        if (parent === outcomeNode) {
          hasParentFromOutcomeSide = true;
        } else {
          const pathsToOutcome = findAllDirectedPaths(parent, outcomeNode, adj);
          // Check if any path doesn't go through the controlled node
          for (const path of pathsToOutcome) {
            if (!path.includes(controlledId)) {
              hasParentFromOutcomeSide = true;
              break;
            }
          }
        }
      }

      // If both sides are connected, this is a problematic collider
      if (hasParentFromTreatmentSide && hasParentFromOutcomeSide) {
        controlledColliders.add(controlledId);
      }
    }
  }

  // Compute node positions for all nodes
  const nodePositions = new Map<string, CausalPosition>();
  for (const node of nodes) {
    const nodeId = node.id;
    if (treatmentSet.has(nodeId)) {
      nodePositions.set(nodeId, 'treatment');
    } else if (nodeId === outcomeNode) {
      nodePositions.set(nodeId, 'outcome');
    } else if (controlledColliders.has(nodeId)) {
      // Controlled collider - RED warning (introduces bias!)
      nodePositions.set(nodeId, 'collider');
    } else if (unblockedConfounders.has(nodeId)) {
      nodePositions.set(nodeId, 'confounder');
    } else if (controlledSet.has(nodeId)) {
      // Blocked confounder, controlled mediator, or plain controlled variable
      nodePositions.set(nodeId, 'controlled');
    } else if (mediatorNodes.has(nodeId)) {
      nodePositions.set(nodeId, 'mediator');
    } else if (outcomeInfluencers.has(nodeId)) {
      nodePositions.set(nodeId, 'outcomeInfluencer');
    } else {
      nodePositions.set(nodeId, 'irrelevant');
    }
  }

  // Check if all backdoor paths are blocked
  const allBackdoorsBlocked = unblockedConfounders.size === 0;

  // Check if there are collider bias issues
  const hasColliderBias = controlledColliders.size > 0;

  // Determine what type of effect is identified (aggregated across all treatments)
  const hasDirectEffect = directEdges.size > 0;
  const hasMediatedPaths = totalEdges.size > directEdges.size; // There are paths beyond the direct ones
  const hasUnblockedMediatedPaths = totalUnblockedMediatedPaths > 0; // Are any mediated paths still open?

  // Effect identification logic:
  // - Total effect: All backdoors blocked AND no collider bias AND mediated paths exist
  // - Direct effect: All backdoors blocked AND no collider bias AND (no mediated paths OR all blocked)
  // - None: There are unblocked backdoor paths OR there is collider bias
  let effectType: 'total' | 'direct' | 'none' = 'none';
  if (allBackdoorsBlocked && !hasColliderBias) {
    if (hasUnblockedMediatedPaths) {
      // Mediated paths exist and are not blocked - total effect is identified
      effectType = 'total';
    } else if (hasDirectEffect) {
      // Either no mediated paths, or all mediated paths are blocked - only direct effect
      effectType = 'direct';
    }
  }

  return {
    treatmentNodes,
    outcomeNode,
    directEdges,
    totalEdges,
    blockedMediatedEdges,
    backdoorEdges,
    blockedBackdoorEdges,
    outcomeOnlyEdges,
    confounders,
    unblockedConfounders,
    controlledColliders,
    mediatorNodes,
    outcomeInfluencers,
    nodePositions,
    requiredControls,
    controlledNodes: controlledSet,
    allBackdoorsBlocked,
    hasColliderBias,
    hasDirectEffect,
    hasMediatedPaths,
    hasUnblockedMediatedPaths,
    effectType,
    perTreatmentResults,
  };
}

// ── React Flow styling (ported from PlanningPage's styledEdges/styledNodes) ─

/** Apply causal-path coloring to edges. No-op (original array) when analysis is null. */
export function styleEdgesForAnalysis(edges: Edge[], analysis: CausalAnalysis | null): Edge[] {
  if (!analysis) {
    return edges;
  }

  return edges.map((edge) => {
    let color = PATH_COLORS.default;
    let strokeWidth = 2;
    let animated = false;
    let strokeDasharray: string | undefined = undefined;

    // Priority order for edge coloring:
    // 1. Unblocked backdoor (red, animated) - highest priority, dangerous
    // 2. Blocked backdoor (black, dashed) - controlled confounding
    // 3. Blocked mediated (orange, dashed) - controlled mediator
    // 4. Direct effect (blue) - treatment -> outcome directly
    // 5. Open mediated (green) - indirect causal paths
    // 6. Outcome-only (orange) - affects outcome but not treatment path
    // 7. Default (gray) - not on any relevant path

    if (analysis.backdoorEdges.has(edge.id)) {
      // Unblocked backdoor/anti-causal path - RED and animated (dangerous!)
      color = PATH_COLORS.backdoor;
      strokeWidth = 3;
      animated = true;
    } else if (analysis.blockedBackdoorEdges.has(edge.id)) {
      // Blocked backdoor path - BLACK and dashed
      color = PATH_COLORS.backdoorBlocked;
      strokeWidth = 3;
      strokeDasharray = '5,5';
    } else if (analysis.blockedMediatedEdges.has(edge.id)) {
      // Blocked/closed indirect path (controlled mediator) - ORANGE and dashed
      color = PATH_COLORS.mediatedBlocked;
      strokeWidth = 3;
      strokeDasharray = '5,5';
    } else if (analysis.directEdges.has(edge.id)) {
      // Direct causal effect - BLUE
      color = PATH_COLORS.direct;
      strokeWidth = 3;
    } else if (analysis.totalEdges.has(edge.id)) {
      // Open indirect/mediated paths - GREEN
      color = PATH_COLORS.mediated;
      strokeWidth = 3;
    } else if (analysis.outcomeOnlyEdges.has(edge.id)) {
      // Affects outcome but not from treatment - ORANGE
      color = PATH_COLORS.outcomeOnly;
      strokeWidth = 2;
    }

    return {
      ...edge,
      style: {
        ...edge.style,
        stroke: color,
        strokeWidth,
        ...(strokeDasharray ? { strokeDasharray } : {}),
      },
      animated,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color,
        width: 25,
        height: 25,
      },
    };
  });
}

/**
 * Attach `causalPosition` (+ boolean flags) to node data for analysis-mode
 * rendering. When the analysis is null (treatment/outcome not yet assigned),
 * user-assigned roles still surface so role clicks give immediate feedback.
 */
export function styleNodesForAnalysis(
  nodes: Node[],
  roles: Map<string, CausalRole>,
  analysis: CausalAnalysis | null,
): Node[] {
  if (!analysis) {
    // Pre-analysis: reflect raw role assignments only.
    return nodes.map((node) => {
      const role = roles.get(node.id) ?? null;
      const causalPosition: CausalPosition | undefined =
        role === 'treatment' ? 'treatment'
        : role === 'outcome' ? 'outcome'
        : role === 'controlled' ? 'controlled'
        : undefined;
      return { ...node, data: { ...node.data, causalPosition } };
    });
  }

  const treatmentSet = new Set(analysis.treatmentNodes);

  return nodes.map((node) => {
    const causalPosition = analysis.nodePositions.get(node.id) || 'irrelevant';

    return {
      ...node,
      data: {
        ...node.data,
        causalPosition,
        isTreatment: treatmentSet.has(node.id),
        isOutcome: node.id === analysis.outcomeNode,
        isMediator: analysis.mediatorNodes.has(node.id),
        isConfounder: analysis.unblockedConfounders.has(node.id),
        isCollider: analysis.controlledColliders.has(node.id),
        isControlled: analysis.controlledNodes.has(node.id),
        isOutcomeInfluencer: analysis.outcomeInfluencers.has(node.id),
      },
    };
  });
}
