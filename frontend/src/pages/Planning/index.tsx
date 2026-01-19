import { useCallback, useMemo, useState } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  BackgroundVariant,
  Handle,
  Position,
  MarkerType,
} from '@xyflow/react';
import type { Connection, Node, Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Card, Title, Text, Button, Select, SelectItem, Badge } from '@tremor/react';
import { PlayIcon, TrashIcon, DocumentTextIcon, EyeIcon } from '@heroicons/react/24/outline';
import { useDAGStore, type DAGNode, type DAGNodeData, type DAGNodeType } from '../../stores/dagStore';
import { useWorkflowStore } from '../../stores/workflowStore';

// Path colors for causal analysis - based on causal role
const PATH_COLORS = {
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
const NODE_COLORS = {
  treatment: { bg: '#DBEAFE', border: '#3B82F6', text: '#1E40AF' },     // Blue
  outcome: { bg: '#D1FAE5', border: '#10B981', text: '#065F46' },       // Green
  mediator: { bg: '#F3E8FF', border: '#A855F7', text: '#6B21A8' },      // Purple
  confounder: { bg: '#FEE2E2', border: '#EF4444', text: '#991B1B' },    // Red (unblocked confounder)
  collider: { bg: '#FEE2E2', border: '#EF4444', text: '#991B1B' },      // Red (controlled collider - bad!)
  controlled: { bg: '#E5E7EB', border: '#6B7280', text: '#374151' },    // Gray (observed/controlled)
  outcomeInfluencer: { bg: '#FED7AA', border: '#F97316', text: '#C2410C' }, // Orange (impacts outcome only)
  irrelevant: { bg: '#F3F4F6', border: '#D1D5DB', text: '#6B7280' },    // Light gray (no causal role)
};

// Helper: Build adjacency list from edges
function buildAdjacencyList(edges: Edge[]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>();
  for (const edge of edges) {
    if (!adj.has(edge.source)) adj.set(edge.source, new Set());
    adj.get(edge.source)!.add(edge.target);
  }
  return adj;
}

// Helper: Build reverse adjacency list (for finding parents)
function buildReverseAdjacencyList(edges: Edge[]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>();
  for (const edge of edges) {
    if (!adj.has(edge.target)) adj.set(edge.target, new Set());
    adj.get(edge.target)!.add(edge.source);
  }
  return adj;
}

// Find all directed paths from source to target (for total effect)
function findAllDirectedPaths(
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
function findDirectEffectEdges(treatment: string, outcome: string, edges: Edge[]): Set<string> {
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
function findTotalEffectEdges(
  treatment: string,
  outcome: string,
  edges: Edge[],
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
function findConfounders(
  treatment: string,
  outcome: string,
  _nodes: Node[],
  edges: Edge[],
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
function findRequiredControls(
  treatment: string,
  outcome: string,
  nodes: Node[],
  edges: Edge[],
  alreadyControlled: Set<string> = new Set()
): string[] {
  const adj = buildAdjacencyList(edges);
  const { unblockedConfounders } = findConfounders(treatment, outcome, nodes, edges, adj, alreadyControlled);

  // Return only the confounders that are not already controlled
  return Array.from(unblockedConfounders);
}

// Node type configurations
const NODE_TYPES: { type: DAGNodeType; label: string; color: string }[] = [
  { type: 'kpi', label: 'KPI/Outcome', color: '#10B981' },
  { type: 'media', label: 'Media Channel', color: '#3B82F6' },
  { type: 'control', label: 'Control Variable', color: '#8B5CF6' },
  { type: 'mediator', label: 'Mediator', color: '#F59E0B' },
];

// Causal role type for nodes (user-assigned)
type CausalRole = 'treatment' | 'outcome' | 'controlled' | null;

// Computed causal position based on graph structure
type CausalPosition =
  | 'treatment'           // Blue - user-designated treatment
  | 'outcome'             // Green - user-designated outcome
  | 'mediator'            // Purple - on causal path from treatment to outcome
  | 'confounder'          // Red - unblocked backdoor path source
  | 'collider'            // Red - controlled collider (introduces bias!)
  | 'controlled'          // Gray - user-controlled (blocks paths)
  | 'outcomeInfluencer'   // Orange - affects outcome but not treatment
  | 'irrelevant';         // Light gray - not on any relevant path

// Extended node data for causal analysis
interface ExtendedNodeData extends DAGNodeData {
  causalPosition?: CausalPosition;
  isConfounder?: boolean;
  isCollider?: boolean;
  isTreatment?: boolean;
  isOutcome?: boolean;
  isControlled?: boolean;
  isMediator?: boolean;
  isOutcomeInfluencer?: boolean;
}

// Custom node component with connection handles
function CustomNode({ data }: { data: ExtendedNodeData }) {
  // Default colors based on node type (when not in analysis mode)
  const defaultColorMap: Record<DAGNodeType, { bg: string; border: string; handle: string }> = {
    kpi: { bg: 'bg-green-100', border: 'border-green-500', handle: '#10B981' },
    media: { bg: 'bg-blue-100', border: 'border-blue-500', handle: '#3B82F6' },
    control: { bg: 'bg-purple-100', border: 'border-purple-500', handle: '#8B5CF6' },
    mediator: { bg: 'bg-yellow-100', border: 'border-yellow-500', handle: '#F59E0B' },
    transform: { bg: 'bg-gray-100', border: 'border-gray-500', handle: '#6B7280' },
    outcome: { bg: 'bg-green-100', border: 'border-green-500', handle: '#10B981' },
  };

  const defaultColors = defaultColorMap[data.type] || { bg: 'bg-gray-100', border: 'border-gray-300', handle: '#6B7280' };

  // Determine colors based on causal position (computed from graph analysis)
  let bgColor = defaultColors.bg;
  let borderColor = defaultColors.border;
  let handleColor = defaultColors.handle;
  let badge = null;
  let ringStyle = '';

  // Apply causal position-based styling when in analysis mode
  if (data.causalPosition) {
    const nodeColors = NODE_COLORS[data.causalPosition];
    bgColor = ''; // Use inline style instead
    borderColor = ''; // Use inline style instead
    handleColor = nodeColors.border;

    // Badge based on causal position
    switch (data.causalPosition) {
      case 'treatment':
        badge = <span className="absolute -top-2 -right-2 text-white text-xs px-1.5 py-0.5 rounded-full" style={{ backgroundColor: NODE_COLORS.treatment.border }}>T</span>;
        ringStyle = 'ring-2 ring-offset-2';
        break;
      case 'outcome':
        badge = <span className="absolute -top-2 -right-2 text-white text-xs px-1.5 py-0.5 rounded-full" style={{ backgroundColor: NODE_COLORS.outcome.border }}>O</span>;
        ringStyle = 'ring-2 ring-offset-2';
        break;
      case 'mediator':
        badge = <span className="absolute -top-2 -right-2 text-white text-xs px-1.5 py-0.5 rounded-full" style={{ backgroundColor: NODE_COLORS.mediator.border }}>M</span>;
        ringStyle = 'ring-2 ring-offset-2';
        break;
      case 'confounder':
        badge = <span className="absolute -top-2 -right-2 text-white text-xs px-1.5 py-0.5 rounded-full" style={{ backgroundColor: NODE_COLORS.confounder.border }}>!</span>;
        ringStyle = 'ring-2 ring-offset-2';
        break;
      case 'collider':
        badge = <span className="absolute -top-2 -right-2 text-white text-xs px-1.5 py-0.5 rounded-full" style={{ backgroundColor: NODE_COLORS.collider.border }}>⚠</span>;
        ringStyle = 'ring-2 ring-offset-2';
        break;
      case 'controlled':
        badge = <span className="absolute -top-2 -right-2 text-white text-xs px-1.5 py-0.5 rounded-full" style={{ backgroundColor: NODE_COLORS.controlled.border }}>✓</span>;
        ringStyle = 'ring-2 ring-offset-2';
        break;
      case 'outcomeInfluencer':
        badge = <span className="absolute -top-2 -right-2 text-white text-xs px-1.5 py-0.5 rounded-full" style={{ backgroundColor: NODE_COLORS.outcomeInfluencer.border }}>→O</span>;
        break;
    }
  }

  // Get inline styles for causal analysis mode
  const inlineStyle = data.causalPosition
    ? {
        backgroundColor: NODE_COLORS[data.causalPosition].bg,
        borderColor: NODE_COLORS[data.causalPosition].border,
        ...(ringStyle && { boxShadow: `0 0 0 2px white, 0 0 0 4px ${NODE_COLORS[data.causalPosition].border}` }),
      }
    : undefined;

  return (
    <div
      className={`px-4 py-2 rounded-lg border-2 shadow-sm relative ${!data.causalPosition ? `${bgColor} ${borderColor}` : ''}`}
      style={inlineStyle}
    >
      {badge}
      {/* Target handle (top) - for receiving connections */}
      <Handle
        type="target"
        position={Position.Top}
        style={{
          background: handleColor,
          width: 12,
          height: 12,
          border: '2px solid white',
          top: -6,
        }}
      />

      <div className="text-sm font-medium" style={data.causalPosition ? { color: NODE_COLORS[data.causalPosition].text } : undefined}>
        {data.label}
      </div>
      <div className="text-xs" style={{ color: data.causalPosition ? NODE_COLORS[data.causalPosition].text : '#6B7280', opacity: 0.8 }}>
        {data.type}
      </div>

      {/* Source handle (bottom) - for creating connections */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: handleColor,
          width: 12,
          height: 12,
          border: '2px solid white',
          bottom: -6,
        }}
      />
    </div>
  );
}

const nodeTypes = {
  default: CustomNode,
};

export function PlanningPage() {
  const {
    nodes: storeNodes,
    edges: storeEdges,
    addNode,
    removeNode,
    setNodes: setStoreNodes,
    setEdges: setStoreEdges,
    clearDAG,
    validateDAG,
    generateNarrative,
    loadTemplate,
    selectedNodeId,
    setSelectedNode,
  } = useDAGStore();

  const { logDecision } = useWorkflowStore();

  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState(storeNodes as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges);

  // Causal analysis state
  const [analysisMode, setAnalysisMode] = useState(false);
  const [causalRoles, setCausalRoles] = useState<Map<string, CausalRole>>(new Map());

  // Helper to get nodes by role
  const treatmentNodes = useMemo(
    () => Array.from(causalRoles.entries()).filter(([, role]) => role === 'treatment').map(([id]) => id),
    [causalRoles]
  );
  const outcomeNodes = useMemo(
    () => Array.from(causalRoles.entries()).filter(([, role]) => role === 'outcome').map(([id]) => id),
    [causalRoles]
  );
  const controlledNodes = useMemo(
    () => Array.from(causalRoles.entries()).filter(([, role]) => role === 'controlled').map(([id]) => id),
    [causalRoles]
  );

  // Set causal role for a node
  const setCausalRole = useCallback((nodeId: string, role: CausalRole) => {
    setCausalRoles((prev) => {
      const next = new Map(prev);
      if (role === null) {
        next.delete(nodeId);
      } else {
        next.set(nodeId, role);
      }
      return next;
    });
  }, []);

  // Toggle controlled status for a node
  const toggleControlled = useCallback((nodeId: string) => {
    setCausalRoles((prev) => {
      const next = new Map(prev);
      const currentRole = next.get(nodeId);
      if (currentRole === 'controlled') {
        next.delete(nodeId);
      } else if (!currentRole || currentRole === null) {
        next.set(nodeId, 'controlled');
      }
      return next;
    });
  }, []);

  // Sync with store
  const syncToStore = useCallback(() => {
    setStoreNodes(nodes as DAGNode[]);
    setStoreEdges(edges);
  }, [nodes, edges, setStoreNodes, setStoreEdges]);

  // Handle new connections - add directed edge with arrow
  const onConnect = useCallback(
    (params: Connection) => {
      const newEdge: Edge = {
        ...params,
        id: `edge_${params.source}_${params.target}`,
        type: 'smoothstep',
        animated: false,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: '#374151',
          width: 25,
          height: 25,
        },
        style: {
          stroke: '#374151',
          strokeWidth: 2,
        },
      };
      setEdges((eds) => addEdge(newEdge, eds));
    },
    [setEdges]
  );

  // Handle node selection
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNode(node.id);
    },
    [setSelectedNode]
  );

  // Add new node
  const handleAddNode = (type: DAGNodeType) => {
    const newNode: DAGNode = {
      id: `node_${Date.now()}`,
      type: 'default',
      position: { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 },
      data: {
        label: `New ${type}`,
        type,
        variableName: `var_${Date.now()}`,
        config: {},
      },
    };
    addNode(newNode);
    setNodes((nds) => [...nds, newNode as Node]);
  };

  // Delete selected node
  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      // Remove the node from local state
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      // Remove all edges connected to this node
      setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
      // Remove from store
      removeNode(nodeId);
      // Clear selection
      setSelectedNode(null);
    },
    [setNodes, setEdges, removeNode, setSelectedNode]
  );

  // Validate and log decision
  const handleValidateAndContinue = () => {
    syncToStore();
    const validation = validateDAG();

    if (validation.valid) {
      logDecision({
        phase: 'planning',
        type: 'dag_complete',
        rationale: `Defined generative model with ${nodes.length} nodes and ${edges.length} edges. ${generateNarrative()}`,
      });
      alert('DAG is valid! You can proceed to the next phase.');
    } else {
      alert(`Validation errors:\n${validation.errors.join('\n')}`);
    }
  };

  // Load template
  const handleLoadTemplate = (template: 'simple' | 'mediation' | 'multivariate') => {
    loadTemplate(template);
    // Reload nodes from store
    const store = useDAGStore.getState();
    setNodes(store.nodes as Node[]);
    setEdges(store.edges);
  };

  // Narrative
  const narrative = useMemo(() => generateNarrative(), [nodes, edges]);

  // Selected node details
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  // Causal analysis results - supports multiple treatments
  const causalAnalysis = useMemo(() => {
    if (!analysisMode || treatmentNodes.length === 0 || outcomeNodes.length === 0) {
      return null;
    }

    // For now, use the first outcome (can extend to multiple later)
    // But support multiple treatments
    const outcomeNode = outcomeNodes[0];
    const treatmentSet = new Set(treatmentNodes);
    const controlledSet = new Set(controlledNodes);

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
    const perTreatmentResults = new Map<string, {
      directEdges: Set<string>;
      totalEdges: Set<string>;
      blockedMediatedEdges: Set<string>;
      unblockedMediatedPaths: number;
      confounders: Set<string>;
      unblockedConfounders: Set<string>;
      hasDirectEffect: boolean;
      hasMediatedPaths: boolean;
      hasUnblockedMediatedPaths: boolean;
    }>();

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
        nodes,
        edges,
        adj,
        controlledSet
      );
      const treatmentRequiredControls = findRequiredControls(treatmentNode, outcomeNode, nodes, edges, controlledSet);

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
        // Check if it's a blocked confounder or just a regular controlled variable
        if (confounders.has(nodeId)) {
          // Blocked confounder - show as controlled (good)
          nodePositions.set(nodeId, 'controlled');
        } else if (mediatorNodes.has(nodeId)) {
          // Controlled mediator - show as controlled
          nodePositions.set(nodeId, 'controlled');
        } else {
          // Regular controlled variable
          nodePositions.set(nodeId, 'controlled');
        }
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
      treatmentNodes: treatmentNodes, // All treatment nodes
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
      perTreatmentResults, // Detailed per-treatment results
    };
  }, [analysisMode, treatmentNodes, outcomeNodes, controlledNodes, nodes, edges]);

  // Apply styling to edges based on causal analysis
  const styledEdges = useMemo(() => {
    if (!causalAnalysis) {
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

      if (causalAnalysis.backdoorEdges.has(edge.id)) {
        // Unblocked backdoor/anti-causal path - RED and animated (dangerous!)
        color = PATH_COLORS.backdoor;
        strokeWidth = 3;
        animated = true;
      } else if (causalAnalysis.blockedBackdoorEdges.has(edge.id)) {
        // Blocked backdoor path - BLACK and dashed
        color = PATH_COLORS.backdoorBlocked;
        strokeWidth = 3;
        strokeDasharray = '5,5';
      } else if (causalAnalysis.blockedMediatedEdges.has(edge.id)) {
        // Blocked/closed indirect path (controlled mediator) - ORANGE and dashed
        color = PATH_COLORS.mediatedBlocked;
        strokeWidth = 3;
        strokeDasharray = '5,5';
      } else if (causalAnalysis.directEdges.has(edge.id)) {
        // Direct causal effect - BLUE
        color = PATH_COLORS.direct;
        strokeWidth = 3;
      } else if (causalAnalysis.totalEdges.has(edge.id)) {
        // Open indirect/mediated paths - GREEN
        color = PATH_COLORS.mediated;
        strokeWidth = 3;
      } else if (causalAnalysis.outcomeOnlyEdges.has(edge.id)) {
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
          ...(strokeDasharray && { strokeDasharray }),
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
  }, [edges, causalAnalysis]);

  // Style nodes based on causal analysis - assign causalPosition for coloring
  const styledNodes = useMemo(() => {
    if (!causalAnalysis) {
      return nodes;
    }

    const treatmentSet = new Set(causalAnalysis.treatmentNodes);

    return nodes.map((node) => {
      const causalPosition = causalAnalysis.nodePositions.get(node.id) || 'irrelevant';

      return {
        ...node,
        data: {
          ...node.data,
          causalPosition,
          isTreatment: treatmentSet.has(node.id),
          isOutcome: node.id === causalAnalysis.outcomeNode,
          isMediator: causalAnalysis.mediatorNodes.has(node.id),
          isConfounder: causalAnalysis.unblockedConfounders.has(node.id),
          isCollider: causalAnalysis.controlledColliders.has(node.id),
          isControlled: causalAnalysis.controlledNodes.has(node.id),
          isOutcomeInfluencer: causalAnalysis.outcomeInfluencers.has(node.id),
        },
      };
    });
  }, [nodes, causalAnalysis]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <Title>Model Planning</Title>
          <Text>Design your data generative story with a DAG</Text>
        </div>
        <div className="flex gap-2">
          <Select
            placeholder="Load template"
            onValueChange={(v) => handleLoadTemplate(v as 'simple' | 'mediation' | 'multivariate')}
          >
            <SelectItem value="simple">Simple MMM</SelectItem>
            <SelectItem value="mediation">With Mediation</SelectItem>
            <SelectItem value="multivariate">Multi-Outcome</SelectItem>
          </Select>
          <Button
            icon={TrashIcon}
            variant="secondary"
            onClick={() => {
              clearDAG();
              setNodes([]);
              setEdges([]);
            }}
          >
            Clear
          </Button>
          <Button icon={PlayIcon} onClick={handleValidateAndContinue}>
            Validate & Continue
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-4 gap-6">
        {/* Node palette */}
        <Card className="col-span-1">
          <Title className="text-sm">Add Nodes</Title>
          <div className="mt-4 space-y-2">
            {NODE_TYPES.map(({ type, label, color }) => (
              <button
                key={type}
                onClick={() => handleAddNode(type)}
                className="w-full p-3 text-left rounded-lg border-2 border-dashed hover:border-solid transition-colors"
                style={{ borderColor: color, backgroundColor: `${color}20` }}
              >
                <span className="text-sm font-medium">{label}</span>
              </button>
            ))}
          </div>

          {/* Selected node editor */}
          {selectedNode && (
            <div className="mt-6 pt-6 border-t">
              <Title className="text-sm">Edit Node</Title>
              <div className="mt-4 space-y-3">
                {/* Node Label/Name */}
                <div>
                  <label className="text-xs text-gray-500">Display Name</label>
                  <input
                    type="text"
                    className="w-full mt-1 px-2 py-1 text-sm border rounded"
                    value={(selectedNode.data as DAGNodeData).label || ''}
                    onChange={(e) => {
                      setNodes((nds) =>
                        nds.map((n) =>
                          n.id === selectedNode.id
                            ? { ...n, data: { ...n.data, label: e.target.value } }
                            : n
                        )
                      );
                    }}
                    placeholder="Enter display name..."
                  />
                </div>
                {/* Variable Name */}
                <div>
                  <label className="text-xs text-gray-500">Variable Name (for model)</label>
                  <input
                    type="text"
                    className="w-full mt-1 px-2 py-1 text-sm border rounded"
                    value={(selectedNode.data as DAGNodeData).variableName || ''}
                    onChange={(e) => {
                      // Convert to snake_case-friendly format
                      const value = e.target.value.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');
                      setNodes((nds) =>
                        nds.map((n) =>
                          n.id === selectedNode.id
                            ? { ...n, data: { ...n.data, variableName: value } }
                            : n
                        )
                      );
                    }}
                    placeholder="e.g., tv_spend"
                  />
                  <Text className="text-xs text-gray-400 mt-1">
                    Used in model code (lowercase, underscores)
                  </Text>
                </div>
                {/* Node Type (read-only) */}
                <div>
                  <label className="text-xs text-gray-500">Type</label>
                  <div className="mt-1 px-2 py-1 text-sm bg-gray-100 rounded capitalize">
                    {(selectedNode.data as DAGNodeData).type}
                  </div>
                </div>
                {/* Delete Node Button */}
                <div className="pt-2">
                  <Button
                    icon={TrashIcon}
                    variant="secondary"
                    color="red"
                    size="xs"
                    className="w-full"
                    onClick={() => handleDeleteNode(selectedNode.id)}
                  >
                    Delete Node
                  </Button>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* DAG Canvas */}
        <Card className="col-span-2 h-[600px]">
          <div className="mb-2 px-2 flex justify-between items-center">
            <Text className="text-xs text-gray-500">
              Drag from a node's bottom handle to another node's top handle to create a directed connection.
            </Text>
            <Button
              icon={EyeIcon}
              variant={analysisMode ? 'primary' : 'secondary'}
              size="xs"
              onClick={() => {
                setAnalysisMode(!analysisMode);
                if (analysisMode) {
                  setCausalRoles(new Map());
                }
              }}
            >
              {analysisMode ? 'Exit Analysis' : 'Analyze Paths'}
            </Button>
          </div>
          <div className="h-[calc(100%-2rem)]">
            <ReactFlow
              nodes={analysisMode ? styledNodes : nodes}
              edges={analysisMode ? styledEdges : edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              nodeTypes={nodeTypes}
              defaultEdgeOptions={{
                type: 'smoothstep',
                markerEnd: {
                  type: MarkerType.ArrowClosed,
                  color: '#374151',
                  width: 25,
                  height: 25,
                },
                style: {
                  stroke: '#374151',
                  strokeWidth: 2,
                },
              }}
              fitView
              snapToGrid
              snapGrid={[15, 15]}
            >
              <Controls />
              <MiniMap />
              <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
            </ReactFlow>
          </div>
        </Card>

        {/* Narrative panel */}
        <Card className="col-span-1">
          <div className="flex items-center gap-2">
            <DocumentTextIcon className="h-5 w-5 text-gray-500" />
            <Title className="text-sm">Generated Narrative</Title>
          </div>
          <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <Text className="text-sm">
              {narrative || 'Add nodes to the canvas to generate a narrative.'}
            </Text>
          </div>

          {/* Validation status */}
          <div className="mt-6">
            <Title className="text-sm">Validation</Title>
            <div className="mt-2 space-y-1">
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${nodes.length > 0 ? 'bg-green-500' : 'bg-gray-300'}`}
                />
                <Text className="text-xs">Has nodes</Text>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${edges.length > 0 ? 'bg-green-500' : 'bg-gray-300'}`}
                />
                <Text className="text-xs">Has connections</Text>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${nodes.some((n) => (n.data as DAGNodeData).type === 'kpi') ? 'bg-green-500' : 'bg-gray-300'}`}
                />
                <Text className="text-xs">Has KPI/Outcome</Text>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${nodes.some((n) => (n.data as DAGNodeData).type === 'media') ? 'bg-green-500' : 'bg-gray-300'}`}
                />
                <Text className="text-xs">Has media channels</Text>
              </div>
            </div>
          </div>

          {/* Causal Analysis Panel */}
          {analysisMode && (
            <div className="mt-6 pt-6 border-t">
              <Title className="text-sm">Causal Path Analysis</Title>
              <Text className="text-xs text-gray-500 mt-1">
                Assign roles to nodes to analyze causal paths
              </Text>

              <div className="mt-4 space-y-3">
                {/* Node Role Assignment */}
                <div>
                  <label className="text-xs text-gray-500 font-medium">Assign Node Roles</label>
                  <div className="mt-2 space-y-2 max-h-48 overflow-y-auto">
                    {nodes.map((node) => {
                      const nodeData = node.data as DAGNodeData;
                      const currentRole = causalRoles.get(node.id) || null;
                      return (
                        <div key={node.id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                          <span className="text-xs font-medium truncate flex-1">{nodeData.label}</span>
                          <div className="flex gap-1 ml-2">
                            <button
                              onClick={() => setCausalRole(node.id, currentRole === 'treatment' ? null : 'treatment')}
                              className={`px-2 py-0.5 text-xs rounded ${
                                currentRole === 'treatment'
                                  ? 'bg-blue-500 text-white'
                                  : 'bg-gray-200 text-gray-600 hover:bg-blue-100'
                              }`}
                              title="Treatment"
                            >
                              T
                            </button>
                            <button
                              onClick={() => setCausalRole(node.id, currentRole === 'outcome' ? null : 'outcome')}
                              className={`px-2 py-0.5 text-xs rounded ${
                                currentRole === 'outcome'
                                  ? 'bg-green-500 text-white'
                                  : 'bg-gray-200 text-gray-600 hover:bg-green-100'
                              }`}
                              title="Outcome"
                            >
                              O
                            </button>
                            <button
                              onClick={() => setCausalRole(node.id, currentRole === 'controlled' ? null : 'controlled')}
                              className={`px-2 py-0.5 text-xs rounded ${
                                currentRole === 'controlled'
                                  ? 'bg-purple-500 text-white'
                                  : 'bg-gray-200 text-gray-600 hover:bg-purple-100'
                              }`}
                              title="Controlled"
                            >
                              ✓
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Current Selections */}
                <div className="pt-2 border-t">
                  <label className="text-xs text-gray-500">Current Selections</label>
                  <div className="mt-2 space-y-1 text-xs">
                    <div className="flex items-center gap-2">
                      <Badge color="blue" size="xs">T</Badge>
                      <span>
                        {treatmentNodes.length > 0
                          ? treatmentNodes.map((id) => (nodes.find((n) => n.id === id)?.data as DAGNodeData)?.label).join(', ')
                          : 'None selected'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge color="green" size="xs">O</Badge>
                      <span>
                        {outcomeNodes.length > 0
                          ? outcomeNodes.map((id) => (nodes.find((n) => n.id === id)?.data as DAGNodeData)?.label).join(', ')
                          : 'None selected'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge color="purple" size="xs">✓</Badge>
                      <span>
                        {controlledNodes.length > 0
                          ? controlledNodes.map((id) => (nodes.find((n) => n.id === id)?.data as DAGNodeData)?.label).join(', ')
                          : 'None controlled'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Edge Legend */}
                <div className="pt-2 border-t">
                  <label className="text-xs text-gray-500 font-medium">Edge Colors</label>
                  <div className="mt-2 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="w-4 h-1 rounded" style={{ backgroundColor: PATH_COLORS.direct }} />
                      <Text className="text-xs">Direct Effect (Blue)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-4 h-1 rounded" style={{ backgroundColor: PATH_COLORS.mediated }} />
                      <Text className="text-xs">Open Indirect (Green)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className="w-4 h-0.5 rounded"
                        style={{
                          backgroundColor: PATH_COLORS.mediatedBlocked,
                          backgroundImage: `repeating-linear-gradient(90deg, ${PATH_COLORS.mediatedBlocked} 0, ${PATH_COLORS.mediatedBlocked} 3px, transparent 3px, transparent 6px)`,
                        }}
                      />
                      <Text className="text-xs">Closed Indirect (Orange dashed)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-4 h-1 rounded" style={{ backgroundColor: PATH_COLORS.backdoor }} />
                      <Text className="text-xs">Unblocked Backdoor (Red)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className="w-4 h-0.5 rounded"
                        style={{
                          backgroundColor: PATH_COLORS.backdoorBlocked,
                          backgroundImage: `repeating-linear-gradient(90deg, ${PATH_COLORS.backdoorBlocked} 0, ${PATH_COLORS.backdoorBlocked} 3px, transparent 3px, transparent 6px)`,
                        }}
                      />
                      <Text className="text-xs">Blocked Backdoor (Black dashed)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-4 h-1 rounded" style={{ backgroundColor: PATH_COLORS.outcomeOnly }} />
                      <Text className="text-xs">Outcome Only (Orange)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-4 h-1 rounded" style={{ backgroundColor: PATH_COLORS.default }} />
                      <Text className="text-xs">Irrelevant (Gray)</Text>
                    </div>
                  </div>
                </div>

                {/* Node Legend */}
                <div className="pt-2 border-t">
                  <label className="text-xs text-gray-500 font-medium">Node Colors</label>
                  <div className="mt-2 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded border-2" style={{ backgroundColor: NODE_COLORS.treatment.bg, borderColor: NODE_COLORS.treatment.border }} />
                      <Text className="text-xs">Treatment (Blue)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded border-2" style={{ backgroundColor: NODE_COLORS.outcome.bg, borderColor: NODE_COLORS.outcome.border }} />
                      <Text className="text-xs">Outcome (Green)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded border-2" style={{ backgroundColor: NODE_COLORS.mediator.bg, borderColor: NODE_COLORS.mediator.border }} />
                      <Text className="text-xs">Mediator (Purple)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded border-2" style={{ backgroundColor: NODE_COLORS.confounder.bg, borderColor: NODE_COLORS.confounder.border }} />
                      <Text className="text-xs">Confounder (Red)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded border-2" style={{ backgroundColor: NODE_COLORS.collider.bg, borderColor: NODE_COLORS.collider.border }} />
                      <Text className="text-xs">Collider Bias (Red ⚠)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded border-2" style={{ backgroundColor: NODE_COLORS.controlled.bg, borderColor: NODE_COLORS.controlled.border }} />
                      <Text className="text-xs">Controlled (Gray)</Text>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded border-2" style={{ backgroundColor: NODE_COLORS.outcomeInfluencer.bg, borderColor: NODE_COLORS.outcomeInfluencer.border }} />
                      <Text className="text-xs">Outcome Influencer (Orange)</Text>
                    </div>
                  </div>
                </div>

                {/* Analysis results */}
                {causalAnalysis && (
                  <div className="pt-3 border-t">
                    <label className="text-xs text-gray-500">Analysis Results</label>
                    <div className="mt-2 space-y-2">
                      {/* Identification status */}
                      {causalAnalysis.effectType === 'total' ? (
                        <div className="p-2 bg-green-50 rounded border border-green-200">
                          <Text className="text-xs font-medium text-green-700">
                            ✓ Total Effect{causalAnalysis.treatmentNodes.length > 1 ? 's are' : ' is'} identified!
                          </Text>
                          <Text className="text-xs text-green-600 mt-1">
                            All backdoor paths are blocked for {causalAnalysis.treatmentNodes.length > 1 ? 'all ' + causalAnalysis.treatmentNodes.length + ' treatments' : 'the treatment'}. Both direct and mediated effects are estimable.
                          </Text>
                        </div>
                      ) : causalAnalysis.effectType === 'direct' ? (
                        <div className="p-2 bg-blue-50 rounded border border-blue-200">
                          <Text className="text-xs font-medium text-blue-700">
                            ✓ Direct Effect{causalAnalysis.treatmentNodes.length > 1 ? 's are' : ' is'} identified!
                          </Text>
                          <Text className="text-xs text-blue-600 mt-1">
                            {causalAnalysis.hasMediatedPaths && !causalAnalysis.hasUnblockedMediatedPaths
                              ? `Backdoor paths blocked for ${causalAnalysis.treatmentNodes.length > 1 ? 'all ' + causalAnalysis.treatmentNodes.length + ' treatments' : 'the treatment'}. Mediators are controlled, so only direct effects (excluding mediated paths) are estimable.`
                              : `All backdoor paths are blocked for ${causalAnalysis.treatmentNodes.length > 1 ? 'all ' + causalAnalysis.treatmentNodes.length + ' treatments' : 'the treatment'}. Only the direct effect is estimable (no mediated paths exist).`}
                          </Text>
                        </div>
                      ) : (
                        <div className="p-2 bg-yellow-50 rounded border border-yellow-200">
                          <Text className="text-xs font-medium text-yellow-700">
                            ⚠️ Causal effect{causalAnalysis.treatmentNodes.length > 1 ? 's' : ''} NOT identified
                          </Text>
                          <Text className="text-xs text-yellow-600 mt-1">
                            {causalAnalysis.hasColliderBias
                              ? 'Controlling for a collider introduces bias. Remove the controlled status from collider nodes.'
                              : `There are unblocked backdoor paths${causalAnalysis.treatmentNodes.length > 1 ? ' for one or more treatments' : ''}. Control for confounders to identify the effect.`}
                          </Text>
                        </div>
                      )}

                      {/* Collider Bias Warning */}
                      {causalAnalysis.hasColliderBias && (
                        <div className="p-2 bg-red-50 rounded border border-red-200">
                          <Text className="text-xs font-medium text-red-700">
                            ⚠ Collider Bias Detected!
                          </Text>
                          <Text className="text-xs text-red-600 mt-1">
                            Controlling for these nodes introduces spurious associations (collider bias):
                          </Text>
                          <div className="mt-2 flex flex-wrap gap-1">
                            {Array.from(causalAnalysis.controlledColliders).map((colliderId) => {
                              const colliderNode = nodes.find((n) => n.id === colliderId);
                              return (
                                <button
                                  key={colliderId}
                                  onClick={() => toggleControlled(colliderId)}
                                  className="inline-flex items-center gap-1 px-2 py-1 bg-red-100 hover:bg-gray-100 text-red-700 hover:text-gray-700 rounded text-xs transition-colors"
                                >
                                  {colliderNode ? (colliderNode.data as DAGNodeData).label : colliderId}
                                  <span className="text-xs">→ uncontrol</span>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Effect structure info */}
                      {causalAnalysis.allBackdoorsBlocked && (
                        <div className="p-2 bg-gray-50 rounded border border-gray-200">
                          <Text className="text-xs text-gray-600">
                            <strong>Effect structure:</strong>{' '}
                            {causalAnalysis.hasDirectEffect && causalAnalysis.hasMediatedPaths
                              ? causalAnalysis.hasUnblockedMediatedPaths
                                ? 'Direct + Mediated paths (total effect)'
                                : 'Direct path only (mediators controlled)'
                              : causalAnalysis.hasDirectEffect
                              ? 'Direct path only'
                              : causalAnalysis.hasMediatedPaths
                              ? causalAnalysis.hasUnblockedMediatedPaths
                                ? 'Mediated paths only (no direct effect)'
                                : 'All paths blocked (no effect estimable)'
                              : 'No causal paths found'}
                          </Text>
                        </div>
                      )}

                      {/* Unblocked Confounders */}
                      {causalAnalysis.unblockedConfounders.size > 0 && (
                        <div className="p-2 bg-red-50 rounded border border-red-200">
                          <Text className="text-xs font-medium text-red-700">
                            ⚠️ Unblocked Confounders
                          </Text>
                          <Text className="text-xs text-red-600 mt-1">
                            Mark these as controlled to block backdoor paths:
                          </Text>
                          <div className="mt-2 flex flex-wrap gap-1">
                            {Array.from(causalAnalysis.unblockedConfounders).map((confId) => {
                              const confNode = nodes.find((n) => n.id === confId);
                              return (
                                <button
                                  key={confId}
                                  onClick={() => toggleControlled(confId)}
                                  className="inline-flex items-center gap-1 px-2 py-1 bg-red-100 hover:bg-purple-100 text-red-700 hover:text-purple-700 rounded text-xs transition-colors"
                                >
                                  {confNode ? (confNode.data as DAGNodeData).label : confId}
                                  <span className="text-xs">→ ✓</span>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Controlled nodes (blocking confounders) */}
                      {controlledNodes.length > 0 && (
                        <div className="p-2 bg-purple-50 rounded border border-purple-200">
                          <Text className="text-xs font-medium text-purple-700">
                            🎛️ Controlled Variables
                          </Text>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {controlledNodes.map((ctrlId) => {
                              const ctrlNode = nodes.find((n) => n.id === ctrlId);
                              const isBlockingConfounder = causalAnalysis.confounders.has(ctrlId);
                              return (
                                <Badge
                                  key={ctrlId}
                                  color={isBlockingConfounder ? 'purple' : 'gray'}
                                  size="xs"
                                >
                                  {ctrlNode ? (ctrlNode.data as DAGNodeData).label : ctrlId}
                                  {isBlockingConfounder && ' (blocking)'}
                                </Badge>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Path counts */}
                      <div className="text-xs text-gray-600 space-y-0.5 pt-2">
                        <div>Treatments: {causalAnalysis.treatmentNodes.length}</div>
                        <div>Direct effect: {causalAnalysis.directEdges.size > 0 ? 'Yes' : 'No'}</div>
                        <div>Total effect paths: {causalAnalysis.totalEdges.size} edges</div>
                        <div>Backdoor paths: {causalAnalysis.backdoorEdges.size} edges</div>
                        <div>All confounders: {causalAnalysis.confounders.size}</div>
                        <div>Unblocked: {causalAnalysis.unblockedConfounders.size}</div>
                      </div>

                      {/* Per-treatment breakdown (when multiple treatments) */}
                      {causalAnalysis.treatmentNodes.length > 1 && (
                        <div className="pt-2 border-t mt-2">
                          <label className="text-xs text-gray-500 font-medium">Per-Treatment Analysis</label>
                          <div className="mt-2 space-y-2">
                            {causalAnalysis.treatmentNodes.map((treatmentId) => {
                              const treatmentNode = nodes.find((n) => n.id === treatmentId);
                              const treatmentLabel = treatmentNode ? (treatmentNode.data as DAGNodeData).label : treatmentId;
                              const results = causalAnalysis.perTreatmentResults.get(treatmentId);
                              if (!results) return null;

                              const isIdentified = results.unblockedConfounders.size === 0;
                              const effectType = isIdentified
                                ? results.hasUnblockedMediatedPaths
                                  ? 'total'
                                  : results.hasDirectEffect
                                  ? 'direct'
                                  : 'none'
                                : 'none';

                              return (
                                <div
                                  key={treatmentId}
                                  className={`p-2 rounded border ${
                                    effectType === 'total'
                                      ? 'bg-green-50 border-green-200'
                                      : effectType === 'direct'
                                      ? 'bg-blue-50 border-blue-200'
                                      : 'bg-yellow-50 border-yellow-200'
                                  }`}
                                >
                                  <div className="flex items-center justify-between">
                                    <Text className="text-xs font-medium">{treatmentLabel}</Text>
                                    <Badge
                                      color={effectType === 'total' ? 'green' : effectType === 'direct' ? 'blue' : 'yellow'}
                                      size="xs"
                                    >
                                      {effectType === 'total' ? 'Total' : effectType === 'direct' ? 'Direct' : 'Not ID'}
                                    </Badge>
                                  </div>
                                  <div className="mt-1 text-xs text-gray-500">
                                    {results.unblockedConfounders.size > 0 && (
                                      <span className="text-red-600">
                                        {results.unblockedConfounders.size} unblocked confounder{results.unblockedConfounders.size > 1 ? 's' : ''}
                                      </span>
                                    )}
                                    {results.unblockedConfounders.size === 0 && results.hasDirectEffect && (
                                      <span>Direct path exists</span>
                                    )}
                                    {results.unblockedConfounders.size === 0 && results.hasUnblockedMediatedPaths && (
                                      <span>{results.hasDirectEffect ? ' + ' : ''}Mediated paths open</span>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Instructions when no analysis */}
                {!causalAnalysis && (
                  <div className="p-3 bg-gray-50 rounded border border-gray-200">
                    <Text className="text-xs text-gray-600">
                      Select at least one <Badge color="blue" size="xs">T</Badge> Treatment and one <Badge color="green" size="xs">O</Badge> Outcome to analyze causal paths.
                    </Text>
                  </div>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

export default PlanningPage;
