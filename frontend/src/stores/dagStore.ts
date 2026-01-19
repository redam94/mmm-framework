import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Node, Edge } from '@xyflow/react';

// Node types for the DAG
export type DAGNodeType = 'kpi' | 'media' | 'control' | 'mediator' | 'transform' | 'outcome';

// Custom data for DAG nodes - extends Record to satisfy react-flow constraints
export interface DAGNodeData extends Record<string, unknown> {
  label: string;
  type: DAGNodeType;
  variableName: string;
  dimensions?: string[];
  config?: {
    // For media/mediator nodes
    adstockType?: string;
    adstockLMax?: number;
    saturationType?: string;
    // For all nodes
    priorType?: string;
    priorParams?: Record<string, number>;
    // For control nodes
    allowNegative?: boolean;
    // For KPI nodes
    logTransform?: boolean;
    // For mediator nodes
    mediatorType?: string;
    observationNoise?: number;
  };
}

// Type aliases for react-flow
export type DAGNode = Node<DAGNodeData>;
export type DAGEdge = Edge;

// Validation result
export interface DAGValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

interface DAGState {
  nodes: DAGNode[];
  edges: DAGEdge[];
  selectedNodeId: string | null;

  // Actions
  addNode: (node: DAGNode) => void;
  removeNode: (id: string) => void;
  updateNode: (id: string, data: Partial<DAGNodeData>) => void;
  updateNodePosition: (id: string, position: { x: number; y: number }) => void;
  setNodes: (nodes: DAGNode[]) => void;

  addEdge: (edge: DAGEdge) => void;
  removeEdge: (id: string) => void;
  setEdges: (edges: DAGEdge[]) => void;

  setSelectedNode: (id: string | null) => void;

  // Utilities
  clearDAG: () => void;
  validateDAG: () => DAGValidationResult;
  generateNarrative: () => string;

  // Templates
  loadTemplate: (template: 'simple' | 'mediation' | 'multivariate') => void;
}

// Generate unique ID
const generateId = () => `node_${Math.random().toString(36).substring(2, 9)}`;
const generateEdgeId = (source: string, target: string) => `edge_${source}_${target}`;

// Template DAGs
const TEMPLATES = {
  simple: {
    nodes: [
      {
        id: 'kpi_1',
        type: 'default',
        position: { x: 400, y: 300 },
        data: {
          label: 'Sales (KPI)',
          type: 'kpi' as DAGNodeType,
          variableName: 'sales',
          config: { logTransform: false },
        },
      },
      {
        id: 'media_1',
        type: 'default',
        position: { x: 100, y: 100 },
        data: {
          label: 'TV Spend',
          type: 'media' as DAGNodeType,
          variableName: 'tv_spend',
          config: { adstockType: 'geometric', saturationType: 'hill' },
        },
      },
      {
        id: 'media_2',
        type: 'default',
        position: { x: 250, y: 100 },
        data: {
          label: 'Digital Spend',
          type: 'media' as DAGNodeType,
          variableName: 'digital_spend',
          config: { adstockType: 'geometric', saturationType: 'hill' },
        },
      },
      {
        id: 'control_1',
        type: 'default',
        position: { x: 550, y: 100 },
        data: {
          label: 'Price',
          type: 'control' as DAGNodeType,
          variableName: 'price',
          config: { allowNegative: true },
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'media_1', target: 'kpi_1' },
      { id: 'e2', source: 'media_2', target: 'kpi_1' },
      { id: 'e3', source: 'control_1', target: 'kpi_1' },
    ],
  },
  mediation: {
    nodes: [
      {
        id: 'kpi_1',
        type: 'default',
        position: { x: 500, y: 400 },
        data: {
          label: 'Sales (KPI)',
          type: 'kpi' as DAGNodeType,
          variableName: 'sales',
        },
      },
      {
        id: 'media_1',
        type: 'default',
        position: { x: 100, y: 100 },
        data: {
          label: 'TV Spend',
          type: 'media' as DAGNodeType,
          variableName: 'tv_spend',
        },
      },
      {
        id: 'mediator_1',
        type: 'default',
        position: { x: 300, y: 250 },
        data: {
          label: 'Brand Awareness',
          type: 'mediator' as DAGNodeType,
          variableName: 'awareness',
          config: { mediatorType: 'partially_observed' },
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'media_1', target: 'mediator_1' },
      { id: 'e2', source: 'mediator_1', target: 'kpi_1' },
      { id: 'e3', source: 'media_1', target: 'kpi_1' }, // Direct effect
    ],
  },
  multivariate: {
    nodes: [
      {
        id: 'outcome_1',
        type: 'default',
        position: { x: 400, y: 300 },
        data: {
          label: 'Revenue',
          type: 'outcome' as DAGNodeType,
          variableName: 'revenue',
        },
      },
      {
        id: 'outcome_2',
        type: 'default',
        position: { x: 600, y: 300 },
        data: {
          label: 'Volume',
          type: 'outcome' as DAGNodeType,
          variableName: 'volume',
        },
      },
      {
        id: 'media_1',
        type: 'default',
        position: { x: 200, y: 100 },
        data: {
          label: 'Marketing Spend',
          type: 'media' as DAGNodeType,
          variableName: 'marketing',
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'media_1', target: 'outcome_1' },
      { id: 'e2', source: 'media_1', target: 'outcome_2' },
      { id: 'e3', source: 'outcome_1', target: 'outcome_2', animated: true }, // Cross-effect
    ],
  },
};

export const useDAGStore = create<DAGState>()(
  persist(
    (set, get) => ({
      nodes: [],
      edges: [],
      selectedNodeId: null,

      addNode: (node) => {
        set((state) => ({
          nodes: [...state.nodes, { ...node, id: node.id || generateId() }],
        }));
      },

      removeNode: (id) => {
        set((state) => ({
          nodes: state.nodes.filter((n) => n.id !== id),
          edges: state.edges.filter((e) => e.source !== id && e.target !== id),
          selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
        }));
      },

      updateNode: (id, data) => {
        set((state) => ({
          nodes: state.nodes.map((n) =>
            n.id === id ? { ...n, data: { ...n.data, ...data } } : n
          ),
        }));
      },

      updateNodePosition: (id, position) => {
        set((state) => ({
          nodes: state.nodes.map((n) => (n.id === id ? { ...n, position } : n)),
        }));
      },

      setNodes: (nodes) => set({ nodes }),

      addEdge: (edge) => {
        const id = edge.id || generateEdgeId(edge.source, edge.target);
        set((state) => {
          // Prevent duplicate edges
          if (state.edges.some((e) => e.source === edge.source && e.target === edge.target)) {
            return state;
          }
          return { edges: [...state.edges, { ...edge, id }] };
        });
      },

      removeEdge: (id) => {
        set((state) => ({
          edges: state.edges.filter((e) => e.id !== id),
        }));
      },

      setEdges: (edges) => set({ edges }),

      setSelectedNode: (id) => set({ selectedNodeId: id }),

      clearDAG: () => set({ nodes: [], edges: [], selectedNodeId: null }),

      validateDAG: () => {
        const state = get();
        const errors: string[] = [];
        const warnings: string[] = [];

        // Must have at least one KPI or outcome
        const kpis = state.nodes.filter(
          (n) => n.data.type === 'kpi' || n.data.type === 'outcome'
        );
        if (kpis.length === 0) {
          errors.push('DAG must have at least one KPI or outcome node');
        }

        // Must have at least one media channel
        const media = state.nodes.filter((n) => n.data.type === 'media');
        if (media.length === 0) {
          errors.push('DAG must have at least one media channel');
        }

        // All media should connect to at least one KPI/outcome (directly or via mediator)
        for (const m of media) {
          const hasConnection = state.edges.some((e) => e.source === m.id);
          if (!hasConnection) {
            warnings.push(`Media channel "${m.data.label}" has no outgoing connections`);
          }
        }

        // KPIs should have at least one incoming connection
        for (const k of kpis) {
          const hasIncoming = state.edges.some((e) => e.target === k.id);
          if (!hasIncoming) {
            warnings.push(`KPI "${k.data.label}" has no incoming connections`);
          }
        }

        // Check for cycles (simple check)
        // A proper cycle detection would use DFS
        const adjacency: Record<string, string[]> = {};
        for (const edge of state.edges) {
          if (!adjacency[edge.source]) adjacency[edge.source] = [];
          adjacency[edge.source].push(edge.target);
        }

        // Nodes should have unique variable names
        const varNames = state.nodes.map((n) => n.data.variableName).filter(Boolean);
        const uniqueVarNames = new Set(varNames);
        if (varNames.length !== uniqueVarNames.size) {
          errors.push('Duplicate variable names detected');
        }

        return {
          valid: errors.length === 0,
          errors,
          warnings,
        };
      },

      generateNarrative: () => {
        const state = get();
        const parts: string[] = [];

        const kpis = state.nodes.filter(
          (n) => n.data.type === 'kpi' || n.data.type === 'outcome'
        );
        const media = state.nodes.filter((n) => n.data.type === 'media');
        const controls = state.nodes.filter((n) => n.data.type === 'control');
        const mediators = state.nodes.filter((n) => n.data.type === 'mediator');

        if (kpis.length > 0) {
          const kpiNames = kpis.map((k) => k.data.label).join(', ');
          parts.push(`The model predicts ${kpiNames}.`);
        }

        if (media.length > 0) {
          const mediaNames = media.map((m) => m.data.label).join(', ');
          parts.push(
            `Marketing effectiveness is measured across ${media.length} channel(s): ${mediaNames}.`
          );

          // Describe transformations
          const withAdstock = media.filter((m) => m.data.config?.adstockType);
          if (withAdstock.length > 0) {
            parts.push(
              'Media effects include carryover (adstock) to capture delayed impact.'
            );
          }

          const withSaturation = media.filter((m) => m.data.config?.saturationType);
          if (withSaturation.length > 0) {
            parts.push(
              'Diminishing returns (saturation) are modeled to capture efficiency loss at high spend.'
            );
          }
        }

        if (controls.length > 0) {
          const controlNames = controls.map((c) => c.data.label).join(', ');
          parts.push(
            `The model controls for ${controls.length} external factor(s): ${controlNames}.`
          );
        }

        if (mediators.length > 0) {
          const mediatorNames = mediators.map((m) => m.data.label).join(', ');
          parts.push(
            `Indirect effects through ${mediators.length} mediator(s) (${mediatorNames}) are estimated.`
          );
        }

        return parts.join(' ');
      },

      loadTemplate: (template) => {
        const templateData = TEMPLATES[template];
        if (templateData) {
          set({
            nodes: templateData.nodes as DAGNode[],
            edges: templateData.edges as DAGEdge[],
            selectedNodeId: null,
          });
        }
      },
    }),
    {
      name: 'mmm-dag',
      partialize: (state) => ({
        nodes: state.nodes,
        edges: state.edges,
      }),
    }
  )
);
