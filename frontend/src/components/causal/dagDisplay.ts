// Non-component DAG display helpers shared by CausalWidgets and CausalPlanner.
// Kept out of the .tsx component files so react-refresh (HMR) stays happy.

import dagre from '@dagrejs/dagre';
import { MarkerType, type Node as RFNode, type Edge as RFEdge } from '@xyflow/react';
import type { DagPayload } from './CausalWidgets';

export interface RawDagNode { id: string; data?: Record<string, unknown> | null }
export interface RawDagEdge { source: string; target: string }

export const NODE_STYLE: Record<string, { bg: string; border: string; text: string }> = {
  kpi:       { bg: '#eef2ff', border: '#4f46e5', text: '#312e81' },
  media:     { bg: '#ecfdf5', border: '#059669', text: '#064e3b' },
  control:   { bg: '#fef3c7', border: '#d97706', text: '#78350f' },
  mediator:  { bg: '#fce7f3', border: '#db2777', text: '#831843' },
  outcome:   { bg: '#e0e7ff', border: '#6366f1', text: '#1e1b4b' },
};

export const NODE_W = 148, NODE_H = 66;

export function classifyNodes(
  rfNodes: RawDagNode[],
  rfEdges: RawDagEdge[],
) {
  const outOf = new Map<string, Set<string>>();
  rfNodes.forEach(n => outOf.set(n.id, new Set()));
  rfEdges.forEach(e => outOf.get(e.source)?.add(e.target));
  const mediaSet = new Set(rfNodes.filter(n => n.data?.type === 'media').map(n => n.id));
  const isConf = (n: RawDagNode) =>
    n.data?.type === 'control' && [...(outOf.get(n.id) ?? [])].some(t => mediaSet.has(t));
  const isBiz = (n: RawDagNode) =>
    n.data?.type === 'control' && !isConf(n);
  return { isConf, isBiz, mediaSet, outOf };
}

export function computeDAGLayout(
  rfNodes: RawDagNode[],
  rfEdges: RawDagEdge[],
): Record<string, { x: number; y: number }> {
  if (rfNodes.length === 0) return {};

  const { isConf, isBiz } = classifyNodes(rfNodes, rfEdges);

  // Assign dagre rank (column) manually so the semantic order is preserved:
  // confounders → media → mediator → kpi; biz controls in kpi column but lower rank
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: 'LR',
    ranksep: 160,   // wider rank gap forces more pronounced S-curves
    nodesep: 44,
    edgesep: 25,
    marginx: 30,
    marginy: 30,
  });

  // Assign explicit rank constraints via dummy "rank groups"
  const rankOf = (n: RawDagNode) => {
    if (isConf(n)) return 0;
    if (n.data?.type === 'media') return 1;
    if (n.data?.type === 'mediator') return 2;
    if (n.data?.type === 'kpi') return 3;
    if (isBiz(n)) return 3;  // same rank as KPI so they sit beside/below it
    return 2;
  };

  rfNodes.forEach(n => {
    g.setNode(n.id, { width: NODE_W, height: NODE_H, rank: rankOf(n) });
  });

  rfEdges.forEach(e => g.setEdge(e.source, e.target));

  dagre.layout(g);

  const positions: Record<string, { x: number; y: number }> = {};
  rfNodes.forEach(n => {
    const nd = g.node(n.id);
    if (nd) {
      // dagre centers on (x,y); React Flow uses top-left corner
      positions[n.id] = { x: nd.x - NODE_W / 2, y: nd.y - NODE_H / 2 };
    }
  });
  return positions;
}

// ── DAG payload → React Flow converters (used by CausalPlanner) ─────────────

export function dagToRFNodes(dag: DagPayload, posMap: Record<string, { x: number; y: number }>): RFNode[] {
  const rfNodes = dag.react_flow?.nodes ?? [];
  const { isConf, isBiz } = classifyNodes(rfNodes, dag.react_flow?.edges ?? []);
  return rfNodes.map(n => {
    const nt = String(n.data?.type ?? 'control');
    const conf = isConf(n);
    const biz  = isBiz(n);
    return {
      id: n.id,
      position: posMap[n.id] ?? n.position ?? { x: 0, y: 0 },
      type: 'mmmNode',
      data: {
        label:    n.data?.variableName ?? n.id,
        nodeType: nt,
        badge:    conf ? 'confounder' : biz ? 'control' : nt,
        ...(NODE_STYLE[nt] ?? NODE_STYLE.control),
      },
    };
  });
}

export function dagToRFEdges(dag: DagPayload): RFEdge[] {
  return (dag.react_flow?.edges ?? []).map(e => {
    const et = String(e.data?.edgeType ?? 'direct');
    const color = et === 'mediated' ? '#db2777' : et === 'crossEffect' ? '#6366f1' : '#64748b';
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      type: 'default',
      data: { edgeType: et },
      style: { stroke: color, strokeWidth: 1.8 },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
    };
  });
}
