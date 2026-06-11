// CausalPlanner — the embedded causal planning workspace for the agent page.
// Replaces both the old EditableDAGViewer widget and the standalone Planning
// page. Three modes:
//   view    — read-only canvas + SemanticCallouts (same as the old viewer)
//   edit    — drag/connect/add/delete nodes, load templates, debounced autosave
//   analyze — assign Treatment/Outcome/Controlled roles, see causal paths,
//             confounders, mediators and collider-bias warnings (ported from
//             the Planning page's analysis engine, now in ./analysis.ts)

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  MarkerType,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Node as RFNode,
  type Edge as RFEdge,
  type NodeChange,
  type EdgeChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Eye, Lock, Maximize2, Network, Pencil, Plus, Workflow, X, CheckCircle2,
} from 'lucide-react';
import { API_BASE_URL, apiClient } from '../../api/client';
import { Modal } from '../../pages/Agent/components/common/Modal';
import { PanelShell, SemanticCallouts, MMMNode, type DagPayload } from './CausalWidgets';
import { NODE_STYLE, computeDAGLayout, dagToRFNodes, dagToRFEdges } from './dagDisplay';
import {
  NODE_COLORS, PATH_COLORS, computeCausalAnalysis, styleEdgesForAnalysis,
  styleNodesForAnalysis, type CausalAnalysis, type CausalPosition, type CausalRole,
} from './analysis';
import { DAG_TEMPLATES } from './templates';

type PlannerMode = 'view' | 'edit' | 'analyze';
type SaveState = 'idle' | 'saving' | 'saved' | 'error';

const NODE_TYPES_LIST = [
  { value: 'media',    label: 'Media' },
  { value: 'kpi',      label: 'KPI / Outcome' },
  { value: 'control',  label: 'Control' },
  { value: 'mediator', label: 'Mediator' },
];

// ── Node renderer: MMMNode in view/edit, NODE_COLORS styling in analyze ─────

interface PlannerNodeData extends Record<string, unknown> {
  label?: string;
  nodeType?: string;
  badge?: string;
  causalPosition?: CausalPosition;
}

const POSITION_BADGE: Partial<Record<CausalPosition, string>> = {
  treatment: 'T', outcome: 'O', mediator: 'M', confounder: '!',
  collider: '⚠', controlled: '✓', outcomeInfluencer: '→O',
};

function PlannerNodeComponent({ data }: { data: PlannerNodeData }) {
  const pos = data.causalPosition;
  if (!pos) return <MMMNode data={data} />;

  const colors = NODE_COLORS[pos];
  const badge = POSITION_BADGE[pos];
  const ring = pos !== 'outcomeInfluencer' && pos !== 'irrelevant';
  const handleStyle = { background: colors.border, width: 8, height: 8, border: 'none' };

  return (
    <div style={{
      background: colors.bg,
      border: `2px solid ${colors.border}`,
      borderRadius: 12,
      padding: '9px 16px', minWidth: 130, textAlign: 'center',
      position: 'relative', userSelect: 'none',
      ...(ring ? { boxShadow: `0 0 0 2px white, 0 0 0 4px ${colors.border}` } : { boxShadow: '0 2px 10px rgba(0,0,0,0.08)' }),
    }}>
      {badge && (
        <span style={{
          position: 'absolute', top: -9, right: -9, background: colors.border,
          color: 'white', fontSize: 10, fontWeight: 700, borderRadius: 999,
          padding: '1px 6px', lineHeight: '14px',
        }}>{badge}</span>
      )}
      <Handle type="target" position={Position.Left}   id="tgt-left"   style={handleStyle} />
      <Handle type="target" position={Position.Bottom} id="tgt-bottom" style={{ ...handleStyle, opacity: 0, pointerEvents: 'none' as const }} />
      <div style={{ fontWeight: 700, fontSize: 13, color: colors.text, lineHeight: 1.3 }}>{String(data.label ?? '')}</div>
      <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: colors.text, opacity: 0.75, marginTop: 3 }}>
        {pos}
      </div>
      <Handle type="source" position={Position.Right} id="src-right" style={handleStyle} />
      <Handle type="source" position={Position.Top}   id="src-top"   style={{ ...handleStyle, opacity: 0, pointerEvents: 'none' as const }} />
    </div>
  );
}

const PLANNER_NODE_TYPES = { mmmNode: PlannerNodeComponent };

// ── Analyze-mode legend (static; memoized) ──────────────────────────────────

const EDGE_LEGEND: Array<{ color: string; label: string; dashed?: boolean }> = [
  { color: PATH_COLORS.direct, label: 'Direct effect' },
  { color: PATH_COLORS.mediated, label: 'Open indirect' },
  { color: PATH_COLORS.mediatedBlocked, label: 'Closed indirect', dashed: true },
  { color: PATH_COLORS.backdoor, label: 'Unblocked backdoor' },
  { color: PATH_COLORS.backdoorBlocked, label: 'Blocked backdoor', dashed: true },
  { color: PATH_COLORS.outcomeOnly, label: 'Outcome only' },
  { color: PATH_COLORS.default, label: 'Irrelevant' },
];

const NODE_LEGEND: Array<{ pos: CausalPosition; label: string }> = [
  { pos: 'treatment', label: 'Treatment' },
  { pos: 'outcome', label: 'Outcome' },
  { pos: 'mediator', label: 'Mediator' },
  { pos: 'confounder', label: 'Confounder' },
  { pos: 'collider', label: 'Collider bias' },
  { pos: 'controlled', label: 'Controlled' },
  { pos: 'outcomeInfluencer', label: 'Outcome influencer' },
];

const AnalysisLegend = memo(function AnalysisLegend() {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-xl border border-gray-200 bg-white px-4 py-3">
      <div>
        <p className="text-[10px] uppercase tracking-wider font-semibold text-gray-400 mb-1.5">Edges</p>
        {EDGE_LEGEND.map(e => (
          <div key={e.label} className="flex items-center gap-2 py-0.5">
            <span
              className="w-4 h-0.5 rounded shrink-0"
              style={e.dashed
                ? { backgroundImage: `repeating-linear-gradient(90deg, ${e.color} 0, ${e.color} 3px, transparent 3px, transparent 6px)` }
                : { backgroundColor: e.color }}
            />
            <span className="text-[11px] text-gray-600">{e.label}</span>
          </div>
        ))}
      </div>
      <div>
        <p className="text-[10px] uppercase tracking-wider font-semibold text-gray-400 mb-1.5">Nodes</p>
        {NODE_LEGEND.map(n => (
          <div key={n.pos} className="flex items-center gap-2 py-0.5">
            <span className="w-3 h-3 rounded border-2 shrink-0" style={{ backgroundColor: NODE_COLORS[n.pos].bg, borderColor: NODE_COLORS[n.pos].border }} />
            <span className="text-[11px] text-gray-600">{n.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

// ── Analyze-mode summary panel (memoized; props only) ───────────────────────

const AnalysisSummary = memo(function AnalysisSummary({
  analysis, labels, onToggleControlled,
}: {
  analysis: CausalAnalysis | null;
  labels: Record<string, string>;
  onToggleControlled: (nodeId: string) => void;
}) {
  if (!analysis) {
    return (
      <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-3">
        <p className="text-xs text-gray-600">
          Click a node to assign it a role. Select at least one <span className="font-semibold text-blue-600">Treatment</span> and
          one <span className="font-semibold text-emerald-600">Outcome</span> to analyze causal paths.
        </p>
      </div>
    );
  }

  const nameOf = (id: string) => labels[id] ?? id;
  const plural = analysis.treatmentNodes.length > 1;

  return (
    <div className="space-y-2">
      {/* Identification status */}
      {analysis.effectType === 'total' ? (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 px-4 py-3">
          <p className="text-xs font-semibold text-emerald-700">✓ Total effect{plural ? 's are' : ' is'} identified</p>
          <p className="text-[11px] text-emerald-600 mt-1">
            All backdoor paths are blocked for {plural ? `all ${analysis.treatmentNodes.length} treatments` : 'the treatment'}. Both direct and mediated effects are estimable.
          </p>
        </div>
      ) : analysis.effectType === 'direct' ? (
        <div className="rounded-xl border border-blue-200 bg-blue-50/60 px-4 py-3">
          <p className="text-xs font-semibold text-blue-700">✓ Direct effect{plural ? 's are' : ' is'} identified</p>
          <p className="text-[11px] text-blue-600 mt-1">
            {analysis.hasMediatedPaths && !analysis.hasUnblockedMediatedPaths
              ? 'Backdoor paths are blocked and mediators are controlled, so only direct effects (excluding mediated paths) are estimable.'
              : 'All backdoor paths are blocked. Only the direct effect is estimable (no mediated paths exist).'}
          </p>
        </div>
      ) : (
        <div className="rounded-xl border border-amber-200 bg-amber-50/60 px-4 py-3">
          <p className="text-xs font-semibold text-amber-700">⚠ Causal effect{plural ? 's' : ''} NOT identified</p>
          <p className="text-[11px] text-amber-600 mt-1">
            {analysis.hasColliderBias
              ? 'Controlling for a collider introduces bias. Remove the controlled status from collider nodes.'
              : 'There are unblocked backdoor paths. Control for confounders to identify the effect.'}
          </p>
        </div>
      )}

      {/* Collider bias warning */}
      {analysis.hasColliderBias && (
        <div className="rounded-xl border border-red-200 bg-red-50/60 px-4 py-3">
          <p className="text-xs font-semibold text-red-700">⚠ Collider bias detected</p>
          <p className="text-[11px] text-red-600 mt-1">Controlling for these nodes opens spurious associations:</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {Array.from(analysis.controlledColliders).map(id => (
              <button
                key={id}
                onClick={() => onToggleControlled(id)}
                className="inline-flex items-center gap-1 px-2 py-0.5 bg-red-100 hover:bg-gray-100 text-red-700 hover:text-gray-700 rounded-full text-[11px] transition-colors"
              >
                {nameOf(id)} <span>→ uncontrol</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Unblocked confounders */}
      {analysis.unblockedConfounders.size > 0 && (
        <div className="rounded-xl border border-red-200 bg-red-50/60 px-4 py-3">
          <p className="text-xs font-semibold text-red-700">⚠ Unblocked confounders</p>
          <p className="text-[11px] text-red-600 mt-1">Mark these as controlled to block backdoor paths:</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {Array.from(analysis.unblockedConfounders).map(id => (
              <button
                key={id}
                onClick={() => onToggleControlled(id)}
                className="inline-flex items-center gap-1 px-2 py-0.5 bg-red-100 hover:bg-indigo-100 text-red-700 hover:text-indigo-700 rounded-full text-[11px] transition-colors"
              >
                {nameOf(id)} <span>→ control</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Controlled variables */}
      {analysis.controlledNodes.size > 0 && (
        <div className="rounded-xl border border-indigo-200 bg-indigo-50/60 px-4 py-3">
          <p className="text-xs font-semibold text-indigo-700">Controlled variables</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {Array.from(analysis.controlledNodes).map(id => (
              <span key={id} className={`px-2 py-0.5 rounded-full text-[11px] border ${
                analysis.confounders.has(id)
                  ? 'bg-indigo-100 border-indigo-300 text-indigo-700'
                  : 'bg-gray-100 border-gray-300 text-gray-600'
              }`}>
                {nameOf(id)}{analysis.confounders.has(id) ? ' (blocking)' : ''}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Per-treatment breakdown */}
      <div className="rounded-xl border border-gray-200 bg-white px-4 py-3">
        <p className="text-[10px] uppercase tracking-wider font-semibold text-gray-400 mb-2">Per-treatment analysis</p>
        <div className="space-y-1.5">
          {analysis.treatmentNodes.map(tid => {
            const r = analysis.perTreatmentResults.get(tid);
            if (!r) return null;
            const identified = r.unblockedConfounders.size === 0;
            const effect = identified ? (r.hasUnblockedMediatedPaths ? 'total' : r.hasDirectEffect ? 'direct' : 'none') : 'none';
            return (
              <div key={tid} className={`rounded-lg border px-3 py-2 ${
                effect === 'total' ? 'border-emerald-200 bg-emerald-50/50'
                : effect === 'direct' ? 'border-blue-200 bg-blue-50/50'
                : 'border-amber-200 bg-amber-50/50'
              }`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-gray-800 truncate">{nameOf(tid)}</span>
                  <span className={`text-[10px] font-semibold uppercase tracking-wider rounded px-1.5 py-0.5 ${
                    effect === 'total' ? 'bg-emerald-100 text-emerald-700'
                    : effect === 'direct' ? 'bg-blue-100 text-blue-700'
                    : 'bg-amber-100 text-amber-700'
                  }`}>
                    {effect === 'none' ? 'not identified' : `${effect} effect`}
                  </span>
                </div>
                <div className="mt-1 text-[11px] text-gray-500 space-y-0.5">
                  <div>
                    Required controls:{' '}
                    {r.requiredControls.length > 0
                      ? <span className="text-red-600 font-medium">{r.requiredControls.map(nameOf).join(', ')}</span>
                      : <span className="text-emerald-600">none — backdoors blocked</span>}
                  </div>
                  {r.unblockedConfounders.size > 0 && (
                    <div className="text-red-600">
                      {r.unblockedConfounders.size} unblocked confounder{r.unblockedConfounders.size > 1 ? 's' : ''}: {Array.from(r.unblockedConfounders).map(nameOf).join(', ')}
                    </div>
                  )}
                  <div>
                    {r.hasDirectEffect ? 'Direct path exists' : 'No direct path'}
                    {r.hasMediatedPaths ? ` · mediated paths ${r.hasUnblockedMediatedPaths ? 'open' : 'blocked'}` : ''}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Path counts */}
      <div className="text-[11px] text-gray-500 px-1 flex flex-wrap gap-x-4 gap-y-0.5">
        <span>Treatments: {analysis.treatmentNodes.length}</span>
        <span>Total-effect edges: {analysis.totalEdges.size}</span>
        <span>Backdoor edges: {analysis.backdoorEdges.size}</span>
        <span>Confounders: {analysis.confounders.size} ({analysis.unblockedConfounders.size} unblocked)</span>
      </div>
    </div>
  );
});

// ── Helpers ─────────────────────────────────────────────────────────────────

function structuralSignature(nodes: RFNode[], edges: RFEdge[]): string {
  return JSON.stringify({
    n: nodes.map(n => [n.id, Math.round(n.position.x), Math.round(n.position.y), n.data?.label, n.data?.nodeType]),
    e: edges.map(e => [e.id, e.source, e.target]),
  });
}

function buildPutPayload(nodes: RFNode[], edges: RFEdge[]) {
  return {
    nodes: nodes.map(n => ({
      id: n.id,
      position: n.position,
      data: {
        label: n.data.label,
        variableName: n.data.label,
        type: n.data.nodeType,
      },
    })),
    edges: edges.map(e => ({
      id: e.id,
      source: e.source,
      target: e.target,
      data: { edgeType: (e.data as Record<string, unknown> | undefined)?.edgeType ?? 'direct' },
    })),
  };
}

const EDGE_DEFAULTS = {
  type: 'default' as const,
  data: { edgeType: 'direct' },
  style: { stroke: '#64748b', strokeWidth: 1.8 },
  markerEnd: { type: MarkerType.ArrowClosed, color: '#64748b', width: 14, height: 14 },
};

// ── CausalPlanner ───────────────────────────────────────────────────────────

export function CausalPlanner({
  dag,
  threadId,
  chatStreaming,
  onSaved,
}: {
  dag: DagPayload | null;
  threadId: string | null;
  chatStreaming: boolean;
  onSaved: () => void;
}) {
  const [mode, setMode] = useState<PlannerMode>('view');
  const [nodes, setNodes, onNodesChange] = useNodesState<RFNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<RFEdge>([]);
  const [roles, setRoles] = useState<Map<string, CausalRole>>(new Map());
  const [roleMenuNodeId, setRoleMenuNodeId] = useState<string | null>(null);
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState('media');
  const [dirty, setDirty] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>('idle');
  const [expanded, setExpanded] = useState(false);
  const [lockState, setLockState] = useState<'idle' | 'locking' | 'locked' | 'error'>('idle');
  const nameInputRef = useRef<HTMLInputElement>(null);

  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  useEffect(() => {
    nodesRef.current = nodes;
    edgesRef.current = edges;
  }, [nodes, edges]);

  // ── Inbound sync: rebuild canvas from the dag prop when not editing and no
  //    unsaved local changes (so a mid-stream agent DAG update can't clobber
  //    pending edits before they're flushed).
  const syncFromDag = useCallback(() => {
    if (!dag) { setNodes([]); setEdges([]); return; }
    const rfNodes = dag.react_flow?.nodes ?? [];
    const rfEdges = dag.react_flow?.edges ?? [];
    const layout = computeDAGLayout(rfNodes, rfEdges);
    // Prefer round-tripped positions when the payload carries them; fall back
    // to the dagre layout for agent-proposed DAGs without meaningful positions.
    const hasStored = rfNodes.some(n => n.position && (n.position.x !== 0 || n.position.y !== 0));
    const posMap: Record<string, { x: number; y: number }> = hasStored
      ? Object.fromEntries(rfNodes.map(n => [
          n.id,
          n.position && (n.position.x !== 0 || n.position.y !== 0) ? n.position : (layout[n.id] ?? { x: 0, y: 0 }),
        ]))
      : layout;
    setNodes(dagToRFNodes(dag, posMap));
    setEdges(dagToRFEdges(dag));
  }, [dag, setNodes, setEdges]);

  useEffect(() => {
    if (mode === 'edit' || dirty) return;
    syncFromDag();
  }, [mode, dirty, syncFromDag]);

  // ── Outbound: mark dirty on structural change while editing. The change
  //    handlers are only wired up in edit mode, so this can't fire elsewhere.
  //    Selection-only changes are ignored.
  const handleNodesChange = useCallback((changes: NodeChange<RFNode>[]) => {
    onNodesChange(changes);
    if (changes.some(c => c.type === 'position' || c.type === 'remove' || c.type === 'add' || c.type === 'replace')) {
      setDirty(true);
    }
  }, [onNodesChange]);

  const handleEdgesChange = useCallback((changes: EdgeChange<RFEdge>[]) => {
    onEdgesChange(changes);
    if (changes.some(c => c.type === 'remove' || c.type === 'add' || c.type === 'replace')) {
      setDirty(true);
    }
  }, [onEdgesChange]);

  const doSave = useCallback(async () => {
    if (!threadId) { setDirty(false); return; }
    const sigAtStart = structuralSignature(nodesRef.current, edgesRef.current);
    setSaveState('saving');
    try {
      const res = await fetch(`${API_BASE_URL}/dag/${threadId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPutPayload(nodesRef.current, edgesRef.current)),
      });
      if (!res.ok) {
        setSaveState('error');
        setDirty(false); // stop the autosave loop; user can retry from the chip
        return;
      }
      // Only clear dirty if nothing changed while the request was in flight.
      if (structuralSignature(nodesRef.current, edgesRef.current) === sigAtStart) setDirty(false);
      setSaveState('saved');
      onSaved();
    } catch {
      setSaveState('error');
      setDirty(false);
    }
  }, [threadId, onSaved]);

  // Debounced autosave: ~1s after the last change. Deferred while the chat
  // turn is streaming (the agent may be writing this thread's state); fires
  // when the turn settles. Also acts as the flush when leaving edit mode,
  // since it keys off `dirty`, not `mode`.
  useEffect(() => {
    if (!dirty || chatStreaming) return;
    const t = setTimeout(() => { void doSave(); }, 1000);
    return () => clearTimeout(t);
  }, [dirty, chatStreaming, nodes, edges, doSave]);

  // ── Mode switching ────────────────────────────────────────────────────────
  const switchMode = useCallback((m: PlannerMode) => {
    if (m === mode) return;
    if (mode === 'analyze') {
      // Roles are ephemeral: reset whenever we leave analyze mode.
      setRoles(new Map());
      setRoleMenuNodeId(null);
    }
    if (m === 'edit') setSaveState('idle');
    setMode(m);
  }, [mode]);

  const handleCancelEdit = useCallback(() => {
    setDirty(false);
    setSaveState('idle');
    setMode('view');
    syncFromDag(); // restore last server DAG
  }, [syncFromDag]);

  // ── Edit actions ──────────────────────────────────────────────────────────
  const onConnect = useCallback((params: Connection) => {
    setEdges(eds => addEdge({ ...params, ...EDGE_DEFAULTS }, eds));
    setDirty(true);
  }, [setEdges]);

  const handleAddNode = useCallback(() => {
    const name = newName.trim();
    if (!name) return;
    const id = `node_${Date.now()}`;
    const s = NODE_STYLE[newType] ?? NODE_STYLE.control;
    setNodes(prev => [...prev, {
      id,
      position: { x: 80 + Math.random() * 260, y: 60 + Math.random() * 200 },
      type: 'mmmNode',
      data: { label: name, nodeType: newType, badge: newType, ...s },
    }]);
    setDirty(true);
    setNewName('');
    nameInputRef.current?.focus();
  }, [newName, newType, setNodes]);

  const handleLoadTemplate = useCallback((templateId: string) => {
    const tpl = DAG_TEMPLATES.find(t => t.id === templateId);
    if (!tpl) return;
    setNodes(tpl.nodes.map(n => {
      const s = NODE_STYLE[n.data.type] ?? NODE_STYLE.control;
      return {
        id: n.id,
        position: n.position,
        type: 'mmmNode',
        data: { label: n.data.variableName, nodeType: n.data.type, badge: n.data.type, ...s },
      };
    }));
    setEdges(tpl.edges.map(e => ({ id: e.id, source: e.source, target: e.target, ...EDGE_DEFAULTS })));
    setDirty(true);
  }, [setNodes, setEdges]);

  // ── Analyze: roles + analysis ─────────────────────────────────────────────
  const assignRole = useCallback((nodeId: string, role: CausalRole) => {
    setRoles(prev => {
      const next = new Map(prev);
      if (role === null) next.delete(nodeId);
      else next.set(nodeId, role);
      return next;
    });
    setRoleMenuNodeId(null);
  }, []);

  const toggleControlled = useCallback((nodeId: string) => {
    setRoles(prev => {
      const next = new Map(prev);
      const current = next.get(nodeId);
      if (current === 'controlled') next.delete(nodeId);
      else if (!current) next.set(nodeId, 'controlled');
      return next;
    });
  }, []);

  const onNodeClick = useCallback((_: React.MouseEvent, node: RFNode) => {
    if (mode === 'analyze') setRoleMenuNodeId(prev => (prev === node.id ? null : node.id));
  }, [mode]);

  const analysis = useMemo(
    () => (mode === 'analyze' ? computeCausalAnalysis(nodes, edges, roles) : null),
    [mode, nodes, edges, roles],
  );

  const displayNodes = useMemo(
    () => (mode === 'analyze' ? styleNodesForAnalysis(nodes, roles, analysis) : nodes),
    [mode, nodes, roles, analysis],
  );
  const displayEdges = useMemo(
    () => (mode === 'analyze' ? styleEdgesForAnalysis(edges, analysis) : edges),
    [mode, edges, analysis],
  );

  const labels = useMemo(() => {
    const out: Record<string, string> = {};
    for (const n of nodes) out[n.id] = String(n.data?.label ?? n.id);
    return out;
  }, [nodes]);

  // ── Lock plan ─────────────────────────────────────────────────────────────
  const handleLockPlan = useCallback(async () => {
    if (!threadId || lockState === 'locking') return;
    setLockState('locking');
    try {
      await apiClient.post('/analysis-plans', {
        thread_id: threadId,
        name: `Analysis Plan — ${new Date().toLocaleDateString()}`,
        dag: buildPutPayload(nodesRef.current, edgesRef.current),
      });
      setLockState('locked');
    } catch (err) {
      console.error('Lock plan failed:', err);
      setLockState('error');
    }
  }, [threadId, lockState]);

  // ── Header chrome ─────────────────────────────────────────────────────────
  const saveChip = (mode === 'edit' || dirty || saveState !== 'idle') && (
    saveState === 'saving' ? (
      <span className="flex items-center gap-1 text-[11px] text-gray-500">
        <span className="w-3 h-3 border-2 border-gray-300 border-t-indigo-500 rounded-full animate-spin" /> Saving…
      </span>
    ) : saveState === 'error' ? (
      <button onClick={() => void doSave()} className="text-[11px] text-red-600 bg-red-50 border border-red-200 rounded px-1.5 py-0.5 hover:bg-red-100" title="Retry save">
        Save failed — retry
      </button>
    ) : dirty ? (
      <span className="text-[11px] text-amber-600">{chatStreaming ? 'Will save after turn…' : 'Unsaved…'}</span>
    ) : saveState === 'saved' ? (
      <span className="text-[11px] text-emerald-600">Saved ✓</span>
    ) : null
  );

  const modeButton = (m: PlannerMode, icon: React.ReactNode, label: string) => (
    <button
      key={m}
      onClick={() => switchMode(m)}
      className={`flex items-center gap-1 px-2 py-1 text-[11px] font-medium transition-colors ${
        mode === m ? 'bg-indigo-600 text-white' : 'bg-white text-gray-500 hover:bg-gray-50'
      }`}
      title={label}
    >
      {icon} {label}
    </button>
  );

  const headerActions = (
    <div className="flex items-center gap-2 shrink-0" onClick={e => e.stopPropagation()}>
      {saveChip}
      {mode === 'edit' && (
        <button onClick={handleCancelEdit} className="flex items-center gap-1 px-2 py-1 rounded-lg border border-gray-200 text-[11px] text-gray-500 hover:bg-gray-50" title="Discard unsaved changes and restore the last saved DAG">
          <X size={11} /> Cancel
        </button>
      )}
      <div className="flex items-center rounded-lg border border-gray-200 overflow-hidden">
        {modeButton('view', <Eye size={11} />, 'View')}
        {modeButton('edit', <Pencil size={11} />, 'Edit')}
        {modeButton('analyze', <Workflow size={11} />, 'Analyze')}
      </div>
      {lockState === 'locked' ? (
        <span className="flex items-center gap-1 px-2 py-1 rounded-lg bg-emerald-50 border border-emerald-200 text-[11px] text-emerald-700 font-medium">
          <CheckCircle2 size={11} /> Plan locked
        </span>
      ) : (
        <button
          onClick={() => void handleLockPlan()}
          disabled={!threadId || nodes.length === 0 || lockState === 'locking'}
          className="flex items-center gap-1 px-2 py-1 rounded-lg border border-gray-200 text-[11px] text-gray-500 hover:bg-gray-50 disabled:opacity-40"
          title="Snapshot the current DAG as a locked analysis plan"
        >
          <Lock size={11} /> {lockState === 'locking' ? 'Locking…' : lockState === 'error' ? 'Lock failed — retry' : 'Lock plan'}
        </button>
      )}
      <button
        onClick={() => setExpanded(true)}
        className="p-1.5 rounded-lg border border-gray-200 text-gray-400 hover:text-gray-700 hover:bg-gray-50"
        title="Expand to fullscreen"
      >
        <Maximize2 size={12} />
      </button>
    </div>
  );

  const title = (
    <span className="flex items-center gap-2">
      Causal Planner
      {mode === 'edit' && <span className="text-[10px] font-semibold uppercase tracking-wider text-violet-600 bg-violet-50 rounded px-1.5 py-0.5">Editing</span>}
      {mode === 'analyze' && <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-600 bg-indigo-50 rounded px-1.5 py-0.5">Analyzing</span>}
    </span>
  );

  // ── Body content (shared between the panel and the fullscreen modal) ─────
  const roleMenuNode = roleMenuNodeId ? nodes.find(n => n.id === roleMenuNodeId) : null;
  const roleMenuRole = roleMenuNodeId ? (roles.get(roleMenuNodeId) ?? null) : null;

  const editToolbar = mode === 'edit' && (
    <div className="mb-2 flex items-center gap-2 p-2 bg-violet-50 rounded-lg border border-violet-100">
      <input
        ref={nameInputRef}
        type="text"
        placeholder="Node name…"
        value={newName}
        onChange={e => setNewName(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && handleAddNode()}
        className="flex-1 min-w-0 text-xs border border-violet-200 rounded px-2 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-violet-400"
      />
      <select
        value={newType}
        onChange={e => setNewType(e.target.value)}
        className="text-xs border border-violet-200 rounded px-2 py-1.5 bg-white focus:outline-none"
      >
        {NODE_TYPES_LIST.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
      </select>
      <button
        onClick={handleAddNode}
        disabled={!newName.trim()}
        className="flex items-center gap-1 px-2.5 py-1.5 bg-violet-600 text-white text-xs rounded-lg hover:bg-violet-700 disabled:opacity-40"
      >
        <Plus size={12} /> Add
      </button>
      <select
        value=""
        onChange={e => { if (e.target.value) handleLoadTemplate(e.target.value); }}
        className="text-xs border border-violet-200 rounded px-2 py-1.5 bg-white focus:outline-none text-gray-600"
        title="Replace the canvas with a starter template"
      >
        <option value="">Templates…</option>
        {DAG_TEMPLATES.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
      </select>
      <span className="text-[10px] text-violet-400 hidden lg:block shrink-0">Del removes selected</span>
    </div>
  );

  const canvas = (height: number | string) => (
    <div style={{ height }} className="relative rounded-xl border border-gray-200 overflow-hidden bg-gray-50/50">
      <ReactFlow
        nodes={displayNodes}
        edges={displayEdges}
        onNodesChange={mode === 'edit' ? handleNodesChange : undefined}
        onEdgesChange={mode === 'edit' ? handleEdgesChange : undefined}
        onConnect={mode === 'edit' ? onConnect : undefined}
        onNodeClick={mode === 'analyze' ? onNodeClick : undefined}
        nodeTypes={PLANNER_NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={mode === 'edit'}
        nodesConnectable={mode === 'edit'}
        elementsSelectable
        deleteKeyCode={mode === 'edit' ? 'Delete' : null}
        minZoom={0.3}
        maxZoom={2}
      >
        <Background gap={20} size={0.8} color="#e5e7eb" />
        <Controls showInteractive={false} />
      </ReactFlow>

      {/* Role-assignment popover (analyze mode) */}
      {mode === 'analyze' && roleMenuNode && (
        <div className="absolute top-2 left-2 z-10 bg-white rounded-xl border border-gray-200 shadow-lg px-3 py-2.5 w-52">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-gray-800 truncate pr-2">{String(roleMenuNode.data?.label ?? roleMenuNode.id)}</span>
            <button onClick={() => setRoleMenuNodeId(null)} className="text-gray-300 hover:text-gray-600 shrink-0"><X size={12} /></button>
          </div>
          <div className="space-y-1">
            {([
              ['treatment', 'Treatment', 'text-blue-700 bg-blue-50 border-blue-200'],
              ['outcome', 'Outcome', 'text-emerald-700 bg-emerald-50 border-emerald-200'],
              ['controlled', 'Controlled', 'text-gray-700 bg-gray-50 border-gray-300'],
            ] as Array<[CausalRole, string, string]>).map(([role, label, cls]) => (
              <button
                key={label}
                onClick={() => assignRole(roleMenuNode.id, roleMenuRole === role ? null : role)}
                className={`w-full text-left text-[11px] px-2 py-1 rounded-lg border transition-colors ${
                  roleMenuRole === role ? `${cls} font-semibold ring-1 ring-indigo-300` : 'text-gray-600 bg-white border-gray-200 hover:bg-gray-50'
                }`}
              >
                {label}{roleMenuRole === role ? ' ✓' : ''}
              </button>
            ))}
            {roleMenuRole && (
              <button
                onClick={() => assignRole(roleMenuNode.id, null)}
                className="w-full text-left text-[11px] px-2 py-1 rounded-lg border border-gray-200 text-gray-400 hover:bg-gray-50"
              >
                Clear role
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );

  const body = (canvasHeight: number | string) => (
    <>
      {editToolbar}
      {canvas(canvasHeight)}

      {mode === 'view' && dag && (
        <SemanticCallouts
          rfNodes={dag.react_flow?.nodes ?? []}
          rfEdges={dag.react_flow?.edges ?? []}
          identification={dag.identification}
        />
      )}

      {mode === 'analyze' && (
        <div className="mt-3 space-y-2">
          <AnalysisSummary analysis={analysis} labels={labels} onToggleControlled={toggleControlled} />
          <AnalysisLegend />
        </div>
      )}
    </>
  );

  // Empty state — no DAG and nothing drawn yet.
  if (!dag && nodes.length === 0 && mode === 'view') {
    return (
      <PanelShell title="Causal Planner" icon={<Network size={16} className="text-violet-600" />} color="violet" right={headerActions}>
        <p className="text-sm text-gray-400 italic">
          No DAG yet. Ask the agent to <code className="text-xs bg-gray-100 px-1 rounded">propose_dag</code> after inspecting the data,
          or switch to <strong>Edit</strong> to build one manually (templates available).
        </p>
      </PanelShell>
    );
  }

  return (
    <>
      <PanelShell title={title} icon={<Network size={16} className="text-violet-600" />} color="violet" right={headerActions}>
        {expanded
          ? <p className="text-sm text-gray-400 italic">Open in fullscreen…</p>
          : body(520)}
      </PanelShell>

      {expanded && (
        <Modal title="Causal Planner" fullWidth onClose={() => setExpanded(false)}>
          <div className="flex items-center justify-end mb-3">{headerActions}</div>
          {body('68vh')}
        </Modal>
      )}
    </>
  );
}
