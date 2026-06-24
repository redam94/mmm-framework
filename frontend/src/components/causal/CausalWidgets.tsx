import { useCallback, useEffect, useMemo, useState } from 'react';
import { titleColorClass } from '../../theme/uiMaps';
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  type Node as RFNode,
  type Edge as RFEdge,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  CheckCircle2, Circle, Clock, XCircle, ChevronDown, ChevronRight,
  History, FileText, Database, Trash2, Network, BookOpen,
} from 'lucide-react';
import { API_BASE_URL, bearerHeader } from '../../api/client';
import { NODE_STYLE, classifyNodes, computeDAGLayout } from './dagDisplay';

// Same origin as the app: relative "/api" in dev (proxied via vite.config.ts),
// or VITE_API_URL when set.
const API_BASE = API_BASE_URL;

// ── Shared types ────────────────────────────────────────────────────────────

// A dynamic JSON value (assumption payloads, spec blobs) whose shape is
// backend-/agent-driven and only known at runtime.
type JsonValue =
  | string | number | boolean | null
  | JsonValue[]
  | { [key: string]: JsonValue };

// The `data` blob on a React Flow DAG node. The agent/backend attach a small
// set of known fields; an index signature keeps it permissive for extras.
interface DagNodeData {
  label?: React.ReactNode;
  nodeType?: string;
  variableName?: string;
  type?: string;
  badge?: string;
  [key: string]: unknown;
}

// The `data` blob on a React Flow DAG edge.
interface DagEdgeData {
  edgeType?: string;
  [key: string]: unknown;
}

export interface WorkflowStep {
  step: number;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'done' | 'skipped';
  inferred_status: string;
  notes: string | null;
  overridden: boolean;
  updated_at: number | null;
}

export interface Assumption {
  id: string;
  thread_id: string;
  key: string;
  category: string;
  value: JsonValue;
  rationale: string;
  change_note: string | null;
  version: number;
  is_tombstone: boolean;
  created_at: number;
}

export interface DataFile {
  id: string;
  thread_id: string;
  path: string;
  name: string;
  kind: string;
  size_bytes: number | null;
  preview: string | null;
  meta: Record<string, unknown>;
  created_at: number;
}

export interface DagPayload {
  spec: JsonValue;
  react_flow: {
    nodes: Array<{ id: string; position: { x: number; y: number }; data: DagNodeData; type?: string }>;
    edges: Array<{ id: string; source: string; target: string; data?: DagEdgeData }>;
  };
  validation: { valid: boolean; errors: string[]; warnings: string[] };
  identification?: {
    treatment: string; outcome: string; adjustment_set: string[];
    identifiable: boolean; notes: string[]; descendants_of_treatment: string[];
    backdoor_paths: Array<{ path: string; blocked_by: string[] }>;
    open_paths_remaining: Array<{ path: string }>;
  };
}

// ── Small reusable shell ────────────────────────────────────────────────────

export function PanelShell({ title, icon, color = 'gray', defaultOpen = true, children, right }: {
  title: React.ReactNode; icon: React.ReactNode; color?: string; defaultOpen?: boolean;
  children: React.ReactNode; right?: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-white rounded-2xl border border-line-200 shadow-sm overflow-hidden">
      <div className="flex items-center gap-3 px-5 py-4">
        <button onClick={() => setOpen(v => !v)} className="flex items-center gap-3 flex-1 text-left">
          {icon}
          <span className={`font-semibold text-sm flex-1 ${titleColorClass(color)}`}>{title}</span>
          {open ? <ChevronDown size={15} className="text-ink-300" /> : <ChevronRight size={15} className="text-ink-300" />}
        </button>
        {right}
      </div>
      {open && <div className="px-5 pb-5">{children}</div>}
    </div>
  );
}

// ── 1. Workflow Checklist ───────────────────────────────────────────────────

const STEP_ICON: Record<string, React.ReactNode> = {
  pending: <Circle size={16} className="text-ink-300" />,
  in_progress: <Clock size={16} className="text-amber-500 animate-pulse" />,
  done: <CheckCircle2 size={16} className="text-emerald-500" />,
  skipped: <XCircle size={16} className="text-ink-300" />,
};

export function WorkflowChecklist({ steps, onOverride }: {
  steps: WorkflowStep[];
  onOverride: (step: number, status: WorkflowStep['status'], notes?: string) => void;
}) {
  if (steps.length === 0) return null;
  const doneCount = steps.filter(s => s.status === 'done').length;

  return (
    <PanelShell
      title={`Scientific Workflow (${doneCount}/9)`}
      icon={<BookOpen size={16} className="text-indigo-600" />}
      color="indigo"
    >
      <div className="space-y-1.5">
        {steps.map(s => (
          <div
            key={s.step}
            className={`flex items-start gap-3 px-3 py-2 rounded-lg border transition-colors ${
              s.status === 'done' ? 'border-emerald-100 bg-emerald-50/40' :
              s.status === 'in_progress' ? 'border-amber-100 bg-amber-50/40' :
              'border-line-200 bg-white'
            }`}
          >
            <div className="mt-0.5 shrink-0">{STEP_ICON[s.status] ?? STEP_ICON.pending}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="text-xs font-mono text-ink-300">{s.step}</span>
                <span className="text-sm font-semibold text-ink-900">{s.title}</span>
                {s.overridden && (
                  <span className="text-[10px] uppercase tracking-wider text-indigo-500 bg-indigo-50 rounded px-1.5 py-0.5">manual</span>
                )}
              </div>
              <p className="text-xs text-ink-400 mt-0.5">{s.description}</p>
              {s.notes && <p className="text-xs text-ink-600 mt-1 italic">"{s.notes}"</p>}
            </div>
            <div className="shrink-0">
              <select
                value={s.status}
                onChange={e => onOverride(s.step, e.target.value as WorkflowStep['status'])}
                className="text-[11px] bg-transparent border border-line-200 rounded px-1.5 py-1 text-ink-400 hover:border-line-300 focus:outline-none"
                title={`Inferred: ${s.inferred_status}`}
              >
                <option value="pending">pending</option>
                <option value="in_progress">in progress</option>
                <option value="done">done</option>
                <option value="skipped">skipped</option>
              </select>
            </div>
          </div>
        ))}
      </div>
    </PanelShell>
  );
}

// ── 2. Assumptions Log ──────────────────────────────────────────────────────

const CATEGORY_COLOR: Record<string, string> = {
  research_question: 'bg-blue-50 text-blue-700 border-blue-200',
  causal_structure:  'bg-purple-50 text-purple-700 border-purple-200',
  identification:    'bg-indigo-50 text-indigo-700 border-indigo-200',
  data:              'bg-amber-50 text-amber-700 border-amber-200',
  functional_form:   'bg-orange-50 text-orange-700 border-orange-200',
  prior:             'bg-pink-50 text-pink-700 border-pink-200',
  external_evidence: 'bg-teal-50 text-teal-700 border-teal-200',
  other:             'bg-cream-50 text-ink-700 border-line-200',
};

export function AssumptionsLog({ threadId, assumptions, onRefresh }: {
  threadId: string | null;
  assumptions: Assumption[];
  onRefresh: () => void;
}) {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [historyByKey, setHistoryByKey] = useState<Record<string, Assumption[]>>({});

  const loadHistory = useCallback(async (key: string) => {
    if (!threadId) return;
    if (historyByKey[key]) {
      setExpandedKey(expandedKey === key ? null : key);
      return;
    }
    try {
      const hist: Assumption[] = await fetch(`${API_BASE}/assumption_history/${threadId}/${encodeURIComponent(key)}`, { headers: bearerHeader() }).then(r => r.json());
      setHistoryByKey(prev => ({ ...prev, [key]: hist }));
      setExpandedKey(key);
    } catch (e) { console.error(e); }
  }, [threadId, historyByKey, expandedKey]);

  const retract = async (key: string) => {
    if (!threadId) return;
    const reason = prompt(`Retract assumption "${key}"? Enter reason for the change log:`);
    if (!reason) return;
    await fetch(`${API_BASE}/assumption/${threadId}/${encodeURIComponent(key)}`, {
      method: 'DELETE', headers: { 'Content-Type': 'application/json', ...bearerHeader() },
      body: JSON.stringify({ reason }),
    });
    setHistoryByKey(prev => { const c = { ...prev }; delete c[key]; return c; });
    onRefresh();
  };

  if (assumptions.length === 0) {
    return (
      <PanelShell title="Modeling Assumptions" icon={<FileText size={16} className="text-indigo-600" />} color="indigo">
        <p className="text-sm text-ink-300 italic">No assumptions recorded yet. The agent will log them as you make modeling choices.</p>
      </PanelShell>
    );
  }

  // Group by category
  const byCat: Record<string, Assumption[]> = {};
  for (const a of assumptions) (byCat[a.category] ??= []).push(a);

  return (
    <PanelShell
      title={`Modeling Assumptions (${assumptions.length})`}
      icon={<FileText size={16} className="text-indigo-600" />}
      color="indigo"
    >
      <div className="space-y-4">
        {Object.entries(byCat).sort().map(([cat, items]) => (
          <div key={cat}>
            <div className="flex items-center gap-2 mb-1.5">
              <span className={`text-[10px] uppercase tracking-wider font-semibold rounded border px-1.5 py-0.5 ${CATEGORY_COLOR[cat] ?? CATEGORY_COLOR.other}`}>
                {cat.replace(/_/g, ' ')}
              </span>
              <span className="text-[11px] text-ink-300">{items.length}</span>
            </div>
            <div className="space-y-1.5">
              {items.map(a => (
                <div key={a.key} className="rounded-lg border border-line-200 bg-cream-50 overflow-hidden">
                  <div className="flex items-start gap-2 px-3 py-2">
                    <button
                      onClick={() => loadHistory(a.key)}
                      className="p-0.5 rounded text-ink-300 hover:text-indigo-600 shrink-0 mt-0.5"
                      title="Toggle history"
                    >
                      <History size={13} />
                    </button>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2 flex-wrap">
                        <code className="text-xs font-semibold text-ink-900 break-all">{a.key}</code>
                        <span className="text-[10px] text-ink-300">v{a.version}</span>
                        <span className="text-[10px] text-ink-300">{new Date(a.created_at * 1000).toLocaleString()}</span>
                      </div>
                      <p className="text-xs text-ink-600 mt-0.5 italic">{a.rationale}</p>
                      {a.change_note && (
                        <p className="text-[11px] text-indigo-600 mt-0.5">↳ {a.change_note}</p>
                      )}
                      <details className="mt-1">
                        <summary className="text-[11px] text-ink-300 cursor-pointer select-none">value</summary>
                        <pre className="text-[10px] font-mono text-ink-700 bg-white border border-line-200 rounded px-2 py-1 mt-1 overflow-x-auto max-h-32">{JSON.stringify(a.value, null, 2)}</pre>
                      </details>
                    </div>
                    <button
                      onClick={() => retract(a.key)}
                      className="p-1 rounded text-ink-300 hover:text-red-500 shrink-0"
                      title="Retract"
                    ><Trash2 size={12} /></button>
                  </div>
                  {expandedKey === a.key && historyByKey[a.key] && (
                    <div className="border-t border-line-200 bg-white px-3 py-2 space-y-1.5">
                      <p className="text-[10px] uppercase tracking-wider text-ink-300 font-semibold">Change log ({historyByKey[a.key].length})</p>
                      {historyByKey[a.key].map(h => (
                        <div key={h.id} className="text-[11px] flex items-baseline gap-2">
                          <span className="text-ink-300 shrink-0">v{h.version}</span>
                          <span className="text-ink-300 shrink-0">{new Date(h.created_at * 1000).toLocaleTimeString()}</span>
                          <span className={`${h.is_tombstone ? 'text-red-500 italic' : 'text-ink-700'}`}>
                            {h.is_tombstone ? `retracted (${h.change_note})` : (h.change_note || 'recorded')}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </PanelShell>
  );
}

// ── 3. Files Explorer ──────────────────────────────────────────────────────

export function DataFilesWidget({ files, onDelete }: {
  files: DataFile[]; onDelete: (id: string) => void;
}) {
  if (files.length === 0) {
    return (
      <PanelShell title="Session Files" icon={<Database size={16} className="text-amber-600" />} color="amber">
        <p className="text-sm text-ink-300 italic">No files associated with this session. Upload a CSV via the paperclip icon, or the agent can generate synthetic data.</p>
      </PanelShell>
    );
  }

  const fmtSize = (n: number | null) => {
    if (n == null) return '';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <PanelShell title={`Session Files (${files.length})`} icon={<Database size={16} className="text-amber-600" />} color="amber">
      <div className="space-y-2">
        {files.map(f => (
          <div key={f.id} className="rounded-lg border border-line-200 bg-white overflow-hidden">
            <div className="flex items-start gap-2 px-3 py-2">
              <Database size={14} className="text-amber-600 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="text-sm font-semibold text-ink-900 truncate">{f.name}</span>
                  <span className="text-[10px] uppercase tracking-wider text-amber-700 bg-amber-50 rounded px-1.5 py-0.5 border border-amber-200">{f.kind}</span>
                  <span className="text-[10px] text-ink-300">{fmtSize(f.size_bytes)}</span>
                </div>
                <p className="text-[11px] text-ink-300 font-mono mt-0.5 truncate">{f.path}</p>
                {f.preview && (
                  <details className="mt-1">
                    <summary className="text-[11px] text-ink-300 cursor-pointer select-none">preview</summary>
                    <pre className="text-[10px] font-mono text-ink-700 bg-cream-50 border border-line-200 rounded px-2 py-1 mt-1 overflow-x-auto whitespace-pre max-h-32">{f.preview}</pre>
                  </details>
                )}
              </div>
              <button
                onClick={() => onDelete(f.id)}
                className="p-1 rounded text-ink-300 hover:text-red-500 shrink-0"
                title="Remove from session (file on disk is kept)"
              ><Trash2 size={12} /></button>
            </div>
          </div>
        ))}
      </div>
    </PanelShell>
  );
}

// ── 4. DAG Viewer ──────────────────────────────────────────────────────────


const hStyle = (color: string, visible: boolean) => ({
  background: color, width: 8, height: 8, border: 'none',
  ...(visible ? {} : { opacity: 0, pointerEvents: 'none' as const }),
});

export function MMMNode({ data }: { data: DagNodeData }) {
  const s = NODE_STYLE[data.nodeType ?? ''] ?? NODE_STYLE.control;
  return (
    <div style={{
      background: s.bg, border: `2px solid ${s.border}`, borderRadius: 12,
      padding: '9px 16px', minWidth: 130, textAlign: 'center',
      boxShadow: '0 2px 10px rgba(0,0,0,0.08)', userSelect: 'none',
    }}>
      <Handle type="target" position={Position.Left}   id="tgt-left"   style={hStyle(s.border, true)} />
      <Handle type="target" position={Position.Bottom} id="tgt-bottom" style={hStyle(s.border, false)} />
      <div style={{ fontWeight: 700, fontSize: 13, color: s.text, lineHeight: 1.3 }}>{data.label}</div>
      <div style={{
        fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em',
        color: s.border, background: 'white', border: `1px solid ${s.border}55`,
        borderRadius: 20, padding: '1px 8px', display: 'inline-block', marginTop: 5,
      }}>
        {data.badge}
      </div>
      <Handle type="source" position={Position.Right} id="src-right" style={hStyle(s.border, true)} />
      <Handle type="source" position={Position.Top}   id="src-top"   style={hStyle(s.border, false)} />
    </div>
  );
}

const MMM_NODE_TYPES = { mmmNode: MMMNode };

export function SemanticCallouts({
  rfNodes, rfEdges, identification,
}: {
  rfNodes: Array<{ id: string; data: DagNodeData }>;
  rfEdges: Array<{ source: string; target: string }>;
  identification?: DagPayload['identification'];
}) {
  const { isConf, mediaSet } = useMemo(
    () => classifyNodes(rfNodes, rfEdges),
    [rfNodes, rfEdges],
  );

  const inEdgeMap = useMemo(() => {
    const m = new Map<string, string[]>();
    rfNodes.forEach(n => m.set(n.id, []));
    rfEdges.forEach(e => m.get(e.target)?.push(e.source));
    return m;
  }, [rfNodes, rfEdges]);

  const confounders = rfNodes.filter(n => isConf(n));
  const mediators   = rfNodes.filter(n => n.data?.type === 'mediator');

  const colliders = rfNodes.filter(n => {
    if (['kpi', 'media', 'control'].includes(n.data?.type ?? '')) return false;
    const parents = inEdgeMap.get(n.id) ?? [];
    return parents.length >= 2 &&
      parents.some(p => mediaSet.has(p)) &&
      parents.some(p => !mediaSet.has(p));
  });

  const adjustSet = identification?.adjustment_set ?? [];

  return (
    <div className="mt-3 space-y-2">
      {confounders.length > 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50/60 px-4 py-3">
          <p className="text-[11px] font-bold text-amber-800 uppercase tracking-wider mb-2">
            ✓ Include in adjustment set — confounders
          </p>
          <div className="flex flex-wrap gap-1.5 mb-2">
            {confounders.map(n => (
              <span key={n.id} className="text-xs font-semibold text-amber-900 bg-white border border-amber-300 rounded-full px-2.5 py-0.5">
                {n.data?.variableName ?? n.id}
              </span>
            ))}
          </div>
          <p className="text-[11px] text-amber-700 leading-relaxed">
            These affect both media spend decisions and the KPI. Conditioning on them blocks backdoor paths — omitting them biases all channel ROI estimates.
          </p>
        </div>
      )}

      {mediators.length > 0 && (
        <div className="rounded-xl border border-pink-200 bg-pink-50/60 px-4 py-3">
          <p className="text-[11px] font-bold text-pink-800 uppercase tracking-wider mb-2">
            ⊘ Do not control for — mediators
          </p>
          <div className="flex flex-wrap gap-1.5 mb-2">
            {mediators.map(n => (
              <span key={n.id} className="text-xs font-semibold text-pink-900 bg-white border border-pink-300 rounded-full px-2.5 py-0.5">
                {n.data?.variableName ?? n.id}
              </span>
            ))}
          </div>
          <p className="text-[11px] text-pink-700 leading-relaxed">
            Controlling for a mediator blocks the causal path and will underestimate the total media effect. Use a structural mediation model to decompose direct vs indirect effects.
          </p>
        </div>
      )}

      {colliders.length > 0 && (
        <div className="rounded-xl border border-red-200 bg-red-50/60 px-4 py-3">
          <p className="text-[11px] font-bold text-red-800 uppercase tracking-wider mb-2">
            ⚠ Collider — do not condition on
          </p>
          <div className="flex flex-wrap gap-1.5 mb-2">
            {colliders.map(n => (
              <span key={n.id} className="text-xs font-semibold text-red-900 bg-white border border-red-300 rounded-full px-2.5 py-0.5">
                {n.data?.variableName ?? n.id}
              </span>
            ))}
          </div>
          <p className="text-[11px] text-red-700 leading-relaxed">
            A collider is caused by two variables on a path. Conditioning on it opens a spurious association between its parents — including it in the model would introduce bias.
          </p>
        </div>
      )}

      {identification && (
        <div className={`rounded-xl border px-4 py-3 ${
          identification.identifiable
            ? 'border-emerald-200 bg-emerald-50/60'
            : 'border-red-200 bg-red-50/60'
        }`}>
          <p className={`text-[11px] font-bold uppercase tracking-wider mb-1.5 ${
            identification.identifiable ? 'text-emerald-800' : 'text-red-800'
          }`}>
            {identification.identifiable ? '✓ Effect identified' : '✗ Not identified'}
            {' '}— {identification.treatment} → {identification.outcome}
          </p>
          {adjustSet.length > 0 && (
            <p className="text-[11px] text-ink-700 mb-1.5">
              Adjustment set: <span className="font-mono">{`{ ${adjustSet.join(', ')} }`}</span>
            </p>
          )}
          {identification.backdoor_paths.length > 0 && (
            <details className="mt-1">
              <summary className="text-[11px] text-ink-400 cursor-pointer select-none hover:text-ink-700">
                {identification.backdoor_paths.length} backdoor path{identification.backdoor_paths.length !== 1 ? 's' : ''}
              </summary>
              <div className="mt-1.5 pl-2 space-y-1">
                {identification.backdoor_paths.map((p, i) => (
                  <div key={i} className="text-[11px] font-mono text-ink-700">
                    {p.path}
                    {p.blocked_by.length > 0
                      ? <span className="text-emerald-600"> ✓ blocked by {p.blocked_by.join(', ')}</span>
                      : <span className="text-red-600"> ✗ OPEN</span>}
                  </div>
                ))}
              </div>
            </details>
          )}
          {identification.notes.length > 0 && (
            <div className="mt-1.5 space-y-0.5">
              {identification.notes.map((n, i) => (
                <p key={i} className="text-[11px] text-ink-600 italic">{n}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function DAGViewer({ dag }: { dag: DagPayload | null }) {
  const rfNodes = useMemo(() => dag?.react_flow?.nodes ?? [], [dag]);
  const rfEdges = useMemo(() => dag?.react_flow?.edges ?? [], [dag]);

  const posMap = useMemo(() => computeDAGLayout(rfNodes, rfEdges), [rfNodes, rfEdges]);

  const { isConf, isBiz } = useMemo(
    () => classifyNodes(rfNodes, rfEdges),
    [rfNodes, rfEdges],
  );

  const nodes: RFNode[] = useMemo(() => rfNodes.map(n => {
    const nt = n.data?.type ?? 'control';
    const conf = isConf(n);
    const biz  = isBiz(n);
    return {
      id: n.id,
      position: posMap[n.id] ?? { x: 0, y: 0 },
      type: 'mmmNode',
      data: {
        label:    n.data?.variableName ?? n.id,
        nodeType: nt,
        badge:    conf ? 'confounder' : biz ? 'control' : nt,
        ...(NODE_STYLE[nt] ?? NODE_STYLE.control),
      },
    };
  }), [rfNodes, posMap, isConf, isBiz]);

  const edges: RFEdge[] = useMemo(() => rfEdges.map(e => {
    const srcNode = rfNodes.find(n => n.id === e.source);
    const et = e.data?.edgeType ?? 'direct';
    const color = et === 'mediated' ? '#db2777' : et === 'crossEffect' ? '#6366f1' : '#64748b';
    const bizSrc = srcNode ? isBiz(srcNode) : false;
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      type: 'default',
      sourceHandle: bizSrc ? 'src-top' : 'src-right',
      targetHandle: bizSrc ? 'tgt-bottom' : 'tgt-left',
      pathOptions: { curvature: 0.45 },
      style: {
        stroke: color, strokeWidth: 1.8,
        strokeDasharray: et === 'crossEffect' ? '5 3' : undefined,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
    };
  }), [rfEdges, rfNodes, isBiz]);

  if (!dag) {
    return (
      <PanelShell title="Causal DAG" icon={<Network size={16} className="text-violet-600" />} color="violet">
        <p className="text-sm text-ink-300 italic">
          No DAG yet. Ask the agent to <code className="text-xs bg-cream-100 px-1 rounded">propose_dag</code> after inspecting the data.
        </p>
      </PanelShell>
    );
  }

  return (
    <PanelShell title="Causal DAG" icon={<Network size={16} className="text-violet-600" />} color="violet">
      <div style={{ height: 400 }} className="rounded-xl border border-line-200 overflow-hidden bg-cream-50/50">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={MMM_NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.18 }}
          proOptions={{ hideAttribution: true }}
          nodesDraggable
          nodesConnectable={false}
          elementsSelectable
          minZoom={0.3}
          maxZoom={2}
        >
          <Background gap={20} size={0.8} color="#e5e7eb" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <SemanticCallouts rfNodes={rfNodes} rfEdges={rfEdges} identification={dag.identification} />
    </PanelShell>
  );
}

// ── Data-loading hook used by AgentPage ─────────────────────────────────────

// eslint-disable-next-line react-refresh/only-export-components -- shared data-loading hook imported by Agent page + WorkspaceTabs; not a component, kept here with the panel components it feeds.
export function useCausalPanels(threadId: string | null) {
  const [workflow, setWorkflow] = useState<WorkflowStep[]>([]);
  const [assumptions, setAssumptions] = useState<Assumption[]>([]);
  const [files, setFiles] = useState<DataFile[]>([]);
  const [dag, setDag] = useState<DagPayload | null>(null);

  const refresh = useCallback(async () => {
    if (!threadId) {
      setWorkflow([]); setAssumptions([]); setFiles([]); setDag(null);
      return;
    }
    try {
      const [w, a, f, d] = await Promise.all([
        fetch(`${API_BASE}/workflow/${threadId}`, { headers: bearerHeader() }).then(r => r.json()),
        fetch(`${API_BASE}/assumptions/${threadId}`, { headers: bearerHeader() }).then(r => r.json()),
        fetch(`${API_BASE}/files/${threadId}`, { headers: bearerHeader() }).then(r => r.json()),
        fetch(`${API_BASE}/dag/${threadId}`, { headers: bearerHeader() }).then(r => r.json()),
      ]);
      setWorkflow(Array.isArray(w) ? w : []);
      setAssumptions(Array.isArray(a) ? a : []);
      setFiles(Array.isArray(f) ? f : []);
      // /dag returns the dag payload directly (not {dag: ...}) when present; null when absent
      setDag(d && d.spec ? d : (d?.dag ?? null));
    } catch (e) { console.error('Causal panels refresh failed', e); }
  }, [threadId]);

  // eslint-disable-next-line react-hooks/set-state-in-effect -- refresh() fetches panel data over the network and sets it on mount / when threadId changes; state is not derivable during render and an event handler would not fire on thread switch.
  useEffect(() => { refresh(); }, [refresh]);

  const overrideWorkflow = useCallback(async (step: number, status: WorkflowStep['status'], notes?: string) => {
    if (!threadId) return;
    await fetch(`${API_BASE}/workflow/${threadId}/${step}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json', ...bearerHeader() },
      body: JSON.stringify({ status, notes: notes ?? null }),
    });
    refresh();
  }, [threadId, refresh]);

  const deleteFile = useCallback(async (id: string) => {
    await fetch(`${API_BASE}/files/${id}`, { method: 'DELETE', headers: bearerHeader() });
    refresh();
  }, [refresh]);

  return { workflow, assumptions, files, dag, refresh, overrideWorkflow, deleteFile };
}
