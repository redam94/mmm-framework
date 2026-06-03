import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  Send, Bot, User, Loader2, Paperclip, ChevronDown, ChevronRight,
  Wrench, CheckCircle2, Maximize2, Minimize2, X, Download, ExternalLink,
  TrendingUp, Calendar, Layers, Zap, BarChart2, Activity, Pencil, Check, RotateCcw, Plus, Trash2,
  Square, ArrowLeft, Play, Copy, MessagesSquare, FileCode,
  BookOpen, Network, Database, BrainCircuit, Search, UploadCloud, FolderOpen, FileText, File as FileIcon,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import { useAuthStore } from '../stores/authStore';
import { ModelSwitcher } from '../components/common';
import { API_BASE_URL } from '../api/client';
import {
  WorkflowChecklist, AssumptionsLog, DataFilesWidget, EditableDAGViewer, useCausalPanels,
} from '../components/causal/CausalWidgets';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ToolCall {
  id: string;
  name: string;
  args?: Record<string, any>;
  result?: string;
  status: 'running' | 'done' | 'error';
}

interface ChatMessage {
  id: string;
  type: 'human' | 'ai' | 'error';
  content: string;
  toolCalls?: ToolCall[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function normalizeContent(content: any): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map(b => (typeof b === 'string' ? b : b?.text ?? ''))
      .filter(Boolean)
      .join('\n');
  }
  if (content && typeof content === 'object') return JSON.stringify(content);
  return String(content ?? '');
}

function formatToolName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function truncate(str: string, n = 300): string {
  return str.length > n ? str.slice(0, n) + '…' : str;
}

// ─── Plotly light-mode layout helper ─────────────────────────────────────────

function applyLightModeLayout(rawLayout: any): any {
  const layout = { ...(rawLayout || {}) };

  layout.paper_bgcolor = 'rgba(0,0,0,0)';
  layout.plot_bgcolor = '#f9fafb';
  layout.font = { family: 'Inter, system-ui, sans-serif', size: 12, ...(layout.font || {}), color: '#1f2937' };

  if (layout.title) {
    layout.title = typeof layout.title === 'string'
      ? { text: layout.title, font: { color: '#111827', size: 15 } }
      : { ...layout.title, font: { size: 15, ...(layout.title.font || {}), color: '#111827' } };
  }

  const axisBase = {
    automargin: true,
    gridcolor: '#f3f4f6',
    linecolor: '#e5e7eb',
    zerolinecolor: '#e5e7eb',
    zerolinewidth: 1,
    tickfont: { color: '#374151', size: 11 },
    titlefont: { color: '#4b5563', size: 12 },
  };

  Object.keys(layout).forEach(key => {
    if (/^[xy]axis\d*$/.test(key)) {
      const existing = layout[key] || {};
      layout[key] = {
        ...axisBase,
        ...existing,
        automargin: true,
        gridcolor: '#f3f4f6',
        linecolor: '#e5e7eb',
        zerolinecolor: '#e5e7eb',
        tickfont: { color: '#374151', size: 11, ...(existing.tickfont || {}) },
        titlefont: { color: '#4b5563', size: 12, ...(existing.titlefont || {}) },
      };
    }
  });

  if (!layout.xaxis) layout.xaxis = { ...axisBase };
  if (!layout.yaxis) layout.yaxis = { ...axisBase };

  layout.legend = {
    bgcolor: 'rgba(255,255,255,0.95)',
    bordercolor: '#e5e7eb',
    borderwidth: 1,
    ...(layout.legend || {}),
    font: { color: '#374151', size: 11 },
  };

  layout.hoverlabel = {
    bgcolor: '#ffffff',
    bordercolor: '#d1d5db',
    font: { color: '#1f2937', size: 12 },
    ...(layout.hoverlabel || {}),
  };

  if (Array.isArray(layout.annotations)) {
    layout.annotations = layout.annotations.map((a: any) => ({
      ...a,
      font: { size: 11, ...(a.font || {}), color: a.font?.color && !isLightOnWhite(a.font.color) ? a.font.color : '#374151' },
    }));
  }

  if (!layout.colorway) {
    layout.colorway = ['#4f46e5', '#0d9488', '#f59e0b', '#e11d48', '#059669', '#7c3aed', '#0284c7', '#b45309', '#6366f1', '#0f766e'];
  }

  layout.margin = { l: 70, r: 40, t: 90, b: 80, ...(layout.margin || {}) };

  return layout;
}

function isLightOnWhite(hex: string): boolean {
  const clean = hex.replace('#', '');
  if (clean.length !== 6) return false;
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  // relative luminance; light colours are bad on white
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.7;
}

// ─── ToolCallBlock ────────────────────────────────────────────────────────────

// Mapping: tool name → tab it produced an artifact in. Only tools that mutate
// some visible workspace state appear here; pure reads (list_*, get_session_status)
// don't get a navigation link.
const TOOL_TO_TAB: Record<string, { tab: string; label: string }> = {
  // Workflow
  mark_workflow_step:               { tab: 'workflow',  label: 'Workflow' },
  // Causal (DAG + identification + assumptions log)
  propose_dag:                      { tab: 'causal',    label: 'Causal' },
  validate_causal_identification:   { tab: 'causal',    label: 'Causal' },
  define_research_question:         { tab: 'causal',    label: 'Causal' },
  record_assumption:                { tab: 'causal',    label: 'Causal' },
  define_analysis_plan:             { tab: 'causal',    label: 'Causal' },
  prior_predictive_check:           { tab: 'causal',    label: 'Causal' },
  leave_one_out_decomposition:      { tab: 'causal',    label: 'Causal' },
  // Data
  generate_synthetic_data:          { tab: 'data',      label: 'Data' },
  inspect_dataset:                  { tab: 'data',      label: 'Data' },
  // Model
  configure_model:                  { tab: 'model',     label: 'Model' },
  update_model_setting:             { tab: 'model',     label: 'Model' },
  load_config:                      { tab: 'model',     label: 'Model' },
  // Results (fit + analysis)
  fit_mmm_model:                    { tab: 'results',   label: 'Results' },
  load_fitted_model:                { tab: 'results',   label: 'Results' },
  get_roi_metrics:                  { tab: 'results',   label: 'Results' },
  get_component_decomposition:      { tab: 'results',   label: 'Results' },
  get_model_diagnostics:            { tab: 'results',   label: 'Results' },
  get_adstock_weights:              { tab: 'results',   label: 'Results' },
  get_saturation_curves:            { tab: 'results',   label: 'Results' },
  // Artifacts (code snippets, plots, REPL output)
  execute_python:                   { tab: 'plots',     label: 'Plots' },
  // Reporting
  generate_project_report:          { tab: 'artifacts', label: 'Artifacts' },
};

function ToolCallBlock({ toolCall, onNavigate }: { toolCall: ToolCall; onNavigate?: (tab: string) => void }) {
  const [expanded, setExpanded] = useState(false);

  const cls = toolCall.status === 'done'
    ? 'text-emerald-700 border-emerald-200 bg-emerald-50'
    : toolCall.status === 'error'
    ? 'text-red-700 border-red-200 bg-red-50'
    : 'text-amber-700 border-amber-200 bg-amber-50';

  const Icon = toolCall.status === 'done'
    ? <CheckCircle2 size={13} className="text-emerald-600 shrink-0" />
    : toolCall.status === 'error'
    ? <X size={13} className="text-red-600 shrink-0" />
    : <Loader2 size={13} className="text-amber-600 animate-spin shrink-0" />;

  const target = TOOL_TO_TAB[toolCall.name];
  const showJump = toolCall.status === 'done' && target && onNavigate;

  return (
    <div className={`my-2 rounded-xl border text-xs font-mono overflow-hidden ${cls}`}>
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-black/5 transition-colors text-left"
      >
        <Wrench size={13} className="shrink-0 opacity-60" />
        <span className="font-semibold tracking-wide flex-1">{formatToolName(toolCall.name)}</span>
        {Icon}
        {expanded ? <ChevronDown size={13} className="shrink-0 opacity-50" /> : <ChevronRight size={13} className="shrink-0 opacity-50" />}
      </button>
      {expanded && (
        <div className="border-t border-black/10 divide-y divide-black/10">
          {toolCall.args && Object.keys(toolCall.args).length > 0 && (
            <div className="px-3 py-2">
              <p className="text-[10px] uppercase tracking-widest text-gray-400 mb-1">Input</p>
              <pre className="text-gray-700 whitespace-pre-wrap break-all text-[11px] leading-relaxed">{JSON.stringify(toolCall.args, null, 2)}</pre>
            </div>
          )}
          {toolCall.result && (
            <div className="px-3 py-2">
              <p className="text-[10px] uppercase tracking-widest text-gray-400 mb-1">Output</p>
              <pre className="text-gray-700 whitespace-pre-wrap break-all text-[11px] leading-relaxed max-h-64 overflow-y-auto">{toolCall.result}</pre>
            </div>
          )}
        </div>
      )}
      {!expanded && toolCall.result && (
        <div className="px-3 pb-2 text-gray-400 text-[11px] truncate">{truncate(toolCall.result, 120)}</div>
      )}
      {showJump && (
        <button
          onClick={(e) => { e.stopPropagation(); onNavigate!(target.tab); }}
          className="w-full flex items-center justify-end gap-1.5 px-3 py-1.5 border-t border-black/10 text-[11px] font-sans text-emerald-700 hover:bg-emerald-100/60 transition-colors"
          title={`Show this artifact in the ${target.label} tab`}
        >
          <ExternalLink size={11} />
          <span className="font-semibold">View in {target.label}</span>
          <ChevronRight size={11} />
        </button>
      )}
    </div>
  );
}

// ─── Modal ────────────────────────────────────────────────────────────────────

function Modal({ title, onClose, fullWidth = false, children }: {
  title: string; onClose: () => void; fullWidth?: boolean; children: React.ReactNode;
}) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/40 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className={`relative flex flex-col bg-white border border-gray-200 rounded-2xl shadow-2xl overflow-hidden ${fullWidth ? 'w-full h-full max-w-none' : 'w-full max-w-4xl max-h-[90vh]'}`}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 shrink-0">
          <h2 className="text-lg font-bold text-gray-900 truncate pr-4">{title}</h2>
          <button onClick={onClose} className="p-2 rounded-full hover:bg-gray-100 text-gray-400 hover:text-gray-700 transition-colors shrink-0" title="Close (Esc)">
            <X size={18} />
          </button>
        </div>
        <div className="overflow-y-auto flex-1 p-6">{children}</div>
      </div>
    </div>
  );
}

// ─── DashWidget ───────────────────────────────────────────────────────────────

function DashWidget({
  title, icon, color = 'indigo', dotColor = 'bg-indigo-500',
  defaultOpen = true, expandTitle, expandContent, children,
}: {
  title: string; icon?: React.ReactNode; color?: string; dotColor?: string;
  defaultOpen?: boolean; expandTitle?: string; expandContent?: React.ReactNode; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [modal, setModal] = useState(false);

  return (
    <>
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm hover:shadow-md transition-all overflow-hidden">
        <div className="flex items-center gap-3 px-5 py-4">
          <button onClick={() => setOpen(v => !v)} className="flex items-center gap-3 flex-1 text-left">
            {icon || <span className={`w-2 h-2 rounded-full shrink-0 ${dotColor}`} />}
            <span className={`font-semibold text-sm text-${color}-600 flex-1`}>{title}</span>
            {open ? <ChevronDown size={15} className="text-gray-400 shrink-0" /> : <ChevronRight size={15} className="text-gray-400 shrink-0" />}
          </button>
          <button onClick={() => setModal(true)} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-700 transition-colors shrink-0" title="Expand">
            <Maximize2 size={14} />
          </button>
        </div>
        {open && <div className="px-5 pb-5">{children}</div>}
      </div>
      {modal && <Modal title={expandTitle || title} onClose={() => setModal(false)}>{expandContent || children}</Modal>}
    </>
  );
}

// ─── MD_COMPONENTS ────────────────────────────────────────────────────────────

const MD_COMPONENTS: any = {
  table: ({ children }: any) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-sm border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }: any) => <thead className="bg-gray-100">{children}</thead>,
  th: ({ children }: any) => <th className="px-3 py-2 text-left font-semibold text-indigo-600 border border-gray-200">{children}</th>,
  td: ({ children }: any) => <td className="px-3 py-2 text-gray-700 border border-gray-200">{children}</td>,
  tr: ({ children }: any) => <tr className="even:bg-gray-50">{children}</tr>,
  code: ({ inline, className, children }: any) => {
    const raw = String(children ?? '').replace(/\n$/, '');
    const langMatch = /language-(\w+)/.exec(className || '');
    // react-markdown v10 no longer passes `inline`; fall back to detecting a
    // fenced block via the language className or a multi-line body.
    const isBlock = inline === false || !!langMatch || raw.includes('\n');
    if (!isBlock) {
      return <code className="bg-gray-100 px-1 py-0.5 rounded text-indigo-600 text-xs font-mono">{children}</code>;
    }
    return (
      <SyntaxHighlighter
        language={langMatch ? langMatch[1] : 'text'}
        style={oneLight}
        PreTag="div"
        customStyle={{
          margin: '0.5rem 0', borderRadius: '0.5rem', fontSize: '0.75rem',
          border: '1px solid #e5e7eb', background: '#f9fafb',
        }}
        codeTagProps={{ style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' } }}
      >
        {raw}
      </SyntaxHighlighter>
    );
  },
};

// ─── PlotCard ─────────────────────────────────────────────────────────────────

function stripHtml(s: string): string {
  return s.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
}

// Viewport gate: true once the element is within `rootMargin` of the viewport.
// Used to defer heavy work (mounting Plotly, highlighting code) for off-screen
// cards so the number of LIVE heavy widgets stays bounded by what's visible —
// this is what keeps the page responsive as outputs accumulate. Stays true once
// seen (we don't tear widgets back down) so scrolling back is instant; the cap
// is on how many ever mount *at once* during the initial reveal, not lifetime.
function useInView<T extends Element>(rootMargin = '800px'): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || inView) return;
    if (typeof IntersectionObserver === 'undefined') { setInView(true); return; }
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) setInView(true);
      },
      { rootMargin }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [rootMargin, inView]);
  return [ref, inView];
}

// Browser-side plot cache. Plots are content-addressed on the backend and served
// with an immutable cache header, so a given id never changes — we fetch its JSON
// at most once per session (and the browser HTTP-caches it across reloads). A
// plot can arrive either inline ({data, layout}, legacy) or as a ref ({id, title}).
const _plotCache = new Map<string, any>();

function usePlotFigure(plot: any, enabled: boolean = true): any | null {
  const isRef = !!(plot && plot.id && !plot.data);
  const [fig, setFig] = useState<any | null>(
    isRef ? _plotCache.get(plot.id) ?? null : plot
  );
  useEffect(() => {
    if (!isRef) { setFig(plot); return; }
    const cached = _plotCache.get(plot.id);
    if (cached) { setFig(cached); return; }
    if (!enabled) return;  // defer the fetch until the card is near the viewport
    let alive = true;
    // No auth header → maximally cacheable; the agent API serves plots publicly.
    fetch(`${API_BASE}/plots/${plot.id}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => { if (j) { _plotCache.set(plot.id, j); if (alive) setFig(j); } })
      .catch(() => {});
    return () => { alive = false; };
  }, [isRef, plot, enabled]);
  return fig;
}

// React.memo: a plot object keeps its identity once appended to dashboardData
// (we merge, never rebuild existing refs), so memoization stops every chart from
// re-rendering — and react-plotly from re-running Plotly.react() — on every SSE
// chunk while the agent streams. Combined with the viewport gate below, this is
// what fixes the "freezes as more outputs accumulate" symptom.
const PlotCard = React.memo(function PlotCard({ plot, idx }: { plot: any; idx: number }) {
  const [fullscreen, setFullscreen] = useState(false);
  // The observed wrapper is ALWAYS in the DOM (even before reveal) so the
  // IntersectionObserver can fire; only the heavy <Plot> mounts once in view.
  const [wrapRef, inView] = useInView<HTMLDivElement>();
  const fig = usePlotFigure(plot, inView);

  const rawTitle =
    fig?.layout?.title?.text ?? fig?.layout?.title ?? plot?.title ?? `Chart ${idx + 1}`;
  const title = stripHtml(String(rawTitle || `Chart ${idx + 1}`));

  const fixedLayout = useMemo(() => applyLightModeLayout(fig?.layout), [fig]);

  const plotEl = (height: string) =>
    fig ? (
      <Plot
        data={fig.data}
        layout={{ ...fixedLayout, autosize: true }}
        useResizeHandler
        style={{ width: '100%', height }}
        config={{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['sendDataToCloud'] }}
      />
    ) : null;

  return (
    <>
      <div
        ref={wrapRef}
        className="rounded-xl overflow-hidden border border-gray-200 bg-white relative group shadow-sm min-h-[400px]"
      >
        <button
          onClick={() => setFullscreen(true)}
          className="absolute top-2 right-2 z-10 p-1.5 rounded-lg bg-white/90 text-gray-400 hover:text-gray-700 hover:bg-gray-100 opacity-0 group-hover:opacity-100 transition-all border border-gray-200"
          title="Expand chart"
        >
          <Maximize2 size={15} />
        </button>
        <p className="text-xs text-gray-500 px-4 pt-3 pb-0 font-semibold truncate">{title}</p>
        {inView && fig ? (
          plotEl('360px')
        ) : (
          <div className="h-[360px] flex items-center justify-center text-sm text-gray-400">
            {inView ? 'Loading chart…' : ''}
          </div>
        )}
      </div>
      {fullscreen && (
        <Modal title={title} onClose={() => setFullscreen(false)} fullWidth>
          {plotEl('calc(100vh - 120px)')}
        </Modal>
      )}
    </>
  );
}, (a, b) =>
  // Plots are content-addressed by `id`, but each streaming update re-parses the
  // dashboard_data JSON, giving every plot ref a NEW object identity. Compare by
  // id (falling back to reference for legacy inline figures) so existing charts
  // are not needlessly re-rendered when a new plot arrives.
  a.idx === b.idx && (a.plot?.id ?? a.plot) === (b.plot?.id ?? b.plot)
);

// ─── ModelSpecWidget ──────────────────────────────────────────────────────────

// Normalize an incoming (possibly minimal) spec into a full editable form
function specWithDefaults(raw: any) {
  return {
    kpi: raw?.kpi ?? '',
    kpi_level: raw?.kpi_level ?? 'national',
    time_granularity: raw?.time_granularity ?? 'weekly',
    inference: {
      chains: raw?.inference?.chains ?? 4,
      draws: raw?.inference?.draws ?? 1000,
      tune: raw?.inference?.tune ?? 1000,
      target_accept: raw?.inference?.target_accept ?? 0.85,
      random_seed: raw?.inference?.random_seed ?? 42,
    },
    trend: {
      type: raw?.trend?.type ?? 'linear',
      n_changepoints: raw?.trend?.n_changepoints ?? 5,
      changepoint_range: raw?.trend?.changepoint_range ?? 0.8,
      n_knots: raw?.trend?.n_knots ?? 5,
      spline_degree: raw?.trend?.spline_degree ?? 3,
    },
    seasonality: {
      yearly: raw?.seasonality?.yearly ?? 0,
      monthly: raw?.seasonality?.monthly ?? 0,
      weekly: raw?.seasonality?.weekly ?? 0,
    },
    media_channels: (raw?.media_channels ?? []).map((ch: any) => ({
      name: ch.name,
      adstock: { type: ch.adstock?.type ?? 'geometric', l_max: ch.adstock?.l_max ?? 8 },
      saturation: { type: ch.saturation?.type ?? 'hill' },
    })),
    control_variables: (raw?.control_variables ?? []).map((c: any) =>
      typeof c === 'string' ? { name: c } : c
    ),
  };
}

// Compose the notification message sent to the agent when config is applied
function buildApplyMessage(spec: any): string {
  const channels = (spec.media_channels ?? [])
    .filter((c: any) => c != null)
    .map((c: any) => {
      const adsType = c?.adstock?.type ?? 'geometric';
      const adsLmax = c?.adstock?.l_max ?? 8;
      const satType = c?.saturation?.type ?? 'hill';
      return `${c.name} (${adsType} adstock l_max=${adsLmax}, ${satType} saturation)`;
    })
    .join('; ');
  const controls = (spec.control_variables ?? []).filter((c: any) => c != null).map((c: any) => c.name).join(', ') || 'none';
  const { chains = 4, draws = 1000, tune = 1000, target_accept = 0.85, random_seed = 42 } = spec.inference ?? {};
  const { type: ttype = 'linear', n_changepoints = 5, changepoint_range = 0.8, n_knots = 5, spline_degree = 3 } = spec.trend ?? {};
  const { yearly = 0, monthly = 0, weekly = 0 } = spec.seasonality ?? {};

  const trendDetail = ttype === 'piecewise'
    ? ` (${n_changepoints} changepoints, range=${(changepoint_range * 100).toFixed(0)}%)`
    : ttype === 'spline'
    ? ` (${n_knots} knots, degree ${spline_degree})`
    : '';

  return `I've reviewed and updated the model configuration. Please use the following JSON as the **exact** \`model_spec\` parameter when you call \`fit_mmm_model\`:

\`\`\`json
${JSON.stringify(spec, null, 2)}
\`\`\`

Summary of key settings:
- **KPI**: ${spec.kpi} (${spec.kpi_level} level, ${spec.time_granularity} data)
- **Media channels**: ${channels || 'none'}
- **Controls**: ${controls}
- **Inference**: ${chains} chains × ${draws} draws (${tune} tune steps), target_accept=${target_accept}, seed=${random_seed}
- **Trend**: ${ttype}${trendDetail}
- **Seasonality**: yearly=${yearly}, monthly=${monthly}, weekly=${weekly}

Please acknowledge this configuration. When fitting, pass the JSON above directly as the \`model_spec\` argument.`;
}

// ─── Shared form primitives ───────────────────────────────────────────────────

const iCls = 'w-full bg-gray-50 border border-gray-200 rounded-lg px-2.5 py-1.5 text-xs text-gray-800 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition-all';
const sCls = iCls + ' cursor-pointer';

function FLabel({ children }: { children: React.ReactNode }) {
  return <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-0.5">{children}</p>;
}

function SpecRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-center py-1.5 border-b border-gray-100 last:border-0">
      <span className="text-xs text-gray-500 font-medium">{label}</span>
      <span className="text-xs text-gray-900 font-semibold text-right max-w-[60%]">{value}</span>
    </div>
  );
}

function SpecSection({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mb-3">
      <button onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-2 text-xs font-bold text-gray-600 uppercase tracking-wider mb-1.5 hover:text-gray-900 transition-colors">
        {icon}
        <span className="flex-1 text-left">{title}</span>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {open && <div className="bg-gray-50 rounded-lg px-3 py-1 border border-gray-100">{children}</div>}
    </div>
  );
}

function Badge({ label, color = 'gray' }: { label: string; color?: 'blue' | 'indigo' | 'gray' | 'green' | 'amber' }) {
  const cls = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    indigo: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    gray: 'bg-gray-100 text-gray-600 border-gray-200',
    green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    amber: 'bg-amber-50 text-amber-700 border-amber-200',
  }[color];
  return <span className={`inline-block px-2.5 py-0.5 text-xs rounded-full border font-medium ${cls}`}>{label}</span>;
}

// ─── EditSection: collapsible form section ────────────────────────────────────

function EditSection({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <button onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-2 px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left">
        <span className="text-gray-500">{icon}</span>
        <span className="flex-1 text-xs font-bold text-gray-700 uppercase tracking-wider">{title}</span>
        {open ? <ChevronDown size={13} className="text-gray-400" /> : <ChevronRight size={13} className="text-gray-400" />}
      </button>
      {open && <div className="px-4 py-3 space-y-3 bg-white">{children}</div>}
    </div>
  );
}

// ─── ModelSpecWidget ──────────────────────────────────────────────────────────

interface ModelSpecWidgetProps {
  spec: any;
  editable: boolean;
  onApplySpec: (newSpec: any) => void;
}

function ModelSpecWidget({ spec, editable, onApplySpec }: ModelSpecWidgetProps) {
  const [editMode, setEditMode] = useState(false);
  const [draft, setDraft] = useState(() => specWithDefaults(spec));
  const [newChannel, setNewChannel] = useState('');
  const [newControl, setNewControl] = useState('');

  // Re-sync draft when spec prop changes (e.g. agent updates it)
  useEffect(() => {
    if (!editMode) setDraft(specWithDefaults(spec));
  }, [spec, editMode]);

  const setDraftField = (path: string[], value: any) =>
    setDraft((prev: any) => {
      const next = { ...prev };
      let cur: any = next;
      for (let i = 0; i < path.length - 1; i++) {
        cur[path[i]] = { ...cur[path[i]] };
        cur = cur[path[i]];
      }
      cur[path[path.length - 1]] = value;
      return next;
    });

  const setChannel = (idx: number, field: string, subfield: string | null, value: any) =>
    setDraft((prev: any) => {
      const channels = prev.media_channels.map((ch: any, i: number) => {
        if (i !== idx) return ch;
        if (subfield) return { ...ch, [field]: { ...ch[field], [subfield]: value } };
        return { ...ch, [field]: value };
      });
      return { ...prev, media_channels: channels };
    });

  const addChannel = () => {
    const name = newChannel.trim();
    if (!name) return;
    setDraft((prev: any) => ({
      ...prev,
      media_channels: [
        ...prev.media_channels,
        { name, adstock: { type: 'geometric', l_max: 8 }, saturation: { type: 'hill' } },
      ],
    }));
    setNewChannel('');
  };

  const removeChannel = (idx: number) =>
    setDraft((prev: any) => ({ ...prev, media_channels: prev.media_channels.filter((_: any, i: number) => i !== idx) }));

  const addControl = () => {
    const name = newControl.trim();
    if (!name) return;
    setDraft((prev: any) => ({ ...prev, control_variables: [...prev.control_variables, { name }] }));
    setNewControl('');
  };

  const removeControl = (idx: number) =>
    setDraft((prev: any) => ({ ...prev, control_variables: prev.control_variables.filter((_: any, i: number) => i !== idx) }));

  const handleApply = () => {
    onApplySpec(draft);
    setEditMode(false);
  };

  const handleDiscard = () => {
    setDraft(specWithDefaults(spec));
    setEditMode(false);
  };

  // ── View mode ──────────────────────────────────────────────────────────────
  const displaySpec = editMode ? draft : specWithDefaults(spec);
  const trendType = displaySpec.trend?.type ?? 'linear';
  const trendLabel = trendType.charAt(0).toUpperCase() + trendType.slice(1).replace('_', ' ');
  const seasonality = displaySpec.seasonality;
  const inference = displaySpec.inference;

  const viewContent = (
    <div className="space-y-3 pt-1">
      <SpecSection title="KPI & Data" icon={<BarChart2 size={13} />}>
        <SpecRow label="KPI Variable" value={displaySpec.kpi || '—'} />
        <SpecRow label="Level" value={(displaySpec.kpi_level || 'national').replace(/\b\w/g, (c: string) => c.toUpperCase())} />
        <SpecRow label="Granularity" value={displaySpec.time_granularity || 'weekly'} />
      </SpecSection>
      <SpecSection title="Inference" icon={<Activity size={13} />}>
        <SpecRow label="Chains" value={inference?.chains ?? 4} />
        <SpecRow label="Draws" value={inference?.draws ?? 1000} />
        <SpecRow label="Tune" value={inference?.tune ?? 1000} />
        <SpecRow label="Target Accept" value={inference?.target_accept ?? 0.85} />
        <SpecRow label="Seed" value={inference?.random_seed ?? 42} />
      </SpecSection>
      <SpecSection title="Trend" icon={<TrendingUp size={13} />}>
        <SpecRow label="Type" value={trendLabel} />
        {trendType === 'piecewise' && <><SpecRow label="Changepoints" value={displaySpec.trend?.n_changepoints ?? 5} /><SpecRow label="Range" value={`${((displaySpec.trend?.changepoint_range ?? 0.8) * 100).toFixed(0)}%`} /></>}
        {trendType === 'spline' && <><SpecRow label="Knots" value={displaySpec.trend?.n_knots ?? 5} /><SpecRow label="Degree" value={displaySpec.trend?.spline_degree ?? 3} /></>}
      </SpecSection>
      <SpecSection title="Seasonality" icon={<Calendar size={13} />}>
        <SpecRow label="Yearly" value={seasonality?.yearly ? `${seasonality.yearly} terms` : 'Off'} />
        <SpecRow label="Monthly" value={seasonality?.monthly ? `${seasonality.monthly} terms` : 'Off'} />
        <SpecRow label="Weekly" value={seasonality?.weekly ? `${seasonality.weekly} terms` : 'Off'} />
      </SpecSection>
      {displaySpec.media_channels?.length > 0 && (
        <SpecSection title="Media Channels" icon={<Zap size={13} />}>
          {displaySpec.media_channels.map((ch: any) => (
            <div key={ch.name} className="py-2 border-b border-gray-100 last:border-0">
              <p className="text-xs font-semibold text-gray-800 mb-1">{ch.name}</p>
              <div className="flex flex-wrap gap-1.5">
                <Badge label={`${ch.adstock?.type ?? 'geometric'} adstock`} color="indigo" />
                <Badge label={`l_max=${ch.adstock?.l_max ?? 8}`} color="gray" />
                <Badge label={`${ch.saturation?.type ?? 'hill'} sat.`} color="blue" />
              </div>
            </div>
          ))}
        </SpecSection>
      )}
      {displaySpec.control_variables?.length > 0 && (
        <SpecSection title="Controls" icon={<Layers size={13} />}>
          <div className="flex flex-wrap gap-1.5 py-1">
            {displaySpec.control_variables.map((c: any) => <Badge key={c.name} label={c.name} />)}
          </div>
        </SpecSection>
      )}
    </div>
  );

  // ── Edit mode ──────────────────────────────────────────────────────────────
  const editForm = (
    <div className="space-y-3 pt-1">
      {/* KPI & Data */}
      <EditSection title="KPI & Data" icon={<BarChart2 size={13} />}>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <FLabel>KPI Variable</FLabel>
            <input className={iCls} value={draft.kpi}
              onChange={e => setDraftField(['kpi'], e.target.value)} placeholder="e.g. Sales" />
          </div>
          <div>
            <FLabel>Level</FLabel>
            <select className={sCls} value={draft.kpi_level} onChange={e => setDraftField(['kpi_level'], e.target.value)}>
              <option value="national">National</option>
              <option value="geo">Geo</option>
            </select>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <FLabel>Granularity</FLabel>
            <select className={sCls} value={draft.time_granularity} onChange={e => setDraftField(['time_granularity'], e.target.value)}>
              <option value="weekly">Weekly</option>
              <option value="daily">Daily</option>
              <option value="monthly">Monthly</option>
            </select>
          </div>
        </div>
      </EditSection>

      {/* Inference */}
      <EditSection title="Inference" icon={<Activity size={13} />}>
        <div className="grid grid-cols-3 gap-2">
          {([['Chains', 'chains', 1, 8], ['Draws', 'draws', 100, 10000], ['Tune', 'tune', 100, 5000]] as const).map(([label, key, min, max]) => (
            <div key={key}>
              <FLabel>{label}</FLabel>
              <input className={iCls} type="number" min={min} max={max}
                value={draft.inference[key]}
                onChange={e => setDraftField(['inference', key], Number(e.target.value))} />
            </div>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <FLabel>Target Accept</FLabel>
            <input className={iCls} type="number" min={0.5} max={0.99} step={0.01}
              value={draft.inference.target_accept}
              onChange={e => setDraftField(['inference', 'target_accept'], Number(e.target.value))} />
          </div>
          <div>
            <FLabel>Random Seed</FLabel>
            <input className={iCls} type="number"
              value={draft.inference.random_seed}
              onChange={e => setDraftField(['inference', 'random_seed'], Number(e.target.value))} />
          </div>
        </div>
      </EditSection>

      {/* Trend */}
      <EditSection title="Trend Model" icon={<TrendingUp size={13} />}>
        <div>
          <FLabel>Type</FLabel>
          <select className={sCls} value={draft.trend.type} onChange={e => setDraftField(['trend', 'type'], e.target.value)}>
            <option value="linear">Linear</option>
            <option value="piecewise">Piecewise Linear</option>
            <option value="spline">Spline</option>
            <option value="gaussian_process">Gaussian Process</option>
            <option value="none">None</option>
          </select>
        </div>
        {draft.trend.type === 'piecewise' && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <FLabel>Changepoints</FLabel>
              <input className={iCls} type="number" min={1} max={50}
                value={draft.trend.n_changepoints}
                onChange={e => setDraftField(['trend', 'n_changepoints'], Number(e.target.value))} />
            </div>
            <div>
              <FLabel>Changepoint Range (0–1)</FLabel>
              <input className={iCls} type="number" min={0.1} max={1} step={0.05}
                value={draft.trend.changepoint_range}
                onChange={e => setDraftField(['trend', 'changepoint_range'], Number(e.target.value))} />
            </div>
          </div>
        )}
        {draft.trend.type === 'spline' && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <FLabel>Knots</FLabel>
              <input className={iCls} type="number" min={2} max={50}
                value={draft.trend.n_knots}
                onChange={e => setDraftField(['trend', 'n_knots'], Number(e.target.value))} />
            </div>
            <div>
              <FLabel>Degree</FLabel>
              <select className={sCls} value={draft.trend.spline_degree}
                onChange={e => setDraftField(['trend', 'spline_degree'], Number(e.target.value))}>
                {[1, 2, 3, 4, 5].map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
          </div>
        )}
      </EditSection>

      {/* Seasonality */}
      <EditSection title="Seasonality (Fourier terms, 0 = off)" icon={<Calendar size={13} />}>
        <div className="grid grid-cols-3 gap-2">
          {(['yearly', 'monthly', 'weekly'] as const).map(period => (
            <div key={period}>
              <FLabel>{period.charAt(0).toUpperCase() + period.slice(1)}</FLabel>
              <input className={iCls} type="number" min={0} max={10}
                value={draft.seasonality[period]}
                onChange={e => setDraftField(['seasonality', period], Number(e.target.value))} />
            </div>
          ))}
        </div>
      </EditSection>

      {/* Media Channels */}
      <EditSection title="Media Channels" icon={<Zap size={13} />}>
        <div className="space-y-2">
          {draft.media_channels.map((ch: any, idx: number) => (
            <div key={idx} className="flex gap-2 items-end bg-gray-50 rounded-lg p-2 border border-gray-100">
              <div className="flex-1">
                <FLabel>Name</FLabel>
                <input className={iCls} value={ch.name}
                  onChange={e => setChannel(idx, 'name', null, e.target.value)} />
              </div>
              <div className="w-28">
                <FLabel>Adstock</FLabel>
                <select className={sCls} value={ch.adstock.type}
                  onChange={e => setChannel(idx, 'adstock', 'type', e.target.value)}>
                  <option value="geometric">Geometric</option>
                  <option value="weibull">Weibull</option>
                  <option value="delayed">Delayed</option>
                </select>
              </div>
              <div className="w-16">
                <FLabel>L-max</FLabel>
                <input className={iCls} type="number" min={1} max={52}
                  value={ch.adstock.l_max}
                  onChange={e => setChannel(idx, 'adstock', 'l_max', Number(e.target.value))} />
              </div>
              <div className="w-32">
                <FLabel>Saturation</FLabel>
                <select className={sCls} value={ch.saturation.type}
                  onChange={e => setChannel(idx, 'saturation', 'type', e.target.value)}>
                  <option value="hill">Hill</option>
                  <option value="logistic">Logistic</option>
                  <option value="michaelis_menten">Michaelis-Menten</option>
                  <option value="tanh">Tanh</option>
                </select>
              </div>
              <button onClick={() => removeChannel(idx)}
                className="p-1.5 text-red-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors shrink-0" title="Remove channel">
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
        <div className="flex gap-2 mt-2">
          <input className={iCls + ' flex-1'} placeholder="New channel name…"
            value={newChannel}
            onChange={e => setNewChannel(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addChannel()} />
          <button onClick={addChannel}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-medium rounded-lg border border-indigo-200 transition-colors">
            <Plus size={13} /> Add
          </button>
        </div>
      </EditSection>

      {/* Controls */}
      <EditSection title="Control Variables" icon={<Layers size={13} />}>
        <div className="flex flex-wrap gap-1.5">
          {draft.control_variables.map((c: any, idx: number) => (
            <span key={idx} className="flex items-center gap-1 px-2.5 py-1 bg-gray-100 text-gray-700 text-xs rounded-full border border-gray-200">
              {c.name}
              <button onClick={() => removeControl(idx)} className="text-gray-400 hover:text-red-500 transition-colors ml-0.5">
                <X size={11} />
              </button>
            </span>
          ))}
        </div>
        <div className="flex gap-2 mt-1">
          <input className={iCls + ' flex-1'} placeholder="New control variable…"
            value={newControl}
            onChange={e => setNewControl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addControl()} />
          <button onClick={addControl}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-50 hover:bg-gray-100 text-gray-700 text-xs font-medium rounded-lg border border-gray-200 transition-colors">
            <Plus size={13} /> Add
          </button>
        </div>
      </EditSection>
    </div>
  );

  // ── Widget wrapper ─────────────────────────────────────────────────────────
  const headerActions = editMode ? (
    <div className="flex items-center gap-2">
      <button onClick={handleApply}
        className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg transition-colors">
        <Check size={13} /> Apply
      </button>
      <button onClick={handleDiscard}
        className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-600 text-xs font-medium rounded-lg transition-colors border border-gray-200">
        <RotateCcw size={13} /> Discard
      </button>
    </div>
  ) : editable ? (
    <button onClick={() => setEditMode(true)}
      className="flex items-center gap-1.5 px-2.5 py-1 bg-gray-50 hover:bg-gray-100 text-gray-500 hover:text-indigo-600 text-xs font-medium rounded-lg border border-gray-200 transition-colors">
      <Pencil size={12} /> Edit
    </button>
  ) : null;

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm hover:shadow-md transition-all overflow-hidden">
      <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-100">
        <Activity size={15} className="text-blue-500 shrink-0" />
        <span className="font-semibold text-sm text-blue-600 flex-1">Model Configuration</span>
        {editMode && (
          <span className="px-2 py-0.5 bg-amber-50 text-amber-600 text-[10px] font-bold uppercase tracking-wide rounded-full border border-amber-200">
            Editing
          </span>
        )}
        {headerActions}
      </div>
      <div className="px-5 py-4 max-h-[600px] overflow-y-auto">
        {editMode ? editForm : viewContent}
      </div>
    </div>
  );
}

// ─── SeasonalityTrendWidget ───────────────────────────────────────────────────

function generateFourierTraces(order: number, period: number, label: string): any[] {
  const t = Array.from({ length: period }, (_, i) => i);
  const PALETTE = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];
  const traces: any[] = [];
  const combined = t.map(() => 0);

  for (let k = 1; k <= Math.min(order, 6); k++) {
    const phase = Math.PI / (2 * k); // deterministic phase offset
    const y = t.map(ti => Math.sin(2 * Math.PI * k * ti / period + phase));
    y.forEach((v, i) => { combined[i] += v; });
    traces.push({
      x: t, y, name: `${label} H${k}`,
      type: 'scatter', mode: 'lines',
      line: { color: PALETTE[(k - 1) % PALETTE.length], width: 1.5, dash: k > 1 ? 'dot' : 'solid' },
      opacity: 0.6,
    });
  }

  const maxAmp = Math.max(...combined.map(Math.abs)) || 1;
  traces.push({
    x: t, y: combined.map(v => v / maxAmp),
    name: `${label} Combined`,
    type: 'scatter', mode: 'lines',
    line: { color: '#111827', width: 2.5 },
    opacity: 1,
  });

  return traces;
}

function generateTrendTrace(type: string, spec: any): { traces: any[]; shapes: any[] } {
  const t = Array.from({ length: 100 }, (_, i) => i / 99);

  if (type === 'linear' || !type) {
    return {
      traces: [{ x: t, y: t.map(x => 0.15 + 0.7 * x), name: 'Linear Trend', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } }],
      shapes: [],
    };
  }

  if (type === 'piecewise') {
    const n = spec?.n_changepoints ?? 5;
    const range = spec?.changepoint_range ?? 0.8;
    // Evenly space n changepoints within (0, range) — mirrors backend linspace logic
    const cps = Array.from({ length: n }, (_, i) => range * (i + 1) / (n + 1));
    // Enough slope variety for up to 25 changepoints (26 segments)
    const slopes = [0.8, 0.3, 1.1, -0.2, 0.6, 0.9, -0.1, 0.7, 0.4, -0.3, 0.5, -0.4, 1.0, 0.2, -0.5, 0.8, 0.3, 1.1, -0.2, 0.6, 0.9, -0.1, 0.7, 0.4, -0.3, 0.5];
    // Walk through sorted t values, advancing segments at each changepoint
    const y: number[] = [];
    let segStart = 0, segBaseY = 0.1, segIdx = 0;
    for (const x of t) {
      while (segIdx < cps.length && x > cps[segIdx]) {
        segBaseY += slopes[segIdx % slopes.length] * (cps[segIdx] - segStart);
        segStart = cps[segIdx];
        segIdx++;
      }
      y.push(segBaseY + slopes[segIdx % slopes.length] * (x - segStart));
    }
    return {
      traces: [{ x: t, y, name: 'Piecewise Trend', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } }],
      shapes: cps.map(cp => ({
        type: 'line', x0: cp, x1: cp, y0: 0, y1: 1, yref: 'paper',
        line: { color: '#f59e0b', width: 1.5, dash: 'dash' },
      })),
    };
  }

  if (type === 'spline') {
    const nk = spec?.n_knots ?? 4;
    const knots = Array.from({ length: nk }, (_, i) => (i + 1) / (nk + 1));
    const y = t.map(x => {
      let val = 0.15 + 0.6 * x;
      knots.forEach(k => { val += 0.08 * Math.exp(-50 * Math.pow(x - k, 2)) * Math.sin(12 * (x - k)); });
      return val;
    });
    return {
      traces: [{ x: t, y, name: 'Spline Trend', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } }],
      shapes: knots.map(k => ({
        type: 'line', x0: k, x1: k, y0: 0, y1: 1, yref: 'paper',
        line: { color: '#10b981', width: 1, dash: 'dot' },
      })),
    };
  }

  if (type === 'gaussian_process') {
    // Smooth GP realisation using summed cosines
    const y = t.map(x => {
      let v = 0.35;
      for (let k = 1; k <= 6; k++) v += (0.08 / k) * Math.cos(k * Math.PI * x + k * 0.7);
      return v;
    });
    const y_upper = y.map(v => v + 0.12);
    const y_lower = y.map(v => v - 0.12);
    return {
      traces: [
        { x: t, y, name: 'GP Mean', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } },
        {
          x: [...t, ...t.slice().reverse()],
          y: [...y_upper, ...y_lower.slice().reverse()],
          name: '±1 σ', type: 'scatter', mode: 'lines', fill: 'toself',
          fillcolor: 'rgba(99,102,241,0.12)', line: { color: 'transparent' },
        },
      ],
      shapes: [],
    };
  }

  return { traces: [], shapes: [] };
}

interface SeasonalityTrendWidgetProps {
  spec: any;
  onQuickAction: (msg: string) => void;
  modelCompleted: boolean;
}

function SeasonalityTrendWidget({ spec, onQuickAction, modelCompleted }: SeasonalityTrendWidgetProps) {
  const [tab, setTab] = useState<'trend' | 'yearly' | 'monthly' | 'weekly'>('trend');

  const trendType = spec?.trend?.type ?? 'linear';
  const seasonality = spec?.seasonality;
  const granularity = spec?.time_granularity ?? 'weekly';
  const period = granularity === 'daily' ? 365 : granularity === 'monthly' ? 12 : 52;

  const allTabs: { key: typeof tab; label: string; disabled?: boolean }[] = [
    { key: 'trend' as const, label: 'Trend' },
    { key: 'yearly' as const, label: 'Yearly Season', disabled: !(seasonality?.yearly > 0) },
    { key: 'monthly' as const, label: 'Monthly Season', disabled: !(seasonality?.monthly > 0) },
    { key: 'weekly' as const, label: 'Weekly Season', disabled: !(seasonality?.weekly > 0) },
  ];
  const tabs = allTabs.filter(t => !t.disabled || t.key === 'trend');

  const chartData = useMemo(() => {
    if (tab === 'trend') {
      const { traces, shapes } = generateTrendTrace(trendType, spec?.trend);
      return {
        data: traces,
        layout: applyLightModeLayout({
          shapes,
          xaxis: { title: { text: 'Relative Time' }, showticklabels: false },
          yaxis: { title: { text: 'Trend Value' } },
          legend: { orientation: 'h', y: -0.25 },
          title: `Trend Model: ${trendType.charAt(0).toUpperCase() + trendType.slice(1).replace('_', ' ')}${shapes.length ? ` (${shapes.length} changepoints)` : ''}`,
          margin: { t: 55, b: 55 },
        }),
      };
    }
    const orders: Record<string, number> = {
      yearly: seasonality?.yearly ?? 0,
      monthly: seasonality?.monthly ?? 0,
      weekly: seasonality?.weekly ?? 0,
    };
    const periods: Record<string, number> = { yearly: period, monthly: Math.round(period / 4), weekly: 7 };
    const order = orders[tab];
    const p = periods[tab];
    const traces = generateFourierTraces(order, p, tab.charAt(0).toUpperCase() + tab.slice(1));
    return {
      data: traces,
      layout: applyLightModeLayout({
        xaxis: { title: { text: tab === 'yearly' ? `Week of Year (period = ${p})` : tab === 'monthly' ? 'Week of Month' : 'Day of Week' } },
        yaxis: { title: { text: 'Normalised Amplitude' } },
        legend: { orientation: 'h', y: -0.3, font: { size: 10 } },
        title: `${tab.charAt(0).toUpperCase() + tab.slice(1)} Seasonality — ${order} Fourier Term${order !== 1 ? 's' : ''}`,
        margin: { t: 55, b: 70 },
      }),
    };
  }, [tab, trendType, seasonality, spec, period]);

  const quickActions = [
    { label: 'Trend over time', msg: 'Using execute_python, extract the fitted linear trend component from the `mmm` object and plot it as a Plotly time series. Show the posterior mean and 89% HDI band.' },
    { label: 'Channel contributions', msg: 'Using execute_python with the `mmm` and `results` objects, create a stacked area Plotly chart showing each media channel\'s contribution to the KPI over time.' },
    { label: 'Fitted vs Actual', msg: 'Using execute_python, plot the model\'s posterior predictive mean against the actual KPI values as a Plotly time series.' },
    { label: 'Saturation curves', msg: 'Using execute_python, plot the Hill saturation curves for each media channel showing diminishing returns.' },
  ];

  return (
    <DashWidget
      title="Trend & Seasonality"
      icon={<TrendingUp size={15} className="text-violet-500 shrink-0" />}
      color="violet"
      expandTitle="Trend & Seasonality Preview"
      expandContent={
        <div className="space-y-4">
          <div className="flex gap-2 flex-wrap">
            {tabs.map(t => (
              <button key={t.key} onClick={() => setTab(t.key)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${tab === t.key ? 'bg-violet-600 text-white border-violet-600' : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'}`}
              >{t.label}</button>
            ))}
          </div>
          <Plot data={chartData.data} layout={{ ...chartData.layout, autosize: true }}
            useResizeHandler style={{ width: '100%', height: '420px' }}
            config={{ responsive: true, displayModeBar: true, displaylogo: false }}
          />
          {modelCompleted && (
            <div className="border-t border-gray-100 pt-4">
              <p className="text-xs font-semibold text-gray-600 mb-2">Generate from fitted model:</p>
              <div className="flex flex-wrap gap-2">
                {quickActions.map(qa => (
                  <button key={qa.label} onClick={() => onQuickAction(qa.msg)}
                    className="px-3 py-1.5 bg-violet-50 hover:bg-violet-100 text-violet-700 text-xs font-medium rounded-lg border border-violet-200 transition-colors">
                    {qa.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      }
    >
      {/* Mini preview */}
      <div className="space-y-3">
        <div className="flex gap-2 flex-wrap">
          {tabs.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${tab === t.key ? 'bg-violet-600 text-white border-violet-600' : 'bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100'}`}
            >{t.label}</button>
          ))}
        </div>
        <Plot data={chartData.data}
          layout={{ ...chartData.layout, autosize: true, margin: { l: 50, r: 20, t: 45, b: 55 } }}
          useResizeHandler style={{ width: '100%', height: '260px' }}
          config={{ responsive: true, displayModeBar: false }}
        />
        {modelCompleted && (
          <div>
            <p className="text-[10px] text-gray-400 uppercase tracking-widest mb-1.5">Quick Actions (from fitted model)</p>
            <div className="flex flex-wrap gap-1.5">
              {quickActions.map(qa => (
                <button key={qa.label} onClick={() => onQuickAction(qa.msg)}
                  className="px-2.5 py-1 bg-violet-50 hover:bg-violet-100 text-violet-700 text-xs rounded-lg border border-violet-200 transition-colors">
                  {qa.label}
                </button>
              ))}
            </div>
          </div>
        )}
        {!modelCompleted && (
          <p className="text-[11px] text-gray-400 italic">Fit the model to generate actual component plots.</p>
        )}
      </div>
    </DashWidget>
  );
}

// ─── Prior Configuration ──────────────────────────────────────────────────────

// --- Math helpers ---

function linspace(a: number, b: number, n: number): number[] {
  if (n <= 1) return [a];
  const step = (b - a) / (n - 1);
  return Array.from({ length: n }, (_, i) => a + i * step);
}

function normalizeDensity(x: number[], y: number[]): number[] {
  if (x.length < 2) return y;
  const dx = (x[x.length - 1] - x[0]) / (x.length - 1);
  const total = y.reduce((s, v) => s + v * dx, 0);
  return total > 1e-12 ? y.map(v => v / total) : y;
}

function computeDensity(dist: string, params: Record<string, number>): { x: number[]; y: number[] } {
  const N = 160;
  const normalPDF = (xi: number, mu: number, sig: number) => {
    const z = (xi - mu) / sig;
    return Math.exp(-0.5 * z * z) / (sig * Math.sqrt(2 * Math.PI));
  };

  if (dist === 'normal') {
    const mu = params.mu ?? 0; const sigma = Math.max(params.sigma ?? 1, 1e-6);
    const x = linspace(mu - 4 * sigma, mu + 4 * sigma, N);
    return { x, y: x.map(xi => normalPDF(xi, mu, sigma)) };
  }
  if (dist === 'half_normal') {
    const sigma = Math.max(params.sigma ?? 1, 1e-6);
    const x = linspace(0, 4.5 * sigma, N);
    return { x, y: x.map(xi => 2 * normalPDF(xi, 0, sigma)) };
  }
  if (dist === 'log_normal') {
    const mu = params.mu ?? 0; const sigma = Math.max(params.sigma ?? 1, 1e-6);
    const xMax = Math.exp(mu + 3.5 * sigma);
    const x = linspace(1e-6, xMax, N);
    const y = x.map(xi => Math.exp(-0.5 * ((Math.log(xi) - mu) / sigma) ** 2) / (xi * sigma * Math.sqrt(2 * Math.PI)));
    return { x, y };
  }
  if (dist === 'beta') {
    const alpha = Math.max(params.alpha ?? 2, 1e-3); const beta = Math.max(params.beta ?? 2, 1e-3);
    const x = linspace(1e-4, 1 - 1e-4, N);
    const unnorm = x.map(xi => Math.exp((alpha - 1) * Math.log(xi) + (beta - 1) * Math.log(1 - xi)));
    return { x, y: normalizeDensity(x, unnorm) };
  }
  if (dist === 'gamma') {
    const alpha = Math.max(params.alpha ?? 2, 1e-3); const rate = Math.max(params.beta ?? 1, 1e-6);
    const xMax = (alpha + 4 * Math.sqrt(alpha)) / rate;
    const x = linspace(1e-6, xMax, N);
    const unnorm = x.map(xi => xi <= 0 ? 0 : Math.exp((alpha - 1) * Math.log(xi) - rate * xi));
    return { x, y: normalizeDensity(x, unnorm) };
  }
  if (dist === 'truncated_normal') {
    const mu = params.mu ?? 0; const sigma = Math.max(params.sigma ?? 1, 1e-6); const lower = params.lower ?? 0;
    const xMax = Math.max(mu + 4 * sigma, lower + 0.1);
    const x = linspace(lower, xMax, N);
    return { x, y: normalizeDensity(x, x.map(xi => normalPDF(xi, mu, sigma))) };
  }
  if (dist === 'half_student_t') {
    const nu = Math.max(params.nu ?? 3, 0.5); const sigma = Math.max(params.sigma ?? 1, 1e-6);
    const x = linspace(0, sigma * 7, N);
    const unnorm = x.map(xi => Math.pow(1 + (xi / sigma) ** 2 / nu, -(nu + 1) / 2));
    return { x, y: normalizeDensity(x, unnorm) };
  }
  return { x: [], y: [] };
}

// --- Distribution parameter definitions ---

type DistKey = 'normal' | 'half_normal' | 'log_normal' | 'gamma' | 'beta' | 'truncated_normal' | 'half_student_t';

interface ParamDef { key: string; label: string; min: number; max: number; step: number; default: number }
interface DistDef { label: string; params: ParamDef[] }

const DIST_DEFS: Record<DistKey, DistDef> = {
  normal:            { label: 'Normal',           params: [{ key: 'mu', label: 'μ', min: -10, max: 10, step: 0.1, default: 0 }, { key: 'sigma', label: 'σ', min: 0.01, max: 10, step: 0.1, default: 1 }] },
  half_normal:       { label: 'Half-Normal',       params: [{ key: 'sigma', label: 'σ', min: 0.01, max: 10, step: 0.1, default: 1 }] },
  log_normal:        { label: 'Log-Normal',        params: [{ key: 'mu', label: 'μ (log)', min: -5, max: 5, step: 0.1, default: 0 }, { key: 'sigma', label: 'σ (log)', min: 0.01, max: 5, step: 0.1, default: 1 }] },
  gamma:             { label: 'Gamma',             params: [{ key: 'alpha', label: 'α (shape)', min: 0.1, max: 20, step: 0.1, default: 2 }, { key: 'beta', label: 'β (rate)', min: 0.01, max: 10, step: 0.1, default: 1 }] },
  beta:              { label: 'Beta',              params: [{ key: 'alpha', label: 'α', min: 0.1, max: 20, step: 0.1, default: 2 }, { key: 'beta', label: 'β', min: 0.1, max: 20, step: 0.1, default: 2 }] },
  truncated_normal:  { label: 'Truncated Normal',  params: [{ key: 'mu', label: 'μ', min: -10, max: 10, step: 0.1, default: 0 }, { key: 'sigma', label: 'σ', min: 0.01, max: 10, step: 0.1, default: 1 }, { key: 'lower', label: 'Lower bound', min: -10, max: 10, step: 0.1, default: 0 }] },
  half_student_t:    { label: 'Half-Student t',    params: [{ key: 'nu', label: 'ν (df)', min: 1, max: 30, step: 0.5, default: 3 }, { key: 'sigma', label: 'σ', min: 0.01, max: 10, step: 0.1, default: 1 }] },
};

// Allowed distributions by prior type
const POSITIVE_DISTS: DistKey[] = ['half_normal', 'half_student_t', 'log_normal', 'gamma', 'truncated_normal'];
const UNIT_DISTS: DistKey[]     = ['beta', 'truncated_normal'];
const ANY_DISTS: DistKey[]      = ['normal', 'half_normal', 'log_normal', 'gamma', 'beta', 'truncated_normal', 'half_student_t'];

interface PriorValue { distribution: string; params: Record<string, number> }

// --- DensitySparkline ---

function DensitySparkline({ x, y, color = '#6366f1' }: { x: number[]; y: number[]; color?: string }) {
  const valid = x.length >= 2 && y.length >= 2
    && x.every(v => isFinite(v)) && y.every(v => isFinite(v))
    && Math.max(...y) > 0;
  if (!valid) return <div className="h-[52px] bg-gray-50 rounded flex items-center justify-center text-[10px] text-gray-300">no preview</div>;
  const W = 280; const H = 52;
  const maxY = Math.max(...y, 1e-12);
  const minX = x[0]; const rangeX = x[x.length - 1] - x[0] || 1;
  const pts = x.map((xi, i) => {
    const px = ((xi - minX) / rangeX) * W;
    const py = H - (y[i] / maxY) * (H - 4) - 2;
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');
  const fill = `0,${H} ${pts} ${W},${H}`;
  const xFmt = (v: number) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
  return (
    <div className="w-full">
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="overflow-visible">
        <polyline points={fill} fill={color} fillOpacity={0.12} stroke="none" />
        <polyline points={pts} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="flex justify-between text-[9px] text-gray-400 mt-0.5 px-0.5">
        <span>{xFmt(x[0])}</span>
        <span>{xFmt(x[Math.floor(x.length / 2)])}</span>
        <span>{xFmt(x[x.length - 1])}</span>
      </div>
    </div>
  );
}

// --- PriorEditor ---

interface PriorEditorProps {
  label: string;
  hint: string;
  value: PriorValue;
  onChange: (v: PriorValue) => void;
  disabled?: boolean;
  allowed?: DistKey[];
  color?: string;
}

function PriorEditor({ label, hint, value, onChange, disabled, allowed = ANY_DISTS, color = '#6366f1' }: PriorEditorProps) {
  const dist = value.distribution as DistKey;
  const defn = DIST_DEFS[dist] ?? DIST_DEFS.half_normal;
  const density = useMemo(() => computeDensity(dist, value.params), [dist, value.params]);

  const changeDist = (newDist: string) => {
    const d = DIST_DEFS[newDist as DistKey];
    onChange({ distribution: newDist, params: Object.fromEntries(d.params.map(p => [p.key, p.default])) });
  };

  const changeParam = (key: string, val: number) =>
    onChange({ ...value, params: { ...value.params, [key]: val } });

  const selectCls = 'bg-gray-50 border border-gray-200 rounded-lg px-2 py-1 text-xs text-gray-800 cursor-pointer focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition-all';

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-3 space-y-2">
      {/* Row 1: label + distribution select */}
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold text-gray-800 truncate">{label}</p>
        <select
          className={selectCls + ' w-36 shrink-0'}
          value={dist}
          disabled={disabled}
          onChange={e => changeDist(e.target.value)}
        >
          {allowed.map(k => <option key={k} value={k}>{DIST_DEFS[k].label}</option>)}
        </select>
      </div>

      {/* Row 2: hint */}
      <p className="text-[10px] text-gray-400 leading-snug">{hint}</p>

      {/* Row 3: parameter inputs */}
      <div className="flex gap-2 flex-wrap">
        {defn.params.map(p => (
          <div key={p.key} className="flex-1 min-w-[70px]">
            <FLabel>{p.label}</FLabel>
            <input className={iCls} type="number" min={p.min} max={p.max} step={p.step}
              value={value.params[p.key] ?? p.default}
              disabled={disabled}
              onChange={e => changeParam(p.key, Number(e.target.value))} />
          </div>
        ))}
      </div>

      {/* Row 4: density sparkline */}
      <DensitySparkline x={density.x} y={density.y} color={color} />
    </div>
  );
}

// --- Prior defaults per context ---

const PRIOR_DEFAULTS = {
  media_coefficient: { distribution: 'half_normal', params: { sigma: 2.0 } },
  adstock_alpha:     { distribution: 'beta',        params: { alpha: 1.0, beta: 3.0 } },
  sat_kappa:         { distribution: 'beta',        params: { alpha: 2.0, beta: 2.0 } },
  sat_slope:         { distribution: 'half_normal', params: { sigma: 1.5 } },
  control_coef:      { distribution: 'normal',      params: { mu: 0.0, sigma: 1.0 } },
};

function initPriors(spec: any): any {
  const media: Record<string, any> = {};
  for (const ch of (spec?.media_channels ?? [])) {
    const name = ch.name;
    const existing = spec?.priors?.media?.[name] ?? {};
    media[name] = {
      coefficient:      existing.coefficient      ?? { ...PRIOR_DEFAULTS.media_coefficient },
      adstock_alpha:    existing.adstock_alpha    ?? { ...PRIOR_DEFAULTS.adstock_alpha },
      saturation_kappa: existing.saturation_kappa ?? { ...PRIOR_DEFAULTS.sat_kappa },
      saturation_slope: existing.saturation_slope ?? { ...PRIOR_DEFAULTS.sat_slope },
    };
  }

  const controls: Record<string, any> = {};
  for (const cv of (spec?.control_variables ?? [])) {
    const name = cv.name;
    const existing = spec?.priors?.controls?.[name] ?? {};
    controls[name] = {
      coefficient:    existing.coefficient  ?? { ...PRIOR_DEFAULTS.control_coef },
      allow_negative: existing.allow_negative ?? true,
    };
  }

  const trendType = spec?.trend?.type ?? 'linear';
  const existingTrend = spec?.priors?.trend ?? {};
  const trend = {
    growth_prior_mu:            existingTrend.growth_prior_mu            ?? 0.0,
    growth_prior_sigma:         existingTrend.growth_prior_sigma         ?? 0.1,
    changepoint_prior_scale:    existingTrend.changepoint_prior_scale    ?? 0.05,
    spline_prior_sigma:         existingTrend.spline_prior_sigma         ?? 1.0,
    gp_lengthscale_prior_mu:    existingTrend.gp_lengthscale_prior_mu    ?? 0.3,
    gp_lengthscale_prior_sigma: existingTrend.gp_lengthscale_prior_sigma ?? 0.2,
    gp_amplitude_prior_sigma:   existingTrend.gp_amplitude_prior_sigma   ?? 0.5,
    _type: trendType,
  };

  return { media, controls, trend };
}

// ─── PriorConfigWidget ────────────────────────────────────────────────────────

interface PriorConfigWidgetProps { spec: any; editable: boolean; onApplySpec: (s: any) => void }

function PriorConfigWidget({ spec, editable, onApplySpec }: PriorConfigWidgetProps) {
  const [priors, setPriors] = useState(() => initPriors(spec));
  const [tab, setTab] = useState<'media' | 'controls' | 'trend'>('media');
  const [openChannel, setOpenChannel] = useState<string | null>(null);
  const [openControl, setOpenControl] = useState<string | null>(null);

  // Re-sync when spec changes (new channels added via ModelSpecWidget)
  useEffect(() => { setPriors(initPriors(spec)); }, [spec?.media_channels?.length, spec?.control_variables?.length, spec?.trend?.type]);

  const setMediaPrior = (channel: string, key: string, val: PriorValue) =>
    setPriors((p: any) => ({ ...p, media: { ...p.media, [channel]: { ...p.media[channel], [key]: val } } }));

  const setControlPrior = (control: string, key: string, val: any) =>
    setPriors((p: any) => ({ ...p, controls: { ...p.controls, [control]: { ...p.controls[control], [key]: val } } }));

  const setTrendPrior = (key: string, val: any) =>
    setPriors((p: any) => ({ ...p, trend: { ...p.trend, [key]: val } }));

  const handleApply = () => {
    const { _type, ...trendPriors } = priors.trend;
    onApplySpec({ ...spec, priors: { media: priors.media, controls: priors.controls, trend: trendPriors } });
  };

  const channels = spec?.media_channels ?? [];
  const ctrls = spec?.control_variables ?? [];
  const trendType = spec?.trend?.type ?? 'linear';

  const TABS = [
    { key: 'media' as const,    label: `Media (${channels.length})` },
    { key: 'controls' as const, label: `Controls (${ctrls.length})` },
    { key: 'trend' as const,    label: 'Trend' },
  ];

  const adstockType = (chName: string) =>
    (spec?.media_channels?.find((c: any) => c.name === chName)?.adstock?.type ?? 'geometric').toLowerCase();
  const satType = (chName: string) =>
    (spec?.media_channels?.find((c: any) => c.name === chName)?.saturation?.type ?? 'hill').toLowerCase();

  const content = (
    <div className="space-y-3">
      {/* Tab bar */}
      <div className="flex gap-1.5 flex-wrap">
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${tab === t.key ? 'bg-fuchsia-600 text-white border-fuchsia-600' : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'}`}>
            {t.label}
          </button>
        ))}
        {editable && (
          <button onClick={handleApply}
            className="ml-auto flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg transition-colors">
            <Check size={12} /> Apply Priors
          </button>
        )}
      </div>

      {/* ── Media tab ─────────────────────────────────────────────────────── */}
      {tab === 'media' && (
        <div className="space-y-2">
          {channels.length === 0 && <p className="text-xs text-gray-400 italic py-2">No media channels configured yet.</p>}
          {channels.map((ch: any) => {
            const isOpen = openChannel === ch.name;
            const aSat = satType(ch.name);
            const aAds = adstockType(ch.name);
            const chPriors = priors.media[ch.name] ?? {};
            return (
              <div key={ch.name} className="border border-gray-200 rounded-xl overflow-hidden">
                <button onClick={() => setOpenChannel(isOpen ? null : ch.name)}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left">
                  <Zap size={13} className="text-indigo-500 shrink-0" />
                  <span className="flex-1 text-xs font-bold text-gray-700">{ch.name}</span>
                  <div className="flex gap-1.5 mr-2">
                    <Badge label={aAds} color="indigo" />
                    <Badge label={aSat} color="blue" />
                  </div>
                  {isOpen ? <ChevronDown size={13} className="text-gray-400" /> : <ChevronRight size={13} className="text-gray-400" />}
                </button>
                {isOpen && (
                  <div className="px-4 py-3 space-y-3 bg-white">
                    <PriorEditor
                      label="Channel Coefficient" hint="Scale of this channel's contribution to the KPI. Use Half-Normal to enforce positivity."
                      value={chPriors.coefficient ?? PRIOR_DEFAULTS.media_coefficient}
                      onChange={v => setMediaPrior(ch.name, 'coefficient', v)}
                      disabled={!editable} allowed={POSITIVE_DISTS} color="#6366f1"
                    />
                    {aAds !== 'none' && (
                      <PriorEditor
                        label="Adstock Decay (α)" hint="Decay rate of advertising carryover. Beta(1,3) favours fast decay; Beta(3,1) favours slow decay."
                        value={chPriors.adstock_alpha ?? PRIOR_DEFAULTS.adstock_alpha}
                        onChange={v => setMediaPrior(ch.name, 'adstock_alpha', v)}
                        disabled={!editable} allowed={UNIT_DISTS} color="#10b981"
                      />
                    )}
                    {(aSat === 'hill' || aSat === 'logistic') && (
                      <PriorEditor
                        label="Saturation κ (half-saturation)" hint="Spend level at which 50% of max effect is reached, relative to observed range."
                        value={chPriors.saturation_kappa ?? PRIOR_DEFAULTS.sat_kappa}
                        onChange={v => setMediaPrior(ch.name, 'saturation_kappa', v)}
                        disabled={!editable} allowed={UNIT_DISTS} color="#f59e0b"
                      />
                    )}
                    {aSat !== 'none' && (
                      <PriorEditor
                        label="Saturation Slope (steepness)" hint="How steeply the response curve rises. Larger σ allows more extreme saturation shapes."
                        value={chPriors.saturation_slope ?? PRIOR_DEFAULTS.sat_slope}
                        onChange={v => setMediaPrior(ch.name, 'saturation_slope', v)}
                        disabled={!editable} allowed={POSITIVE_DISTS} color="#ef4444"
                      />
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Controls tab ──────────────────────────────────────────────────── */}
      {tab === 'controls' && (
        <div className="space-y-2">
          {ctrls.length === 0 && <p className="text-xs text-gray-400 italic py-2">No control variables configured yet.</p>}
          {ctrls.map((cv: any) => {
            const isOpen = openControl === cv.name;
            const cvPriors = priors.controls[cv.name] ?? {};
            const allowNeg = cvPriors.allow_negative ?? true;
            return (
              <div key={cv.name} className="border border-gray-200 rounded-xl overflow-hidden">
                <button onClick={() => setOpenControl(isOpen ? null : cv.name)}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left">
                  <Layers size={13} className="text-gray-500 shrink-0" />
                  <span className="flex-1 text-xs font-bold text-gray-700">{cv.name}</span>
                  <Badge label={allowNeg ? 'any sign' : 'positive only'} color={allowNeg ? 'gray' : 'green'} />
                  {isOpen ? <ChevronDown size={13} className="text-gray-400" /> : <ChevronRight size={13} className="text-gray-400" />}
                </button>
                {isOpen && (
                  <div className="px-4 py-3 space-y-3 bg-white">
                    <div className="flex items-center gap-3 py-1">
                      <span className="text-xs text-gray-600 font-medium">Allow negative coefficient</span>
                      <button onClick={() => editable && setControlPrior(cv.name, 'allow_negative', !allowNeg)}
                        className={`relative w-9 h-5 rounded-full transition-colors ${allowNeg ? 'bg-indigo-500' : 'bg-gray-300'} ${!editable ? 'cursor-default opacity-60' : 'cursor-pointer'}`}>
                        <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${allowNeg ? 'translate-x-4' : 'translate-x-0.5'}`} />
                      </button>
                    </div>
                    <PriorEditor
                      label="Coefficient Prior" hint={allowNeg ? 'Normal prior centred at zero for controls with uncertain direction.' : 'Half-Normal for controls expected to have positive-only effects.'}
                      value={cvPriors.coefficient ?? PRIOR_DEFAULTS.control_coef}
                      onChange={v => setControlPrior(cv.name, 'coefficient', v)}
                      disabled={!editable} allowed={allowNeg ? ANY_DISTS : POSITIVE_DISTS} color="#6366f1"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Trend tab ─────────────────────────────────────────────────────── */}
      {tab === 'trend' && (
        <div className="space-y-3">
          {(trendType === 'linear' || trendType === 'none') && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <FLabel>Growth Prior μ</FLabel>
                  <input className={iCls} type="number" step={0.01}
                    value={priors.trend.growth_prior_mu}
                    disabled={!editable}
                    onChange={e => setTrendPrior('growth_prior_mu', Number(e.target.value))} />
                  <p className="text-[10px] text-gray-400 mt-0.5">Expected average growth rate. Use 0 for no expected trend.</p>
                </div>
                <div>
                  <FLabel>Growth Prior σ</FLabel>
                  <input className={iCls} type="number" step={0.01} min={0.001}
                    value={priors.trend.growth_prior_sigma}
                    disabled={!editable}
                    onChange={e => setTrendPrior('growth_prior_sigma', Number(e.target.value))} />
                  <p className="text-[10px] text-gray-400 mt-0.5">Uncertainty in growth rate. Smaller = tighter prior.</p>
                </div>
              </div>
              {(() => {
                const { x, y } = computeDensity('normal', { mu: priors.trend.growth_prior_mu, sigma: priors.trend.growth_prior_sigma });
                return (
                  <div className="bg-white rounded-xl border border-gray-200 p-3">
                    <p className="text-[10px] text-gray-500 font-semibold mb-1.5">Growth Rate Prior Distribution</p>
                    <DensitySparkline x={x} y={y} color="#6366f1" />
                  </div>
                );
              })()}
            </div>
          )}

          {trendType === 'piecewise' && (
            <div>
              <FLabel>Changepoint Prior Scale</FLabel>
              <input className={iCls} type="number" step={0.001} min={0.001} max={1}
                value={priors.trend.changepoint_prior_scale}
                disabled={!editable}
                onChange={e => setTrendPrior('changepoint_prior_scale', Number(e.target.value))} />
              <p className="text-[10px] text-gray-400 mt-0.5">Controls how sharply the trend can change at each changepoint. Smaller = smoother.</p>
            </div>
          )}

          {trendType === 'spline' && (
            <div>
              <FLabel>Spline Coefficient Prior σ</FLabel>
              <input className={iCls} type="number" step={0.1} min={0.01}
                value={priors.trend.spline_prior_sigma}
                disabled={!editable}
                onChange={e => setTrendPrior('spline_prior_sigma', Number(e.target.value))} />
              <p className="text-[10px] text-gray-400 mt-0.5">Controls how far spline coefficients can deviate from zero. Larger = more flexible trend.</p>
            </div>
          )}

          {trendType === 'gaussian_process' && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <FLabel>Lengthscale μ</FLabel>
                  <input className={iCls} type="number" step={0.05} min={0.01}
                    value={priors.trend.gp_lengthscale_prior_mu}
                    disabled={!editable}
                    onChange={e => setTrendPrior('gp_lengthscale_prior_mu', Number(e.target.value))} />
                </div>
                <div>
                  <FLabel>Lengthscale σ</FLabel>
                  <input className={iCls} type="number" step={0.05} min={0.01}
                    value={priors.trend.gp_lengthscale_prior_sigma}
                    disabled={!editable}
                    onChange={e => setTrendPrior('gp_lengthscale_prior_sigma', Number(e.target.value))} />
                </div>
              </div>
              <div>
                <FLabel>Amplitude Prior σ</FLabel>
                <input className={iCls} type="number" step={0.05} min={0.01}
                  value={priors.trend.gp_amplitude_prior_sigma}
                  disabled={!editable}
                  onChange={e => setTrendPrior('gp_amplitude_prior_sigma', Number(e.target.value))} />
                <p className="text-[10px] text-gray-400 mt-0.5">Controls the overall magnitude of the GP trend component.</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  return (
    <DashWidget
      title="Prior Configuration"
      icon={<Calendar size={15} className="text-fuchsia-500 shrink-0" />}
      color="fuchsia"
      expandTitle="Prior Configuration"
      expandContent={<div className="max-w-3xl mx-auto py-2">{content}</div>}
    >
      {content}
    </DashWidget>
  );
}

// ─── DecompositionWidget ──────────────────────────────────────────────────────

function DecompositionWidget({ decomposition }: { decomposition: Array<{ component: string; total_contribution: number; pct_of_total: number }> }) {
  const sorted = [...decomposition].sort((a, b) => b.pct_of_total - a.pct_of_total);
  const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#06b6d4', '#84cc16'];

  const barLayout = applyLightModeLayout({
    xaxis: { title: { text: '% of Total KPI' }, range: [0, 100], ticksuffix: '%' },
    yaxis: { autorange: 'reversed', tickfont: { size: 11 } },
    margin: { l: 120, r: 30, t: 30, b: 50 },
    showlegend: false,
  });

  return (
    <DashWidget
      title="Component Decomposition"
      icon={<Layers size={15} className="text-emerald-500 shrink-0" />}
      color="emerald"
      expandContent={
        <Plot
          data={[{
            type: 'bar', orientation: 'h',
            x: sorted.map(d => +(d.pct_of_total * 100).toFixed(1)),
            y: sorted.map(d => d.component),
            text: sorted.map(d => `${(d.pct_of_total * 100).toFixed(1)}%`),
            textposition: 'outside',
            textfont: { color: '#374151', size: 12 },
            marker: { color: sorted.map((_, i) => COLORS[i % COLORS.length]) },
          }]}
          layout={{ ...barLayout, autosize: true }}
          useResizeHandler style={{ width: '100%', height: '420px' }}
          config={{ responsive: true, displayModeBar: false }}
        />
      }
    >
      <div className="space-y-2">
        {sorted.map((d, i) => (
          <div key={d.component} className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
            <span className="text-xs text-gray-700 flex-1 font-medium">{d.component}</span>
            <div className="flex-1 max-w-[120px] bg-gray-100 rounded-full h-1.5 overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${(d.pct_of_total * 100).toFixed(1)}%`, backgroundColor: COLORS[i % COLORS.length] }} />
            </div>
            <span className="text-xs font-semibold text-gray-900 w-10 text-right">{(d.pct_of_total * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </DashWidget>
  );
}

// ─── Python output helpers ────────────────────────────────────────────────────

interface PythonOutput {
  id: string;
  code: string;
  output: string;
  hasError: boolean;
  plotCount: number;
}

function extractPythonOutput(raw: string): string {
  const m = raw.match(/```(?:text|python)?\n([\s\S]*?)\n?```/);
  return m ? m[1] : raw.replace(/^###[^\n]*\n/, '').trim();
}

function pyOutputKind(text: string): 'error' | 'table' | 'text' {
  if (/Traceback \(most recent call last\)|^\w+Error:|^\w+Exception:/m.test(text)) return 'error';
  const lines = text.split('\n').filter(l => l.trim());
  if (lines.length >= 3) {
    const dataLines = lines.slice(1);
    const indexed = dataLines.filter(l => /^\s*\d+\s/.test(l)).length;
    if (indexed > dataLines.length * 0.5) return 'table';
  }
  return 'text';
}

function parseTextTable(text: string): { headers: string[]; rows: string[][] } | null {
  const lines = text.split('\n').filter(l => l.trim());
  if (lines.length < 2) return null;
  // Try to split by 2+ spaces (common pandas formatting)
  const splitLine = (l: string) => l.trim().split(/\s{2,}/).map(s => s.trim()).filter(Boolean);
  const headers = splitLine(lines[0]);
  if (headers.length < 2) return null;
  const rows = lines.slice(1).map(l => splitLine(l));
  if (rows.some(r => r.length === 0)) return null;
  return { headers, rows };
}

function PythonCodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="rounded-t-lg overflow-hidden border border-gray-700">
      <div className="flex items-center justify-between px-3 py-1.5 bg-gray-800">
        <div className="flex gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
          <span className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
        </div>
        <span className="text-[10px] text-gray-400 font-medium">Python</span>
        <button
          onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
          className="text-[10px] text-gray-500 hover:text-gray-200 transition-colors flex items-center gap-1"
        >
          {copied ? <><Check size={10} />Copied</> : 'Copy'}
        </button>
      </div>
      <div className="overflow-x-auto max-h-64 bg-[#fafafa]">
        <SyntaxHighlighter
          language="python"
          style={oneLight}
          showLineNumbers
          PreTag="div"
          customStyle={{ margin: 0, padding: '0.5rem 0', fontSize: '0.6875rem', background: '#fafafa', lineHeight: '1.25rem' }}
          lineNumberStyle={{ minWidth: '2.25em', paddingRight: '0.75em', color: '#9ca3af', userSelect: 'none' }}
          codeTagProps={{ style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' } }}
        >
          {code.replace(/\n$/, '')}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}

function PythonOutputBlock({ output, hasError }: { output: string; hasError: boolean }) {
  const kind = hasError ? 'error' : pyOutputKind(output);

  if (kind === 'error') {
    const lines = output.split('\n');
    return (
      <div className="rounded-b-lg border border-t-0 border-red-200 bg-red-50 overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-1.5 bg-red-100 border-b border-red-200">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
          <span className="text-[10px] font-semibold text-red-700 uppercase tracking-widest">Error</span>
        </div>
        <div className="overflow-x-auto max-h-48 p-3">
          <pre className="text-[11px] font-mono text-red-800 whitespace-pre leading-5">
            {lines.map((l, i) => (
              <span key={i} className={/^\w+Error:|^\w+Exception:/.test(l) ? 'font-bold text-red-900' : ''}>
                {l}{'\n'}
              </span>
            ))}
          </pre>
        </div>
      </div>
    );
  }

  if (kind === 'table') {
    const parsed = parseTextTable(output);
    if (parsed) {
      return (
        <div className="rounded-b-lg border border-t-0 border-gray-200 bg-white overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 border-b border-gray-200">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest">DataFrame</span>
            <span className="ml-auto text-[10px] text-gray-400">{parsed.rows.length} rows × {parsed.headers.length} cols</span>
          </div>
          <div className="overflow-x-auto max-h-56">
            <table className="w-full text-[11px] font-mono">
              <thead className="sticky top-0 bg-gray-50 border-b border-gray-200">
                <tr>
                  {parsed.headers.map((h, i) => (
                    <th key={i} className="px-3 py-1.5 text-right font-semibold text-gray-600 whitespace-nowrap first:text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {parsed.rows.map((row, i) => (
                  <tr key={i} className="hover:bg-blue-50/40">
                    {row.map((cell, j) => (
                      <td key={j} className="px-3 py-1 text-right text-gray-700 whitespace-nowrap first:text-left first:font-medium first:text-gray-500">{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }
  }

  // Plain terminal output
  return (
    <div className="rounded-b-lg border border-t-0 border-gray-700 bg-gray-950 overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 border-b border-gray-700">
        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Output</span>
      </div>
      <div className="overflow-x-auto max-h-56 p-3">
        <pre className="text-[11px] font-mono text-green-300 whitespace-pre leading-5">{output}</pre>
      </div>
    </div>
  );
}

// One REPL cell. Memoized + viewport-gated: the expensive SyntaxHighlighter and
// output block only mount when the cell scrolls near view, and a streaming update
// to OTHER cells (or the collapse of a sibling) never re-highlights this one.
const PythonCell = React.memo(function PythonCell({
  out,
  index,
  isCollapsed,
  onToggle,
}: {
  out: PythonOutput;
  index: number;
  isCollapsed: boolean;
  onToggle: (id: string) => void;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  const hasCode = !!out.code.trim();
  const firstLine = hasCode ? out.code.trim().split('\n')[0] : '(output only)';
  return (
    <div ref={ref} className="rounded-xl overflow-hidden shadow-sm">
      {/* Cell header */}
      <button
        onClick={() => onToggle(out.id)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 transition-colors text-left border border-gray-200 rounded-t-xl"
      >
        <span className="text-[10px] font-mono text-gray-400 shrink-0">In [{index + 1}]</span>
        <span className="flex-1 text-[11px] font-mono text-gray-600 truncate">{firstLine}</span>
        {out.hasError && <span className="text-[9px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded font-semibold">ERROR</span>}
        {out.plotCount > 0 && <span className="text-[9px] bg-fuchsia-100 text-fuchsia-600 px-1.5 py-0.5 rounded font-semibold">{out.plotCount} plot{out.plotCount > 1 ? 's' : ''}</span>}
        {isCollapsed ? <ChevronRight size={13} className="text-gray-400 shrink-0" /> : <ChevronDown size={13} className="text-gray-400 shrink-0" />}
      </button>
      {!isCollapsed && (
        <div className="border-l border-r border-b border-gray-200 rounded-b-xl overflow-hidden min-h-[2.5rem]">
          {inView ? (
            <>
              {hasCode && <PythonCodeBlock code={out.code} />}
              <PythonOutputBlock output={out.output} hasError={out.hasError} />
            </>
          ) : (
            <div className="h-10 bg-gray-50" />
          )}
        </div>
      )}
    </div>
  );
});

function PythonOutputWidget({ outputs, onClear, onExport }: { outputs: PythonOutput[]; onClear: () => void; onExport?: () => void }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggle = useCallback((id: string) => setCollapsed(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  }), []);

  if (outputs.length === 0) return null;

  return (
    <DashWidget
      title={`Python REPL (${outputs.length} run${outputs.length > 1 ? 's' : ''})`}
      dotColor="bg-emerald-500"
      color="emerald"
    >
      <div className="space-y-1 mb-2 flex items-center justify-between">
        <p className="text-xs text-gray-500">{outputs.length} execution{outputs.length > 1 ? 's' : ''} recorded this session.</p>
        <div className="flex items-center gap-3">
          {onExport && (
            <button onClick={onExport} className="text-[10px] text-gray-400 hover:text-indigo-600 flex items-center gap-1 transition-colors" title="Download this session's work as a standalone, runnable Python script">
              <Download size={11} /> Download .py
            </button>
          )}
          <button onClick={onClear} className="text-[10px] text-gray-400 hover:text-red-500 flex items-center gap-1 transition-colors">
            <Trash2 size={11} /> Clear
          </button>
        </div>
      </div>
      <div className="space-y-4">
        {outputs.map((out, idx) => (
          <PythonCell
            key={out.id}
            out={out}
            index={idx}
            isCollapsed={collapsed.has(out.id)}
            onToggle={toggle}
          />
        ))}
      </div>
    </DashWidget>
  );
}

// One chat message. Memoized so a streaming update — which fires setMessages on
// every node-step — only re-renders the message that actually changed (the one
// being streamed) instead of re-parsing markdown + re-highlighting code for the
// ENTIRE conversation each step. `pending` is computed in the parent as
// `loading && isLast`, so the global `loading` flip doesn't invalidate the
// already-rendered history (only the last bubble depends on it).
const ChatMessageBubble = React.memo(function ChatMessageBubble({
  msg,
  pending,
  onNavigate,
}: {
  msg: ChatMessage;
  pending: boolean;
  onNavigate: (tab: string) => void;
}) {
  return (
    <div className={`flex gap-3 ${msg.type === 'human' ? 'justify-end' : msg.type === 'error' ? 'justify-center' : 'justify-start'}`}>
      {msg.type === 'error' && (
        <div className="rounded-xl px-4 py-3 bg-amber-50 border border-amber-200 text-amber-800 text-sm max-w-[90%] flex gap-2 items-start">
          <span className="shrink-0 mt-0.5">⚠️</span>
          <span>{msg.content}</span>
        </div>
      )}
      {msg.type !== 'error' && msg.type === 'ai' && (
        <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 mt-1">
          <Bot size={16} className="text-white" />
        </div>
      )}
      {msg.type !== 'error' && (
        <div className="max-w-[82%] flex flex-col gap-1">
          {msg.type === 'ai' && msg.toolCalls && msg.toolCalls.length > 0 && (
            <div className="space-y-1">
              {msg.toolCalls.map(tc => (
                <ToolCallBlock key={tc.id} toolCall={tc} onNavigate={onNavigate} />
              ))}
            </div>
          )}
          {(msg.content || (pending && msg.type === 'ai')) && (
            <div className={`rounded-2xl p-4 ${msg.type === 'human'
              ? 'bg-blue-600 text-white rounded-br-none'
              : 'bg-white text-gray-800 rounded-bl-none border border-gray-200 shadow-sm'}`}>
              {msg.type === 'human'
                ? <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                : <div className="prose prose-sm max-w-none text-sm">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                      {msg.content || (pending ? 'Thinking…' : '')}
                    </ReactMarkdown>
                  </div>}
            </div>
          )}
        </div>
      )}
      {msg.type === 'human' && (
        <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center shrink-0 mt-1">
          <User size={16} className="text-white" />
        </div>
      )}
    </div>
  );
});

// ─── Session + Artifact types ────────────────────────────────────────────────

interface Session {
  thread_id: string;
  name: string;
  created_at: number;
  updated_at: number;
}

interface Artifact {
  id: string;
  thread_id: string;
  kind: 'code_snippet' | 'report' | 'model_run' | 'project_report' | 'project_slides' | 'text_output' | string;
  payload: any;
  created_at: number;
}

// ─── Project + KB + Workspace types ──────────────────────────────────────────

interface Project {
  project_id: string;
  name: string;
  description?: string | null;
  session_count?: number;
  doc_count?: number;
  created_at: number;
  updated_at: number;
}

interface KbDocument {
  id: string;
  name: string;
  kind: string;
  size_bytes: number | null;
  n_chunks: number;
  status: 'pending' | 'ready' | 'error' | string;
  created_at: number;
}

interface KbSearchResult {
  document: string;
  chunk_index: number;
  text: string;
  score: number;
}

interface WorkspaceFile {
  id: string;
  name: string;
  path: string;
  kind: string;
  size_bytes: number | null;
  created_at: number;
}

// Single source of truth from the API client: relative "/api" in dev (proxied),
// or VITE_API_URL when set. Keeps all fetch() calls on the same origin as the app.
const API_BASE = API_BASE_URL;

function authHeaders(apiKey: string | null, modelName: string | null): HeadersInit {
  const h: Record<string, string> = { 'X-API-Key': apiKey || '', 'X-Model-Name': modelName || '' };
  // Optional provider/base-url overrides; harmless for non-chat routes.
  const baseUrl = (typeof localStorage !== 'undefined' && localStorage.getItem('mmm_base_url')) || '';
  const provider = (typeof localStorage !== 'undefined' && localStorage.getItem('mmm_provider')) || '';
  if (baseUrl) h['X-Base-Url'] = baseUrl;
  if (provider) h['X-Provider'] = provider;
  return h;
}

function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

// ─── ProjectPicker (in SessionSidebar) ───────────────────────────────────────

function ProjectPicker({ projects, projectId, onSelect, onCreate }: {
  projects: Project[];
  projectId: string | null;
  onSelect: (id: string) => void;
  onCreate: (name: string, description?: string) => Promise<void> | void;
}) {
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);

  // Degraded mode: /projects unavailable. Hide the picker entirely.
  if (projects.length === 0 && projectId == null) return null;

  const submit = async () => {
    const n = name.trim();
    if (!n) return;
    setBusy(true);
    try { await onCreate(n); setName(''); setCreating(false); }
    finally { setBusy(false); }
  };

  return (
    <div className="px-3 py-2.5 border-b border-gray-200 bg-gray-50/60">
      <div className="flex items-center gap-1 mb-1.5">
        <FolderOpen size={12} className="text-indigo-500 shrink-0" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider flex-1">Project</span>
        <button
          onClick={() => setCreating(v => !v)}
          className="p-1 rounded hover:bg-indigo-50 text-indigo-600"
          title="New project"
        ><Plus size={12} /></button>
      </div>
      {creating ? (
        <div className="flex items-center gap-1">
          <input
            autoFocus value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') submit(); if (e.key === 'Escape') { setCreating(false); setName(''); } }}
            placeholder="Project name…"
            disabled={busy}
            className="flex-1 text-xs bg-white border border-indigo-300 rounded px-1.5 py-1 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          />
          <button onClick={submit} disabled={busy || !name.trim()}
            className="p-1 rounded bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40" title="Create">
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
          </button>
        </div>
      ) : (
        <select
          value={projectId ?? ''}
          onChange={e => onSelect(e.target.value)}
          className="w-full text-xs bg-white border border-gray-200 rounded-lg px-2 py-1.5 text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {projects.map(p => (
            <option key={p.project_id} value={p.project_id}>
              {p.name}{typeof p.session_count === 'number' ? ` (${p.session_count})` : ''}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

// ─── SessionSidebar ──────────────────────────────────────────────────────────

function SessionSidebar({
  sessions, activeId, onSelect, onCreate, onRename, onDelete, collapsed, onToggle,
  projects, projectId, onProjectSelect, onProjectCreate,
}: {
  sessions: Session[]; activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
  collapsed: boolean;
  onToggle: () => void;
  projects: Project[];
  projectId: string | null;
  onProjectSelect: (id: string) => void;
  onProjectCreate: (name: string, description?: string) => Promise<void> | void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');

  if (collapsed) {
    return (
      <div className="w-12 border-r border-gray-200 bg-white flex flex-col items-center py-3 gap-2 shrink-0">
        <button onClick={onToggle} className="p-2 rounded-lg hover:bg-gray-100 text-gray-500" title="Show sessions">
          <MessagesSquare size={16} />
        </button>
        <button onClick={onCreate} className="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white" title="New session">
          <Plus size={16} />
        </button>
      </div>
    );
  }

  return (
    <div className="w-56 border-r border-gray-200 bg-white flex flex-col shrink-0">
      <ProjectPicker
        projects={projects}
        projectId={projectId}
        onSelect={onProjectSelect}
        onCreate={onProjectCreate}
      />
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Sessions</span>
        <div className="flex items-center gap-1">
          <button onClick={onCreate} className="p-1.5 rounded-md hover:bg-indigo-50 text-indigo-600" title="New session">
            <Plus size={14} />
          </button>
          <button onClick={onToggle} className="p-1.5 rounded-md hover:bg-gray-100 text-gray-400" title="Collapse">
            <ChevronRight size={14} className="rotate-180" />
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto py-2">
        {sessions.length === 0 && (
          <div className="px-3 py-6 text-xs text-gray-400 text-center">No sessions yet.</div>
        )}
        {sessions.map(s => {
          const active = s.thread_id === activeId;
          const isEditing = editingId === s.thread_id;
          return (
            <div key={s.thread_id} className={`group mx-2 mb-1 rounded-lg ${active ? 'bg-indigo-50 border border-indigo-200' : 'hover:bg-gray-50 border border-transparent'}`}>
              <div className="flex items-center gap-1 px-2 py-1.5">
                {isEditing ? (
                  <input
                    autoFocus value={editName}
                    onChange={e => setEditName(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') { onRename(s.thread_id, editName.trim() || s.name); setEditingId(null); }
                      if (e.key === 'Escape') setEditingId(null);
                    }}
                    onBlur={() => { onRename(s.thread_id, editName.trim() || s.name); setEditingId(null); }}
                    className="flex-1 text-xs bg-white border border-indigo-300 rounded px-1.5 py-1 focus:outline-none"
                  />
                ) : (
                  <button onClick={() => onSelect(s.thread_id)} className="flex-1 text-left text-xs text-gray-700 truncate">
                    {s.name}
                  </button>
                )}
                {!isEditing && (
                  <>
                    <button
                      onClick={() => { setEditName(s.name); setEditingId(s.thread_id); }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded text-gray-400 hover:text-indigo-600"
                      title="Rename"
                    ><Pencil size={11} /></button>
                    <button
                      onClick={() => { if (confirm(`Delete "${s.name}"?`)) onDelete(s.thread_id); }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded text-gray-400 hover:text-red-600"
                      title="Delete"
                    ><Trash2 size={11} /></button>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── ProjectDocsWidget ───────────────────────────────────────────────────────

function ProjectDocsWidget({ artifacts, onDelete }: {
  artifacts: Artifact[];
  onDelete: (id: string) => void;
}) {
  const reports = artifacts.filter(a => a.kind === 'project_report');
  const slides  = artifacts.filter(a => a.kind === 'project_slides');
  if (reports.length === 0 && slides.length === 0) return null;

  const latest = (arr: Artifact[]) => arr.sort((a, b) => b.created_at - a.created_at)[0];
  const reportArt = latest(reports);
  const slidesArt = latest(slides);

  const DocCard = ({ art, icon, label, viewUrl, downloadUrl }: {
    art: Artifact; icon: React.ReactNode; label: string;
    viewUrl: string; downloadUrl: string;
  }) => (
    <div className="flex items-center gap-3 p-4 bg-white rounded-xl border border-gray-200 shadow-sm">
      <div className="w-10 h-10 rounded-xl bg-indigo-50 flex items-center justify-center text-indigo-600 shrink-0">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-gray-900">{label}</p>
        <p className="text-xs text-gray-400 truncate">{new Date(art.created_at * 1000).toLocaleString()}</p>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <a
          href={viewUrl}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-colors"
        >
          <ExternalLink size={12} /> Open
        </a>
        <a
          href={downloadUrl}
          download
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-semibold transition-colors border border-gray-200"
        >
          <Download size={12} /> Save
        </a>
        <button
          onClick={() => onDelete(art.id)}
          className="p-1.5 rounded-lg hover:bg-red-50 text-red-400 hover:text-red-600 transition-colors"
          title="Remove"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );

  return (
    <DashWidget
      title="Project Documents"
      dotColor="bg-indigo-500"
      color="indigo"
      icon={<BookOpen size={15} className="text-indigo-600 shrink-0" />}
    >
      <div className="space-y-2.5">
        {reportArt && (
          <DocCard
            art={reportArt}
            icon={<BookOpen size={18} />}
            label="Project Report (HTML)"
            viewUrl={`${API_BASE}/project-report`}
            downloadUrl={`${API_BASE}/project-report/download`}
          />
        )}
        {slidesArt && (
          <DocCard
            art={slidesArt}
            icon={<Layers size={18} />}
            label="Presentation Slides (Reveal.js)"
            viewUrl={`${API_BASE}/project-slides`}
            downloadUrl={`${API_BASE}/project-slides/download`}
          />
        )}
        {(reports.length > 1 || slides.length > 1) && (
          <p className="text-xs text-gray-400 text-center pt-1">Showing latest version of each. Older versions available in history.</p>
        )}
      </div>
    </DashWidget>
  );
}

// ─── ModelRunsWidget ─────────────────────────────────────────────────────────

function ModelRunsWidget({
  runs,
  onLoad,
  onDelete,
}: {
  runs: Artifact[];
  onLoad: (runName: string) => void;
  onDelete: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (runs.length === 0) return null;

  return (
    <DashWidget
      title={`Model Run History (${runs.length})`}
      dotColor="bg-emerald-500"
      color="emerald"
      icon={<BarChart2 size={15} className="text-emerald-600 shrink-0" />}
    >
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full text-xs">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-3 py-2 text-left font-semibold text-emerald-700 border-b border-gray-200">Run</th>
              <th className="px-3 py-2 text-left font-semibold text-emerald-700 border-b border-gray-200">Timestamp</th>
              <th className="px-3 py-2 text-left font-semibold text-emerald-700 border-b border-gray-200">KPI</th>
              <th className="px-3 py-2 text-center font-semibold text-emerald-700 border-b border-gray-200">Channels</th>
              <th className="px-3 py-2 text-center font-semibold text-emerald-700 border-b border-gray-200">Draws</th>
              <th className="px-3 py-2 text-center font-semibold text-emerald-700 border-b border-gray-200">Actions</th>
            </tr>
          </thead>
          <tbody>
            {runs.map(a => {
              const r = a.payload ?? {};
              const isExp = expanded === a.id;
              const ts = r.timestamp_iso
                ? new Date(r.timestamp_iso).toLocaleString()
                : new Date(a.created_at * 1000).toLocaleString();
              const channels: string[] = r.channels ?? [];
              const draws = r.inference?.draws ?? r.draws ?? '—';
              return (
                <React.Fragment key={a.id}>
                  <tr
                    className="even:bg-gray-50 hover:bg-emerald-50 transition-colors cursor-pointer"
                    onClick={() => setExpanded(isExp ? null : a.id)}
                  >
                    <td className="px-3 py-2 text-gray-700 border-b border-gray-100 font-mono font-semibold">
                      <span className="flex items-center gap-1.5">
                        {isExp ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                        {r.run_name ?? a.id.slice(0, 8)}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-500 border-b border-gray-100">{ts}</td>
                    <td className="px-3 py-2 text-gray-700 border-b border-gray-100 font-medium">{r.kpi ?? '—'}</td>
                    <td className="px-3 py-2 text-center text-gray-600 border-b border-gray-100">{channels.length}</td>
                    <td className="px-3 py-2 text-center text-gray-600 border-b border-gray-100">{draws}</td>
                    <td className="px-3 py-2 text-center border-b border-gray-100">
                      <div className="flex items-center justify-center gap-1.5" onClick={e => e.stopPropagation()}>
                        <button
                          onClick={() => onLoad(r.run_name)}
                          className="px-2 py-1 rounded bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 font-semibold text-[11px] flex items-center gap-1"
                          title="Load this run into the agent"
                        >
                          <Play size={10} /> Load
                        </button>
                        <button
                          onClick={() => onDelete(a.id)}
                          className="p-1 rounded hover:bg-red-50 text-red-400 hover:text-red-600"
                          title="Remove from history"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </td>
                  </tr>
                  {isExp && (
                    <tr>
                      <td colSpan={6} className="px-4 py-3 bg-emerald-50/60 border-b border-gray-200">
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Channels</p>
                            <div className="flex flex-wrap gap-1">
                              {channels.length > 0
                                ? channels.map(ch => <span key={ch} className="px-2 py-0.5 rounded-full bg-white border border-emerald-200 text-emerald-700 text-[11px]">{ch}</span>)
                                : <span className="text-gray-400">none</span>}
                            </div>
                          </div>
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Controls</p>
                            <div className="flex flex-wrap gap-1">
                              {(r.controls ?? []).length > 0
                                ? (r.controls as string[]).map(c => <span key={c} className="px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-600 text-[11px]">{c}</span>)
                                : <span className="text-gray-400">none</span>}
                            </div>
                          </div>
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Inference</p>
                            <p className="text-gray-700 font-mono">
                              {r.inference?.chains ?? '?'} chains × {r.inference?.draws ?? '?'} draws
                              {r.inference?.tune ? ` (${r.inference.tune} tune)` : ''}
                            </p>
                          </div>
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Trend / Seasonality</p>
                            <p className="text-gray-700">{r.trend ?? '—'}</p>
                          </div>
                          {r.model_path && (
                            <div className="col-span-2">
                              <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Saved path</p>
                              <p className="text-gray-500 font-mono text-[11px] truncate">{r.model_path}</p>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </DashWidget>
  );
}

// ─── ArtifactsPanel ──────────────────────────────────────────────────────────

function ArtifactsPanel({ artifacts, onRerun, onDelete, onLoadRun }: {
  artifacts: Artifact[];
  onRerun: (a: Artifact) => void;
  onDelete: (id: string) => void;
  onLoadRun: (runName: string) => void;
}) {
  if (artifacts.length === 0) return null;
  const codeArtifacts = artifacts.filter(a => a.kind === 'code_snippet');
  const reportArtifacts = artifacts.filter(a => a.kind === 'report');
  const modelRunArtifacts = artifacts.filter(a => a.kind === 'model_run');
  const projectDocArtifacts = artifacts.filter(a => a.kind === 'project_report' || a.kind === 'project_slides');

  return (
    <div className="space-y-4">
      <ProjectDocsWidget artifacts={projectDocArtifacts} onDelete={onDelete} />
      <ModelRunsWidget runs={modelRunArtifacts} onLoad={onLoadRun} onDelete={onDelete} />
      {(codeArtifacts.length > 0 || reportArtifacts.length > 0) && (
        <DashWidget title={`Code & Reports (${codeArtifacts.length + reportArtifacts.length})`} dotColor="bg-amber-500" color="amber">
          <div className="space-y-3">
            {codeArtifacts.length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5">Code Snippets</p>
                <div className="space-y-2">
                  {codeArtifacts.map(a => {
                    const code = String(a.payload?.code ?? '');
                    return (
                      <div key={a.id} className="rounded-lg border border-gray-200 bg-gray-50 overflow-hidden">
                        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-200 bg-white">
                          <FileCode size={12} className="text-amber-600 shrink-0" />
                          <span className="text-[11px] text-gray-500 flex-1 truncate">
                            {new Date(a.created_at * 1000).toLocaleString()}
                          </span>
                          <button
                            onClick={() => navigator.clipboard.writeText(code)}
                            className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-800"
                            title="Copy"
                          ><Copy size={11} /></button>
                          <a
                            href={`${API_BASE}/artifacts/${a.id}/download`}
                            download
                            className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-800"
                            title="Download"
                          ><Download size={11} /></a>
                          <button
                            onClick={() => onRerun(a)}
                            className="p-1 rounded hover:bg-indigo-50 text-indigo-600"
                            title="Rerun"
                          ><Play size={11} /></button>
                          <button
                            onClick={() => onDelete(a.id)}
                            className="p-1 rounded hover:bg-red-50 text-red-500"
                            title="Delete"
                          ><Trash2 size={11} /></button>
                        </div>
                        <div className="overflow-x-auto max-h-40 bg-[#fafafa]">
                          <SyntaxHighlighter
                            language="python"
                            style={oneLight}
                            PreTag="div"
                            customStyle={{ margin: 0, padding: '0.5rem 0.75rem', fontSize: '0.6875rem', background: '#fafafa' }}
                            codeTagProps={{ style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' } }}
                          >
                            {truncate(code, 600)}
                          </SyntaxHighlighter>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {reportArtifacts.length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5">Reports</p>
                <div className="space-y-1.5">
                  {reportArtifacts.map(a => (
                    <div key={a.id} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
                      <ExternalLink size={12} className="text-violet-600 shrink-0" />
                      <span className="text-xs text-gray-700 flex-1 truncate font-mono">{a.payload?.path ?? a.id}</span>
                      <a
                        href={`${API_BASE}/artifacts/${a.id}/download`}
                        download
                        className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-800"
                        title="Download"
                      ><Download size={11} /></a>
                      <button onClick={() => onDelete(a.id)} className="p-1 rounded hover:bg-red-50 text-red-500" title="Delete">
                        <Trash2 size={11} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </DashWidget>
      )}
    </div>
  );
}

// ─── DatasetPanel ─────────────────────────────────────────────────────────────

interface DatasetInfo {
  rows: number;
  columns: string[];
  date_range?: { min: string; max: string } | null;
  variable_names?: string[];
  geographies?: string[];
  column_stats?: Record<string, { unique: number; top_values: { value: string; count: number }[]; truncated: boolean }>;
  active_dimensions?: string[];
}

function DatasetPanel({ dataset, threadId }: { dataset: DatasetInfo; threadId: string | null }) {
  const [selectedVar, setSelectedVar] = useState<string | null>(null);
  const [dimFilters, setDimFilters] = useState<Record<string, string>>({});
  const [series, setSeries] = useState<{ date: string; value: number }[] | null>(null);
  const [loadingSeries, setLoadingSeries] = useState(false);

  const activeDims = dataset.active_dimensions ?? [];
  const variables = dataset.variable_names ?? [];

  useEffect(() => {
    if (!selectedVar || !threadId) return;
    setLoadingSeries(true);
    const params = new URLSearchParams({ variable: selectedVar });
    const activeDimFilters = activeDims.filter(d => dimFilters[d]);
    if (activeDimFilters.length > 0) {
      params.set('dim', activeDimFilters[0]);
      params.set('value', dimFilters[activeDimFilters[0]]);
    }
    fetch(`${API_BASE}/dataset/preview/${encodeURIComponent(threadId)}?${params}`)
      .then(r => r.json())
      .then(data => { setSeries(data.series ?? null); })
      .catch(() => setSeries(null))
      .finally(() => setLoadingSeries(false));
  }, [selectedVar, JSON.stringify(dimFilters), threadId]);

  const plotData = series ? [{
    x: series.map(p => p.date),
    y: series.map(p => p.value),
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { color: '#6366f1', width: 2 },
    name: selectedVar ?? '',
  }] : [];

  const plotLayout = applyLightModeLayout({
    title: selectedVar ? `${selectedVar}${dimFilters && Object.keys(dimFilters).length ? ` — ${Object.entries(dimFilters).map(([k, v]) => `${k}: ${v}`).join(', ')}` : ''}` : '',
    xaxis: { title: 'Date' },
    yaxis: { title: 'Value' },
    height: 280,
    margin: { l: 60, r: 20, t: 40, b: 60 },
  });

  return (
    <div className="space-y-4">
      {/* Summary */}
      <DashWidget title="Dataset Summary" dotColor="bg-indigo-500" color="indigo">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-gray-50 p-3 rounded-xl border border-gray-100">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Rows</p>
            <p className="text-2xl font-bold text-gray-900">{dataset.rows.toLocaleString()}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded-xl border border-gray-100">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Columns</p>
            <p className="text-2xl font-bold text-gray-900">{dataset.columns.length}</p>
          </div>
          {dataset.date_range && (
            <div className="bg-gray-50 p-3 rounded-xl border border-gray-100 col-span-2">
              <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Date Range</p>
              <p className="text-sm font-medium text-gray-700">{dataset.date_range.min} → {dataset.date_range.max}</p>
            </div>
          )}
        </div>
      </DashWidget>

      {/* Variable names + preview */}
      {variables.length > 0 && (
        <DashWidget title={`Variables (${variables.length})`} dotColor="bg-violet-500" color="violet">
          {/* Dimension filters */}
          {activeDims.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {activeDims.map(dim => {
                const stat = dataset.column_stats?.[dim];
                const opts = stat?.top_values ?? [];
                return (
                  <div key={dim} className="flex items-center gap-1.5">
                    <span className="text-xs text-gray-500 font-medium">{dim}:</span>
                    <select
                      value={dimFilters[dim] ?? ''}
                      onChange={e => setDimFilters(prev => ({ ...prev, [dim]: e.target.value }))}
                      className="text-xs border border-gray-200 rounded-lg px-2 py-1 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-violet-400"
                    >
                      <option value="">All</option>
                      {opts.map(o => (
                        <option key={o.value} value={o.value}>{o.value}</option>
                      ))}
                    </select>
                  </div>
                );
              })}
            </div>
          )}
          {/* Variable chips */}
          <div className="flex flex-wrap gap-2 mb-3">
            {variables.map(v => (
              <button
                key={v}
                onClick={() => setSelectedVar(v === selectedVar ? null : v)}
                className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                  selectedVar === v
                    ? 'bg-violet-600 text-white border-violet-600'
                    : 'bg-white text-gray-600 border-gray-200 hover:border-violet-400 hover:text-violet-600'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
          {/* Chart */}
          {selectedVar && (
            <div className="rounded-xl overflow-hidden border border-gray-100 bg-gray-50">
              {loadingSeries ? (
                <div className="flex items-center justify-center h-[280px] text-gray-400">
                  <Loader2 size={22} className="animate-spin" />
                </div>
              ) : series ? (
                <Plot
                  data={plotData}
                  layout={{ ...plotLayout, autosize: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '280px' }}
                  config={{ responsive: true, displayModeBar: false }}
                />
              ) : (
                <div className="flex items-center justify-center h-[280px] text-gray-400 text-sm">No data</div>
              )}
            </div>
          )}
        </DashWidget>
      )}

      {/* Dimension value counts */}
      {Object.entries(dataset.column_stats ?? {}).map(([dim, stat]) => (
        <DashWidget
          key={dim}
          title={`${dim} (${stat.unique} unique${stat.truncated ? ', top 20 shown' : ''})`}
          dotColor="bg-sky-500"
          color="sky"
          defaultOpen={false}
        >
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-3 py-2 text-left font-semibold text-sky-600 border-b border-gray-200">Value</th>
                  <th className="px-3 py-2 text-right font-semibold text-sky-600 border-b border-gray-200">Count</th>
                  <th className="px-3 py-2 text-right font-semibold text-sky-600 border-b border-gray-200">%</th>
                </tr>
              </thead>
              <tbody>
                {stat.top_values.map((row, i) => {
                  const total = stat.top_values.reduce((s, r) => s + r.count, 0);
                  const pct = total > 0 ? ((row.count / total) * 100).toFixed(1) : '0.0';
                  return (
                    <tr key={i} className="even:bg-gray-50 hover:bg-sky-50 transition-colors">
                      <td className="px-3 py-1.5 text-gray-700 border-b border-gray-100 font-mono">{row.value}</td>
                      <td className="px-3 py-1.5 text-right text-gray-600 border-b border-gray-100">{row.count.toLocaleString()}</td>
                      <td className="px-3 py-1.5 text-right text-gray-400 border-b border-gray-100">{pct}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </DashWidget>
      ))}
    </div>
  );
}

// ─── WorkspaceFilesWidget (Data tab) ─────────────────────────────────────────

function WorkspaceFilesWidget({ threadId, apiKey, modelName, refreshKey }: {
  threadId: string | null;
  apiKey: string | null;
  modelName: string | null;
  refreshKey: number;
}) {
  const [files, setFiles] = useState<WorkspaceFile[]>([]);

  useEffect(() => {
    if (!threadId) { setFiles([]); return; }
    let cancelled = false;
    fetch(`${API_BASE}/workspace/${encodeURIComponent(threadId)}/files`, { headers: authHeaders(apiKey, modelName) })
      .then(r => r.json())
      .then(data => { if (!cancelled) setFiles(Array.isArray(data?.files) ? data.files : []); })
      .catch(() => { if (!cancelled) setFiles([]); });
    return () => { cancelled = true; };
  }, [threadId, apiKey, modelName, refreshKey]);

  if (files.length === 0) {
    return (
      <PanelShellLite title="Workspace Outputs" icon={<FolderOpen size={16} className="text-teal-600" />} color="teal">
        <p className="text-sm text-gray-400 italic">No generated files yet. When the agent writes reports, CSVs, or PNGs via <code className="text-xs bg-gray-100 px-1 rounded">execute_python</code>, they appear here for download.</p>
      </PanelShellLite>
    );
  }

  return (
    <PanelShellLite title={`Workspace Outputs (${files.length})`} icon={<FolderOpen size={16} className="text-teal-600" />} color="teal">
      <div className="space-y-2">
        {files.map(f => (
          <div key={f.id} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
            <FileIcon size={14} className="text-teal-600 shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className="text-sm font-semibold text-gray-800 truncate">{f.name}</span>
                <span className="text-[10px] uppercase tracking-wider text-teal-700 bg-teal-50 rounded px-1.5 py-0.5 border border-teal-200">{f.kind}</span>
                <span className="text-[10px] text-gray-400">{fmtBytes(f.size_bytes)}</span>
              </div>
              <p className="text-[11px] text-gray-400 font-mono mt-0.5 truncate">{f.path}</p>
            </div>
            <a
              href={`${API_BASE}/files/${f.id}/download`}
              download
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-semibold transition-colors border border-gray-200 shrink-0"
            >
              <Download size={12} /> Save
            </a>
          </div>
        ))}
      </div>
    </PanelShellLite>
  );
}

// Small collapsible shell (mirrors CausalWidgets PanelShell) for new widgets.
function PanelShellLite({ title, icon, color = 'gray', children }: {
  title: React.ReactNode; icon: React.ReactNode; color?: string; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <button onClick={() => setOpen(v => !v)} className="w-full flex items-center gap-3 px-5 py-4 text-left">
        {icon}
        <span className={`font-semibold text-sm text-${color}-600 flex-1`}>{title}</span>
        {open ? <ChevronDown size={15} className="text-gray-400" /> : <ChevronRight size={15} className="text-gray-400" />}
      </button>
      {open && <div className="px-5 pb-5">{children}</div>}
    </div>
  );
}

// ─── KnowledgeTab ─────────────────────────────────────────────────────────────

const KB_STATUS_STYLE: Record<string, string> = {
  ready:   'bg-emerald-50 text-emerald-700 border-emerald-200',
  pending: 'bg-amber-50 text-amber-700 border-amber-200',
  error:   'bg-red-50 text-red-700 border-red-200',
};

function KnowledgeTab({ projectId, apiKey, modelName }: {
  projectId: string | null;
  apiKey: string | null;
  modelName: string | null;
}) {
  const [docs, setDocs] = useState<KbDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<KbSearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    if (!projectId) { setDocs([]); return; }
    try {
      const data = await fetch(`${API_BASE}/projects/${projectId}/kb`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json());
      setDocs(Array.isArray(data?.documents) ? data.documents : []);
    } catch { /* leave as-is */ }
  }, [projectId, apiKey, modelName]);

  useEffect(() => {
    if (!projectId) { setDocs([]); return; }
    setLoading(true);
    fetchDocs().finally(() => setLoading(false));
  }, [projectId, fetchDocs]);

  // Poll while any document is still ingesting (pending → ready/error).
  useEffect(() => {
    if (!projectId) return;
    if (!docs.some(d => d.status === 'pending')) return;
    const t = setInterval(fetchDocs, 3000);
    return () => clearInterval(t);
  }, [projectId, docs, fetchDocs]);

  const uploadFile = useCallback(async (file: File) => {
    if (!projectId) return;
    setError(null);
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      // NOTE: do not set Content-Type — the browser sets the multipart boundary.
      const res = await fetch(`${API_BASE}/projects/${projectId}/kb`, {
        method: 'POST', headers: authHeaders(apiKey, modelName), body: fd,
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        setError(e?.detail ?? e?.error ?? `Upload failed (${res.status})`);
      }
      await fetchDocs();
    } catch {
      setError('Upload failed — is the API running?');
    } finally {
      setUploading(false);
    }
  }, [projectId, apiKey, modelName, fetchDocs]);

  const onPickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) uploadFile(f);
    if (fileRef.current) fileRef.current.value = '';
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) uploadFile(f);
  };

  const deleteDoc = async (id: string) => {
    if (!confirm('Remove this document from the knowledge base?')) return;
    await fetch(`${API_BASE}/kb/${id}`, { method: 'DELETE', headers: authHeaders(apiKey, modelName) });
    fetchDocs();
  };

  const runSearch = async () => {
    const q = query.trim();
    if (!q || !projectId) return;
    setSearching(true);
    setError(null);
    try {
      const params = new URLSearchParams({ q, k: '6' });
      const data = await fetch(`${API_BASE}/projects/${projectId}/kb/search?${params}`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json());
      setResults(Array.isArray(data?.results) ? data.results : []);
    } catch {
      setError('Search failed.');
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  if (!projectId) {
    return (
      <EmptyTabState
        icon={<BrainCircuit size={28} />}
        title="No project selected"
        hint="Select or create a project in the sidebar to manage its knowledge base."
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* Upload zone */}
      <PanelShellLite title="Knowledge Base" icon={<BrainCircuit size={16} className="text-indigo-600" />} color="indigo">
        <input ref={fileRef} type="file" className="hidden" onChange={onPickFile}
          accept=".txt,.md,.markdown,.csv,.pdf,.docx,.xlsx" />
        <div
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`flex flex-col items-center justify-center gap-2 py-8 px-4 rounded-xl border-2 border-dashed cursor-pointer transition-colors ${
            dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-gray-200 hover:border-indigo-300 hover:bg-gray-50'
          }`}
        >
          {uploading ? (
            <><Loader2 size={22} className="text-indigo-500 animate-spin" />
              <p className="text-sm text-gray-500">Uploading & ingesting…</p></>
          ) : (
            <><UploadCloud size={22} className="text-indigo-400" />
              <p className="text-sm text-gray-600 font-medium">Drop a file or click to upload</p>
              <p className="text-xs text-gray-400">txt · md · csv · pdf · docx · xlsx</p></>
          )}
        </div>
        {error && <p className="mt-2 text-xs text-red-600 bg-red-50 border border-red-200 rounded px-3 py-1.5">{error}</p>}

        {/* Document list */}
        <div className="mt-4 space-y-2">
          {loading && docs.length === 0 ? (
            <div className="flex items-center justify-center py-6 text-gray-400"><Loader2 size={18} className="animate-spin" /></div>
          ) : docs.length === 0 ? (
            <p className="text-sm text-gray-400 italic text-center py-2">No documents yet. Upload context files the agent can look up.</p>
          ) : docs.map(d => (
            <div key={d.id} className="flex items-start gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
              <FileText size={14} className="text-indigo-600 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="text-sm font-semibold text-gray-800 truncate">{d.name}</span>
                  <span className="text-[10px] uppercase tracking-wider text-indigo-700 bg-indigo-50 rounded px-1.5 py-0.5 border border-indigo-200">{d.kind}</span>
                  <span className="text-[10px] text-gray-400">{fmtBytes(d.size_bytes)}</span>
                  {d.n_chunks > 0 && <span className="text-[10px] text-gray-400">{d.n_chunks} chunk{d.n_chunks !== 1 ? 's' : ''}</span>}
                  <span className={`text-[10px] uppercase tracking-wider font-semibold rounded border px-1.5 py-0.5 ${KB_STATUS_STYLE[d.status] ?? KB_STATUS_STYLE.error}`}>
                    {d.status === 'pending' && <Loader2 size={9} className="inline animate-spin mr-1" />}
                    {d.status}
                  </span>
                </div>
              </div>
              <button onClick={() => deleteDoc(d.id)} className="p-1 rounded text-gray-300 hover:text-red-500 shrink-0" title="Delete">
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      </PanelShellLite>

      {/* Search */}
      <PanelShellLite title="Search Knowledge Base" icon={<Search size={16} className="text-teal-600" />} color="teal">
        <div className="flex items-center gap-2">
          <input
            type="text" value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && runSearch()}
            placeholder="Search the knowledge base…"
            className="flex-1 text-sm border border-gray-200 rounded-lg px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-teal-400"
          />
          <button onClick={runSearch} disabled={searching || !query.trim()}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-teal-600 text-white text-sm font-medium hover:bg-teal-500 disabled:opacity-40">
            {searching ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />} Search
          </button>
        </div>
        {results != null && (
          <div className="mt-3 space-y-2">
            {results.length === 0 ? (
              <p className="text-sm text-gray-400 italic">No matches found.</p>
            ) : results.map((r, i) => (
              <div key={i} className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2">
                <div className="flex items-baseline gap-2 mb-1">
                  <FileText size={12} className="text-teal-600 shrink-0" />
                  <span className="text-xs font-semibold text-gray-700 truncate flex-1">{r.document}</span>
                  <span className="text-[10px] text-gray-400">#{r.chunk_index}</span>
                  <span className="text-[10px] font-mono text-teal-600 bg-teal-50 rounded px-1.5 py-0.5 border border-teal-200">{r.score.toFixed(3)}</span>
                </div>
                <p className="text-xs text-gray-600 leading-relaxed whitespace-pre-wrap line-clamp-6">{r.text}</p>
              </div>
            ))}
          </div>
        )}
      </PanelShellLite>
    </div>
  );
}

// ─── EmptyTabState ────────────────────────────────────────────────────────────

function EmptyTabState({ icon, title, hint }: { icon: React.ReactNode; title: string; hint: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-gray-400 space-y-3">
      <div className="w-14 h-14 rounded-2xl bg-white flex items-center justify-center border border-gray-200 shadow-sm text-gray-300">
        {icon}
      </div>
      <p className="text-base text-gray-500 font-medium">{title}</p>
      <p className="text-sm text-gray-400 max-w-sm text-center">{hint}</p>
    </div>
  );
}

// ─── Main AgentPage ───────────────────────────────────────────────────────────

export function AgentPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [dashboardData, setDashboardData] = useState<any>({});
  const [pythonOutputs, setPythonOutputs] = useState<PythonOutput[]>([]);
  const [loading, setLoading] = useState(false);
  const [rightExpanded, setRightExpanded] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [threadId, setThreadId] = useState<string | null>(() => localStorage.getItem('mmm.activeThreadId'));
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<string | null>(null);
  // Gate session loading until /projects has resolved (success or failure) so we
  // know whether to filter by project. Degraded mode (error) → projectId stays null.
  const [projectsReady, setProjectsReady] = useState(false);
  const [workspaceRefreshKey, setWorkspaceRefreshKey] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const { apiKey, modelName } = useAuthStore();
  const causal = useCausalPanels(threadId);
  const [activeTab, setActiveTab] = useState<string>(() => localStorage.getItem('mmm.activeTab') || 'workflow');

  useEffect(() => { localStorage.setItem('mmm.activeTab', activeTab); }, [activeTab]);

  // Persist the selected project id
  useEffect(() => {
    if (projectId) localStorage.setItem('mmm.projectId', projectId);
  }, [projectId]);

  const loadProjects = useCallback(async (): Promise<Project[]> => {
    const data = await fetch(`${API_BASE}/projects`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json());
    return Array.isArray(data?.projects) ? data.projects : [];
  }, [apiKey, modelName]);

  // Effect P (mount): resolve the project list + selected project, then unlock
  // session loading. On error, degrade to a single implicit project (projectId=null).
  useEffect(() => {
    (async () => {
      try {
        const list = await loadProjects();
        setProjects(list);
        if (list.length > 0) {
          const stored = localStorage.getItem('mmm.projectId');
          const pick = stored && list.some(p => p.project_id === stored) ? stored : list[0].project_id;
          setProjectId(pick);
        } else {
          setProjectId(null);
        }
      } catch (e) {
        console.error('Failed to load projects (degrading to single implicit project)', e);
        setProjects([]);
        setProjectId(null);
      } finally {
        setProjectsReady(true);
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Resume session from ?session=<thread_id> query param (e.g. launched from Dashboard)
  useEffect(() => {
    const sessionParam = searchParams.get('session');
    if (sessionParam) setThreadId(sessionParam);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist active session
  useEffect(() => {
    if (threadId) localStorage.setItem('mmm.activeThreadId', threadId);
  }, [threadId]);

  // Load session list (filtered by project); auto-create one if none, auto-select
  // first if no active. Gated on projectsReady so the filter is known up front.
  useEffect(() => {
    if (!projectsReady) return;
    (async () => {
      try {
        const url = projectId
          ? `${API_BASE}/sessions?project_id=${encodeURIComponent(projectId)}`
          : `${API_BASE}/sessions`;
        const raw = await fetch(url).then(r => r.json());
        let list: Session[] = Array.isArray(raw) ? raw : (raw?.sessions ?? []);
        if (list.length === 0) {
          const created = await fetch(`${API_BASE}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(projectId ? { project_id: projectId } : {}),
          }).then(r => r.json());
          list = [created];
        }
        setSessions(list);
        if (!threadId || !list.some(s => s.thread_id === threadId)) {
          setThreadId(list[0].thread_id);
        }
      } catch (e) { console.error('Failed to load sessions', e); }
    })();
  }, [projectId, projectsReady]); // eslint-disable-line react-hooks/exhaustive-deps

  const refreshSessions = useCallback(async () => {
    try {
      const url = projectId
        ? `${API_BASE}/sessions?project_id=${encodeURIComponent(projectId)}`
        : `${API_BASE}/sessions`;
      const raw = await fetch(url).then(r => r.json());
      setSessions(Array.isArray(raw) ? raw : (raw?.sessions ?? []));
    } catch (e) { console.error(e); }
  }, [projectId]);

  // Project actions
  const handleProjectSelect = useCallback((id: string) => {
    setProjectId(id);
  }, []);

  const handleProjectCreate = useCallback(async (name: string, description?: string) => {
    try {
      const created: Project = await fetch(`${API_BASE}/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ name, description }),
      }).then(r => r.json());
      const list = await loadProjects().catch(() => null);
      if (list) setProjects(list);
      else setProjects(prev => [...prev, created]);
      if (created?.project_id) setProjectId(created.project_id);
    } catch (e) { console.error('Failed to create project', e); }
  }, [apiKey, modelName, loadProjects]);

  const loadThreadState = useCallback(async (tid: string) => {
    try {
      const [stateRes, artRes] = await Promise.all([
        fetch(`${API_BASE}/state/${tid}`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json()),
        fetch(`${API_BASE}/artifacts/${tid}`).then(r => r.json()),
      ]);

      // Build messages, pairing AI tool_calls with their ToolMessage results.
      const parsed: ChatMessage[] = [];
      // Maps tool_call_id → index in parsed so tool results can be stitched back in.
      const tcIdToMsgIdx: Record<string, number> = {};

      (stateRes.messages || []).forEach((m: any, i: number) => {
        if (m.type === 'human') {
          const content = normalizeContent(m.content);
          if (content) parsed.push({ id: `loaded-${tid}-${i}`, type: 'human', content });
        } else if (m.type === 'ai') {
          const content = normalizeContent(m.content);
          const toolCalls: ToolCall[] = (m.tool_calls || []).map((tc: any) => ({
            id: tc.id ?? `tc-${i}-${tc.name}`,
            name: tc.name ?? 'unknown',
            args: tc.args ?? {},
            status: 'done' as const,
            result: undefined,
          }));
          if (!content && !toolCalls.length) return;
          const msgIdx = parsed.length;
          parsed.push({ id: `loaded-${tid}-${i}`, type: 'ai', content, toolCalls });
          toolCalls.forEach(tc => { tcIdToMsgIdx[tc.id] = msgIdx; });
        } else if (m.type === 'tool') {
          // Stitch the result back into the matching tool call on the AI message.
          const tcId = m.tool_call_id;
          const msgIdx = tcId != null ? tcIdToMsgIdx[tcId] : undefined;
          if (msgIdx != null) {
            const msg = parsed[msgIdx];
            const tc = msg?.toolCalls?.find(t => t.id === tcId);
            if (tc) tc.result = normalizeContent(m.content);
          }
        }
      });

      setMessages(parsed);
      setDashboardData(stateRes.dashboard_data || {});
      const arts: Artifact[] = Array.isArray(artRes) ? artRes : [];
      setArtifacts(arts);

      // Rehydrate persisted python outputs from text_output artifacts, pairing
      // each with its code_snippet by call_id (code may be unavailable for some).
      const codeByCall: Record<string, string> = {};
      for (const a of arts) {
        if (a.kind === 'code_snippet' && a.payload?.call_id) {
          codeByCall[a.payload.call_id] = String(a.payload.code ?? '');
        }
      }
      const rehydrated: PythonOutput[] = arts
        .filter(a => a.kind === 'text_output')
        .sort((x, y) => x.created_at - y.created_at)
        .map(a => {
          const callId = String(a.payload?.call_id ?? a.id);
          const output = String(a.payload?.stdout ?? '');
          return {
            id: callId,
            code: codeByCall[callId] ?? '',
            output,
            hasError: !!a.payload?.is_error,
            plotCount: Number(a.payload?.plot_count ?? 0),
          };
        });
      setPythonOutputs(rehydrated);
    } catch (e) { console.error('Failed to load thread state', e); }
  }, [apiKey, modelName]);

  // Re-load whenever active session changes
  useEffect(() => {
    if (threadId) loadThreadState(threadId);
  }, [threadId, loadThreadState]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── Session actions ────────────────────────────────────────────────────────
  const handleCreateSession = async () => {
    const created: Session = await fetch(`${API_BASE}/sessions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(projectId ? { project_id: projectId } : {}),
    }).then(r => r.json());
    await refreshSessions();
    setThreadId(created.thread_id);
  };

  const handleRenameSession = async (id: string, name: string) => {
    await fetch(`${API_BASE}/sessions/${id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }),
    });
    refreshSessions();
  };

  const handleDeleteSession = async (id: string) => {
    await fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE' });
    const remaining = sessions.filter(s => s.thread_id !== id);
    setSessions(remaining);
    if (id === threadId) {
      if (remaining.length > 0) setThreadId(remaining[0].thread_id);
      else handleCreateSession();
    }
  };

  // ── Chat actions ───────────────────────────────────────────────────────────
  const handleClearChat = async () => {
    if (!threadId) return;
    setMessages([]);
    setDashboardData({});
    setPythonOutputs([]);
    try {
      await fetch(`${API_BASE}/state/${threadId}`, {
        method: 'DELETE', headers: authHeaders(apiKey, modelName),
      });
    } catch { /* ignore */ }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
  };

  const handleBack = async () => {
    if (!threadId || loading) return;
    // Fetch timeline, find a checkpoint before the latest human message
    try {
      const timeline: any[] = await fetch(`${API_BASE}/history/${threadId}`).then(r => r.json());
      if (!Array.isArray(timeline) || timeline.length < 2) return;
      // timeline is newest-first. Find checkpoints in chronological order
      // and rewind to the one before the latest user-visible state change.
      // Strategy: target = the checkpoint where message_count drops by ≥1
      // compared to current head — equivalent to "previous turn boundary".
      const head = timeline[0];
      let target: any = null;
      for (let i = 1; i < timeline.length; i++) {
        if (timeline[i].message_count < head.message_count) {
          target = timeline[i];
          break;
        }
      }
      if (!target) target = timeline[timeline.length - 1];
      await fetch(`${API_BASE}/rewind/${threadId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_id: target.checkpoint_id }),
      });
      await loadThreadState(threadId);
    } catch (e) { console.error('Back failed', e); }
  };

  const handleRetry = async () => {
    if (!threadId || loading) return;
    // Find the last human message in current state
    const lastHuman = [...messages].reverse().find(m => m.type === 'human');
    if (!lastHuman) return;
    try {
      const timeline: any[] = await fetch(`${API_BASE}/history/${threadId}`).then(r => r.json());
      if (!Array.isArray(timeline)) return;
      // Rewind to the checkpoint with the smallest message_count that is
      // still ≥ (messages_before_last_human). That's the state right before
      // the most recent user turn. Walk oldest→newest.
      const ordered = [...timeline].reverse(); // oldest first
      const targetCount = messages.findIndex(m => m.id === lastHuman.id);
      // Find first checkpoint whose message_count ≥ targetCount but whose
      // last_human_index < targetCount (i.e. the human is not yet in it).
      let chosen: any = null;
      for (const cp of ordered) {
        if ((cp.last_human_index ?? -1) < targetCount) chosen = cp;
        else break;
      }
      if (!chosen) chosen = ordered[0];
      await fetch(`${API_BASE}/rewind/${threadId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_id: chosen.checkpoint_id }),
      });
      await loadThreadState(threadId);
      handleSend(lastHuman.content);
    } catch (e) { console.error('Retry failed', e); }
  };

  const handleSend = async (messageOverride?: string) => {
    if (!threadId) return;
    const textToSend = messageOverride || input;
    if (!textToSend.trim()) return;

    const humanId = crypto.randomUUID();
    setMessages(prev => [...prev, { id: humanId, type: 'human', content: textToSend }]);
    if (!messageOverride) setInput('');
    setLoading(true);

    const tempAiId = crypto.randomUUID();
    setMessages(prev => [...prev, { id: tempAiId, type: 'ai', content: '', toolCalls: [] }]);
    const toolCallMap: Record<string, ToolCall> = {};

    const updateMsg = (updater: (m: ChatMessage) => ChatMessage) =>
      setMessages(prev => prev.map(m => m.id === tempAiId ? updater(m) : m));

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ message: textToSend, thread_id: threadId }),
        signal: controller.signal,
      });
      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiContent = '';
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line === 'data: [DONE]') {
            // Mark any tools still spinning as done (stream ended without their result)
            const runningKeys = Object.keys(toolCallMap).filter(k => toolCallMap[k].status === 'running');
            if (runningKeys.length > 0) {
              runningKeys.forEach(k => { toolCallMap[k] = { ...toolCallMap[k], status: 'done', result: toolCallMap[k].result ?? '' }; });
              updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
            }
            continue;
          }
          try {
            const data = JSON.parse(line.substring(6));
            if (data.dashboard_data && Object.keys(data.dashboard_data).length > 0)
              setDashboardData((prev: any) => ({ ...prev, ...data.dashboard_data }));
            if (data.type === 'dashboard_update') continue;
            if (data.type === 'error') {
              // Replace the pending AI bubble with an error notice
              setMessages(prev => prev.map(m =>
                m.id === tempAiId
                  ? { ...m, type: 'error' as const, content: data.content || 'An unknown error occurred.' }
                  : m
              ));
              continue;
            }
            if (data.type === 'ai') {
              if (Array.isArray(data.tool_calls)) {
                for (const tc of data.tool_calls) {
                  const id = tc.id || tc.name + '_' + Date.now();
                  toolCallMap[id] = { id, name: tc.name || 'unknown', args: tc.args || {}, status: 'running' };
                }
                updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
              }
              const cs = normalizeContent(data.content);
              if (cs) { aiContent += cs + '\n'; updateMsg(m => ({ ...m, content: aiContent })); }
            }
            if (data.type === 'tool') {
              const rs = normalizeContent(data.content);
              const key = data.tool_call_id && toolCallMap[data.tool_call_id]
                ? data.tool_call_id
                : Object.keys(toolCallMap).find(k => toolCallMap[k].status === 'running');
              if (key) {
                toolCallMap[key] = { ...toolCallMap[key], result: rs, status: 'done' };
                updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
                // Capture execute_python output for the right-panel REPL widget
                if (toolCallMap[key].name === 'execute_python') {
                  const code = toolCallMap[key].args?.code ?? '';
                  const output = extractPythonOutput(rs);
                  const plotMatch = rs.match(/Generated (\d+) Plotly/);
                  setPythonOutputs(prev => [...prev, {
                    id: key,
                    code,
                    output,
                    hasError: /Traceback \(most recent call last\)|^\w+Error:|^\w+Exception:/m.test(output),
                    plotCount: plotMatch ? parseInt(plotMatch[1], 10) : 0,
                  }]);
                }
              }
            }
          } catch { /* ignore parse errors */ }
        }
      }
    } catch (e: any) {
      const aborted = e?.name === 'AbortError';
      if (!aborted) console.error(e);
      const runningKeys = Object.keys(toolCallMap).filter(k => toolCallMap[k].status === 'running');
      if (runningKeys.length > 0) {
        runningKeys.forEach(k => {
          toolCallMap[k] = {
            ...toolCallMap[k],
            status: aborted ? 'done' : 'error',
            result: toolCallMap[k].result ?? (aborted ? 'Stopped by user' : 'Connection error'),
          };
        });
        updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
      }
      if (aborted) {
        updateMsg(m => ({ ...m, content: (m.content || '') + (m.content ? '\n\n' : '') + '_⏹ Stopped by user._' }));
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
      // Refresh artifacts + causal panels after the turn so newly-saved
      // snippets, assumptions, files, DAG, and workflow status all show up.
      if (threadId) {
        try {
          const arts = await fetch(`${API_BASE}/artifacts/${threadId}`).then(r => r.json());
          if (Array.isArray(arts)) setArtifacts(arts);
        } catch { /* ignore */ }
        causal.refresh();
        // Refresh workspace output files (newly generated reports/CSVs/PNGs).
        setWorkspaceRefreshKey(k => k + 1);
      }
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    setLoading(true);
    try {
      const url = threadId ? `${API_BASE}/upload?thread_id=${encodeURIComponent(threadId)}` : `${API_BASE}/upload`;
      const res = await fetch(url, {
        method: 'POST',
        headers: authHeaders(apiKey, modelName),
        body: fd,
      });
      const data = await res.json();
      if (data.path) {
        causal.refresh();
        handleSend(`I have uploaded a dataset at \`${data.path}\`. Please load it using the execute_python tool and run some basic EDA on it. Don't build a model yet.`);
      }
    } catch (e) { console.error('File upload failed', e); }
    finally { setLoading(false); if (fileInputRef.current) fileInputRef.current.value = ''; }
  };

  const modelCompleted = dashboardData.model_status === 'completed';
  const hasSpec = !!dashboardData.model_spec;

  const handleApplySpec = useCallback((newSpec: any) => {
    setDashboardData((prev: any) => ({ ...prev, model_spec: newSpec }));
    handleSend(buildApplyMessage(newSpec));
  }, [threadId]);

  const handleRerunArtifact = (a: Artifact) => {
    if (a.kind !== 'code_snippet') return;
    const code = String(a.payload?.code ?? '');
    if (!code.trim()) return;
    handleSend(
      `Please re-run this saved code snippet using \`execute_python\`:\n\n\`\`\`python\n${code}\n\`\`\``
    );
  };

  const handleDeleteArtifact = async (id: string) => {
    await fetch(`${API_BASE}/artifacts/${id}`, { method: 'DELETE' });
    setArtifacts(prev => prev.filter(a => a.id !== id));
  };

  const handleLoadRun = (runName: string) => {
    if (!runName) return;
    handleSend(`Please load the fitted model from run \`${runName}\` using \`load_fitted_model\`.`);
  };

  const hasDecomp = dashboardData.decomposition?.length > 0;
  const lastAssistantHasContent = messages.length > 0 && messages[messages.length - 1].type === 'ai';
  const canRetry = !loading && messages.some(m => m.type === 'human');
  const canBack = !loading && messages.length >= 2;

  const activeSession = sessions.find(s => s.thread_id === threadId);

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-900 overflow-hidden font-sans">
      {/* ── Shared top bar (matches main app Header) ── */}
      <header className="flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm sm:px-6 z-20">
        <button
          onClick={() => navigate('/dashboard')}
          className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-indigo-600 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          <span className="hidden sm:inline">Dashboard</span>
        </button>
        <div className="h-6 w-px bg-gray-200" />
        <h1 className="text-lg font-semibold text-gray-900 flex-1">MMM Chat</h1>
        <span className="text-sm text-gray-400 truncate max-w-48 hidden md:block">
          {activeSession?.name ?? ''}
        </span>
        <div className="h-6 w-px bg-gray-200 hidden md:block" />
        <ModelSwitcher theme="light" />
      </header>

      {/* ── Panel row ── */}
      <div className="flex flex-1 overflow-hidden">
      <SessionSidebar
        sessions={sessions}
        activeId={threadId}
        onSelect={setThreadId}
        onCreate={handleCreateSession}
        onRename={handleRenameSession}
        onDelete={handleDeleteSession}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(v => !v)}
        projects={projects}
        projectId={projectId}
        onProjectSelect={handleProjectSelect}
        onProjectCreate={handleProjectCreate}
      />

      {/* ── Left: Chat (1/3 width) ── */}
      {!rightExpanded && (
        <div className="w-1/3 border-r border-gray-200 flex flex-col bg-white shadow-md relative z-10 shrink-0">
          <div className="p-4 border-b border-gray-200 bg-white sticky top-0 flex items-start justify-between gap-2">
            <div className="min-w-0">
              <h1 className="text-lg font-semibold text-gray-800">
                MMM Copilot
              </h1>
            </div>
            <div className="flex items-center gap-1 shrink-0 mt-0.5">
              <button
                onClick={handleBack}
                disabled={!canBack}
                title="Back to previous turn"
                className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ArrowLeft size={15} />
              </button>
              <button
                onClick={handleRetry}
                disabled={!canRetry || !lastAssistantHasContent}
                title="Regenerate last response"
                className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <RotateCcw size={15} />
              </button>
              <button
                onClick={handleClearChat}
                disabled={loading || messages.length === 0}
                title="Clear conversation"
                className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <Trash2 size={15} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-gray-50">
            {messages.map((msg, i) => (
              <ChatMessageBubble
                key={msg.id}
                msg={msg}
                pending={loading && i === messages.length - 1}
                onNavigate={setActiveTab}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-4 border-t border-gray-200 bg-white">
            <div className="relative flex items-center">
              <input type="file" ref={fileInputRef} className="hidden" onChange={handleFileUpload} accept=".csv,.xlsx,.xls" />
              <button onClick={() => fileInputRef.current?.click()} disabled={loading}
                className="absolute left-2 p-2 text-gray-400 hover:text-indigo-500 transition-colors disabled:opacity-50" title="Upload Dataset">
                <Paperclip size={18} />
              </button>
              <input
                type="text" value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
                placeholder="Ask the agent to generate data, configure models, or explain ROI…"
                className="w-full bg-gray-100 border border-gray-200 rounded-full py-3 px-12 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 transition-all text-gray-900 placeholder-gray-400"
                disabled={loading}
              />
              {loading ? (
                <button onClick={handleStop}
                  className="absolute right-2 p-2 bg-red-500 hover:bg-red-400 rounded-full text-white transition-colors"
                  title="Stop generation"
                >
                  <Square size={16} fill="white" />
                </button>
              ) : (
                <button onClick={() => handleSend()} disabled={!input.trim()}
                  className="absolute right-2 p-2 bg-indigo-600 hover:bg-indigo-500 rounded-full text-white transition-colors disabled:opacity-50">
                  <Send size={18} />
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Right: Workspace Dashboard (2/3 or full) ── */}
      <div className={`${rightExpanded ? 'w-full' : 'w-2/3'} bg-gray-50 overflow-hidden flex flex-col`}>
        {/* Header — title + tab bar + fullscreen toggle */}
        <div className="px-5 pt-5 pb-0 bg-gray-50 sticky top-0 z-10 border-b border-gray-200">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-bold text-gray-900">Project Workspace</h2>
            <button
              onClick={() => setRightExpanded(v => !v)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-gray-200 text-gray-500 hover:text-gray-800 shadow-sm transition-all text-sm font-medium"
            >
              {rightExpanded ? <><Minimize2 size={14} /> Restore Chat</> : <><Maximize2 size={14} /> Full Screen</>}
            </button>
          </div>

          {/* Tab bar */}
          {(() => {
            const tabs = [
              { id: 'workflow',  label: 'Workflow',  icon: <BookOpen size={14} />,
                badge: `${causal.workflow.filter(s => s.status === 'done').length}/9` },
              { id: 'causal',    label: 'Causal',    icon: <Network size={14} />,
                badge: causal.assumptions.length > 0 ? String(causal.assumptions.length) : null,
                dot: !!causal.dag },
              { id: 'data',      label: 'Data',      icon: <Database size={14} />,
                badge: causal.files.length > 0 ? String(causal.files.length) : null,
                dot: !!dashboardData.dataset },
              { id: 'knowledge', label: 'Knowledge', icon: <BrainCircuit size={14} />,
                badge: null, dot: false },
              { id: 'model',     label: 'Model',     icon: <Layers size={14} />,
                dot: hasSpec },
              { id: 'results',   label: 'Results',   icon: <BarChart2 size={14} />,
                dot: modelCompleted || hasDecomp || !!dashboardData.roi_metrics },
              { id: 'plots',     label: 'Plots',     icon: <Activity size={14} />,
                badge: (dashboardData.plots?.length || 0) > 0 ? String(dashboardData.plots.length) : null,
                dot: (dashboardData.plots?.length || 0) > 0 },
              { id: 'artifacts', label: 'Artifacts', icon: <FileCode size={14} />,
                badge: (artifacts.length + pythonOutputs.length) > 0
                  ? String(artifacts.length + pythonOutputs.length) : null },
            ];
            return (
              <div className="flex items-center gap-1 overflow-x-auto -mb-px">
                {tabs.map(t => {
                  const active = activeTab === t.id;
                  return (
                    <button
                      key={t.id}
                      onClick={() => setActiveTab(t.id)}
                      className={`flex items-center gap-2 px-3.5 py-2.5 text-sm font-medium rounded-t-lg border-b-2 transition-colors shrink-0 ${
                        active
                          ? 'border-indigo-500 text-indigo-700 bg-white'
                          : 'border-transparent text-gray-500 hover:text-gray-800 hover:bg-gray-100'
                      }`}
                    >
                      <span className={active ? 'text-indigo-600' : 'text-gray-400'}>{t.icon}</span>
                      <span>{t.label}</span>
                      {t.badge && (
                        <span className={`text-[10px] font-semibold rounded-full px-1.5 py-0.5 ${
                          active ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-100 text-gray-600'
                        }`}>
                          {t.badge}
                        </span>
                      )}
                      {!t.badge && t.dot && (
                        <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-indigo-500' : 'bg-gray-300'}`} />
                      )}
                    </button>
                  );
                })}
              </div>
            );
          })()}
        </div>

        {/* Tab panels — only one is rendered at a time */}
        <div className="flex-1 overflow-y-auto p-5">
          <div className="grid grid-cols-1 gap-4">

            {activeTab === 'workflow' && (
              causal.workflow.length > 0
                ? <WorkflowChecklist steps={causal.workflow} onOverride={causal.overrideWorkflow} />
                : <EmptyTabState
                    icon={<BookOpen size={28} />}
                    title="Start a conversation to begin"
                    hint="Type a message to the copilot and the scientific workflow checklist will appear here as you progress."
                  />
            )}

            {activeTab === 'causal' && (
              <>
                <EditableDAGViewer
                  dag={causal.dag}
                  threadId={threadId}
                  onSaved={causal.refresh}
                />
                <AssumptionsLog
                  threadId={threadId}
                  assumptions={causal.assumptions}
                  onRefresh={causal.refresh}
                />
              </>
            )}

            {activeTab === 'data' && (
              <>
                <DataFilesWidget files={causal.files} onDelete={causal.deleteFile} />
                <WorkspaceFilesWidget
                  threadId={threadId}
                  apiKey={apiKey}
                  modelName={modelName}
                  refreshKey={workspaceRefreshKey}
                />
                {dashboardData.dataset ? (
                  <DatasetPanel dataset={dashboardData.dataset} threadId={threadId} />
                ) : (
                  <EmptyTabState
                    icon={<Database size={28} />}
                    title="No dataset loaded yet"
                    hint="Ask the agent to generate synthetic data or upload a CSV — it'll call `inspect_dataset` and details will appear here."
                  />
                )}
              </>
            )}

            {activeTab === 'knowledge' && (
              <KnowledgeTab
                projectId={projectId}
                apiKey={apiKey}
                modelName={modelName}
              />
            )}

            {activeTab === 'model' && (
              <>
                {hasSpec ? (
                  <>
                    <ModelSpecWidget
                      spec={dashboardData.model_spec}
                      editable={!modelCompleted}
                      onApplySpec={handleApplySpec}
                    />
                    <SeasonalityTrendWidget
                      spec={dashboardData.model_spec}
                      onQuickAction={handleSend}
                      modelCompleted={modelCompleted}
                    />
                    <PriorConfigWidget
                      spec={dashboardData.model_spec}
                      editable={!modelCompleted}
                      onApplySpec={handleApplySpec}
                    />
                    {modelCompleted && (
                      <DashWidget title="Model Successfully Fit" dotColor="bg-green-500 animate-pulse" color="green">
                        <p className="text-sm text-gray-700">{dashboardData.summary}</p>
                      </DashWidget>
                    )}
                  </>
                ) : (
                  <EmptyTabState
                    icon={<Layers size={28} />}
                    title="No model configured yet"
                    hint="Ask the agent to configure a model (Step 3 of the workflow) — it'll call `configure_model` and a spec will appear here."
                  />
                )}
              </>
            )}

            {activeTab === 'results' && (
              <>
                {!modelCompleted && !hasDecomp && !dashboardData.roi_metrics && !dashboardData.report_path && (
                  <EmptyTabState
                    icon={<BarChart2 size={28} />}
                    title="No results yet"
                    hint="Fit a model (Step 5), then ask for the decomposition or ROI."
                  />
                )}

                {hasDecomp && <DecompositionWidget decomposition={dashboardData.decomposition} />}

                {dashboardData.roi_metrics && (() => {
                  const table = (
                    <div className="overflow-x-auto rounded-xl border border-gray-200">
                      <table className="w-full text-left text-sm">
                        <thead className="bg-gray-50 text-gray-500 uppercase text-xs">
                          <tr>
                            {['Channel', 'Mean ROI', '94% HDI', 'Prob. Profitable'].map(h => (
                              <th key={h} className="px-4 py-3 font-medium">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {dashboardData.roi_metrics.map((row: any) => (
                            <tr key={row.channel} className="bg-white hover:bg-gray-50 transition-colors">
                              <td className="px-4 py-3 font-medium text-gray-900">{row.channel}</td>
                              <td className="px-4 py-3 text-emerald-600 font-semibold">{row.roi_mean?.toFixed(2)}x</td>
                              <td className="px-4 py-3 text-gray-500">[{row.roi_hdi_low?.toFixed(2)}, {row.roi_hdi_high?.toFixed(2)}]</td>
                              <td className="px-4 py-3">
                                <div className="flex items-center gap-2">
                                  <div className="w-14 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                    <div className="h-full bg-emerald-500" style={{ width: `${(row.prob_profitable || 0) * 100}%` }} />
                                  </div>
                                  <span className="text-gray-700 text-xs font-medium">{((row.prob_profitable || 0) * 100).toFixed(1)}%</span>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  );
                  return (
                    <DashWidget title="ROI Performance" dotColor="bg-emerald-500" color="emerald" expandContent={table}>
                      {table}
                    </DashWidget>
                  );
                })()}

                {dashboardData.report_path && (
                  <DashWidget title="Full MMM Report" dotColor="bg-violet-500" color="violet"
                    expandTitle="MMM Report"
                    expandContent={
                      <div className="h-[80vh]">
                        <iframe src={`${API_BASE}/report`} className="w-full h-full rounded-xl border border-gray-200" title="MMM Report" sandbox="allow-scripts allow-same-origin" />
                      </div>
                    }
                  >
                    <div className="flex flex-col gap-3">
                      <p className="text-sm text-gray-500">Full analysis report with diagnostics, ROI, and channel decomposition.</p>
                      <div className="flex gap-3">
                        <a href={`${API_BASE}/report/download`} download="mmm_report.html"
                          className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm rounded-xl transition-colors font-medium">
                          <Download size={15} /> Download
                        </a>
                        <a href={`${API_BASE}/report`} target="_blank" rel="noreferrer"
                          className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-xl transition-colors font-medium border border-gray-200">
                          <ExternalLink size={15} /> Open Tab
                        </a>
                      </div>
                      <div className="rounded-xl overflow-hidden border border-gray-200" style={{ height: '340px' }}>
                        <iframe src={`${API_BASE}/report`} className="w-full h-full" title="Preview" sandbox="allow-scripts allow-same-origin" />
                      </div>
                    </div>
                  </DashWidget>
                )}
              </>
            )}

            {activeTab === 'plots' && (
              dashboardData.plots?.length > 0 ? (
                <DashWidget title={`Visualizations (${dashboardData.plots.length})`} dotColor="bg-fuchsia-500" color="fuchsia">
                  <div className="space-y-4">
                    {dashboardData.plots.map((plot: any, idx: number) => (
                      <PlotCard key={plot?.id ?? idx} plot={plot} idx={idx} />
                    ))}
                  </div>
                </DashWidget>
              ) : (
                <EmptyTabState
                  icon={<Activity size={28} />}
                  title="No plots yet"
                  hint="Ask the agent to run execute_python with fig.show() — charts appear here automatically."
                />
              )
            )}

            {activeTab === 'artifacts' && (
              <>
                <ArtifactsPanel
                  artifacts={artifacts}
                  onRerun={handleRerunArtifact}
                  onDelete={handleDeleteArtifact}
                  onLoadRun={handleLoadRun}
                />
                <PythonOutputWidget
                  outputs={pythonOutputs}
                  onClear={() => setPythonOutputs([])}
                  onExport={threadId ? () => window.open(`${API_BASE}/sessions/${threadId}/export`, '_blank') : undefined}
                />
              </>
            )}

          </div>
        </div>
      </div>
      </div>
    </div>
  );
}
