import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Send, Bot, User, Loader2, Paperclip, ChevronDown, ChevronRight,
  Wrench, CheckCircle2, Maximize2, Minimize2, X, Download, ExternalLink,
  TrendingUp, Calendar, Layers, Zap, BarChart2, Activity, Pencil, Check, RotateCcw, Plus, Trash2,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import { useAuthStore } from '../stores/authStore';

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
  type: 'human' | 'ai';
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
    layout.colorway = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316'];
  }

  layout.margin = { l: 60, r: 30, t: 60, b: 70, ...(layout.margin || {}) };

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

function ToolCallBlock({ toolCall }: { toolCall: ToolCall }) {
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
  code: ({ inline, children }: any) => inline
    ? <code className="bg-gray-100 px-1 py-0.5 rounded text-indigo-600 text-xs font-mono">{children}</code>
    : <pre className="bg-gray-50 rounded-lg p-3 overflow-x-auto text-xs font-mono text-gray-700 border border-gray-200 my-2"><code>{children}</code></pre>,
};

// ─── PlotCard ─────────────────────────────────────────────────────────────────

function PlotCard({ plot, idx }: { plot: any; idx: number }) {
  const [fullscreen, setFullscreen] = useState(false);
  const rawTitle = plot.layout?.title?.text ?? plot.layout?.title ?? `Chart ${idx + 1}`;
  const title = String(rawTitle);

  const fixedLayout = useMemo(() => applyLightModeLayout(plot.layout), [plot.layout]);

  const plotEl = (height: string) => (
    <Plot
      data={plot.data}
      layout={{ ...fixedLayout, autosize: true }}
      useResizeHandler
      style={{ width: '100%', height }}
      config={{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['sendDataToCloud'] }}
    />
  );

  return (
    <>
      <div className="rounded-xl overflow-hidden border border-gray-200 bg-white relative group shadow-sm">
        <button
          onClick={() => setFullscreen(true)}
          className="absolute top-2 right-2 z-10 p-1.5 rounded-lg bg-white/90 text-gray-400 hover:text-gray-700 hover:bg-gray-100 opacity-0 group-hover:opacity-100 transition-all border border-gray-200"
          title="Expand chart"
        >
          <Maximize2 size={15} />
        </button>
        <p className="text-xs text-gray-500 px-4 pt-3 pb-0 font-semibold truncate">{title}</p>
        {plotEl('360px')}
      </div>
      {fullscreen && (
        <Modal title={title} onClose={() => setFullscreen(false)} fullWidth>
          {plotEl('calc(100vh - 120px)')}
        </Modal>
      )}
    </>
  );
}

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
  const KNOTS = [0.2, 0.35, 0.55, 0.75];

  if (type === 'linear' || !type) {
    return {
      traces: [{ x: t, y: t.map(x => 0.15 + 0.7 * x), name: 'Linear Trend', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } }],
      shapes: [],
    };
  }

  if (type === 'piecewise') {
    const cps = spec?.changepoint_range
      ? KNOTS.filter(k => k < spec.changepoint_range)
      : KNOTS.slice(0, spec?.n_changepoints ?? 3);
    const segments = [0, ...cps, 1];
    const slopes = [0.8, 0.3, 1.1, -0.2, 0.6];
    let y0 = 0.1;
    const xAll: number[] = [];
    const yAll: number[] = [];
    for (let s = 0; s < segments.length - 1; s++) {
      const xs = t.filter(x => x >= segments[s] && x <= segments[s + 1]);
      const sl = slopes[s % slopes.length];
      xs.forEach(x => { xAll.push(x); yAll.push(y0 + sl * (x - segments[s])); });
      y0 = yAll[yAll.length - 1];
    }
    return {
      traces: [{ x: xAll, y: yAll, name: 'Piecewise Trend', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } }],
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
  const lines = code.split('\n');
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
      <div className="bg-gray-900 overflow-x-auto max-h-64">
        <table className="w-full border-collapse">
          <tbody>
            {lines.map((line, i) => (
              <tr key={i} className="hover:bg-white/[0.03]">
                <td className="select-none text-right text-[10px] font-mono text-gray-600 pl-3 pr-3 w-7 align-top leading-5 border-r border-gray-800">{i + 1}</td>
                <td className="pl-3 pr-3 text-[11px] font-mono text-gray-100 whitespace-pre leading-5">{line || '​'}</td>
              </tr>
            ))}
          </tbody>
        </table>
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

function PythonOutputWidget({ outputs, onClear }: { outputs: PythonOutput[]; onClear: () => void }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggle = (id: string) => setCollapsed(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  if (outputs.length === 0) return null;

  return (
    <DashWidget
      title={`Python REPL (${outputs.length} run${outputs.length > 1 ? 's' : ''})`}
      dotColor="bg-emerald-500"
      color="emerald"
    >
      <div className="space-y-1 mb-2 flex items-center justify-between">
        <p className="text-xs text-gray-500">{outputs.length} execution{outputs.length > 1 ? 's' : ''} recorded this session.</p>
        <button onClick={onClear} className="text-[10px] text-gray-400 hover:text-red-500 flex items-center gap-1 transition-colors">
          <Trash2 size={11} /> Clear
        </button>
      </div>
      <div className="space-y-4">
        {outputs.map((out, idx) => {
          const isCollapsed = collapsed.has(out.id);
          const firstLine = out.code.trim().split('\n')[0];
          return (
            <div key={out.id} className="rounded-xl overflow-hidden shadow-sm">
              {/* Cell header */}
              <button
                onClick={() => toggle(out.id)}
                className="w-full flex items-center gap-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 transition-colors text-left border border-gray-200 rounded-t-xl"
              >
                <span className="text-[10px] font-mono text-gray-400 shrink-0">In [{idx + 1}]</span>
                <span className="flex-1 text-[11px] font-mono text-gray-600 truncate">{firstLine}</span>
                {out.hasError && <span className="text-[9px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded font-semibold">ERROR</span>}
                {out.plotCount > 0 && <span className="text-[9px] bg-fuchsia-100 text-fuchsia-600 px-1.5 py-0.5 rounded font-semibold">{out.plotCount} plot{out.plotCount > 1 ? 's' : ''}</span>}
                {isCollapsed ? <ChevronRight size={13} className="text-gray-400 shrink-0" /> : <ChevronDown size={13} className="text-gray-400 shrink-0" />}
              </button>
              {!isCollapsed && (
                <div className="border-l border-r border-b border-gray-200 rounded-b-xl overflow-hidden">
                  <PythonCodeBlock code={out.code} />
                  <PythonOutputBlock output={out.output} hasError={out.hasError} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </DashWidget>
  );
}

// ─── Main AgentPage ───────────────────────────────────────────────────────────

export function AgentPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [dashboardData, setDashboardData] = useState<any>({});
  const [pythonOutputs, setPythonOutputs] = useState<PythonOutput[]>([]);
  const [loading, setLoading] = useState(false);
  const [rightExpanded, setRightExpanded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { apiKey, modelName } = useAuthStore();

  useEffect(() => {
    fetch('http://localhost:8000/state/default_thread', {
      headers: { 'X-API-Key': apiKey || '', 'X-Model-Name': modelName || '' },
    })
      .then(r => r.json())
      .then(data => {
        if (data.messages) {
          const parsed: ChatMessage[] = [];
          data.messages.forEach((m: any, i: number) => {
            if (m.type === 'tool') return;
            const content = normalizeContent(m.content);
            if (!content && !m.tool_calls?.length) return;
            parsed.push({ id: `loaded-${i}`, type: m.type, content });
          });
          setMessages(parsed);
        }
        if (data.dashboard_data) setDashboardData(data.dashboard_data);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleClearChat = async () => {
    setMessages([]);
    setDashboardData({});
    setPythonOutputs([]);
    try {
      await fetch('http://localhost:8000/state/default_thread', {
        method: 'DELETE',
        headers: { 'X-API-Key': apiKey || '', 'X-Model-Name': modelName || '' },
      });
    } catch { /* silently ignore if backend doesn't support DELETE */ }
  };

  const handleSend = async (messageOverride?: string) => {
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

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey || '', 'X-Model-Name': modelName || '' },
        body: JSON.stringify({ message: textToSend, thread_id: 'default_thread' }),
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
    } catch (e) {
      console.error(e);
      // Stream error — mark any running tools as errored
      const runningKeys = Object.keys(toolCallMap).filter(k => toolCallMap[k].status === 'running');
      if (runningKeys.length > 0) {
        runningKeys.forEach(k => { toolCallMap[k] = { ...toolCallMap[k], status: 'error', result: 'Connection error' }; });
        updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
      }
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        headers: { 'X-API-Key': apiKey || '', 'X-Model-Name': modelName || '' },
        body: fd,
      });
      const data = await res.json();
      if (data.path)
        handleSend(`I have uploaded a dataset at \`${data.path}\`. Please load it using the execute_python tool and run some basic EDA on it. Don't build a model yet.`);
    } catch (e) { console.error('File upload failed', e); }
    finally { setLoading(false); if (fileInputRef.current) fileInputRef.current.value = ''; }
  };

  const modelCompleted = dashboardData.model_status === 'completed';
  const hasSpec = !!dashboardData.model_spec;

  const handleApplySpec = useCallback((newSpec: any) => {
    setDashboardData((prev: any) => ({ ...prev, model_spec: newSpec }));
    handleSend(buildApplyMessage(newSpec));
  }, []);
  const hasDecomp = dashboardData.decomposition?.length > 0;

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 overflow-hidden font-sans">
      {/* ── Left: Chat (1/3 width) ── */}
      {!rightExpanded && (
        <div className="w-1/3 border-r border-gray-200 flex flex-col bg-white shadow-md relative z-10 shrink-0">
          <div className="p-4 border-b border-gray-200 bg-white sticky top-0 flex items-start justify-between gap-2">
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Agentic MMM Copilot
              </h1>
              {modelName && <p className="text-xs text-gray-400 mt-0.5">{modelName}</p>}
            </div>
            <button
              onClick={handleClearChat}
              disabled={loading || messages.length === 0}
              title="Clear conversation"
              className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed shrink-0 mt-0.5"
            >
              <Trash2 size={15} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-gray-50">
            {messages.map(msg => (
              <div key={msg.id} className={`flex gap-3 ${msg.type === 'human' ? 'justify-end' : 'justify-start'}`}>
                {msg.type === 'ai' && (
                  <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 mt-1">
                    <Bot size={16} className="text-white" />
                  </div>
                )}
                <div className="max-w-[82%] flex flex-col gap-1">
                  {msg.type === 'ai' && msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div className="space-y-1">
                      {msg.toolCalls.map(tc => <ToolCallBlock key={tc.id} toolCall={tc} />)}
                    </div>
                  )}
                  {(msg.content || (loading && msg.type === 'ai')) && (
                    <div className={`rounded-2xl p-4 ${msg.type === 'human'
                      ? 'bg-blue-600 text-white rounded-br-none'
                      : 'bg-white text-gray-800 rounded-bl-none border border-gray-200 shadow-sm'}`}>
                      {msg.type === 'human'
                        ? <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                        : <div className="prose prose-sm max-w-none text-sm">
                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                              {msg.content || (loading ? 'Thinking…' : '')}
                            </ReactMarkdown>
                          </div>}
                    </div>
                  )}
                </div>
                {msg.type === 'human' && (
                  <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center shrink-0 mt-1">
                    <User size={16} className="text-white" />
                  </div>
                )}
              </div>
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
              <button onClick={() => handleSend()} disabled={loading || !input.trim()}
                className="absolute right-2 p-2 bg-indigo-600 hover:bg-indigo-500 rounded-full text-white transition-colors disabled:opacity-50">
                {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Right: Workspace Dashboard (2/3 or full) ── */}
      <div className={`${rightExpanded ? 'w-full' : 'w-2/3'} bg-gray-50 p-5 overflow-y-auto`}>
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-xl font-bold text-gray-900">Project Workspace</h2>
          <button
            onClick={() => setRightExpanded(v => !v)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-gray-200 text-gray-500 hover:text-gray-800 shadow-sm transition-all text-sm font-medium"
          >
            {rightExpanded ? <><Minimize2 size={14} /> Restore Chat</> : <><Maximize2 size={14} /> Full Screen</>}
          </button>
        </div>

        {!Object.keys(dashboardData).length && (
          <div className="flex flex-col items-center justify-center h-[calc(100%-4rem)] text-gray-400 space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-white flex items-center justify-center border border-gray-200 shadow-sm">
              <Bot size={32} className="text-gray-300" />
            </div>
            <p className="text-base text-gray-500">Waiting for agent insights…</p>
            <p className="text-sm text-gray-400 max-w-sm text-center">Charts and metrics will appear here as the agent works on your model.</p>
          </div>
        )}

        <div className="grid grid-cols-1 gap-4">
          {/* Dataset */}
          {dashboardData.dataset && (
            <DashWidget title="Dataset Details" dotColor="bg-indigo-500" color="indigo"
              expandContent={
                <div className="grid grid-cols-2 gap-6 max-w-2xl mx-auto pt-4">
                  <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
                    <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">Total Rows</p>
                    <p className="text-4xl font-bold text-gray-900">{dashboardData.dataset.rows}</p>
                  </div>
                  <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
                    <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">Geographies</p>
                    <p className="text-2xl font-medium text-gray-700">{dashboardData.dataset.geographies?.join(', ')}</p>
                  </div>
                </div>
              }
            >
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 p-3 rounded-xl border border-gray-100">
                  <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Rows</p>
                  <p className="text-2xl font-bold text-gray-900">{dashboardData.dataset.rows}</p>
                </div>
                <div className="bg-gray-50 p-3 rounded-xl border border-gray-100">
                  <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Geographies</p>
                  <p className="text-sm font-medium text-gray-700">{dashboardData.dataset.geographies?.join(', ')}</p>
                </div>
              </div>
            </DashWidget>
          )}

          {/* Model Configuration — full detail */}
          {hasSpec && (
            <ModelSpecWidget
              spec={dashboardData.model_spec}
              editable={!modelCompleted}
              onApplySpec={handleApplySpec}
            />
          )}

          {/* Trend & Seasonality Preview */}
          {hasSpec && (
            <SeasonalityTrendWidget
              spec={dashboardData.model_spec}
              onQuickAction={handleSend}
              modelCompleted={modelCompleted}
            />
          )}

          {/* Prior Configuration */}
          {hasSpec && (
            <PriorConfigWidget
              spec={dashboardData.model_spec}
              editable={!modelCompleted}
              onApplySpec={handleApplySpec}
            />
          )}

          {/* Fit status */}
          {modelCompleted && (
            <DashWidget title="Model Successfully Fit" dotColor="bg-green-500 animate-pulse" color="green">
              <p className="text-sm text-gray-700">{dashboardData.summary}</p>
            </DashWidget>
          )}

          {/* Decomposition */}
          {hasDecomp && <DecompositionWidget decomposition={dashboardData.decomposition} />}

          {/* Report */}
          {dashboardData.report_path && (
            <DashWidget title="Full MMM Report" dotColor="bg-violet-500" color="violet"
              expandTitle="MMM Report"
              expandContent={
                <div className="h-[80vh]">
                  <iframe src="http://localhost:8000/report" className="w-full h-full rounded-xl border border-gray-200" title="MMM Report" sandbox="allow-scripts allow-same-origin" />
                </div>
              }
            >
              <div className="flex flex-col gap-3">
                <p className="text-sm text-gray-500">Full analysis report with diagnostics, ROI, and channel decomposition.</p>
                <div className="flex gap-3">
                  <a href="http://localhost:8000/report/download" download="mmm_report.html"
                    className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm rounded-xl transition-colors font-medium">
                    <Download size={15} /> Download
                  </a>
                  <a href="http://localhost:8000/report" target="_blank" rel="noreferrer"
                    className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-xl transition-colors font-medium border border-gray-200">
                    <ExternalLink size={15} /> Open Tab
                  </a>
                </div>
                <div className="rounded-xl overflow-hidden border border-gray-200" style={{ height: '340px' }}>
                  <iframe src="http://localhost:8000/report" className="w-full h-full" title="Preview" sandbox="allow-scripts allow-same-origin" />
                </div>
              </div>
            </DashWidget>
          )}

          {/* ROI Metrics */}
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

          {/* Python REPL output */}
          <PythonOutputWidget
            outputs={pythonOutputs}
            onClear={() => setPythonOutputs([])}
          />

          {/* Generated Plots */}
          {dashboardData.plots?.length > 0 && (
            <DashWidget title={`Generated Visualizations (${dashboardData.plots.length})`} dotColor="bg-fuchsia-500 animate-pulse" color="fuchsia">
              <div className="space-y-4">
                {dashboardData.plots.map((plot: any, idx: number) => (
                  <PlotCard key={idx} plot={plot} idx={idx} />
                ))}
              </div>
            </DashWidget>
          )}
        </div>
      </div>
    </div>
  );
}
