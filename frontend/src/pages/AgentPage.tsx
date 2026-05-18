import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Bot, User, Loader2, Paperclip, ChevronDown, ChevronRight, Wrench, CheckCircle2, Maximize2, X, Download, ExternalLink, FileText } from 'lucide-react';
import Plot from 'react-plotly.js';
import { useAuthStore } from '../stores/authStore';

// --- Types ---

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

// --- Helpers ---

function normalizeContent(content: any): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map(block => {
      if (typeof block === 'string') return block;
      if (block && typeof block.text === 'string') return block.text;
      return '';
    }).filter(Boolean).join('\n');
  }
  if (typeof content === 'object' && content !== null) return JSON.stringify(content);
  return String(content || '');
}

// Prettify tool name for display
function formatToolName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// Truncate long output for preview
function truncate(str: string, n = 300): string {
  return str.length > n ? str.slice(0, n) + '…' : str;
}

// --- ToolCallBlock Component ---

function ToolCallBlock({ toolCall }: { toolCall: ToolCall }) {
  const [expanded, setExpanded] = useState(false);

  const statusColor = toolCall.status === 'done'
    ? 'text-emerald-400 border-emerald-800/60 bg-emerald-950/30'
    : toolCall.status === 'error'
    ? 'text-red-400 border-red-800/60 bg-red-950/30'
    : 'text-amber-400 border-amber-800/60 bg-amber-950/30';

  const StatusIcon = toolCall.status === 'done'
    ? <CheckCircle2 size={13} className="text-emerald-400 shrink-0" />
    : toolCall.status === 'error'
    ? <CheckCircle2 size={13} className="text-red-400 shrink-0" />
    : <Loader2 size={13} className="text-amber-400 animate-spin shrink-0" />;

  return (
    <div className={`my-2 rounded-xl border text-xs font-mono overflow-hidden ${statusColor}`}>
      {/* Header — always visible, click to expand */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-white/5 transition-colors text-left"
      >
        <Wrench size={13} className="shrink-0 opacity-70" />
        <span className="font-semibold tracking-wide flex-1">{formatToolName(toolCall.name)}</span>
        {StatusIcon}
        {expanded
          ? <ChevronDown size={13} className="shrink-0 opacity-60" />
          : <ChevronRight size={13} className="shrink-0 opacity-60" />}
      </button>

      {/* Expandable body */}
      {expanded && (
        <div className="border-t border-white/10 divide-y divide-white/10">
          {toolCall.args && Object.keys(toolCall.args).length > 0 && (
            <div className="px-3 py-2">
              <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Input</p>
              <pre className="text-gray-300 whitespace-pre-wrap break-all text-[11px] leading-relaxed">
                {JSON.stringify(toolCall.args, null, 2)}
              </pre>
            </div>
          )}
          {toolCall.result && (
            <div className="px-3 py-2">
              <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Output</p>
              <pre className="text-gray-300 whitespace-pre-wrap break-all text-[11px] leading-relaxed max-h-64 overflow-y-auto">
                {toolCall.result}
              </pre>
            </div>
          )}
        </div>
      )}
      {/* Collapsed preview when result exists */}
      {!expanded && toolCall.result && (
        <div className="px-3 pb-2 text-gray-500 text-[11px] truncate">
          {truncate(toolCall.result, 120)}
        </div>
      )}
    </div>
  );
}

// --- Modal Dialog (centered, with backdrop) ---

function Modal({ title, onClose, fullWidth = false, children }: {
  title: string;
  onClose: () => void;
  fullWidth?: boolean;
  children: React.ReactNode;
}) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/80 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className={`relative flex flex-col bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl overflow-hidden ${fullWidth ? 'w-full h-full max-w-none' : 'w-full max-w-4xl max-h-[90vh]'}`}>
        {/* Modal header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700 shrink-0">
          <h2 className="text-lg font-bold text-gray-100 truncate pr-4">{title}</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-700 text-gray-400 hover:text-white transition-colors shrink-0"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>
        {/* Scrollable content */}
        <div className="overflow-y-auto flex-1 p-6">
          {children}
        </div>
      </div>
    </div>
  );
}

// --- Collapsible Dashboard Widget (always expandable) ---

function DashWidget({
  title, color = 'indigo', dotColor = 'bg-indigo-400',
  defaultOpen = true, expandTitle, expandContent, children
}: {
  title: string;
  color?: string;
  dotColor?: string;
  defaultOpen?: boolean;
  expandTitle?: string;
  expandContent?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [modal, setModal] = useState(false);

  return (
    <>
      <div className={`bg-gray-800/80 backdrop-blur-sm rounded-2xl border border-gray-700 shadow-xl transition-all hover:border-${color}-500/40 overflow-hidden`}>
        {/* Card header — click left side to collapse, expand button on right */}
        <div className="flex items-center gap-3 px-6 py-4">
          <button
            onClick={() => setOpen(v => !v)}
            className="flex items-center gap-3 flex-1 text-left hover:opacity-80 transition-opacity"
          >
            <span className={`w-2 h-2 rounded-full shrink-0 ${dotColor}`} />
            <span className={`font-semibold text-base text-${color}-300 flex-1`}>{title}</span>
            {open
              ? <ChevronDown size={16} className="text-gray-500 shrink-0" />
              : <ChevronRight size={16} className="text-gray-500 shrink-0" />}
          </button>
          {/* Expand button — always visible */}
          <button
            onClick={() => setModal(true)}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-500 hover:text-white transition-colors shrink-0"
            title="Expand in modal (Esc to close)"
          >
            <Maximize2 size={15} />
          </button>
        </div>

        {/* Collapsible body */}
        {open && (
          <div className="px-6 pb-6">
            {children}
          </div>
        )}
      </div>

      {/* Modal overlay */}
      {modal && (
        <Modal title={expandTitle || title} onClose={() => setModal(false)}>
          {expandContent || children}
        </Modal>
      )}
    </>
  );
}

// --- Markdown render config (shared) ---
const MD_COMPONENTS: any = {
  table: ({ children }: any) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-sm border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }: any) => <thead className="bg-gray-700/60">{children}</thead>,
  th: ({ children }: any) => <th className="px-3 py-2 text-left font-semibold text-indigo-300 border border-gray-600">{children}</th>,
  td: ({ children }: any) => <td className="px-3 py-2 text-gray-200 border border-gray-700">{children}</td>,
  tr: ({ children }: any) => <tr className="even:bg-gray-800/40">{children}</tr>,
  code: ({ inline, children }: any) => inline
    ? <code className="bg-gray-700 px-1 py-0.5 rounded text-indigo-300 text-xs font-mono">{children}</code>
    : <pre className="bg-gray-900 rounded-lg p-3 overflow-x-auto text-xs font-mono text-gray-300 border border-gray-700 my-2"><code>{children}</code></pre>,
};

// --- PlotCard: single chart with expand button ---

function PlotCard({ plot, idx }: { plot: any; idx: number }) {
  const [fullscreen, setFullscreen] = useState(false);
  const title = plot.layout?.title?.text || plot.layout?.title || `Chart ${idx + 1}`;

  const plotEl = (height: string) => (
    <Plot
      data={plot.data}
      layout={{
        ...plot.layout,
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e5e7eb' },
        margin: { l: 50, r: 30, t: 50, b: 50 },
      }}
      useResizeHandler={true}
      style={{ width: '100%', height }}
      config={{ responsive: true, displayModeBar: true, displaylogo: false }}
    />
  );

  return (
    <>
      <div className="rounded-xl overflow-hidden border border-gray-700/50 bg-gray-900 relative group">
        {/* Expand button */}
        <button
          onClick={() => setFullscreen(true)}
          className="absolute top-2 right-2 z-10 p-1.5 rounded-lg bg-gray-800/80 text-gray-400 hover:text-white hover:bg-gray-700 opacity-0 group-hover:opacity-100 transition-all"
          title="Expand chart"
        >
          <Maximize2 size={15} />
        </button>
        <p className="text-xs text-gray-500 px-4 pt-3 pb-0 font-medium truncate">{String(title)}</p>
        {plotEl('360px')}
      </div>

      {fullscreen && (
        <Modal title={String(title)} onClose={() => setFullscreen(false)} fullWidth>
          {plotEl('calc(100vh - 120px)')}
        </Modal>
      )}
    </>
  );
}

// --- Main AgentPage Component ---

export function AgentPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [dashboardData, setDashboardData] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { apiKey, modelName } = useAuthStore();

  // Fetch persisted state on mount
  useEffect(() => {
    fetch('http://localhost:8000/state/default_thread', {
      headers: { 'X-API-Key': apiKey || '', 'X-Model-Name': modelName || '' }
    })
      .then(res => res.json())
      .then(data => {
        if (data.messages) {
          const parsed: ChatMessage[] = [];
          data.messages.forEach((m: any, i: number) => {
            if (m.type === 'tool') return; // skip raw tool results at init
            const content = normalizeContent(m.content);
            if (!content && !m.tool_calls?.length) return;
            parsed.push({ id: i.toString(), type: m.type, content });
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

  const handleSend = async (messageOverride?: string) => {
    const textToSend = messageOverride || input;
    if (!textToSend.trim()) return;

    const userMsg: ChatMessage = { id: Date.now().toString(), type: 'human', content: textToSend };
    setMessages(prev => [...prev, userMsg]);
    if (!messageOverride) setInput('');
    setLoading(true);

    const tempAiId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, { id: tempAiId, type: 'ai', content: '', toolCalls: [] }]);

    // Track tool calls keyed by tool_call_id
    const toolCallMap: Record<string, ToolCall> = {};

    const updateMsg = (updater: (m: ChatMessage) => ChatMessage) => {
      setMessages(prev => prev.map(m => m.id === tempAiId ? updater(m) : m));
    };

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey || '',
          'X-Model-Name': modelName || ''
        },
        body: JSON.stringify({ message: textToSend, thread_id: 'default_thread' })
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
          if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
          try {
            const data = JSON.parse(line.substring(6));

            // Update dashboard data — backend is the source of truth, always overwrite
            if (data.dashboard_data && Object.keys(data.dashboard_data).length > 0) {
              setDashboardData((prev: any) => ({ ...prev, ...data.dashboard_data }));
            }

            // Pure dashboard update event (e.g. plots) — no message processing needed
            if (data.type === 'dashboard_update') continue;

            // AI message: may carry tool_calls (with or without text content)
            if (data.type === 'ai') {
              // Register any tool calls in the map
              if (Array.isArray(data.tool_calls) && data.tool_calls.length > 0) {
                for (const tc of data.tool_calls) {
                  const id = tc.id || (tc.name + '_' + Date.now());
                  toolCallMap[id] = {
                    id,
                    name: tc.name || 'unknown',
                    args: tc.args || {},
                    status: 'running',
                  };
                }
                updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
              }
              // Append text content if present
              const contentStr = normalizeContent(data.content);
              if (contentStr) {
                aiContent += contentStr + '\n';
                updateMsg(m => ({ ...m, content: aiContent }));
              }
            }

            // Tool result — match by tool_call_id, then fall back to first running
            if (data.type === 'tool') {
              const resultStr = normalizeContent(data.content);
              const matchKey = data.tool_call_id && toolCallMap[data.tool_call_id]
                ? data.tool_call_id
                : Object.keys(toolCallMap).find(k => toolCallMap[k].status === 'running');
              if (matchKey) {
                toolCallMap[matchKey] = {
                  ...toolCallMap[matchKey],
                  result: resultStr,
                  status: 'done',
                };
                updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
              }
            }
          } catch {
            // ignore parse errors for partial lines
          }
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        headers: { 'X-API-Key': apiKey || '', 'X-Model-Name': modelName || '' },
        body: formData,
      });
      const data = await res.json();
      if (data.path) {
        handleSend(`I have uploaded a dataset at \`${data.path}\`. Please load it using the execute_python tool and run some basic EDA on it. Don't build a model yet.`);
      }
    } catch (e) {
      console.error('File upload failed', e);
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 overflow-hidden font-sans">
      {/* ── Left Pane: Chat ── */}
      <div className="w-1/2 border-r border-gray-800 flex flex-col bg-gray-950 shadow-2xl relative z-10">
        {/* Header */}
        <div className="p-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur-md sticky top-0">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
            Agentic MMM Copilot
          </h1>
          {modelName && (
            <p className="text-xs text-gray-500 mt-0.5">{modelName}</p>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map(msg => (
            <div key={msg.id} className={`flex gap-3 ${msg.type === 'human' ? 'justify-end' : 'justify-start'}`}>
              {msg.type === 'ai' && (
                <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 mt-1">
                  <Bot size={16} />
                </div>
              )}

              <div className="max-w-[82%] flex flex-col gap-1">
                {/* Tool call blocks – shown ABOVE the AI reply text */}
                {msg.type === 'ai' && msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="space-y-1">
                    {msg.toolCalls.map(tc => (
                      <ToolCallBlock key={tc.id} toolCall={tc} />
                    ))}
                  </div>
                )}

                {/* Message bubble (only if there's text content) */}
                {(msg.content || (loading && msg.type === 'ai')) && (
                  <div className={`rounded-2xl p-4 ${
                    msg.type === 'human'
                      ? 'bg-blue-600 text-white rounded-br-none'
                      : 'bg-gray-800 text-gray-200 rounded-bl-none border border-gray-700 shadow-lg'
                  }`}>
                    {msg.type === 'human' ? (
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    ) : (
                      <div className="prose prose-invert max-w-none text-sm">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                          {msg.content || (loading ? 'Thinking…' : '')}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {msg.type === 'human' && (
                <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center shrink-0 mt-1">
                  <User size={16} />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input bar */}
        <div className="p-4 border-t border-gray-800 bg-gray-900">
          <div className="relative flex items-center">
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={handleFileUpload}
              accept=".csv,.xlsx,.xls"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="absolute left-2 p-2 text-gray-400 hover:text-indigo-400 transition-colors disabled:opacity-50"
              title="Upload Dataset"
            >
              <Paperclip size={18} />
            </button>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Ask the agent to generate data, configure models, or explain ROI…"
              className="w-full bg-gray-800 border border-gray-700 rounded-full py-3 px-12 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all shadow-inner text-gray-100 placeholder-gray-400"
              disabled={loading}
            />
            <button
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              className="absolute right-2 p-2 bg-indigo-600 hover:bg-indigo-500 rounded-full text-white transition-colors disabled:opacity-50"
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
            </button>
          </div>
        </div>
      </div>

      {/* ── Right Pane: Workspace Dashboard ── */}
      <div className="w-1/2 bg-gray-900 p-6 overflow-y-auto">
        <h2 className="text-2xl font-bold mb-6 text-gray-100">Project Workspace</h2>

        {!Object.keys(dashboardData).length && (
          <div className="flex flex-col items-center justify-center h-[calc(100%-4rem)] text-gray-500 space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-gray-800 flex items-center justify-center border border-gray-700">
              <Bot size={32} className="text-gray-600" />
            </div>
            <p className="text-lg">Waiting for agent insights…</p>
            <p className="text-sm text-gray-600 max-w-sm text-center">Charts and metrics will appear here as the agent works on your model.</p>
          </div>
        )}

        <div className="grid grid-cols-1 gap-4">
          {dashboardData.dataset && (
            <DashWidget
              title="Dataset Details"
              color="indigo" dotColor="bg-indigo-400"
              expandTitle="Dataset Details"
              expandContent={
                <div className="grid grid-cols-2 gap-6 max-w-2xl mx-auto pt-4">
                  <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-700/50">
                    <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">Total Rows</p>
                    <p className="text-4xl font-bold text-gray-100">{dashboardData.dataset.rows}</p>
                  </div>
                  <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-700/50">
                    <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">Geographies</p>
                    <p className="text-2xl font-medium text-gray-200">{dashboardData.dataset.geographies?.join(", ")}</p>
                  </div>
                </div>
              }
            >
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-700/50">
                  <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Total Rows</p>
                  <p className="text-2xl font-bold text-gray-100">{dashboardData.dataset.rows}</p>
                </div>
                <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-700/50">
                  <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Geographies</p>
                  <p className="text-lg font-medium text-gray-200">{dashboardData.dataset.geographies?.join(", ")}</p>
                </div>
              </div>
            </DashWidget>
          )}

          {dashboardData.model_spec && (
            <DashWidget title="Model Configuration" color="blue" dotColor="bg-blue-400"
              expandContent={
                <div className="max-w-2xl mx-auto space-y-4 pt-4">
                  <div className="flex justify-between items-center bg-gray-900/50 p-4 rounded-xl border border-gray-700/50">
                    <span className="text-gray-400">KPI</span>
                    <span className="font-semibold text-gray-100 text-lg">{dashboardData.model_spec.kpi}</span>
                  </div>
                  <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-700/50">
                    <p className="text-sm text-gray-400 mb-3">Media Channels</p>
                    <div className="flex flex-wrap gap-2">
                      {dashboardData.model_spec.media_channels?.map((c: any) => (
                        <span key={c.name} className="px-4 py-2 bg-blue-900/30 text-blue-300 text-sm rounded-full border border-blue-800/50">{c.name}</span>
                      ))}
                    </div>
                  </div>
                  <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-700/50">
                    <p className="text-sm text-gray-400 mb-3">Control Variables</p>
                    <div className="flex flex-wrap gap-2">
                      {dashboardData.model_spec.control_variables?.map((c: any) => (
                        <span key={c.name} className="px-4 py-2 bg-gray-700 text-gray-300 text-sm rounded-full border border-gray-600">{c.name}</span>
                      ))}
                    </div>
                  </div>
                </div>
              }
            >
              <div className="space-y-3">
                <div className="flex justify-between items-center bg-gray-900/50 p-3 rounded-lg border border-gray-700/50">
                  <span className="text-sm text-gray-400">KPI</span>
                  <span className="font-medium text-gray-100">{dashboardData.model_spec.kpi}</span>
                </div>
                <div className="bg-gray-900/50 p-3 rounded-lg border border-gray-700/50">
                  <span className="text-sm text-gray-400 block mb-2">Media Channels</span>
                  <div className="flex flex-wrap gap-2">
                    {dashboardData.model_spec.media_channels?.map((c: any) => (
                      <span key={c.name} className="px-3 py-1 bg-blue-900/30 text-blue-300 text-xs rounded-full border border-blue-800/50">{c.name}</span>
                    ))}
                  </div>
                </div>
                <div className="bg-gray-900/50 p-3 rounded-lg border border-gray-700/50">
                  <span className="text-sm text-gray-400 block mb-2">Control Variables</span>
                  <div className="flex flex-wrap gap-2">
                    {dashboardData.model_spec.control_variables?.map((c: any) => (
                      <span key={c.name} className="px-3 py-1 bg-gray-700 text-gray-300 text-xs rounded-full border border-gray-600">{c.name}</span>
                    ))}
                  </div>
                </div>
              </div>
            </DashWidget>
          )}

          {dashboardData.model_status === "completed" && (
            <DashWidget title="Model Successfully Fit" color="green" dotColor="bg-green-400 animate-pulse">
              <p className="text-sm text-gray-300">{dashboardData.summary}</p>
            </DashWidget>
          )}

          {dashboardData.report_path && (
            <DashWidget
              title="Full MMM Report"
              color="violet" dotColor="bg-violet-400"
              expandTitle="MMM Report"
              expandContent={
                <div className="h-[80vh] flex flex-col gap-0">
                  <iframe
                    src="http://localhost:8000/report"
                    className="flex-1 w-full rounded-xl border border-gray-700"
                    title="MMM Report"
                    sandbox="allow-scripts allow-same-origin"
                  />
                </div>
              }
            >
              <div className="flex flex-col gap-3">
                <p className="text-sm text-gray-400">Your full analysis report is ready with all model diagnostics, ROI analysis, and channel decomposition.</p>
                <div className="flex gap-3">
                  <a
                    href="http://localhost:8000/report/download"
                    download="mmm_report.html"
                    className="flex items-center gap-2 px-4 py-2 bg-violet-700 hover:bg-violet-600 text-white text-sm rounded-xl transition-colors font-medium"
                  >
                    <Download size={15} />
                    Download Report
                  </a>
                  <a
                    href="http://localhost:8000/report"
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm rounded-xl transition-colors font-medium"
                  >
                    <ExternalLink size={15} />
                    Open in New Tab
                  </a>
                </div>
                <div className="rounded-xl overflow-hidden border border-gray-700 bg-white" style={{height: '340px'}}>
                  <iframe
                    src="http://localhost:8000/report"
                    className="w-full h-full"
                    title="MMM Report Preview"
                    sandbox="allow-scripts allow-same-origin"
                  />
                </div>
              </div>
            </DashWidget>
          )}

          {dashboardData.roi_metrics && (() => {
            const roiTable = (
              <div className="overflow-x-auto rounded-xl border border-gray-700/50">
                <table className="w-full text-left text-sm">
                  <thead className="bg-gray-900 text-gray-400 uppercase text-xs">
                    <tr>
                      <th className="px-4 py-3 font-medium">Channel</th>
                      <th className="px-4 py-3 font-medium">Mean ROI</th>
                      <th className="px-4 py-3 font-medium">94% HDI</th>
                      <th className="px-4 py-3 font-medium">Prob. Profitable</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700/50">
                    {dashboardData.roi_metrics.map((row: any) => (
                      <tr key={row.channel} className="bg-gray-800/50 hover:bg-gray-700/50 transition-colors">
                        <td className="px-4 py-3 font-medium text-gray-100">{row.channel}</td>
                        <td className="px-4 py-3 text-emerald-400 font-semibold">{row.roi_mean?.toFixed(2)}x</td>
                        <td className="px-4 py-3 text-gray-400">[{row.roi_hdi_low?.toFixed(2)}, {row.roi_hdi_high?.toFixed(2)}]</td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                              <div className="h-full bg-emerald-500" style={{ width: `${(row.prob_profitable || 0) * 100}%` }} />
                            </div>
                            <span className="text-gray-300 text-xs">{((row.prob_profitable || 0) * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
            return (
              <DashWidget title="ROI Performance" color="emerald" dotColor="bg-emerald-400" expandContent={roiTable}>
                {roiTable}
              </DashWidget>
            );
          })()}

          {dashboardData.plots && dashboardData.plots.length > 0 && (
            <DashWidget
              title={`Generated Visualizations (${dashboardData.plots.length})`}
              color="fuchsia" dotColor="bg-fuchsia-400 animate-pulse"
            >
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
