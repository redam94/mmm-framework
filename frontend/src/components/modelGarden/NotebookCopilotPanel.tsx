/**
 * Notebook copilot — the same Bayesian-modeling + PyMC expert as the editor
 * copilot, but wired into the Atelier notebook so it can DIAGNOSE a failed cell.
 * It streams from `POST /model-garden/copilot` grounded in the model source PLUS
 * the failing cell's code + traceback + dataset preview + sibling cells, and
 * surfaces "Apply to cell" / "Insert as new cell" on any code block it returns.
 */
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { remarkPlugins, rehypePlugins, normalizeMath } from '../../lib/markdownMath';
import { ClipboardCheck, FileCode, Plus, RotateCcw, Send, Sparkles, Square, X } from 'lucide-react';
import {
  copilotService,
  readCopilotStream,
  type CopilotTurn,
  type NotebookCopilotContext,
} from '../../api/services/copilotService';
import { useCopilotChatState, type PersistedMsg } from '../../api/hooks';
import { lastCodeBlock, MD_COMPONENTS } from './copilotMarkdown';
import type { NotebookCell as Cell } from '../../api/services/atelierNotebookService';

/** A request to diagnose a specific cell, raised from the cell's error output.
 * `nonce` re-triggers the auto-send even when the same cell is diagnosed twice. */
export interface DiagnoseRequest {
  nonce: number;
  cellId: string;
  cellIndex: number;
  code: string;
  traceback: string;
}

/** Assistant turns carry the cell they target (Apply writes back there). */
type Msg = PersistedMsg;

interface Props {
  sourceCode: string;
  datasetPreview?: string | null;
  cells: Cell[];
  diagnoseRequest: DiagnoseRequest | null;
  onApplyCode: (code: string, targetCellId: string | null) => void;
  /** Apply a code block back to the MODEL SOURCE editor (for model-class fixes —
   * the diagnosis prompt can return a corrected `_build_model`). */
  onApplyToEditor?: (code: string) => void;
  /** Model identity the chat is scoped to (per model/version memory). */
  name: string | null;
  version?: number | null;
  /** Whether the rail is open — gates loading the persisted chat. */
  active?: boolean;
  onClose?: () => void;
  className?: string;
}

/** Heuristic: does this code block look like the MODEL SOURCE (a CustomMMM
 * subclass / GARDEN_MODEL / _build_model) rather than a notebook cell? Those
 * belong in the editor, not a cell — the kernel imports the model from the editor
 * buffer, so pasting a class into a cell would be a dead apply. */
function looksLikeModelSource(code: string): boolean {
  return (
    /class\s+\w+\s*\([^)]*(CustomMMM|BayesianMMM|BaseExtendedMMM)/.test(code) ||
    /\bGARDEN_MODEL\s*=/.test(code) ||
    /def\s+_build_model\b/.test(code)
  );
}

const QUICK_PROMPTS = [
  'Why did my last fit fail to start (logp -inf at the initial point)?',
  'My fit is very slow — is a pytensor.scan the cause, and how do I vectorize it?',
  'Add a cell that plots the prior vs posterior for the media coefficients',
  'Map my uploaded MFF columns into the spec correctly',
];

function ApplyActions({
  code,
  targetCellId,
  cellIndex,
  onApplyCode,
  onApplyToEditor,
}: {
  code: string;
  targetCellId: string | null | undefined;
  cellIndex: number | null;
  onApplyCode: (code: string, targetCellId: string | null) => void;
  onApplyToEditor?: (code: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const canTargetCell = !!targetCellId && cellIndex != null;
  // A model-class fix belongs in the editor source (the kernel imports the model
  // from there), not a notebook cell — offer the editor write-back instead.
  const isModelSource = !!onApplyToEditor && looksLikeModelSource(code);
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {isModelSource ? (
        <button
          onClick={() => onApplyToEditor!(code)}
          className="inline-flex items-center gap-1.5 rounded-md border border-sage-300 bg-sage-100/70 px-2.5 py-1 text-xs font-medium text-sage-800 transition-colors hover:bg-sage-100"
        >
          <FileCode size={13} /> Apply to editor (model source)
        </button>
      ) : (
        <button
          onClick={() => onApplyCode(code, canTargetCell ? targetCellId! : null)}
          className="inline-flex items-center gap-1.5 rounded-md border border-sage-300 bg-sage-100/70 px-2.5 py-1 text-xs font-medium text-sage-800 transition-colors hover:bg-sage-100"
        >
          {canTargetCell ? (
            <>
              <ClipboardCheck size={13} /> Apply to cell [{cellIndex! + 1}]
            </>
          ) : (
            <>
              <Plus size={13} /> Insert as new cell
            </>
          )}
        </button>
      )}
      <button
        onClick={() => {
          navigator.clipboard?.writeText(code).then(
            () => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            },
            () => {},
          );
        }}
        className="inline-flex items-center gap-1.5 rounded-md border border-line-300 bg-white px-2.5 py-1 text-xs font-medium text-ink-600 transition-colors hover:bg-cream-100"
      >
        {copied ? 'Copied ✓' : 'Copy'}
      </button>
    </div>
  );
}

function NotebookCopilotMessage({
  msg,
  cells,
  onApplyCode,
  onApplyToEditor,
}: {
  msg: Msg;
  cells: Cell[];
  onApplyCode: (code: string, targetCellId: string | null) => void;
  onApplyToEditor?: (code: string) => void;
}) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[88%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-sage-700 px-3.5 py-2 text-sm leading-relaxed text-white">
          {msg.content}
        </div>
      </div>
    );
  }
  if (msg.role === 'error') {
    return (
      <div className="flex justify-start">
        <div className="max-w-[92%] rounded-2xl rounded-bl-md border border-rust-600/30 bg-rust-100 px-3.5 py-2 text-sm leading-relaxed text-rust-700">
          {msg.content || 'Something went wrong.'}
        </div>
      </div>
    );
  }
  const code = lastCodeBlock(msg.content);
  // Resolve the target cell's CURRENT index (it may have moved/been deleted).
  const idx = msg.targetCellId ? cells.findIndex((c) => c.id === msg.targetCellId) : -1;
  return (
    <div className="flex flex-col items-start gap-1.5">
      <div className="max-w-full rounded-2xl rounded-bl-md border border-line-200 bg-white px-3.5 py-2 text-sm text-ink-700">
        <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={MD_COMPONENTS}>
          {normalizeMath(msg.content || '…')}
        </ReactMarkdown>
      </div>
      {code && (
        <ApplyActions
          code={code}
          targetCellId={idx >= 0 ? msg.targetCellId : null}
          cellIndex={idx >= 0 ? idx : null}
          onApplyCode={onApplyCode}
          onApplyToEditor={onApplyToEditor}
        />
      )}
    </div>
  );
}

export function NotebookCopilotPanel({
  sourceCode,
  datasetPreview,
  cells,
  diagnoseRequest,
  onApplyCode,
  onApplyToEditor,
  name,
  version = null,
  active = true,
  onClose,
  className,
}: Props) {
  const { messages, setMessages, clear: clearChat } = useCopilotChatState({
    name,
    version,
    surface: 'notebook',
    enabled: active,
  });
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the latest notebook state for the request without re-rendering on every
  // keystroke (mirrors CopilotPanel's sourceRef pattern).
  const sourceRef = useRef(sourceCode);
  sourceRef.current = sourceCode;
  const cellsRef = useRef(cells);
  cellsRef.current = cells;
  const previewRef = useRef(datasetPreview);
  previewRef.current = datasetPreview;
  const streamingRef = useRef(streaming);
  streamingRef.current = streaming;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streaming]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const send = async (
    visibleText: string,
    opts: {
      targetCellId?: string | null;
      cellCode?: string;
      traceback?: string;
      isError?: boolean;
    } = {},
  ) => {
    const q = visibleText.trim();
    if (!q || streamingRef.current) return;
    setInput('');
    const targetCellId = opts.targetCellId ?? null;
    const aiId = `a-${Date.now()}`;

    // Build the notebook context for grounding: the sibling code cells (so the
    // assistant sees where `data`/`spec`/`mmm` come from), the dataset preview,
    // and (on diagnose) the failing cell + traceback.
    const others = cellsRef.current
      .filter((c) => c.type === 'code' && c.id !== targetCellId && c.source.trim())
      .map((c) => c.source)
      .slice(0, 12);
    const notebook: NotebookCopilotContext = {
      cell_code: opts.cellCode ?? '',
      traceback: opts.traceback ?? '',
      dataset_preview: previewRef.current ?? null,
      other_cells: others,
      is_error: !!opts.isError,
    };

    const history: CopilotTurn[] = [
      ...messages
        .filter((m) => m.role !== 'error')
        .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content })),
      { role: 'user', content: q },
    ];
    setMessages((prev) => [
      ...prev,
      { id: `u-${Date.now()}`, role: 'user', content: q },
      { id: aiId, role: 'assistant', content: '', targetCellId },
    ]);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await copilotService.streamCopilot(
        history,
        sourceRef.current,
        controller.signal,
        notebook,
      );
      await readCopilotStream(
        res,
        (acc) => setMessages((prev) => prev.map((m) => (m.id === aiId ? { ...m, content: acc } : m))),
        (content) =>
          setMessages((prev) =>
            prev.map((m) => (m.id === aiId ? { ...m, role: 'error', content } : m)),
          ),
      );
    } catch (e) {
      const err = e as Error;
      if (err.name !== 'AbortError') {
        setMessages((prev) =>
          prev.map((m) => (m.id === aiId ? { ...m, role: 'error', content: err.message } : m)),
        );
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  // Auto-send a diagnosis when the notebook raises a DiagnoseRequest (nonce-keyed
  // so re-diagnosing the same cell fires again). If the copilot is mid-stream, we
  // do NOT consume the nonce — `streaming` is a dep, so the effect re-runs and the
  // queued diagnosis fires the moment the current stream finishes (no dropped
  // request, no need to click Diagnose twice).
  const lastNonceRef = useRef<number>(0);
  useEffect(() => {
    if (!diagnoseRequest || diagnoseRequest.nonce === lastNonceRef.current) return;
    if (streaming) return;
    lastNonceRef.current = diagnoseRequest.nonce;
    void send(`Diagnose & fix the error in cell [${diagnoseRequest.cellIndex + 1}].`, {
      targetCellId: diagnoseRequest.cellId,
      cellCode: diagnoseRequest.code,
      traceback: diagnoseRequest.traceback,
      isError: true,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [diagnoseRequest, streaming]);

  const stop = () => abortRef.current?.abort();
  const clear = () => {
    abortRef.current?.abort();
    clearChat();
  };

  return (
    <div className={`flex flex-col bg-cream-50 ${className ?? ''}`}>
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-line-200 bg-sage-100 px-3 py-2.5">
        <Sparkles size={16} className="text-sage-700" />
        <div className="min-w-0 flex-1">
          <h3 className="font-display text-sm font-semibold text-ink-900">Notebook copilot</h3>
          <p className="truncate text-[11px] text-ink-500">Diagnose cell errors · tips · rewrites</p>
        </div>
        {messages.length > 0 && (
          <button onClick={clear} title="Clear this model's chat" className="text-ink-400 hover:text-ink-700">
            <RotateCcw size={14} />
          </button>
        )}
        {onClose && (
          <button onClick={onClose} title="Close" className="text-ink-400 hover:text-ink-700">
            <X size={16} />
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 space-y-3 overflow-y-auto p-3 scrollbar-thin">
        {messages.length === 0 && !streaming && (
          <p className="px-1 pt-2 text-xs leading-relaxed text-ink-400">
            Run a cell — if it errors, hit <strong className="text-ink-600">Diagnose</strong> on the
            output and I'll explain the root cause and rewrite the cell. I read your model source,
            the failing cell + traceback, and your other cells.
          </p>
        )}
        {messages.map((m) => (
          <NotebookCopilotMessage
            key={m.id}
            msg={m}
            cells={cells}
            onApplyCode={onApplyCode}
            onApplyToEditor={onApplyToEditor}
          />
        ))}
        {streaming && messages[messages.length - 1]?.content === '' && (
          <div className="flex items-center gap-2 px-1 text-xs text-ink-400">
            <span className="flex gap-1">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:0ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:150ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:300ms]" />
            </span>
            <span className="animate-pulse">diagnosing…</span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Quick prompts */}
      {messages.length === 0 && !streaming && (
        <div className="flex flex-col gap-1 border-t border-line-200 bg-white px-3 py-2">
          {QUICK_PROMPTS.map((p) => (
            <button
              key={p}
              onClick={() => send(p)}
              className="rounded-md border border-sage-200 bg-sage-100/50 px-2.5 py-1.5 text-left text-xs text-sage-800 transition-colors hover:bg-sage-100"
            >
              {p}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="flex items-end gap-1.5 border-t border-line-200 bg-white p-2.5">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          rows={2}
          placeholder="Ask about a cell, or fix an error…  (Enter to send)"
          disabled={streaming}
          className="max-h-28 min-h-[2.5rem] flex-1 resize-none rounded-md border border-line-300 bg-cream-50 px-3 py-2 text-sm text-ink-900 placeholder-ink-300 focus:outline-none focus:ring-2 focus:ring-sage-600 disabled:opacity-60"
        />
        {streaming ? (
          <button
            onClick={stop}
            title="Stop"
            className="rounded-md bg-rust-600 p-2 text-white transition-colors hover:bg-rust-700"
          >
            <Square size={15} />
          </button>
        ) : (
          <button
            onClick={() => send(input)}
            disabled={!input.trim()}
            title="Send"
            className="rounded-md bg-sage-700 p-2 text-white transition-colors hover:bg-sage-600 disabled:opacity-50"
          >
            <Send size={15} />
          </button>
        )}
      </div>
    </div>
  );
}

export default NotebookCopilotPanel;
