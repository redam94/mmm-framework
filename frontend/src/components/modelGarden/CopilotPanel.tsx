/**
 * Modeling copilot — a Bayesian-modeling + PyMC + mmm-framework expert chat that
 * lives beside the Atelier editor. It streams from `POST /model-garden/copilot`
 * (grounded in the current editor source) and surfaces an "Apply to editor"
 * action on any code block it returns.
 */
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ClipboardCheck, RotateCcw, Send, Sparkles, Square, X } from 'lucide-react';
import {
  copilotService,
  readCopilotStream,
  type CopilotTurn,
} from '../../api/services/copilotService';
import { lastCodeBlock, MD_COMPONENTS } from './copilotMarkdown';

interface Msg {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
}

interface Props {
  sourceCode: string;
  onApplyCode: (code: string) => void;
  onClose?: () => void;
  className?: string;
}

const QUICK_PROMPTS = [
  'Explain what _build_model must register for the read-ops to work',
  'Add a non-centered hierarchical geo random effect to this model',
  'Convert any adstock/state recursion here to a vectorized form',
  'Why might my media coefficients be unidentified, and how do I fix it?',
];

function CopilotMessage({ msg, onApplyCode }: { msg: Msg; onApplyCode: (c: string) => void }) {
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
  return (
    <div className="flex flex-col items-start gap-1.5">
      <div className="max-w-full rounded-2xl rounded-bl-md border border-line-200 bg-white px-3.5 py-2 text-sm text-ink-700">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
          {msg.content || '…'}
        </ReactMarkdown>
      </div>
      {code && (
        <button
          onClick={() => onApplyCode(code)}
          className="inline-flex items-center gap-1.5 rounded-md border border-sage-300 bg-sage-100/70 px-2.5 py-1 text-xs font-medium text-sage-800 transition-colors hover:bg-sage-100"
        >
          <ClipboardCheck size={13} /> Apply to editor
        </button>
      )}
    </div>
  );
}

export function CopilotPanel({ sourceCode, onApplyCode, onClose, className }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  // Keep the latest source for the request without re-rendering on every keystroke.
  const sourceRef = useRef(sourceCode);
  sourceRef.current = sourceCode;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streaming]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || streaming) return;
    setInput('');
    const aiId = `a-${Date.now()}`;
    const history: CopilotTurn[] = [
      ...messages
        .filter((m) => m.role !== 'error')
        .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content })),
      { role: 'user', content: q },
    ];
    setMessages((prev) => [
      ...prev,
      { id: `u-${Date.now()}`, role: 'user', content: q },
      { id: aiId, role: 'assistant', content: '' },
    ]);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await copilotService.streamCopilot(history, sourceRef.current, controller.signal);
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

  const stop = () => abortRef.current?.abort();
  const clear = () => {
    abortRef.current?.abort();
    setMessages([]);
  };

  return (
    <div className={`flex flex-col bg-cream-50 ${className ?? ''}`}>
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-line-200 bg-sage-100 px-3 py-2.5">
        <Sparkles size={16} className="text-sage-700" />
        <div className="min-w-0 flex-1">
          <h3 className="font-display text-sm font-semibold text-ink-900">Modeling copilot</h3>
          <p className="truncate text-[11px] text-ink-500">Bayesian + PyMC + this framework's contract</p>
        </div>
        {messages.length > 0 && (
          <button onClick={clear} title="Clear" className="text-ink-400 hover:text-ink-700">
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
            Ask about priors, identifiability, adstock/saturation, or the Model Garden contract.
            I read your editor source — when I write code you can apply it in one click.
          </p>
        )}
        {messages.map((m) => (
          <CopilotMessage key={m.id} msg={m} onApplyCode={onApplyCode} />
        ))}
        {streaming && messages[messages.length - 1]?.content === '' && (
          <div className="flex items-center gap-2 px-1 text-xs text-ink-400">
            <span className="flex gap-1">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:0ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:150ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:300ms]" />
            </span>
            <span className="animate-pulse">thinking…</span>
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
          placeholder="Ask the modeling copilot…  (Enter to send)"
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

export default CopilotPanel;
