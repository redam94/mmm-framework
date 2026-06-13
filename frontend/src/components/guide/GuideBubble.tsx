import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ExternalLink, MessageCircleQuestion, Send, X } from 'lucide-react';
import { clsx } from 'clsx';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';
import { useProjectStore } from '../../stores/projectStore';
import { useGuideChat } from './useGuideChat';
import type { GuideMessage } from './useGuideChat';

// ─── Route-aware context ──────────────────────────────────────────────────────

function pageContextFor(pathname: string): string {
  if (pathname.startsWith('/program')) {
    return 'The user is viewing the Orrery page (the measurement-program home): measurement-cycle stage (T0–T5), headline KPIs (portfolio mROI, misallocation, % spend experiment-backed, mean ROI CI width), next-best-actions, calibration coverage map, identification contract, recent activity.';
  }
  if (pathname.startsWith('/experiments')) {
    return 'The user is viewing the Auspices page (experiments): EIG/EVOI priority matrix, experiment lifecycle board (draft→planned→running→completed→calibrated), re-test schedule with information decay, and the experiment design studio (randomized geo lift, matched-market DiD, randomized flighting).';
  }
  if (pathname.startsWith('/performance')) {
    return 'The user is viewing the Chronicle page (performance): cycle-over-cycle trajectories (ROI CI contraction, budget share migration, misallocation, portfolio mROI), saturation & avg-vs-marginal ROAS curves, the model–experiment agreement log, model health diagnostics, and the run lineage timeline.';
  }
  if (pathname.startsWith('/team')) {
    return "The user is viewing the College page (team): the project's user roster and roles.";
  }
  return 'The user is in Augur, the causal marketing-measurement app.';
}

function quickPromptsFor(pathname: string): string[] {
  if (pathname.startsWith('/program')) {
    return ["What are this client's goals?", 'What should we do next?'];
  }
  if (pathname.startsWith('/experiments')) {
    return ['Explain the priority matrix', 'Which experiment should we run next and why?'];
  }
  if (pathname.startsWith('/performance')) {
    return ['Summarize how measurement improved', 'Explain the agreement log'];
  }
  return ['How does the measurement loop work?', "What's in the project brief?"];
}

/** Friendly labels for the tool-activity line (raw tool names read poorly). */
const TOOL_LABELS: Record<string, string> = {
  search_knowledge_base: 'searching project docs',
  list_knowledge_base: 'checking project docs',
};

function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? name.replace(/_/g, ' ');
}

// ─── Markdown styling (compact, bubble-sized) ────────────────────────────────

const MD_COMPONENTS: Components = {
  p: ({ children }) => <p className="my-1.5 leading-relaxed first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="my-1.5 list-disc space-y-0.5 pl-4">{children}</ul>,
  ol: ({ children }) => <ol className="my-1.5 list-decimal space-y-0.5 pl-4">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-ink-900">{children}</strong>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-sage-800 underline decoration-sage-300 hover:decoration-sage-600"
    >
      {children}
    </a>
  ),
  code: ({ children }) => (
    <code className="rounded bg-cream-100 px-1 py-0.5 font-mono text-xs text-sage-800">
      {children}
    </code>
  ),
  h1: ({ children }) => <p className="my-1.5 font-semibold text-ink-900">{children}</p>,
  h2: ({ children }) => <p className="my-1.5 font-semibold text-ink-900">{children}</p>,
  h3: ({ children }) => <p className="my-1.5 font-semibold text-ink-900">{children}</p>,
  table: ({ children }) => (
    <div className="my-1.5 overflow-x-auto">
      <table className="min-w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-line-200 bg-cream-100 px-2 py-1 text-left font-semibold text-sage-800">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="border border-line-200 px-2 py-1">{children}</td>,
};

function MessageBubble({ msg }: { msg: GuideMessage }) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-sage-700 px-3.5 py-2 text-sm leading-relaxed text-white">
          {msg.content}
        </div>
      </div>
    );
  }
  if (msg.role === 'error') {
    return (
      <div className="flex justify-start">
        <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-rust-600/30 bg-rust-100 px-3.5 py-2 text-sm leading-relaxed text-rust-700">
          {msg.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-line-200 bg-white px-3.5 py-2 text-sm text-ink-700">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
          {msg.content}
        </ReactMarkdown>
      </div>
    </div>
  );
}

// ─── Bubble + panel ───────────────────────────────────────────────────────────

/**
 * Floating per-project guide chat. Mounted in AppShell, so it appears on every
 * shell page (Program / Experiments / Performance / Team) but not on the
 * full-screen workspace or login.
 */
export function GuideBubble() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  // Quick prompts hide once a message has been sent during this open.
  const [sentThisOpen, setSentThisOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { currentProjectId } = useProjectStore();
  const { messages, threadId, streaming, activeTool, init, send } =
    useGuideChat(currentProjectId);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const pageContext = useMemo(() => pageContextFor(location.pathname), [location.pathname]);
  const quickPrompts = useMemo(() => quickPromptsFor(location.pathname), [location.pathname]);

  // Create/hydrate the guide session lazily on first open (per project).
  useEffect(() => {
    if (open && currentProjectId) void init();
  }, [open, currentProjectId, init]);

  useEffect(() => {
    if (open) messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [open, messages, streaming]);

  const handleSend = (text: string) => {
    if (!text.trim() || streaming) return;
    setInput('');
    setSentThisOpen(true);
    void send(text, pageContext);
  };

  const toggleOpen = () => {
    setOpen((v) => {
      if (!v) setSentThisOpen(false);
      return !v;
    });
  };

  return (
    <>
      {open && (
        <div className="fixed bottom-24 right-6 z-40 flex max-h-[60vh] w-[380px] flex-col overflow-hidden rounded-xl border border-line-200 bg-white shadow-2xl">
          {/* Header */}
          <div className="flex items-start justify-between gap-2 border-b border-line-200 bg-sage-100 px-4 py-3">
            <div className="min-w-0">
              <h2 className="font-display text-base font-semibold text-ink-900">Project guide</h2>
              <p className="mt-0.5 text-xs text-ink-600">
                Grounded in this page and your project docs
              </p>
            </div>
            {threadId && (
              <button
                onClick={() => navigate(`/workspace?session=${threadId}`)}
                title="Open full workspace"
                className="mt-0.5 shrink-0 rounded-md p-1.5 text-sage-800 transition-colors hover:bg-sage-300/40"
              >
                <ExternalLink size={15} />
              </button>
            )}
          </div>

          {!currentProjectId ? (
            <div className="flex flex-col items-center bg-cream-50 px-6 py-10 text-center">
              <MessageCircleQuestion className="mb-3 h-7 w-7 text-ink-300" strokeWidth={1.5} />
              <p className="font-display text-sm font-semibold text-ink-900">
                Pick a project to start the guide
              </p>
              <p className="mt-1 text-xs text-ink-400">
                Choose a project from the switcher in the header — the guide is scoped to it.
              </p>
            </div>
          ) : (
            <>
              {/* Messages */}
              <div className="flex-1 space-y-3 overflow-y-auto bg-cream-50 p-4 scrollbar-thin">
                {messages.length === 0 && !streaming && (
                  <p className="px-2 pt-4 text-center text-xs leading-relaxed text-ink-400">
                    Ask what a chart means, what to do next, or about this client's brief and
                    goals — the guide draws on your project's knowledge base.
                  </p>
                )}
                {messages.map((m) => (
                  <MessageBubble key={m.id} msg={m} />
                ))}
                {streaming && (
                  <div className="flex items-center gap-2 px-1 text-xs text-ink-400">
                    <span className="flex gap-1">
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:0ms]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:150ms]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-600 [animation-delay:300ms]" />
                    </span>
                    <span className="animate-pulse">
                      {activeTool ? `⚙ ${toolLabel(activeTool)}…` : 'thinking…'}
                    </span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Quick prompts (until first message this open) */}
              {!sentThisOpen && !streaming && (
                <div className="flex flex-wrap gap-1.5 border-t border-line-200 bg-white px-3 pt-2.5">
                  {quickPrompts.map((p) => (
                    <button
                      key={p}
                      onClick={() => handleSend(p)}
                      className="rounded-full border border-sage-300 bg-sage-100/60 px-2.5 py-1 text-xs text-sage-800 transition-colors hover:bg-sage-100"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              )}

              {/* Input */}
              <div className="relative flex items-center bg-white p-3">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend(input)}
                  placeholder="Ask the guide…"
                  disabled={streaming}
                  className="w-full rounded-full border border-line-300 bg-cream-100 py-2.5 pl-4 pr-11 text-sm text-ink-900 placeholder-ink-300 transition-all focus:outline-none focus:ring-2 focus:ring-sage-600 disabled:opacity-60"
                />
                <button
                  onClick={() => handleSend(input)}
                  disabled={!input.trim() || streaming}
                  className="absolute right-5 rounded-full bg-sage-700 p-1.5 text-white transition-colors hover:bg-sage-600 disabled:opacity-50"
                >
                  <Send size={14} />
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Floating toggle */}
      <button
        onClick={toggleOpen}
        title={open ? 'Close guide' : 'Project guide'}
        className={clsx(
          'fixed bottom-6 right-6 z-40 flex h-13 w-13 items-center justify-center rounded-full',
          'bg-sage-700 text-white shadow-lg transition-all duration-200 hover:bg-sage-800 hover:scale-105 active:scale-95',
        )}
      >
        {open ? <X size={22} /> : <MessageCircleQuestion size={22} />}
      </button>
    </>
  );
}
