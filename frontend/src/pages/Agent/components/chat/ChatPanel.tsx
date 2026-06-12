import React, { useEffect, useRef } from 'react';
import { ArrowLeft, Paperclip, RotateCcw, Send, Square, Trash2 } from 'lucide-react';
import { ChatMessageBubble } from './ChatMessageBubble';
import { WorkflowSuggestions } from './WorkflowSuggestions';
import type { WorkflowStep } from '../../../../components/causal/CausalWidgets';
import type { ChatMessage, DashboardData } from '../../types';

export function ChatPanel({
  messages, loading, input, onInputChange,
  canBack, canRetry, lastAssistantHasContent,
  onBack, onRetry, onClear, onSend, onStop, onNavigate,
  fileInputRef, onFileUpload,
  dashboardData, projectId, workflow,
}: {
  messages: ChatMessage[];
  loading: boolean;
  input: string;
  onInputChange: (v: string) => void;
  canBack: boolean;
  canRetry: boolean;
  lastAssistantHasContent: boolean;
  onBack: () => void;
  onRetry: () => void;
  onClear: () => void;
  onSend: (messageOverride?: string) => void;
  onStop: () => void;
  onNavigate: (tab: string) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  dashboardData: DashboardData;
  projectId: string | null;
  workflow: WorkflowStep[];
}) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const done = workflow.filter((s) => s.status === 'done').length;
  const next = workflow.find((s) => s.status !== 'done' && s.status !== 'skipped');

  return (
    <div className="w-1/3 border-r border-line-200 flex flex-col bg-white shadow-sm relative z-10 shrink-0">
      <div className="p-4 border-b border-line-200 bg-white sticky top-0 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold font-display text-ink-900">
            MMM Copilot
          </h1>
          {/* Workflow position: where the session sits in the 9-step Bayesian
              workflow; click jumps to the checklist. */}
          {workflow.length > 0 && (
            <button
              onClick={() => onNavigate('workflow')}
              className="mt-1 flex items-center gap-1.5 text-xs text-ink-400 transition-colors hover:text-sage-800"
              title="Open the workflow checklist"
            >
              <span className="num font-medium text-sage-800">{done}/{workflow.length}</span>
              <span className="h-1.5 w-16 overflow-hidden rounded-full bg-cream-200">
                <span
                  className="block h-full rounded-full bg-sage-600 transition-all"
                  style={{ width: `${(100 * done) / Math.max(workflow.length, 1)}%` }}
                />
              </span>
              {next && <span className="truncate max-w-40">next: {next.title}</span>}
            </button>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0 mt-0.5">
          <button
            onClick={onBack}
            disabled={!canBack}
            title="Back to previous turn"
            className="p-1.5 rounded-lg text-ink-300 hover:text-sage-800 hover:bg-sage-100 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <ArrowLeft size={15} />
          </button>
          <button
            onClick={onRetry}
            disabled={!canRetry || !lastAssistantHasContent}
            title="Regenerate last response"
            className="p-1.5 rounded-lg text-ink-300 hover:text-sage-800 hover:bg-sage-100 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <RotateCcw size={15} />
          </button>
          <button
            onClick={onClear}
            disabled={loading || messages.length === 0}
            title="Clear conversation"
            className="p-1.5 rounded-lg text-ink-300 hover:text-rust-600 hover:bg-rust-100 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Trash2 size={15} />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-cream-50 scrollbar-thin">
        {messages.length === 0 && !loading && (
          <div className="mt-10 px-4 text-center">
            <p className="font-display text-base font-semibold text-ink-900">
              Work the measurement loop with the copilot
            </p>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-400">
              Upload or generate data, build the causal structure, fit and check the
              model, then plan the experiments that earn the next calibration.
              Use the suggestions below to take the next step.
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <ChatMessageBubble
            key={msg.id}
            msg={msg}
            pending={loading && i === messages.length - 1}
            onNavigate={onNavigate}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-line-200 bg-white pt-2.5">
        <WorkflowSuggestions
          dashboardData={dashboardData}
          projectId={projectId}
          disabled={loading}
          onSelect={(prompt) => onSend(prompt)}
        />
        <div className="relative flex items-center px-4 pb-4">
          <input type="file" ref={fileInputRef} className="hidden" onChange={onFileUpload} accept=".csv,.xlsx,.xls" />
          <button onClick={() => fileInputRef.current?.click()} disabled={loading}
            className="absolute left-6 p-2 text-ink-300 hover:text-sage-700 transition-colors disabled:opacity-50" title="Upload Dataset">
            <Paperclip size={18} />
          </button>
          <input
            type="text" value={input} onChange={e => onInputChange(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && onSend()}
            placeholder="Ask the agent to generate data, configure models, or explain ROI…"
            className="w-full bg-cream-100 border border-line-300 rounded-full py-3 px-12 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600 transition-all text-ink-900 placeholder-ink-300"
            disabled={loading}
          />
          {loading ? (
            <button onClick={onStop}
              className="absolute right-6 p-2 bg-rust-600 hover:bg-rust-700 rounded-full text-white transition-colors"
              title="Stop generation"
            >
              <Square size={16} fill="white" />
            </button>
          ) : (
            <button onClick={() => onSend()} disabled={!input.trim()}
              className="absolute right-6 p-2 bg-sage-700 hover:bg-sage-600 rounded-full text-white transition-colors disabled:opacity-50">
              <Send size={18} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
