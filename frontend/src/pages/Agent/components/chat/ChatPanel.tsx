import React, { useEffect, useRef } from 'react';
import { ArrowLeft, Paperclip, RotateCcw, Send, Square, Trash2 } from 'lucide-react';
import { ChatMessageBubble } from './ChatMessageBubble';
import type { ChatMessage } from '../../types';

export function ChatPanel({
  messages, loading, input, onInputChange,
  canBack, canRetry, lastAssistantHasContent,
  onBack, onRetry, onClear, onSend, onStop, onNavigate,
  fileInputRef, onFileUpload,
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
}) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="w-1/3 border-r border-gray-200 flex flex-col bg-white shadow-md relative z-10 shrink-0">
      <div className="p-4 border-b border-gray-200 bg-white sticky top-0 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold text-gray-800">
            MMM Copilot
          </h1>
        </div>
        <div className="flex items-center gap-1 shrink-0 mt-0.5">
          <button
            onClick={onBack}
            disabled={!canBack}
            title="Back to previous turn"
            className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <ArrowLeft size={15} />
          </button>
          <button
            onClick={onRetry}
            disabled={!canRetry || !lastAssistantHasContent}
            title="Regenerate last response"
            className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <RotateCcw size={15} />
          </button>
          <button
            onClick={onClear}
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
            onNavigate={onNavigate}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-gray-200 bg-white">
        <div className="relative flex items-center">
          <input type="file" ref={fileInputRef} className="hidden" onChange={onFileUpload} accept=".csv,.xlsx,.xls" />
          <button onClick={() => fileInputRef.current?.click()} disabled={loading}
            className="absolute left-2 p-2 text-gray-400 hover:text-indigo-500 transition-colors disabled:opacity-50" title="Upload Dataset">
            <Paperclip size={18} />
          </button>
          <input
            type="text" value={input} onChange={e => onInputChange(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && onSend()}
            placeholder="Ask the agent to generate data, configure models, or explain ROI…"
            className="w-full bg-gray-100 border border-gray-200 rounded-full py-3 px-12 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 transition-all text-gray-900 placeholder-gray-400"
            disabled={loading}
          />
          {loading ? (
            <button onClick={onStop}
              className="absolute right-2 p-2 bg-red-500 hover:bg-red-400 rounded-full text-white transition-colors"
              title="Stop generation"
            >
              <Square size={16} fill="white" />
            </button>
          ) : (
            <button onClick={() => onSend()} disabled={!input.trim()}
              className="absolute right-2 p-2 bg-indigo-600 hover:bg-indigo-500 rounded-full text-white transition-colors disabled:opacity-50">
              <Send size={18} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
