import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User } from 'lucide-react';
import { mdComponents } from '../common/markdown';
import { ToolCallBlock } from './ToolCallBlock';
import type { ChatMessage } from '../../types';

// One chat message. Memoized so a streaming update — which fires setMessages on
// every node-step — only re-renders the message that actually changed (the one
// being streamed) instead of re-parsing markdown + re-highlighting code for the
// ENTIRE conversation each step. `pending` is computed in the parent as
// `loading && isLast`, so the global `loading` flip doesn't invalidate the
// already-rendered history (only the last bubble depends on it).
export const ChatMessageBubble = React.memo(function ChatMessageBubble({
  msg,
  pending,
  onNavigate,
}: {
  msg: ChatMessage;
  pending: boolean;
  onNavigate: (tab: string) => void;
}) {
  // Built once per onNavigate identity (a stable useCallback/setState in the
  // page), so markdown components keep referential stability across renders
  // and the React.memo above stays effective.
  const components = useMemo(() => mdComponents(onNavigate), [onNavigate]);
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
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
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
