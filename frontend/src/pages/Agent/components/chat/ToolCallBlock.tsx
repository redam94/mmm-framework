import { useState } from 'react';
import {
  CheckCircle2, ChevronDown, ChevronRight, ExternalLink, Loader2, Wrench, X,
} from 'lucide-react';
import { TOOL_TO_TAB } from '../../constants';
import { formatToolName, truncate } from '../../utils/text';
import type { ToolCall } from '../../types';

export function ToolCallBlock({ toolCall, onNavigate }: { toolCall: ToolCall; onNavigate?: (tab: string) => void }) {
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
