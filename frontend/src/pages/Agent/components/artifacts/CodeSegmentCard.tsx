import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Copy, FileCode } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { PythonOutput } from '../../types';

/** A collapsed-by-default executed-code segment for the grouped Results
 *  timeline: header shows the first code line + error/plot badges + copy;
 *  expanding reveals the syntax-highlighted source and its output.
 *
 *  React.memo by python.id: a segment is immutable once produced, but every
 *  SSE chunk rebuilds the group objects — without the memo an expanded card
 *  would re-run Prism tokenization on every streamed token. */
export const CodeSegmentCard = React.memo(function CodeSegmentCard({ python }: { python: PythonOutput }) {
  const [open, setOpen] = useState(false);
  const firstLine = (python.code.split('\n').find(l => l.trim()) ?? '(no code)').trim();

  return (
    <div className="rounded-xl border border-line-200 bg-white shadow-sm overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          onClick={() => setOpen(v => !v)}
          className="flex items-center gap-2 flex-1 min-w-0 text-left"
          title={open ? 'Collapse code' : 'Expand code'}
        >
          {open
            ? <ChevronDown size={13} className="text-ink-300 shrink-0" />
            : <ChevronRight size={13} className="text-ink-300 shrink-0" />}
          <FileCode size={12} className={`shrink-0 ${python.hasError ? 'text-red-500' : 'text-amber-600'}`} />
          <span className="text-[11px] font-mono text-ink-400 truncate flex-1">{firstLine}</span>
        </button>
        {python.hasError && (
          <span className="text-[10px] font-semibold text-red-600 bg-red-50 border border-red-200 rounded-full px-2 py-0.5 shrink-0">
            error
          </span>
        )}
        {python.plotCount > 0 && (
          <span className="text-[10px] text-ink-400 bg-cream-100 border border-line-200 rounded-full px-2 py-0.5 shrink-0">
            {python.plotCount} plot{python.plotCount === 1 ? '' : 's'}
          </span>
        )}
        <button
          onClick={() => navigator.clipboard.writeText(python.code)}
          className="p-1 rounded hover:bg-cream-100 text-ink-400 hover:text-ink-900 shrink-0"
          title="Copy code"
        >
          <Copy size={11} />
        </button>
      </div>
      {open && (
        <>
          <div className="overflow-x-auto max-h-64 overflow-y-auto bg-[#fafafa] border-t border-line-200">
            <SyntaxHighlighter
              language="python"
              style={oneLight}
              PreTag="div"
              customStyle={{ margin: 0, padding: '0.5rem 0.75rem', fontSize: '0.6875rem', background: '#fafafa' }}
              codeTagProps={{ style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' } }}
            >
              {python.code}
            </SyntaxHighlighter>
          </div>
          {python.output.trim() && (
            <pre className={`text-[11px] font-mono px-3 py-2 border-t border-line-200 max-h-40 overflow-auto whitespace-pre-wrap ${
              python.hasError ? 'bg-red-50 text-red-700' : 'bg-cream-50 text-ink-700'
            }`}>
              {python.output}
            </pre>
          )}
        </>
      )}
    </div>
  );
}, (a, b) => a.python.id === b.python.id);
