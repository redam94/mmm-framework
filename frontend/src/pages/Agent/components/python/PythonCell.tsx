import React from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useInView } from '../../hooks/useInView';
import { PythonCodeBlock } from './PythonCodeBlock';
import { PythonOutputBlock } from './PythonOutputBlock';
import type { PythonOutput } from '../../types';

// One REPL cell. Memoized + viewport-gated: the expensive SyntaxHighlighter and
// output block only mount when the cell scrolls near view, and a streaming update
// to OTHER cells (or the collapse of a sibling) never re-highlights this one.
export const PythonCell = React.memo(function PythonCell({
  out,
  index,
  isCollapsed,
  onToggle,
}: {
  out: PythonOutput;
  index: number;
  isCollapsed: boolean;
  onToggle: (id: string) => void;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  const hasCode = !!out.code.trim();
  const firstLine = hasCode ? out.code.trim().split('\n')[0] : '(output only)';
  return (
    <div ref={ref} className="rounded-xl overflow-hidden shadow-sm">
      {/* Cell header */}
      <button
        onClick={() => onToggle(out.id)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-cream-100 hover:bg-gray-200 transition-colors text-left border border-line-200 rounded-t-xl"
      >
        <span className="text-[10px] font-mono text-ink-300 shrink-0">In [{index + 1}]</span>
        <span className="flex-1 text-[11px] font-mono text-ink-600 truncate">{firstLine}</span>
        {out.hasError && <span className="text-[9px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded font-semibold">ERROR</span>}
        {out.plotCount > 0 && <span className="text-[9px] bg-fuchsia-100 text-fuchsia-600 px-1.5 py-0.5 rounded font-semibold">{out.plotCount} plot{out.plotCount > 1 ? 's' : ''}</span>}
        {isCollapsed ? <ChevronRight size={13} className="text-ink-300 shrink-0" /> : <ChevronDown size={13} className="text-ink-300 shrink-0" />}
      </button>
      {!isCollapsed && (
        <div className="border-l border-r border-b border-line-200 rounded-b-xl overflow-hidden min-h-[2.5rem]">
          {inView ? (
            <>
              {hasCode && <PythonCodeBlock code={out.code} />}
              <PythonOutputBlock output={out.output} hasError={out.hasError} />
            </>
          ) : (
            <div className="h-10 bg-cream-50" />
          )}
        </div>
      )}
    </div>
  );
});
