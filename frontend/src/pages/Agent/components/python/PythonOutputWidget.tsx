import { useCallback, useState } from 'react';
import { Download, Trash2 } from 'lucide-react';
import { DashWidget } from '../common/DashWidget';
import { PythonCell } from './PythonCell';
import type { PythonOutput } from '../../types';

export function PythonOutputWidget({ outputs, onClear, onExport }: { outputs: PythonOutput[]; onClear: () => void; onExport?: () => void }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggle = useCallback((id: string) => setCollapsed(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  }), []);

  if (outputs.length === 0) return null;

  return (
    <DashWidget
      title={`Python REPL (${outputs.length} run${outputs.length > 1 ? 's' : ''})`}
      dotColor="bg-emerald-500"
      color="emerald"
    >
      <div className="space-y-1 mb-2 flex items-center justify-between">
        <p className="text-xs text-ink-400">{outputs.length} execution{outputs.length > 1 ? 's' : ''} recorded this session.</p>
        <div className="flex items-center gap-3">
          {onExport && (
            <button onClick={onExport} className="text-[10px] text-ink-300 hover:text-indigo-600 flex items-center gap-1 transition-colors" title="Download this session's work as a standalone, runnable Python script">
              <Download size={11} /> Download .py
            </button>
          )}
          <button onClick={onClear} className="text-[10px] text-ink-300 hover:text-red-500 flex items-center gap-1 transition-colors">
            <Trash2 size={11} /> Clear
          </button>
        </div>
      </div>
      <div className="space-y-4">
        {outputs.map((out, idx) => (
          <PythonCell
            key={out.id}
            out={out}
            index={idx}
            isCollapsed={collapsed.has(out.id)}
            onToggle={toggle}
          />
        ))}
      </div>
    </DashWidget>
  );
}
