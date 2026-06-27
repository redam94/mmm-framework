import { ArrowDown, ArrowUp, X } from 'lucide-react';
import type { TransformStep } from '../../../types';

function summarize(step: TransformStep): string {
  switch (step.op) {
    case 'rename': return `Rename ${step.from} → ${step.to}`;
    case 'drop_columns': return `Drop ${step.columns.join(', ')}`;
    case 'cast': return `Cast ${step.column} → ${step.dtype}`;
    case 'parse_date': return `Parse date ${step.column ?? '(auto)'}`;
    case 'fill_missing': return `Fill missing (${step.strategy})${step.columns?.length ? ` in ${step.columns.join(', ')}` : ''}`;
    case 'drop_duplicates': return `Drop duplicate rows`;
    case 'filter_rows': return `Filter ${step.column} ${step.operator} ${String(step.value ?? '')}`;
    case 'date_range': return `Date range ${step.start ?? '…'} → ${step.end ?? '…'}`;
    case 'winsorize': return `Winsorize ${step.column} ≤ ${step.cap_value}`;
    case 'impute': return `Impute ${step.column} = ${step.value}`;
    case 'event_dummy': return `Event dummy ${step.name} (${step.periods.length})`;
    default: return JSON.stringify(step);
  }
}

export function StepList({
  steps, onReorder, onRemove, disabled,
}: {
  steps: TransformStep[];
  onReorder: (from: number, to: number) => void;
  onRemove: (idx: number) => void;
  disabled?: boolean;
}) {
  if (steps.length === 0) {
    return <p className="text-xs text-ink-300 py-3 text-center">No cleaning steps yet — add one below.</p>;
  }
  return (
    <ol className="space-y-1.5">
      {steps.map((step, i) => (
        <li key={i} className="flex items-center gap-2 bg-white border border-line-200 rounded-lg px-3 py-2">
          <span className="text-[10px] font-mono text-ink-300 w-5 shrink-0">{i + 1}</span>
          <span className="text-xs text-ink-700 flex-1 truncate" title={summarize(step)}>{summarize(step)}</span>
          <div className="flex items-center gap-0.5 shrink-0">
            <button onClick={() => onReorder(i, i - 1)} disabled={disabled || i === 0}
              className="p-1 rounded text-ink-300 hover:text-ink-700 disabled:opacity-30" title="Move up">
              <ArrowUp size={13} />
            </button>
            <button onClick={() => onReorder(i, i + 1)} disabled={disabled || i === steps.length - 1}
              className="p-1 rounded text-ink-300 hover:text-ink-700 disabled:opacity-30" title="Move down">
              <ArrowDown size={13} />
            </button>
            <button onClick={() => onRemove(i)} disabled={disabled}
              className="p-1 rounded text-ink-300 hover:text-red-500" title="Remove step">
              <X size={13} />
            </button>
          </div>
        </li>
      ))}
    </ol>
  );
}
