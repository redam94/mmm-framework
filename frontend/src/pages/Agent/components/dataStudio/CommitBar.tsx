import { CheckCircle2, Loader2, Trash2 } from 'lucide-react';
import type { DataStudioState } from '../../types';

// Sticky footer: convert the cleaned staged frame into the session's working
// dataset. Disabled while a chat turn is streaming (so the commit-merge can't
// race a concurrent dashboard_update).
export function CommitBar({
  state, busy, chatLoading, error, onCommit, onDiscard,
}: {
  state: DataStudioState | null;
  busy: boolean;
  chatLoading: boolean;
  error: string | null;
  onCommit: () => void;
  onDiscard: () => void;
}) {
  const kpis = state ? Object.values(state.roles).filter(r => r === 'kpi').length : 0;
  const media = state ? Object.values(state.roles).filter(r => r === 'media').length : 0;
  const ready = kpis >= 1 && media >= 1;

  return (
    <div className="shrink-0 border-t border-line-200 bg-white px-5 py-3 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <button onClick={onDiscard} disabled={busy}
          className="flex items-center gap-1.5 text-xs text-ink-400 hover:text-red-500 transition-colors disabled:opacity-40">
          <Trash2 size={13} /> Discard
        </button>
        {state && (
          <span className="text-xs text-ink-300 truncate">
            {state.n_rows.toLocaleString()} rows · {kpis} KPI · {media} media · {state.steps.length} step(s)
          </span>
        )}
        {error && <span className="text-xs text-red-600 truncate">{error}</span>}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {!ready && <span className="text-xs text-ink-300">Assign a KPI + ≥1 media role to continue</span>}
        <button
          onClick={onCommit}
          disabled={busy || chatLoading || !ready}
          title={chatLoading ? 'Wait for the assistant to finish' : undefined}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-sage-700 hover:bg-sage-800 text-white text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {busy ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle2 size={15} />}
          Use as dataset
        </button>
      </div>
    </div>
  );
}
