import { AlertTriangle, Check, Lock, X } from 'lucide-react';
import { fmtVal, lockPathLabel } from '../../utils/spec';

// ─── PendingSpecChangesModal ──────────────────────────────────────────────────
// When the LLM proposes a change to a field the user locked manually, the change
// is deferred server-side and surfaced here for explicit confirmation.
export interface PendingChange { path: string; current: unknown; proposed: unknown; reason?: string | null }

export function PendingSpecChangesModal({
  changes, onResolve,
}: {
  changes: PendingChange[];
  onResolve: (path: string, action: 'approve' | 'reject') => void;
}) {
  if (!changes || changes.length === 0) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-lg bg-white rounded-2xl shadow-xl border border-line-200 overflow-hidden">
        <div className="flex items-center gap-2.5 px-5 py-4 border-b border-line-200 bg-amber-50">
          <AlertTriangle size={16} className="text-amber-600 shrink-0" />
          <span className="font-semibold text-sm text-amber-700 flex-1">
            Confirm change{changes.length > 1 ? 's' : ''} to your locked settings
          </span>
        </div>
        <div className="px-5 py-4 space-y-3 max-h-[60vh] overflow-y-auto">
          <p className="text-xs text-ink-400">
            The assistant wants to change setting{changes.length > 1 ? 's' : ''} you set manually.
            Approve to apply, or keep your value.
          </p>
          {changes.map((c) => (
            <div key={c.path} className="rounded-xl border border-line-200 p-3">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Lock size={11} className="text-amber-500 shrink-0" />
                <span className="text-xs font-semibold text-ink-900">{lockPathLabel(c.path)}</span>
              </div>
              <div className="flex items-center gap-2 text-xs mb-1.5">
                <span className="px-2 py-0.5 rounded-md bg-green-50 text-green-700 border border-green-200">
                  yours: {fmtVal(c.current)}
                </span>
                <span className="text-ink-300">→</span>
                <span className="px-2 py-0.5 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200">
                  proposed: {fmtVal(c.proposed)}
                </span>
              </div>
              {c.reason && <p className="text-[11px] text-ink-400 italic mb-2">“{c.reason}”</p>}
              <div className="flex items-center gap-2">
                <button onClick={() => onResolve(c.path, 'approve')}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg transition-colors">
                  <Check size={12} /> Approve
                </button>
                <button onClick={() => onResolve(c.path, 'reject')}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-cream-100 hover:bg-gray-200 text-ink-600 text-xs font-medium rounded-lg border border-line-200 transition-colors">
                  <X size={12} /> Keep mine
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
