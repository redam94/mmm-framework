import { clsx } from 'clsx';
import type { LearningWave, LearningWaveSource } from '../../api/services/learningService';
import { fmtDollars, fmtSignedDollars } from './format';

const SOURCE_BADGE: Record<LearningWaveSource, { label: string; cls: string }> = {
  wave: { label: 'designed wave', cls: 'bg-sage-100 text-sage-800' },
  experiment_import: { label: 'experiment import', cls: 'bg-steel-100 text-steel-700' },
  manual: { label: 'manual', cls: 'bg-cream-200 text-ink-600' },
};

function fmtUnix(ts: number | null | undefined): string {
  return ts ? new Date(ts * 1000).toLocaleDateString() : '—';
}

/** Total $/geo-period the recommended allocation moved vs the previous wave's snapshot. */
function recommendationShift(
  rec: Record<string, number> | undefined,
  prev: Record<string, number> | undefined,
): number | null {
  if (!rec || !prev) return null;
  const keys = new Set([...Object.keys(rec), ...Object.keys(prev)]);
  let total = 0;
  for (const k of keys) total += Math.abs((rec[k] ?? 0) - (prev[k] ?? 0));
  return total;
}

/**
 * Per-wave history of the learning loop (RunsTimeline idiom): design → ingest
 * → refit, with the post-fit movement in E[regret] and the recommendation.
 */
export function WaveTimeline({ waves }: { waves: LearningWave[] }) {
  const ordered = [...waves].sort((a, b) => b.wave_index - a.wave_index);

  if (ordered.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-line-300 bg-cream-200/60 px-4 py-6 text-center text-sm text-ink-400">
        No waves yet — design the first wave or import past experiments.
      </p>
    );
  }

  // chronological snapshots for delta computation
  const chrono = [...waves].sort((a, b) => a.wave_index - b.wave_index);
  const prevSnapshotOf = (wave: LearningWave) => {
    let prev = null;
    for (const w of chrono) {
      if (w.wave_index >= wave.wave_index) break;
      if (w.snapshot) prev = w.snapshot;
    }
    return prev;
  };

  return (
    <div className="relative">
      <div className="absolute left-[6px] top-6 bottom-6 w-px bg-line-200" />
      {ordered.map((w, i) => {
        const badge = w.source ? SOURCE_BADGE[w.source] : null;
        const snap = w.snapshot;
        const prev = prevSnapshotOf(w);
        const regretDelta =
          snap && prev ? snap.regret.e_regret_dollars - prev.regret.e_regret_dollars : null;
        const shift = snap ? recommendationShift(snap.recommendation, prev?.recommendation) : null;
        return (
          <div key={w.id} className="relative pl-8">
            <div
              className={clsx(
                'absolute left-0 top-5 h-3.5 w-3.5 rounded-full border-2',
                i === 0 ? 'border-sage-700 bg-sage-700' : 'border-line-300 bg-white',
              )}
            />
            <div className="mb-4 rounded-lg border border-line-200 bg-white p-4 shadow-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-display text-sm font-semibold text-ink-900">
                  Wave <span className="num">{w.wave_index}</span>
                </span>
                {badge && (
                  <span className={clsx('rounded-full px-2 py-0.5 text-[10px] font-medium', badge.cls)}>
                    {badge.label}
                  </span>
                )}
                <span
                  className={clsx(
                    'rounded-full px-2 py-0.5 text-[10px] font-medium',
                    w.status === 'ingested'
                      ? 'bg-sage-100 text-sage-800'
                      : 'bg-gold-100 text-gold-700',
                  )}
                >
                  {w.status}
                </span>
                <span className="ml-auto num text-xs text-ink-400">{fmtUnix(w.created_at)}</span>
              </div>

              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-600">
                {w.design?.n_cells != null && (
                  <span>
                    <span className="num font-medium text-ink-900">{w.design.n_cells}</span> design
                    cells{w.design.delta != null && (
                      <>
                        {' '}· δ <span className="num">{w.design.delta}</span>
                      </>
                    )}
                  </span>
                )}
                {w.experiment_ids != null && w.experiment_ids.length > 0 && (
                  <span>
                    <span className="num font-medium text-ink-900">{w.experiment_ids.length}</span>{' '}
                    imported readouts
                  </span>
                )}
              </div>

              {snap && (
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 border-t border-line-200 pt-2 text-xs">
                  <span className="text-ink-600">
                    E[regret]{' '}
                    <span className="num font-medium text-ink-900">
                      {fmtDollars(snap.regret.e_regret_dollars)}
                    </span>
                    {regretDelta != null && (
                      <span
                        className={clsx(
                          'num ml-1',
                          regretDelta <= 0 ? 'text-sage-800' : 'text-rust-700',
                        )}
                      >
                        ({fmtSignedDollars(regretDelta)})
                      </span>
                    )}
                  </span>
                  {shift != null && (
                    <span className="text-ink-600">
                      recommendation shift{' '}
                      <span className="num font-medium text-ink-900">{fmtDollars(shift)}</span>
                      /geo-period
                    </span>
                  )}
                  <span
                    className={clsx(
                      'rounded-full px-2 py-0.5 text-[10px] font-medium',
                      snap.regret.stop ? 'bg-cream-200 text-ink-600' : 'bg-sage-100 text-sage-800',
                    )}
                  >
                    {snap.regret.stop ? 'stop' : 'keep testing'}
                  </span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
