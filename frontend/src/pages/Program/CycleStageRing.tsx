import { clsx } from 'clsx';

export type CycleStage = 0 | 1 | 2 | 3 | 4 | 5;

export const STAGES: { t: string; name: string; detail: string }[] = [
  { t: 'T₀', name: 'Fit baseline', detail: 'Fit the MMM on history; document posterior widths.' },
  { t: 'T₁', name: 'Prioritize', detail: 'Score channels by EIG × EVOI; pick the experiment portfolio.' },
  { t: 'T₂', name: 'Run experiments', detail: 'Execute pre-registered geo-lift / pulse tests.' },
  { t: 'T₃', name: 'Calibrate', detail: 'Fold readouts into the next fit as likelihoods.' },
  { t: 'T₄', name: 'Allocate', detail: 'Budget from the calibrated posterior, with confidence tiers.' },
  { t: 'T₅', name: 'Re-evaluate', detail: 'Recompute priorities with tightened posteriors; loop.' },
];

interface CycleStageRingProps {
  current: CycleStage;
  /** one-line reason the program is at this stage */
  reason?: string;
}

/**
 * The measurement cycle as a horizontal stepper (doubles as onboarding: future
 * stages stay visible with their one-line descriptions).
 */
export function CycleStageRing({ current, reason }: CycleStageRingProps) {
  return (
    <div className="rounded-lg border border-line-200 bg-white p-5 shadow-sm">
      <div className="flex items-baseline justify-between">
        <h2 className="font-display text-lg font-semibold text-ink-900">Measurement cycle</h2>
        {reason && <span className="text-xs text-ink-400">{reason}</span>}
      </div>
      <ol className="mt-4 grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-3 xl:grid-cols-6">
        {STAGES.map((s, i) => {
          const state = i < current ? 'done' : i === current ? 'active' : 'todo';
          return (
            <li key={s.t} className="relative">
              {/* connector */}
              {i < STAGES.length - 1 && (
                <span
                  className={clsx(
                    'absolute left-9 top-4 hidden h-px w-[calc(100%-2rem)] xl:block',
                    i < current ? 'bg-sage-600' : 'bg-line-300',
                  )}
                />
              )}
              <div className="flex items-start gap-3 xl:flex-col xl:gap-2">
                <span
                  className={clsx(
                    'flex h-8 w-8 shrink-0 items-center justify-center rounded-full font-display text-sm font-semibold ring-1',
                    state === 'done' && 'bg-sage-600 text-white ring-sage-600',
                    state === 'active' && 'bg-gold-100 text-gold-700 ring-gold-600',
                    state === 'todo' && 'bg-cream-100 text-ink-400 ring-line-300',
                  )}
                >
                  {s.t}
                </span>
                <div className="min-w-0">
                  <div
                    className={clsx(
                      'text-sm font-medium',
                      state === 'active' ? 'text-ink-900' : state === 'done' ? 'text-ink-700' : 'text-ink-400',
                    )}
                  >
                    {s.name}
                    {state === 'active' && (
                      <span className="ml-1.5 rounded-full bg-gold-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-gold-700">
                        now
                      </span>
                    )}
                  </div>
                  <p className={clsx('mt-0.5 text-xs leading-snug', state === 'active' ? 'text-ink-600' : 'text-ink-300')}>
                    {s.detail}
                  </p>
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
