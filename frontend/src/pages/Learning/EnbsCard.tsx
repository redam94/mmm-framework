import { clsx } from 'clsx';
import { CircleStop, Play } from 'lucide-react';
import type { LearningRegret } from '../../api/services/learningService';
import { fmtDollars, fmtNum } from './format';

/**
 * Stopping-rule verdict card: expected net benefit of sampling (ENBS) =
 * E[regret] in dollars − the cost of running another wave. `stop: false`
 * means learning still pays — run the next wave.
 */
export function EnbsCard({ regret }: { regret: LearningRegret }) {
  const keepTesting = !regret.stop;

  return (
    <div className="rounded-lg border border-line-200 bg-white p-5 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wider text-ink-400">
        Stopping rule · ENBS
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2.5">
        <span className="font-display text-2xl font-semibold tracking-tight text-ink-900">
          {keepTesting ? 'Run next wave' : 'Stop testing'}
        </span>
        <span
          className={clsx(
            'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
            keepTesting ? 'bg-sage-100 text-sage-800' : 'bg-cream-200 text-ink-600',
          )}
        >
          {keepTesting ? (
            <>
              <Play className="h-3 w-3" /> learning still pays
            </>
          ) : (
            <>
              <CircleStop className="h-3 w-3" /> learning has paid out
            </>
          )}
        </span>
      </div>
      <dl className="mt-3 space-y-1.5 text-sm">
        <div className="flex items-baseline justify-between gap-3">
          <dt className="text-xs text-ink-400">
            E[regret] of acting now{' '}
            <span title="expected value-$ left on the table per geo-period by the current best allocation">
              ($<span className="num">{fmtNum(regret.e_regret_kpi)}</span> / geo-period)
            </span>
          </dt>
          <dd className="num font-medium text-ink-900">{fmtDollars(regret.e_regret_dollars)}</dd>
        </div>
        <div className="flex items-baseline justify-between gap-3">
          <dt className="text-xs text-ink-400">Cost of one more wave</dt>
          <dd className="num font-medium text-ink-900">−{fmtDollars(regret.wave_cost)}</dd>
        </div>
        <div className="flex items-baseline justify-between gap-3 border-t border-line-200 pt-1.5">
          <dt className="text-xs font-semibold text-ink-600">Expected net benefit</dt>
          <dd
            className={clsx(
              'num font-semibold',
              regret.enbs > 0 ? 'text-sage-800' : 'text-rust-700',
            )}
          >
            {fmtDollars(regret.enbs)}
          </dd>
        </div>
      </dl>
      <p className="mt-2.5 text-[11px] leading-snug text-ink-300">
        E[regret] per geo-period × margin (
        <span className="num">{fmtNum(regret.margin, 1)}</span>) × population (
        <span className="num">{regret.population}</span> geo-periods = geos × horizon) vs the
        wave cost — stop when the value of what's left to learn no longer covers the next
        test.
      </p>
    </div>
  );
}
