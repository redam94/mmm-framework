import { clsx } from 'clsx';
import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react';

interface StatHeroProps {
  label: string;
  /** Pre-formatted value — rendered in display serif with mono fallback for digits */
  value: string;
  /** Smaller trailing unit, e.g. "%" or "/wk" */
  unit?: string;
  /** Signed delta vs previous cycle; direction inferred from sign */
  delta?: number | null;
  /** Format the delta (default: ±x.x%) */
  deltaLabel?: string;
  /** Whether an increase is good (mROI up = sage) or bad (misallocation up = rust) */
  increaseIsGood?: boolean;
  hint?: string;
  className?: string;
}

/** Hero metric: big Fraunces number, mono-formatted, with a semantic delta. */
export function StatHero({
  label,
  value,
  unit,
  delta,
  deltaLabel,
  increaseIsGood = true,
  hint,
  className,
}: StatHeroProps) {
  const hasDelta = delta !== undefined && delta !== null && Number.isFinite(delta);
  const up = hasDelta && (delta as number) > 0;
  const flat = hasDelta && Math.abs(delta as number) < 1e-9;
  const good = flat ? null : up === increaseIsGood;
  const DeltaIcon = flat ? Minus : up ? ArrowUpRight : ArrowDownRight;

  return (
    <div className={clsx('rounded-lg border border-line-200 bg-white p-5 shadow-sm', className)}>
      <div className="text-xs font-medium uppercase tracking-wider text-ink-400">{label}</div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className="font-display text-3xl font-semibold tracking-tight text-ink-900 num">
          {value}
        </span>
        {unit && <span className="text-sm text-ink-400">{unit}</span>}
      </div>
      <div className="mt-2 flex items-center gap-2 text-xs">
        {hasDelta && (
          <span
            className={clsx(
              'inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 font-medium num',
              flat && 'bg-cream-200 text-ink-600',
              good === true && 'bg-sage-100 text-sage-800',
              good === false && 'bg-rust-100 text-rust-700',
            )}
          >
            <DeltaIcon className="h-3 w-3" />
            {deltaLabel ?? `${(delta as number) > 0 ? '+' : ''}${(delta as number).toFixed(1)}%`}
          </span>
        )}
        {hint && <span className="text-ink-400">{hint}</span>}
      </div>
    </div>
  );
}
