import { clsx } from 'clsx';
import { Card } from '../../components/ui';
import { useUsage } from '../../api/hooks/useAccount';
import type { UsageBucket } from '../../api/services/accountService';

function fmtLimit(v: number | null): string {
  return v == null ? '∞' : String(v);
}

function Tile({ label, bucket }: { label: string; bucket: UsageBucket }) {
  return (
    <Card padding="md">
      <p className="text-xs font-medium uppercase tracking-wider text-ink-400">{label}</p>
      <p className={clsx('mt-1 font-display text-2xl font-semibold tabular-nums', bucket.over ? 'text-rust-700' : 'text-ink-900')}>
        {bucket.used}
        <span className="text-base font-normal text-ink-400"> / {fmtLimit(bucket.limit)}</span>
      </p>
      {bucket.over && <p className="mt-0.5 text-xs text-rust-600">over limit</p>}
    </Card>
  );
}

export function SeatUsage() {
  const { data, isLoading } = useUsage();
  if (isLoading || !data) {
    return (
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-20 animate-pulse rounded-lg border border-line-200 bg-cream-100" />
        ))}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      <Card padding="md">
        <p className="text-xs font-medium uppercase tracking-wider text-ink-400">Plan</p>
        <p className="mt-1 font-display text-2xl font-semibold text-ink-900">{data.plan_name}</p>
      </Card>
      <Tile label="Seats" bucket={data.seats} />
      <Tile label="Projects" bucket={data.projects} />
      <Tile label="Fits this month" bucket={data.fits_this_month} />
    </div>
  );
}
