import { clsx } from 'clsx';
import { ArrowUp, ArrowDown, FlaskConical } from 'lucide-react';
import { DataTable, type Column } from '../../components/ui';
import type { PortfolioBrand } from '../../api/services/benchmarkService';
import { fmtRoi, fmtAge, leaderCount, laggardCount } from './helpers';

function StalePill({ brand }: { brand: PortfolioBrand }) {
  if (brand.last_fit_at == null) {
    return <span className="rounded bg-cream-200 px-1.5 py-0.5 text-[11px] font-medium text-ink-500">No fit</span>;
  }
  const stale = brand.stale === true;
  return (
    <span
      className={clsx(
        'rounded px-1.5 py-0.5 text-[11px] font-medium',
        stale ? 'bg-rust-100 text-rust-700' : 'bg-sage-100 text-sage-700',
      )}
    >
      {stale ? 'Stale' : 'Fresh'}
    </span>
  );
}

const COLUMNS: Column<PortfolioBrand>[] = [
  {
    key: 'name',
    header: 'Brand',
    render: (b) => (
      <div className="min-w-0">
        <p className="truncate font-medium text-ink-900">{b.name ?? b.project_id}</p>
        <p className="truncate text-[11px] text-ink-400">{b.n_runs} {b.n_runs === 1 ? 'run' : 'runs'}</p>
      </div>
    ),
  },
  {
    key: 'fit',
    header: 'Last fit',
    render: (b) => (
      <div className="flex items-center gap-2">
        <span className="tabular-nums text-ink-700">{fmtAge(b.age_days)}</span>
        <StalePill brand={b} />
      </div>
    ),
  },
  { key: 'channels', header: 'Channels', numeric: true, render: (b) => b.n_channels || '—' },
  {
    key: 'pmroi',
    header: 'Portfolio mROI',
    numeric: true,
    render: (b) => <span className="font-semibold text-ink-900">{fmtRoi(b.portfolio_marginal_roi)}</span>,
  },
  {
    key: 'top',
    header: 'Top channel',
    render: (b) =>
      b.top_channel ? (
        <div className="flex items-baseline gap-1.5">
          <span className="text-ink-800">{b.top_channel.channel}</span>
          <span className="text-[11px] tabular-nums text-ink-400">{fmtRoi(b.top_channel.roi_mean)}</span>
        </div>
      ) : (
        <span className="text-ink-300">—</span>
      ),
  },
  {
    key: 'vs',
    header: 'vs. portfolio',
    render: (b) => {
      const up = leaderCount(b);
      const down = laggardCount(b);
      if (up === 0 && down === 0) return <span className="text-ink-300">—</span>;
      return (
        <div className="flex items-center gap-2.5 text-[12px] tabular-nums">
          <span className={clsx('flex items-center gap-0.5', up > 0 ? 'text-sage-700' : 'text-ink-300')}>
            <ArrowUp size={12} />
            {up}
          </span>
          <span className={clsx('flex items-center gap-0.5', down > 0 ? 'text-rust-600' : 'text-ink-300')}>
            <ArrowDown size={12} />
            {down}
          </span>
        </div>
      );
    },
  },
  {
    key: 'calibrated',
    header: 'Calibrated',
    numeric: true,
    render: (b) =>
      b.n_calibrated > 0 ? (
        <span className="inline-flex items-center gap-1 rounded bg-gold-100 px-1.5 py-0.5 text-[11px] font-medium text-gold-700">
          <FlaskConical size={11} />
          {b.n_calibrated}
        </span>
      ) : (
        <span className="text-ink-300">0</span>
      ),
  },
];

export function BrandTable({ brands }: { brands: PortfolioBrand[] }) {
  // Fitted brands first, then by portfolio marginal ROI (desc), unfitted last.
  const rows = [...brands].sort((a, b) => {
    const av = a.portfolio_marginal_roi ?? -Infinity;
    const bv = b.portfolio_marginal_roi ?? -Infinity;
    return bv - av;
  });
  return <DataTable columns={COLUMNS} rows={rows} rowKey={(b) => b.project_id} />;
}
