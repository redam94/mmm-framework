import { DataTable, type Column } from '../../components/ui';
import type {
  AllocationRow,
  BudgetPlanResult,
  GeoAllocationRow,
} from '../../api/services/plannerService';
import { FlightingCalendar } from './FlightingCalendar';
import { fmtInt, fmtSignedPct } from './format';

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-line-200 bg-white px-3 py-2.5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div className="mt-0.5 font-display text-xl font-semibold text-ink-900 num">{value}</div>
      {hint && <div className="num text-[10px] text-ink-400">{hint}</div>}
    </div>
  );
}

const ALLOC_COLS: Column<AllocationRow>[] = [
  { key: 'channel', header: 'Channel', render: (r) => r.channel },
  {
    key: 'current_spend',
    header: 'Current',
    numeric: true,
    render: (r) => fmtInt(r.current_spend),
  },
  {
    key: 'optimal_spend',
    header: 'Recommended',
    numeric: true,
    render: (r) => fmtInt(r.optimal_spend),
  },
  {
    key: 'change_pct',
    header: 'Change',
    numeric: true,
    render: (r) => (
      <span className={changeClass(r.change_pct)}>{fmtSignedPct(r.change_pct)}</span>
    ),
  },
];

const GEO_COLS: Column<GeoAllocationRow>[] = [
  { key: 'geo', header: 'Geography', render: (r) => r.geo },
  { key: 'channel', header: 'Channel', render: (r) => r.channel },
  {
    key: 'optimal_spend',
    header: 'Recommended',
    numeric: true,
    render: (r) => fmtInt(r.optimal_spend),
  },
  {
    key: 'change_pct',
    header: 'Change',
    numeric: true,
    render: (r) => (
      <span className={changeClass(r.change_pct)}>{fmtSignedPct(r.change_pct)}</span>
    ),
  },
];

function changeClass(pct: number | undefined): string {
  if (pct == null) return 'text-ink-400';
  if (pct > 1) return 'text-sage-800';
  if (pct < -1) return 'text-rust-600';
  return 'text-ink-500';
}

export function AllocationResult({ plan }: { plan: BudgetPlanResult }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <Metric label="Budget allocated" value={fmtInt(plan.total_budget)} />
        <Metric
          label="Expected uplift"
          value={fmtInt(plan.expected_uplift)}
          hint={`90% [${fmtInt(plan.uplift_hdi?.[0])}, ${fmtInt(plan.uplift_hdi?.[1])}]`}
        />
        <Metric
          label="P(uplift > 0)"
          value={`${Math.round((plan.prob_positive_uplift ?? 0) * 100)}%`}
          hint={`${plan.n_draws} draws`}
        />
      </div>

      <div>
        <h4 className="mb-1.5 text-sm font-semibold text-ink-900">Recommended allocation</h4>
        <DataTable columns={ALLOC_COLS} rows={plan.allocation} rowKey={(r) => r.channel} />
      </div>

      {plan.by_geo && plan.geo_allocation && plan.geo_allocation.length > 0 && (
        <div>
          <h4 className="mb-1.5 text-sm font-semibold text-ink-900">
            Allocation by geography{' '}
            <span className="font-normal text-ink-400">({plan.geos?.length ?? 0} geos)</span>
          </h4>
          <DataTable
            columns={GEO_COLS}
            rows={plan.geo_allocation}
            rowKey={(r) => `${r.geo}|${r.channel}`}
          />
        </div>
      )}

      {plan.flighting && plan.flighting.schedule.length > 0 && (
        <div>
          <h4 className="mb-1.5 text-sm font-semibold text-ink-900">
            Flighting calendar{' '}
            <span className="font-normal text-ink-400">({plan.flighting.pattern})</span>
          </h4>
          <FlightingCalendar flighting={plan.flighting} />
        </div>
      )}

      {plan.notes.length > 0 && (
        <ul className="space-y-0.5 text-xs text-gold-700">
          {plan.notes.map((n, i) => (
            <li key={i}>⚠ {n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
