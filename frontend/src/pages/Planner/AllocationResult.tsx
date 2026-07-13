import { DataTable, type Column } from '../../components/ui';
import type {
  AllocationRow,
  BudgetPlanResult,
  FrontierResult,
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

function fmt2(v: number | undefined | null): string {
  return v == null || !Number.isFinite(v) ? '—' : v.toFixed(2);
}

/**
 * The efficient frontier (#139): optimized return as budget scales, with a
 * credible band and the marginal ROI (next-dollar return) at each budget. A
 * dependency-free inline-bar table — invest while the marginal ROI stays above
 * your breakeven.
 */
function FrontierTable({ frontier }: { frontier: FrontierResult }) {
  const maxRet = Math.max(...frontier.points.map((p) => p.return_p95), 1e-9);
  return (
    <div>
      <h4 className="mb-1.5 text-sm font-semibold text-ink-900">
        Efficient frontier{' '}
        <span className="font-normal text-ink-400">
          (return vs budget — {frontier.objective_label})
        </span>
      </h4>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">
              <th className="py-1 text-left">Budget</th>
              <th className="py-1 text-right">Return</th>
              <th className="py-1 text-left pl-3">Return (90% band)</th>
              <th className="py-1 text-right">Marginal ROI</th>
            </tr>
          </thead>
          <tbody>
            {frontier.points.map((p, i) => {
              const lo = (100 * p.return_p5) / maxRet;
              const w = (100 * (p.return_p95 - p.return_p5)) / maxRet;
              const mid = (100 * p.expected_return) / maxRet;
              return (
                <tr key={i} className="border-t border-line-100">
                  <td className="py-1 num">{fmtInt(p.total_budget)}</td>
                  <td className="py-1 text-right num font-medium">
                    {fmtInt(p.expected_return)}
                  </td>
                  <td className="py-1 pl-3">
                    <div className="relative h-2.5 w-full min-w-[120px] rounded bg-cream-100">
                      <div
                        className="absolute h-2.5 rounded bg-sage-200"
                        style={{ left: `${lo}%`, width: `${Math.max(w, 1)}%` }}
                      />
                      <div
                        className="absolute h-2.5 w-0.5 bg-sage-700"
                        style={{ left: `${mid}%` }}
                      />
                    </div>
                  </td>
                  <td className="py-1 text-right num">
                    {Number.isFinite(p.marginal_roi) ? fmt2(p.marginal_roi) : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function AllocationResult({ plan }: { plan: BudgetPlanResult }) {
  // Budget optimizer v2 (#139): append a Marginal ROAS column when available
  // (the funding line — fund a channel while its marginal return exceeds 1).
  const marg = plan.marginal_roas;
  const allocCols: Column<AllocationRow>[] = marg
    ? [
        ...ALLOC_COLS,
        {
          key: 'marginal_roas',
          header: 'Marg. ROAS',
          numeric: true,
          render: (r) => (
            <span className={(marg[r.channel] ?? 0) >= 1 ? 'text-sage-800' : 'text-ink-400'}>
              {fmt2(marg[r.channel])}
            </span>
          ),
        },
      ]
    : ALLOC_COLS;
  const advanced =
    (plan.objective && plan.objective !== 'mean') ||
    (plan.mode && plan.mode !== 'fixed') ||
    plan.shadow_price != null;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <Metric
          label={plan.mode === 'free' ? 'Budget (breakeven)' : 'Budget allocated'}
          value={fmtInt(plan.total_budget)}
          hint={plan.objective_label}
        />
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

      {advanced && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {plan.objective_label && (
            <span className="rounded-full bg-steel-100 px-2.5 py-1 text-steel-700">
              Objective: {plan.objective_label}
            </span>
          )}
          {plan.mode === 'free' && (
            <span className="rounded-full bg-gold-100 px-2.5 py-1 text-gold-700">
              Breakeven mode
            </span>
          )}
          {plan.shadow_price != null && (
            <span className="rounded-full bg-sage-100 px-2.5 py-1 text-sage-800">
              Shadow price (next-$ return): {fmt2(plan.shadow_price)}
            </span>
          )}
        </div>
      )}

      {plan.goal_seek && (
        <div className="rounded-lg border border-line-200 bg-cream-50 px-3 py-2.5 text-sm">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-ink-400">
            Goal-seek to KPI {fmtInt(plan.goal_seek.target_kpi)}
          </div>
          {plan.goal_seek.feasible ? (
            <div className="text-ink-700">
              Needs{' '}
              <span className="font-semibold num">
                {fmtInt(plan.goal_seek.required_budget ?? 0)}
              </span>{' '}
              budget — probability it clears the target:{' '}
              <span className="font-semibold">
                {Math.round((plan.goal_seek.prob_hit_target ?? 0) * 100)}%
              </span>
            </div>
          ) : (
            <div className="text-rust-600">
              Not reachable within the supported spend range.
            </div>
          )}
        </div>
      )}

      {plan.frontier && plan.frontier.points.length > 0 && (
        <FrontierTable frontier={plan.frontier} />
      )}

      <div>
        <h4 className="mb-1.5 text-sm font-semibold text-ink-900">Recommended allocation</h4>
        <DataTable columns={allocCols} rows={plan.allocation} rowKey={(r) => r.channel} />
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
