import { useMemo } from 'react';
import { DataTable, type Column } from '../../components/ui';
import type { BudgetPlanInfo } from '../../api/hooks/useBudgetPlans';
import type { BudgetPlanResult } from '../../api/services/plannerService';
import { fmtInt, fmtSignedInt } from './format';

interface Row {
  channel: string;
  a: number;
  b: number;
  delta: number;
}

/** Side-by-side per-channel allocation delta between two saved optimization
 * plans — the analyst's "why did the recommendation change?" view. */
export function PlanCompare({ a, b }: { a: BudgetPlanInfo; b: BudgetPlanInfo }) {
  const rows = useMemo<Row[]>(() => {
    const pa = (a.plan_payload as BudgetPlanResult)?.allocation ?? [];
    const pb = (b.plan_payload as BudgetPlanResult)?.allocation ?? [];
    const mapA = new Map(pa.map((r) => [r.channel, r.optimal_spend]));
    const mapB = new Map(pb.map((r) => [r.channel, r.optimal_spend]));
    const channels = Array.from(new Set([...mapA.keys(), ...mapB.keys()]));
    return channels.map((ch) => {
      const av = mapA.get(ch) ?? 0;
      const bv = mapB.get(ch) ?? 0;
      return { channel: ch, a: av, b: bv, delta: bv - av };
    });
  }, [a, b]);

  const cols: Column<Row>[] = [
    { key: 'channel', header: 'Channel', render: (r) => r.channel },
    { key: 'a', header: a.name, numeric: true, render: (r) => fmtInt(r.a) },
    { key: 'b', header: b.name, numeric: true, render: (r) => fmtInt(r.b) },
    {
      key: 'delta',
      header: 'Δ (B − A)',
      numeric: true,
      render: (r) => (
        <span
          className={r.delta > 0 ? 'text-sage-800' : r.delta < 0 ? 'text-rust-600' : 'text-ink-500'}
        >
          {fmtSignedInt(r.delta)}
        </span>
      ),
    },
  ];

  return <DataTable columns={cols} rows={rows} rowKey={(r) => r.channel} />;
}
