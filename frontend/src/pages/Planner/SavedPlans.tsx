import { useMemo, useState } from 'react';
import { Download, Trash2, Eye, GitCompare } from 'lucide-react';
import { EmptyState } from '../../components/ui';
import {
  budgetPlanCsvUrl,
  useDeleteBudgetPlan,
  type BudgetPlanInfo,
} from '../../api/hooks/useBudgetPlans';
import type { BudgetPlanResult } from '../../api/services/plannerService';
import { AllocationResult } from './AllocationResult';
import { PlanCompare } from './PlanCompare';
import { fmtInt } from './format';

function isOptimization(p: BudgetPlanInfo): boolean {
  return p.kind !== 'scenario' && !!(p.plan_payload as BudgetPlanResult)?.allocation;
}

export function SavedPlans({ plans }: { plans: BudgetPlanInfo[] }) {
  const del = useDeleteBudgetPlan();
  const [openId, setOpenId] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);

  const opened = plans.find((p) => p.plan_id === openId) ?? null;
  const comparePair = useMemo(
    () =>
      compareIds
        .map((id) => plans.find((p) => p.plan_id === id))
        .filter((p): p is BudgetPlanInfo => !!p),
    [compareIds, plans],
  );

  const toggleCompare = (id: string) =>
    setCompareIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id].slice(-2),
    );

  if (plans.length === 0) {
    return (
      <EmptyState
        title="No saved plans yet"
        description="Build a plan or run a what-if above, then save it here to compare and reload."
      />
    );
  }

  return (
    <div className="space-y-4">
      <ul className="divide-y divide-line-200 overflow-hidden rounded-lg border border-line-200 bg-white">
        {plans.map((p) => {
          const payload = p.plan_payload as BudgetPlanResult | undefined;
          const summary =
            p.kind === 'scenario'
              ? `Δ ${fmtInt(p.outcome_change)} KPI`
              : `${fmtInt(payload?.total_budget)} budget · uplift ${fmtInt(payload?.expected_uplift)}`;
          return (
            <li key={p.plan_id} className="flex items-center gap-3 px-3 py-2.5">
              <label className="flex items-center" title="Select to compare (max 2)">
                <input
                  type="checkbox"
                  checked={compareIds.includes(p.plan_id)}
                  onChange={() => toggleCompare(p.plan_id)}
                  disabled={p.kind === 'scenario'}
                  className="h-3.5 w-3.5 rounded border-line-300 text-sage-600 focus:ring-sage-600 disabled:opacity-30"
                />
              </label>
              <div className="min-w-0 grow">
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium text-ink-900">{p.name}</span>
                  <span className="rounded-full bg-cream-200 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-ink-500">
                    {p.kind}
                  </span>
                </div>
                <div className="text-xs text-ink-400">{summary}</div>
              </div>
              <button
                onClick={() => setOpenId(openId === p.plan_id ? null : p.plan_id)}
                className="inline-flex items-center gap-1 rounded-md border border-line-300 px-2 py-1 text-xs text-ink-700 hover:bg-cream-100"
              >
                <Eye className="h-3.5 w-3.5" /> {openId === p.plan_id ? 'Hide' : 'Open'}
              </button>
              <a
                href={budgetPlanCsvUrl(p.plan_id)}
                download
                className="inline-flex items-center gap-1 rounded-md border border-line-300 px-2 py-1 text-xs text-ink-700 hover:bg-cream-100"
              >
                <Download className="h-3.5 w-3.5" /> CSV
              </a>
              <button
                onClick={() => del.mutate(p.plan_id)}
                disabled={del.isPending}
                className="inline-flex items-center gap-1 rounded-md border border-line-300 px-2 py-1 text-xs text-rust-600 hover:bg-rust-50"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </li>
          );
        })}
      </ul>

      {comparePair.length === 2 && (
        <div className="rounded-lg border border-steel-300 bg-steel-50/40 px-4 py-3">
          <h4 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink-900">
            <GitCompare className="h-4 w-4" /> Compare
          </h4>
          <PlanCompare a={comparePair[0]} b={comparePair[1]} />
        </div>
      )}

      {opened && (
        <div className="rounded-lg border border-line-200 bg-cream-50/50 px-4 py-3">
          <h4 className="mb-2 text-sm font-semibold text-ink-900">{opened.name}</h4>
          {isOptimization(opened) ? (
            <AllocationResult plan={opened.plan_payload as unknown as BudgetPlanResult} />
          ) : (
            <ScenarioSummary plan={opened} />
          )}
        </div>
      )}
    </div>
  );
}

function ScenarioSummary({ plan }: { plan: BudgetPlanInfo }) {
  return (
    <div className="space-y-1 text-sm text-ink-700">
      <p>
        Baseline {fmtInt(plan.baseline_outcome)} → scenario {fmtInt(plan.scenario_outcome)} (Δ{' '}
        {fmtInt(plan.outcome_change)}, {plan.outcome_change_pct?.toFixed?.(1) ?? '—'}%).
      </p>
      {plan.spend_changes && (
        <p className="text-xs text-ink-400">
          {Object.entries(plan.spend_changes)
            .map(([ch, v]) => `${ch} ${v >= 0 ? '+' : ''}${Math.round(v * 100)}%`)
            .join(' · ')}
        </p>
      )}
    </div>
  );
}
