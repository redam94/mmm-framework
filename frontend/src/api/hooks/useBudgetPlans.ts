import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, API_BASE_URL } from '../client';
import type { BudgetPlanResult, PlannerScenarioResult } from '../services/plannerService';

/** A saved budget plan. ``plan_payload`` holds the rich studio result (an
 * allocation BudgetPlanResult, or a scenario ScenarioResult) so a saved plan
 * reopens with every panel intact. */
export interface BudgetPlanInfo {
  plan_id: string;
  name: string;
  description?: string | null;
  model_id?: string | null;
  project_id?: string | null;
  kind: 'optimization' | 'scenario' | string;
  spend_changes?: Record<string, number> | null;
  baseline_outcome?: number | null;
  scenario_outcome?: number | null;
  outcome_change?: number | null;
  outcome_change_pct?: number | null;
  channel_details?: Record<string, unknown> | null;
  plan_payload?: BudgetPlanResult | PlannerScenarioResult | Record<string, unknown> | null;
  created_at: number | string;
  updated_at: number | string;
}

/** Upsert body — the FE persists an already-computed studio result (no model
 * load on the backend). Pass ``plan_id`` to update an existing plan in place. */
export interface BudgetPlanUpsertRequest {
  plan_id?: string;
  name: string;
  description?: string | null;
  project_id?: string | null;
  model_id?: string | null;
  kind?: 'optimization' | 'scenario';
  spend_changes?: Record<string, number> | null;
  baseline_outcome?: number | null;
  scenario_outcome?: number | null;
  outcome_change?: number | null;
  outcome_change_pct?: number | null;
  channel_details?: Record<string, unknown> | null;
  plan_payload?: Record<string, unknown> | null;
}

export const budgetPlanKeys = {
  all: ['budget-plans'] as const,
  lists: () => [...budgetPlanKeys.all, 'list'] as const,
  list: (params?: { model_id?: string; project_id?: string | null }) =>
    [...budgetPlanKeys.lists(), params] as const,
  detail: (id: string) => [...budgetPlanKeys.all, 'detail', id] as const,
};

export function useBudgetPlans(params?: { model_id?: string; project_id?: string | null }) {
  return useQuery({
    queryKey: budgetPlanKeys.list(params),
    queryFn: async () => {
      const { data } = await apiClient.get<{ plans: BudgetPlanInfo[]; total: number }>(
        '/budget-plans',
        { params: params ?? undefined },
      );
      return data;
    },
    staleTime: 15000,
  });
}

export function useBudgetPlan(planId: string | undefined) {
  return useQuery({
    queryKey: budgetPlanKeys.detail(planId!),
    queryFn: async () => {
      const { data } = await apiClient.get<BudgetPlanInfo>(`/budget-plans/${planId}`);
      return data;
    },
    enabled: !!planId,
    staleTime: 15000,
  });
}

/** Create OR update a budget plan (POST /budget-plans; pass plan_id to update). */
export function useSaveBudgetPlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (req: BudgetPlanUpsertRequest) => {
      const { data } = await apiClient.post<BudgetPlanInfo>('/budget-plans', req);
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: budgetPlanKeys.lists() }),
  });
}

export function useDeleteBudgetPlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (planId: string) => {
      await apiClient.delete(`/budget-plans/${planId}`);
      return planId;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: budgetPlanKeys.lists() }),
  });
}

/** Absolute URL to a plan's CSV flight-plan export (used by a download link). */
export function budgetPlanCsvUrl(planId: string): string {
  return `${API_BASE_URL}/budget-plans/${planId}/export.csv`;
}
