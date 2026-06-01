import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../client';

export interface BudgetPlanInfo {
  plan_id: string;
  name: string;
  description?: string;
  model_id: string;
  spend_changes: Record<string, number>;
  baseline_outcome: number;
  scenario_outcome: number;
  outcome_change: number;
  outcome_change_pct: number;
  channel_details: Record<string, Record<string, number>>;
  created_at: string;
  project_id?: string;
}

export interface BudgetPlanCreateRequest {
  name: string;
  description?: string;
  model_id: string;
  spend_changes: Record<string, number>;
  time_period?: [number, number];
  project_id?: string;
}

export const budgetPlanKeys = {
  all: ['budget-plans'] as const,
  lists: () => [...budgetPlanKeys.all, 'list'] as const,
  list: (params?: { model_id?: string; project_id?: string }) =>
    [...budgetPlanKeys.lists(), params] as const,
  detail: (id: string) => [...budgetPlanKeys.all, 'detail', id] as const,
};

export function useBudgetPlans(params?: { model_id?: string; project_id?: string }) {
  return useQuery({
    queryKey: budgetPlanKeys.list(params),
    queryFn: async () => {
      const { data } = await apiClient.get<{ plans: BudgetPlanInfo[]; total: number }>(
        '/budget-plans',
        { params }
      );
      return data;
    },
    staleTime: 30000,
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
    staleTime: 30000,
  });
}

export function useCreateBudgetPlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (req: BudgetPlanCreateRequest) => {
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
