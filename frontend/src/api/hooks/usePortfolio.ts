import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { portfolioService } from '../services/portfolioService';
import type { ExperimentUpsert } from '../services/portfolioService';

export const portfolioKeys = {
  all: ['portfolio'] as const,
  byProject: (projectId: string | null) => [...portfolioKeys.all, projectId] as const,
};

export function usePortfolio(projectId: string | null) {
  return useQuery({
    queryKey: portfolioKeys.byProject(projectId),
    queryFn: () => portfolioService.getPortfolio(projectId),
    staleTime: 30000,
  });
}

export function useUpsertExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: ExperimentUpsert) => portfolioService.upsertExperiment(body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: portfolioKeys.all }),
  });
}

export function useDeleteExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => portfolioService.deleteExperiment(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: portfolioKeys.all }),
  });
}
