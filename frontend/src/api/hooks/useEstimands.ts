import { useQuery } from '@tanstack/react-query';
import { estimandsService } from '../services/estimandsService';

export const estimandKeys = {
  all: ['estimands'] as const,
  project: (projectId: string | null) => [...estimandKeys.all, 'project', projectId] as const,
};

/** Grouped declarative estimands for every fitted model in a project. */
export function useProjectEstimands(projectId: string | null) {
  return useQuery({
    queryKey: estimandKeys.project(projectId),
    queryFn: () => estimandsService.getProjectEstimands(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}
