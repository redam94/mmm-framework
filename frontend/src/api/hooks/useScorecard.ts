import { useQuery } from '@tanstack/react-query';
import { scorecardService } from '../services/scorecardService';

export const scorecardKeys = {
  all: ['scorecard'] as const,
  project: (projectId: string | null) => [...scorecardKeys.all, 'project', projectId] as const,
};

/** Predicted-vs-realized recommendation scorecard for a project. */
export function useProjectScorecard(projectId: string | null) {
  return useQuery({
    queryKey: scorecardKeys.project(projectId),
    queryFn: () => scorecardService.getScorecard(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}
