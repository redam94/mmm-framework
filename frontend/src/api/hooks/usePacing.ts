import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { pacingService } from '../services/pacingService';

export const pacingKeys = {
  all: ['pacing'] as const,
  project: (projectId: string | null) => [...pacingKeys.all, 'project', projectId] as const,
  delivery: (projectId: string | null) => [...pacingKeys.all, 'delivery', projectId] as const,
};

/** Computed in-flight pacing (actual delivery vs the saved plan). */
export function useProjectPacing(projectId: string | null) {
  return useQuery({
    queryKey: pacingKeys.project(projectId),
    queryFn: () => pacingService.getPacing(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}

/** The stored actual-delivery rows for a project. */
export function useProjectDelivery(projectId: string | null) {
  return useQuery({
    queryKey: pacingKeys.delivery(projectId),
    queryFn: () => pacingService.getDelivery(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}

function invalidatePacing(qc: ReturnType<typeof useQueryClient>, projectId: string | null) {
  qc.invalidateQueries({ queryKey: pacingKeys.project(projectId) });
  qc.invalidateQueries({ queryKey: pacingKeys.delivery(projectId) });
}

/** Upload actual delivery, then recompute pacing (invalidates both queries). */
export function useUploadDelivery(projectId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => pacingService.uploadDelivery(projectId!, file),
    onSuccess: () => invalidatePacing(qc, projectId),
  });
}

export function useClearDelivery(projectId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (channel?: string) => pacingService.clearDelivery(projectId!, channel),
    onSuccess: () => invalidatePacing(qc, projectId),
  });
}
