import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { connectionsService } from '../services/connectionsService';
import type { ConnectionInput } from '../services/connectionsService';

export const connectionKeys = {
  all: ['data-connections'] as const,
  byProject: (projectId: string | null) => [...connectionKeys.all, projectId] as const,
};

export function useConnections(projectId: string | null) {
  return useQuery({
    queryKey: connectionKeys.byProject(projectId),
    queryFn: () => connectionsService.list(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}

export function useCreateConnection(projectId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ConnectionInput) => connectionsService.create(projectId!, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: connectionKeys.byProject(projectId) }),
  });
}

export function useDeleteConnection(projectId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => connectionsService.remove(projectId!, id),
    onSuccess: () => qc.invalidateQueries({ queryKey: connectionKeys.byProject(projectId) }),
  });
}

export function useTestConnection(projectId: string | null) {
  return useMutation({
    mutationFn: (id: string) => connectionsService.test(projectId!, id),
  });
}

export function usePreviewConnection(projectId: string | null) {
  return useMutation({
    mutationFn: (id: string) => connectionsService.preview(projectId!, id),
  });
}
