import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { projectService } from '../services/projectService';
import type {
  ProjectCreateRequest,
  ProjectOnboardingRequest,
  ProjectUpdateRequest,
} from '../services/projectService';

export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  list: () => [...projectKeys.lists()] as const,
  detail: (id: string) => [...projectKeys.all, 'detail', id] as const,
};

export function useProjects() {
  return useQuery({
    queryKey: projectKeys.list(),
    queryFn: () => projectService.listProjects(),
    staleTime: 30000,
  });
}

export function useProject(projectId: string | undefined) {
  return useQuery({
    queryKey: projectKeys.detail(projectId!),
    queryFn: () => projectService.getProject(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}

export function useOnboardingStatus(projectId: string | undefined) {
  return useQuery({
    queryKey: [...projectKeys.detail(projectId ?? ''), 'onboarding-status'],
    queryFn: () => projectService.getOnboardingStatus(projectId!),
    enabled: !!projectId,
    staleTime: 15000,
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: ProjectCreateRequest) => projectService.createProject(req),
    onSuccess: () => qc.invalidateQueries({ queryKey: projectKeys.lists() }),
  });
}

export function useUpdateProject(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: ProjectUpdateRequest) => projectService.updateProject(projectId, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: projectKeys.lists() });
      qc.invalidateQueries({ queryKey: projectKeys.detail(projectId) });
    },
  });
}

export function useOnboardProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ projectId, body }: { projectId: string; body: ProjectOnboardingRequest }) =>
      projectService.onboardProject(projectId, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: projectKeys.all }),
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (projectId: string) => projectService.deleteProject(projectId),
    onSuccess: () => qc.invalidateQueries({ queryKey: projectKeys.lists() }),
  });
}
