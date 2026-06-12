import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { teamService } from '../services/teamService';
import type {
  ProjectMemberInput,
  UserCreateRequest,
  UserUpdateRequest,
} from '../services/teamService';

export const teamKeys = {
  all: ['team'] as const,
  users: () => [...teamKeys.all, 'users'] as const,
  members: (projectId: string) => [...teamKeys.all, 'members', projectId] as const,
};

export function useUsers() {
  return useQuery({
    queryKey: teamKeys.users(),
    queryFn: () => teamService.listUsers(),
    staleTime: 30000,
  });
}

export function useCreateUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: UserCreateRequest) => teamService.createUser(req),
    onSuccess: () => qc.invalidateQueries({ queryKey: teamKeys.all }),
  });
}

export function useUpdateUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, body }: { userId: string; body: UserUpdateRequest }) =>
      teamService.updateUser(userId, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: teamKeys.all }),
  });
}

export function useDeleteUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (userId: string) => teamService.deleteUser(userId),
    onSuccess: () => qc.invalidateQueries({ queryKey: teamKeys.all }),
  });
}

export function useProjectMembers(projectId: string | null) {
  return useQuery({
    queryKey: teamKeys.members(projectId ?? ''),
    queryFn: () => teamService.getMembers(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}

export function useSetProjectMembers(projectId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (members: ProjectMemberInput[]) => teamService.setMembers(projectId!, members),
    onSuccess: () => qc.invalidateQueries({ queryKey: teamKeys.all }),
  });
}
