import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { adminService } from '../services/adminService';
import type { OrgRole } from '../services/accountService';
import { accountKeys } from './useAccount';

export const adminKeys = {
  all: ['admin'] as const,
  members: () => [...adminKeys.all, 'members'] as const,
  invites: () => [...adminKeys.all, 'invites'] as const,
};

export function useMembers() {
  return useQuery({
    queryKey: adminKeys.members(),
    queryFn: () => adminService.listMembers(),
    staleTime: 30000,
    retry: false,
  });
}

export function useInvites() {
  return useQuery({
    queryKey: adminKeys.invites(),
    queryFn: () => adminService.listInvites(),
    staleTime: 30000,
    retry: false,
  });
}

export function useSetMemberRole() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, role }: { userId: string; role: OrgRole }) =>
      adminService.setMemberRole(userId, role),
    onSuccess: () => qc.invalidateQueries({ queryKey: adminKeys.members() }),
  });
}

export function useRemoveMember() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (userId: string) => adminService.removeMember(userId),
    onSuccess: () => qc.invalidateQueries({ queryKey: adminKeys.members() }),
  });
}

export function useCreateInvite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ email, role }: { email: string; role: OrgRole }) =>
      adminService.createInvite(email, role),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: adminKeys.invites() });
      qc.invalidateQueries({ queryKey: accountKeys.usage() });
    },
  });
}

export function useRevokeInvite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (token: string) => adminService.revokeInvite(token),
    onSuccess: () => qc.invalidateQueries({ queryKey: adminKeys.invites() }),
  });
}
