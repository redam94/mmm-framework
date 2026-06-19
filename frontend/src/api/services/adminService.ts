import { apiClient } from '../client';
import type { OrgRole } from './accountService';

export interface OrgMember {
  user_id: string;
  role: OrgRole;
  email: string | null;
  name: string | null;
}

export interface OrgInvite {
  token: string;
  org_id: string;
  email: string;
  role: OrgRole;
  invited_by: string | null;
  expires_at: number;
  created_at: number;
}

export interface InviteResult {
  token: string;
  email: string;
  role: OrgRole;
  org_id: string;
  expires_at: number;
}

export const adminService = {
  async listMembers(): Promise<OrgMember[]> {
    const { data } = await apiClient.get<{ members: OrgMember[] }>('/auth/members');
    return data.members;
  },

  async setMemberRole(userId: string, role: OrgRole): Promise<void> {
    await apiClient.patch(`/auth/members/${userId}`, { role });
  },

  async removeMember(userId: string): Promise<void> {
    await apiClient.delete(`/auth/members/${userId}`);
  },

  async createInvite(email: string, role: OrgRole): Promise<InviteResult> {
    const { data } = await apiClient.post<InviteResult>('/auth/invite', { email, role });
    return data;
  },

  async listInvites(): Promise<OrgInvite[]> {
    const { data } = await apiClient.get<{ invites: OrgInvite[] }>('/auth/invites');
    return data.invites;
  },

  async revokeInvite(token: string): Promise<void> {
    await apiClient.delete(`/auth/invites/${encodeURIComponent(token)}`);
  },
};
