import { apiClient } from '../client';

export type TeamRole = 'owner' | 'analyst' | 'viewer';

export interface TeamUser {
  user_id: string;
  name: string;
  email: string | null;
  role: TeamRole;
  created_at: string;
}

export interface TeamUserListResponse {
  users: TeamUser[];
  total: number;
}

export interface UserCreateRequest {
  name: string;
  email?: string;
  role: TeamRole;
}

export interface UserUpdateRequest {
  name?: string;
  email?: string;
  role?: TeamRole;
}

export interface ProjectMember {
  user_id: string;
  name: string;
  email: string | null;
  role: TeamRole;
}

export interface ProjectMemberInput {
  user_id: string;
  role: TeamRole;
}

export interface ProjectMembersResponse {
  members: ProjectMember[];
  total: number;
}

export const teamService = {
  async listUsers(): Promise<TeamUserListResponse> {
    const { data } = await apiClient.get<TeamUserListResponse>('/users');
    return data;
  },

  async createUser(request: UserCreateRequest): Promise<TeamUser> {
    const { data } = await apiClient.post<TeamUser>('/users', request);
    return data;
  },

  async updateUser(userId: string, request: UserUpdateRequest): Promise<TeamUser> {
    const { data } = await apiClient.patch<TeamUser>(`/users/${userId}`, request);
    return data;
  },

  async deleteUser(userId: string): Promise<void> {
    await apiClient.delete(`/users/${userId}`);
  },

  async getMembers(projectId: string): Promise<ProjectMembersResponse> {
    const { data } = await apiClient.get<ProjectMembersResponse>(`/projects/${projectId}/members`);
    return data;
  },

  async setMembers(projectId: string, members: ProjectMemberInput[]): Promise<ProjectMembersResponse> {
    const { data } = await apiClient.put<ProjectMembersResponse>(
      `/projects/${projectId}/members`,
      { members },
    );
    return data;
  },
};
