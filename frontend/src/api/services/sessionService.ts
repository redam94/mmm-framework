import { apiClient } from '../client';

export interface SessionInfo {
  thread_id: string;
  name: string;
  created_at: number;
  updated_at: number;
  project_id: string | null;
  artifact_count: number;
}

export interface SessionDetail extends SessionInfo {
  artifacts: Array<{ id: string; kind: string; created_at: number; payload: Record<string, unknown> }>;
  assumptions: Array<{ key: string; category: string; value: unknown; rationale: string; version: number }>;
  workflow_steps: Record<number, { status: string; notes: string | null }>;
}

export interface SessionListResponse {
  sessions: SessionInfo[];
  total: number;
}

export interface SessionCreateRequest {
  name?: string;
  project_id?: string;
}

export interface SessionUpdateRequest {
  name?: string;
  project_id?: string;
}

export const sessionService = {
  async listSessions(params?: { project_id?: string; skip?: number; limit?: number }): Promise<SessionListResponse> {
    const { data } = await apiClient.get<SessionListResponse>('/sessions', { params });
    return data;
  },

  async getSession(threadId: string): Promise<SessionDetail> {
    const { data } = await apiClient.get<SessionDetail>(`/sessions/${threadId}`);
    return data;
  },

  async createSession(request: SessionCreateRequest): Promise<SessionInfo> {
    const { data } = await apiClient.post<SessionInfo>('/sessions', request);
    return data;
  },

  async updateSession(threadId: string, request: SessionUpdateRequest): Promise<SessionInfo> {
    const { data } = await apiClient.patch<SessionInfo>(`/sessions/${threadId}`, request);
    return data;
  },

  async deleteSession(threadId: string): Promise<void> {
    await apiClient.delete(`/sessions/${threadId}`);
  },
};
