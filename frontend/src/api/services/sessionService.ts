import { apiClient } from '../client';

/** Modeling mode of a session — selects the oracle's prompt framing + tool set. */
export type ModelingMode = 'mmm' | 'causal_inference' | 'general_bayes' | 'descriptive';

export const MODELING_MODES: { value: ModelingMode; label: string; blurb: string }[] = [
  { value: 'mmm', label: 'Marketing Mix Modeling', blurb: 'ROI, budget allocation, the lift-test measurement loop.' },
  { value: 'causal_inference', label: 'Causal Inference', blurb: 'DAGs, identification, estimands & experiment design (no ROI/budget).' },
  { value: 'general_bayes', label: 'General Bayesian', blurb: 'Full Bayesian workflow for any model; causal steps optional.' },
  { value: 'descriptive', label: 'Descriptive / Measurement', blurb: 'CFA / LCA & latent-structure models: fit indices, loadings, class profiles.' },
];

export interface SessionInfo {
  thread_id: string;
  name: string;
  created_at: number;
  updated_at: number;
  project_id: string | null;
  artifact_count: number;
  modeling_mode?: ModelingMode;
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

  /** Switch the session's modeling mode; the next chat turn applies it. */
  async setSessionMode(threadId: string, mode: ModelingMode): Promise<{ status: string; modeling_mode: ModelingMode }> {
    const { data } = await apiClient.patch<{ status: string; modeling_mode: ModelingMode }>(
      `/sessions/${threadId}/mode`,
      { modeling_mode: mode },
    );
    return data;
  },
};
