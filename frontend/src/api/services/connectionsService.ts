import { apiClient } from '../client';

export interface DataConnection {
  id: string;
  project_id: string;
  name: string;
  kind: string; // gcs | bigquery
  config: Record<string, unknown>;
  created_at: number;
  updated_at: number;
  last_synced: number | null;
}

export interface ConnectionInput {
  name: string;
  kind: string;
  config: Record<string, unknown>;
}

export interface ConnectionTestResult {
  ok: boolean;
  detail: string;
  source: string;
  target: string;
}

export interface ConnectionPreview {
  columns: string[];
  rows: Record<string, unknown>[];
  n_preview: number;
}

const base = (projectId: string) => `/projects/${projectId}/data-connections`;

export const connectionsService = {
  async list(projectId: string): Promise<DataConnection[]> {
    const { data } = await apiClient.get<{ connections: DataConnection[] }>(base(projectId));
    return data.connections;
  },

  async create(projectId: string, body: ConnectionInput): Promise<DataConnection> {
    const { data } = await apiClient.post<DataConnection>(base(projectId), body);
    return data;
  },

  async remove(projectId: string, id: string): Promise<void> {
    await apiClient.delete(`${base(projectId)}/${id}`);
  },

  async test(projectId: string, id: string): Promise<ConnectionTestResult> {
    const { data } = await apiClient.post<ConnectionTestResult>(`${base(projectId)}/${id}/test`);
    return data;
  },

  async preview(projectId: string, id: string, rows = 20): Promise<ConnectionPreview> {
    const { data } = await apiClient.post<ConnectionPreview>(
      `${base(projectId)}/${id}/preview`,
      null,
      { params: { rows } },
    );
    return data;
  },
};
