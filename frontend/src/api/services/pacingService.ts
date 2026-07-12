import { apiClient } from '../client';

/**
 * In-flight pacing — actual delivery vs the saved plan (issue #123). The planned
 * series is auto-sourced server-side from the project's latest saved budget plan;
 * the panel uploads actual delivery and reads the computed pacing. Mirrors
 * api/pacing.py + the /projects/{id}/delivery|pacing endpoints.
 */

export type PacingStatus = 'on-track' | 'over-pacing' | 'under-pacing' | 'not-started';

export interface PacingChannel {
  channel: string;
  planned: number;
  actual: number;
  divergence_pct: number;
  status: PacingStatus;
  planned_series: number[];
  actual_series: number[];
}

export interface PacingAlert {
  off_pace: boolean;
  n_flagged: number;
  flagged: string[];
  threshold: number;
  worst: {
    channel: string;
    divergence_pct: number;
    abs_divergence: number;
    status: PacingStatus;
  } | null;
  portfolio_divergence_pct: number | null;
}

export interface Pacing {
  /** false → no saved plan or no delivery yet (see `reason`). */
  available: boolean;
  reason?: 'no_plan' | 'no_delivery' | string;
  plan_basis?: 'flighting' | 'allocation' | null;
  plan_id?: string | null;
  plan_name?: string | null;
  threshold: number;
  planned_total?: number;
  actual_total?: number;
  divergence_pct?: number;
  flagged: string[];
  channels: PacingChannel[];
  periods?: string[];
  /** Present only when computed with a fitted model (agent tool); null here. */
  outcome_delta?: { mean: number; lower: number; upper: number } | null;
  alert?: PacingAlert;
}

export interface DeliveryRow {
  id: string;
  project_id: string;
  channel: string;
  period: string;
  spend: number;
  source: string | null;
  created_at: number;
  updated_at: number;
}

export const pacingService = {
  async getPacing(projectId: string): Promise<Pacing> {
    const { data } = await apiClient.get<Pacing>(`/projects/${projectId}/pacing`);
    return data;
  },

  async getDelivery(projectId: string): Promise<DeliveryRow[]> {
    const { data } = await apiClient.get<{ delivery: DeliveryRow[] }>(
      `/projects/${projectId}/delivery`,
    );
    return data.delivery;
  },

  /** Upload actual delivery (CSV/TSV or JSON: spend by channel/period). */
  async uploadDelivery(
    projectId: string,
    file: File,
  ): Promise<{ delivery: DeliveryRow[]; ingested: number }> {
    const fd = new FormData();
    fd.append('file', file);
    const { data } = await apiClient.post<{ delivery: DeliveryRow[]; ingested: number }>(
      `/projects/${projectId}/delivery`,
      fd,
      // Let the browser set the multipart boundary.
      { headers: { 'Content-Type': undefined as unknown as string }, timeout: 60000 },
    );
    return data;
  },

  async clearDelivery(projectId: string, channel?: string): Promise<number> {
    const { data } = await apiClient.delete<{ deleted: number }>(
      `/projects/${projectId}/delivery`,
      { params: channel ? { channel } : undefined },
    );
    return data.deleted;
  },
};
