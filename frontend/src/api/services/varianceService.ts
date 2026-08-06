import { apiClient } from '../client';

/**
 * Variance to plan (#227) — the committed forecast vs realized KPI, as a bridge
 * that closes exactly. Mirrors mmm_framework/planning/variance.py + the
 * POST/GET /projects/{id}/variance job endpoints. Two buckets only: per-channel
 * delivery variance on the COMMITTED posterior, and a LABELLED unexplained
 * remainder; the refit "effectiveness" split is refused server-side with the
 * reason stated.
 */

export type BridgeProvenance = 'modelled' | 'observed' | 'residual' | 'absorbing' | 'supplied';

export interface BridgeRow {
  name: string;
  value: number;
  provenance: BridgeProvenance;
  basis?: string;
  note?: string;
  source_note?: string;
  lower?: number | null;
  upper?: number | null;
}

export interface VarianceBridge {
  committed_kpi: number;
  actual_kpi: number;
  gap: number;
  rows: BridgeRow[];
  rows_dollars: Record<string, number> | null;
  within_committed_interval: boolean | null;
  committed_lower: number | null;
  committed_upper: number | null;
  interval_mass: number | null;
  delivery_total: number;
  delivery_lower: number | null;
  delivery_upper: number | null;
  unexplained: number;
  period_set: string[];
  value_per_kpi: number | null;
  value_source: string | null;
  dollar_headline_suppressed: boolean;
  closes: boolean;
  caveats: string[];
  refusals: string[];
  run_diff?: Record<string, unknown> | null;
}

export interface SuppliedLine {
  name: string;
  value: number;
  source_note: string;
}

export interface VarianceJob {
  status: 'pending' | 'running' | 'done' | 'error';
  result: VarianceBridge | null;
  error: string | null;
}

export const varianceService = {
  /** POST /projects/{id}/variance — start the bridge job (202 → job_id).
   * A bridge that cannot be built refuses at POST time with a 409 + reason. */
  async startVariance(
    projectId: string,
    supplied: SuppliedLine[] = [],
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/variance`,
      { supplied },
    );
    return data;
  },
  /** GET /projects/{id}/variance/{jobId} — poll the bridge job. */
  async pollVariance(projectId: string, jobId: string): Promise<VarianceJob> {
    const { data } = await apiClient.get<VarianceJob>(
      `/projects/${projectId}/variance/${jobId}`,
    );
    return data;
  },
};
