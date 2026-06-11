import { apiClient } from '../client';

export interface SpecChange {
  path: string;
  old: unknown;
  new: unknown;
}

export interface AssumptionDelta {
  key: string;
  version: number;
  category: string | null;
  rationale: string | null;
  change: 'added' | 'revised';
}

export interface RunChanges {
  spec_changes: SpecChange[];
  data_changed: boolean;
  assumptions_delta: AssumptionDelta[];
  baseline: boolean;
}

export interface RunInfo {
  artifact_id: string;
  run_id: string | null;
  run_name: string | null;
  thread_id: string;
  session_name: string | null;
  project_id: string | null;
  created_at: number;
  timestamp_iso: string | null;
  kpi: string | null;
  channels: string[];
  controls: string[];
  trend: string | null;
  seasonality: Record<string, number> | null;
  inference: Record<string, number> | null;
  n_obs: number | null;
  summary: string | null;
  report_path: string | null;
  model_path: string | null;
  data_fingerprint: { md5: string; size_bytes: number; n_rows: number; path: string } | null;
  spec_hash: string | null;
  parent_run_id: string | null;
  assumptions: { key: string; version: number; category: string | null; rationale: string | null }[];
  changes: RunChanges;
}

export const runsService = {
  async listRuns(projectId?: string | null): Promise<RunInfo[]> {
    const { data } = await apiClient.get<{ runs: RunInfo[] }>('/runs', {
      params: projectId ? { project_id: projectId } : {},
    });
    return data.runs;
  },
};
