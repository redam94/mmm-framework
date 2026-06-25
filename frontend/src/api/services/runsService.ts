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

// ── Model-health snapshot (fit-time diagnostics) ──────────────────────────────

export interface ConvergenceBlock {
  divergences: number | null;
  rhat_max: number | null;
  ess_bulk_min: number | null;
  rhat_threshold: number;
  ess_threshold: number;
  /** which checks failed: 'divergences' | 'rhat' | 'ess' */
  flags: string[];
  ok: boolean;
}

export type LearningVerdict =
  | 'strong'
  | 'moderate'
  | 'weak'
  | 'relocated'
  | 'prior-dominated'
  | 'undetermined';

export interface LearningParameter {
  parameter: string;
  verdict: LearningVerdict;
  /** 1 − Var_post/Var_prior: →1 data pinned it, ~0 prior re-stated, <0 conflict */
  contraction: number | null;
  contraction_robust: number | null;
  /** prior–posterior overlap coefficient in [0,1]; ~1 means nothing learned */
  overlap: number | null;
  /** posterior-mean shift in prior SDs */
  shift_z: number | null;
  post_mean: number | null;
  post_sd: number | null;
  post_ess_bulk: number | null;
}

export interface LearningBlock {
  n_parameters: number;
  verdict_counts: Partial<Record<LearningVerdict, number>>;
  /** worst (least-learned) parameters first; may be truncated */
  parameters: LearningParameter[];
  truncated: boolean;
}

export interface RunDiagnostics {
  schema_version: number;
  convergence?: ConvergenceBlock;
  convergence_error?: string;
  learning?: LearningBlock | null;
  learning_error?: string;
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
  diagnostics: RunDiagnostics | null;
  spec_hash: string | null;
  parent_run_id: string | null;
  assumptions: { key: string; version: number; category: string | null; rationale: string | null }[];
  changes: RunChanges;
}

// ── Run comparison (per-channel ROI/spend delta, B vs A) ──────────────────────

export interface DeltaCell {
  a: number | null;
  b: number | null;
  delta: number | null;
}

export interface RunChannelDelta {
  channel: string;
  in_a: boolean;
  in_b: boolean;
  roi_mean: DeltaCell;
  roi_hdi_low: DeltaCell;
  roi_hdi_high: DeltaCell;
  marginal_roi: DeltaCell;
  spend: DeltaCell;
  spend_share: DeltaCell;
}

export interface RunComparison {
  run_a: { run_id: string; created_at: number | null; project_id: string | null };
  run_b: { run_id: string; created_at: number | null; project_id: string | null };
  channels: RunChannelDelta[];
  portfolio: Record<string, DeltaCell>;
}

export const runsService = {
  async listRuns(projectId?: string | null): Promise<RunInfo[]> {
    const { data } = await apiClient.get<{ runs: RunInfo[] }>('/runs', {
      params: projectId ? { project_id: projectId } : {},
    });
    return data.runs;
  },

  async compareRuns(runA: string, runB: string): Promise<RunComparison> {
    const { data } = await apiClient.get<RunComparison>('/runs/compare', {
      params: { run_a: runA, run_b: runB },
    });
    return data;
  },
};
