import { apiClient } from '../client';

/**
 * Project estimands for the Performance page — the declarative causal lens
 * (contribution ROI, marginal ROAS, incremental contribution, …) realized for
 * every fitted model and grouped into (estimand × KPI) comparability clusters
 * by the backend. Mirrors api/estimands.py.
 */

/** Evidence vs the no-effect reference (1.0 for ratios, 0 otherwise). */
export type Evidence = 'strong' | 'below' | 'uncertain' | 'na';

/**
 * Where a channel number's credibility comes from — the evidence *tier*
 * (distinct from `Evidence`, which is the CI-vs-reference verdict). Mirrors
 * `EvidenceTier` in reporting/evidence.py; rendered as the same chip the report
 * uses so the dashboard and the report never disagree (issue #124).
 */
export type EvidenceTier =
  | 'experiment-validated'
  | 'model-identified'
  | 'prior-dominated';

/** Per-channel evidence tier + identifiability flag (`ChannelEvidence.to_dict`). */
export interface ChannelTier {
  channel: string;
  tier: EvidenceTier;
  label: string;
  short_label: string;
  gloss: string;
  /** false → the channel is not separately identified (collinear). */
  identified: boolean;
  collinear_with: string[];
  contraction: number | null;
  learning_verdict: string | null;
  vif: number | null;
  experiment: boolean;
  /** true when the number should be read with a caveat (prior-dominated or unidentified). */
  gated: boolean;
  caveat: string | null;
}

export interface EstimandCell {
  channel: string;
  mean: number | null;
  lower: number | null;
  upper: number | null;
  units: string;
  status: string;
  evidence: Evidence;
  /** Evidence tier chip; absent for runs fitted before evidence persistence. */
  tier?: ChannelTier | null;
  prob_positive: number | null;
  prob_profitable: number | null;
}

export interface EstimandModel {
  run_id: string;
  label: string;
  model_kind: string;
  model_key: string;
  created_at: number | null;
  rows: EstimandCell[];
}

/** One comparability cluster: a single estimand measured on a single KPI. */
export interface EstimandGroup {
  key: string;
  estimand: string;
  label: string;
  kpi: string;
  kind: string;
  units: string;
  is_ratio: boolean;
  reference: number;
  channels: string[];
  models: EstimandModel[];
  n_models: number;
  n_models_with_data: number;
}

export interface EstimandRunSummary {
  run_id: string;
  label: string;
  model_kind: string;
  model_key: string;
  kpi: string;
  created_at: number | null;
  n_estimands: number;
  is_latest_for_model: boolean;
}

export interface ProjectEstimands {
  runs: EstimandRunSummary[];
  kpis: string[];
  groups: EstimandGroup[];
}

export const estimandsService = {
  async getProjectEstimands(projectId: string): Promise<ProjectEstimands> {
    const { data } = await apiClient.get<ProjectEstimands>(
      `/projects/${projectId}/estimands`,
    );
    return data;
  },
};
