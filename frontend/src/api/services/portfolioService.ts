import { apiClient } from '../client';

// ── Types ─────────────────────────────────────────────────────────────────────

export type ExperimentStatus = 'planned' | 'running' | 'completed' | 'calibrated' | 'cancelled';

export interface ExperimentInfo {
  id: string;
  project_id: string | null;
  thread_id: string | null;
  channel: string;
  design_type: string | null;
  status: ExperimentStatus;
  start_date: string | null;
  end_date: string | null;
  estimand: string | null;
  value: number | null;
  se: number | null;
  notes: string | null;
  created_at: number;
  updated_at: number;
}

export interface ExperimentUpsert {
  id?: string;
  project_id?: string | null;
  channel?: string;
  design_type?: string | null;
  status?: ExperimentStatus;
  start_date?: string | null;
  end_date?: string | null;
  estimand?: string | null;
  value?: number | null;
  se?: number | null;
  notes?: string | null;
}

export interface ModelRunInfo {
  model_id: string;
  thread_id: string;
  project_id: string | null;
  run_name: string | null;
  kpi: string | null;
  channels: string[];
  trend: string | null;
  n_obs: number | null;
  summary: string;
  report_path: string | null;
  created_at: number;
}

export interface ExperimentDesign {
  channel: string;
  priority?: number;
  why?: string;
  design_type?: string;
  min_duration_periods?: number;
  duration_rationale?: string;
  target_se?: number;
  target_se_rationale?: string;
  calibration_snippet?: string;
}

export interface NextAction {
  type: 'calibrate' | 'refresh' | 'fit' | 'experiment';
  urgency: 'high' | 'medium' | 'low';
  title: string;
  detail: string;
  design?: ExperimentDesign;
}

export interface PortfolioResponse {
  model_runs: ModelRunInfo[];
  experiments: ExperimentInfo[];
  latest_experiment_design: ({ created_at: number; thread_id: string; designs?: ExperimentDesign[] } & Record<string, unknown>) | null;
  latest_budget_optimization: Record<string, unknown> | null;
  last_fit_at: number | null;
  next_actions: NextAction[];
}

// ── Service ───────────────────────────────────────────────────────────────────

export const portfolioService = {
  async getPortfolio(projectId?: string | null): Promise<PortfolioResponse> {
    const { data } = await apiClient.get<PortfolioResponse>('/portfolio', {
      params: projectId ? { project_id: projectId } : {},
    });
    return data;
  },

  async listExperiments(projectId?: string | null): Promise<ExperimentInfo[]> {
    const { data } = await apiClient.get<{ experiments: ExperimentInfo[] }>('/experiments', {
      params: projectId ? { project_id: projectId } : {},
    });
    return data.experiments;
  },

  async upsertExperiment(body: ExperimentUpsert): Promise<ExperimentInfo> {
    const { data } = await apiClient.post<ExperimentInfo>('/experiments', body);
    return data;
  },

  async deleteExperiment(id: string): Promise<void> {
    await apiClient.delete(`/experiments/${id}`);
  },
};
