import { apiClient } from '../client';
import type { ExperimentInfo, ExperimentStatus } from './portfolioService';

// ── Experiment lifecycle registry ─────────────────────────────────────────────

export type LifecycleStatus = ExperimentStatus | 'draft' | 'abandoned';

export interface StatusHistoryEntry {
  status: string;
  at: number;
  note?: string;
}

/** Full registry record (extends the legacy ExperimentInfo scalar columns). */
export interface ExperimentRecord extends Omit<ExperimentInfo, 'status'> {
  status: LifecycleStatus;
  recommending_run_id: string | null;
  calibrated_run_id: string | null;
  design: Record<string, any> | null;
  readout: Record<string, any> | null;
  priority: Record<string, any> | null;
  preregistered_at: number | null;
  status_history: StatusHistoryEntry[];
}

export interface ExperimentTransition {
  status: LifecycleStatus;
  note?: string;
  value?: number;
  se?: number;
  estimand?: string;
  start_date?: string;
  end_date?: string;
  readout?: Record<string, any>;
  calibrated_run_id?: string;
}

// ── EIG/EVOI priorities ───────────────────────────────────────────────────────

export type Quadrant = 'test_now' | 'learn_cheaply' | 'monitor' | 'deprioritize';

export interface PriorityChannel {
  channel: string;
  spend: number | null;
  spend_share: number | null;
  roi_mean: number | null;
  roi_sd: number | null;
  roi_hdi_low: number | null;
  roi_hdi_high: number | null;
  sigma_exp: number | null;
  /** ∂contribution/∂spend at current spend — compare to roi_mean (average) */
  marginal_roi: number | null;
  eig: number | null;
  evoi: number | null;
  priority: number | null;
  quadrant: Quadrant;
  calibration_status: 'experiment_backed' | 'model_only';
  weeks_since_evidence: number | null;
  eig_decayed: number | null;
  retest_due: boolean;
}

/** One channel's saturation curve over the spend-multiplier grid. */
export interface ResponseCurve {
  spend: number[];
  mean: number[];
  p5: number[];
  p95: number[];
}

export interface ResponseCurves {
  multipliers: number[];
  /** index of the 1.0× (current spend) grid point */
  current_index: number;
  channels: Record<string, ResponseCurve>;
}

export interface PrioritiesPayload {
  run_id: string;
  computed_at: number;
  as_of: string;
  channels: PriorityChannel[];
  portfolio: Record<string, number>;
  /** present from run-metrics schema v2 onward (older runs: null) */
  response_curves?: ResponseCurves | null;
  matrix: Partial<Record<Quadrant, string[]>>;
  stale: boolean;
}

// ── History (trajectories) ────────────────────────────────────────────────────

export interface RoiPoint {
  run_id: string;
  mean: number | null;
  sd: number | null;
  hdi_low: number | null;
  hdi_high: number | null;
  ci_width: number | null;
}

export interface ValuePoint {
  run_id: string;
  value: number | null;
}

export interface CalibrationPoint {
  run_id: string;
  status: 'experiment_backed' | 'model_only';
  evidence_age_days: number | null;
}

export interface PortfolioPoint {
  run_id: string;
  created_at: number;
  marginal_roi: number | null;
  expected_uplift: number | null;
  mean_ci_width: number | null;
  evpi: number | null;
  v_current: number | null;
  prob_positive_uplift: number | null;
  total_spend: number | null;
}

export interface HistoryPayload {
  runs: { run_id: string; created_at: number; timestamp_iso: string }[];
  channels: string[];
  series: {
    roi: Record<string, RoiPoint[]>;
    spend_share: Record<string, ValuePoint[]>;
    share_gap: Record<string, ValuePoint[]>;
    calibration: Record<string, CalibrationPoint[]>;
  };
  portfolio: PortfolioPoint[];
}

// ── Calibration coverage ──────────────────────────────────────────────────────

export interface CoverageChannel {
  channel: string;
  /** running = a test is in market or its readout awaits calibration */
  tier: 'calibrated' | 'stale' | 'model_only' | 'running';
  in_flight_status?: 'running' | 'completed' | null;
  in_flight_started?: string | null;
  spend_share: number | null;
  n_experiments: number;
  n_calibrated: number;
  last_experiment_end: string | null;
  evidence_age_days: number | null;
  half_life_weeks: number;
  eig_decayed: number | null;
  retest_due: boolean;
  experiment_ids: string[];
}

export interface CoveragePayload {
  channels: CoverageChannel[];
  coverage_pct: number;
  spend_weighted_coverage_pct: number;
  as_of: string;
  run_id: string | null;
}

// ── Experiment design studio ──────────────────────────────────────────────────

export type DesignKey = 'geo_lift' | 'matched_market_did' | 'national_flighting';

export interface DesignOptions {
  n_geos: number;
  geos: string[];
  n_weeks: number;
  designs: DesignKey[];
  recommended: DesignKey;
  kpi: string;
}

export interface DesignRequest {
  channel: string;
  design_key?: DesignKey;
  design?: 'holdout' | 'scaling';
  intensity_pct?: number;
  n_pairs?: number;
  duration?: number;
  amplitude_pct?: number;
  block_weeks?: number;
  seed?: number;
}

export interface PairAssignment {
  treatment: string;
  control: string;
  /** raw pre-period KPI correlation (dominated by shared trend/seasonality) */
  correlation: number;
  /** co-movement AFTER removing trend/seasonality/spend — the DiD's noise floor */
  residual_correlation?: number;
  size_ratio: number;
}

export interface BalanceRow {
  feature: string;
  treatment_mean: number;
  control_mean: number;
  abs_std_diff: number;
}

export interface PowerPoint {
  duration: number;
  se_roas: number;
  mde_roas: number;
}

export interface SchedulePoint {
  week_offset: number;
  multiplier: number;
}

/** Union payload from POST /projects/{id}/experiment-design. */
export interface ExperimentDesignPayload {
  design_key: DesignKey;
  design_type: string;
  channel: string;
  kpi: string;
  duration: number;
  se_roas: number;
  mde_roas: number;
  weekly_spend_delta: number;
  analysis_plan: string;
  seed: number;
  se_source?: 'placebo_calibrated' | 'analytic';
  balance?: BalanceRow[];
  // geo designs
  randomized?: boolean;
  n_pairs?: number;
  assignment?: PairAssignment[];
  treatment_geos?: string[];
  control_geos?: string[];
  intensity_pct?: number;
  power_curve?: PowerPoint[];
  placebo?: { n_windows: number; sd: number | null; p95_abs: number | null };
  diagnostics?: {
    sigma_pair_diff: number;
    pre_period_weeks: number;
    matching?: string;
    calibration_factor?: number;
    min_pair_correlation: number;
    min_residual_correlation?: number;
    max_balance_abs_std_diff?: number;
    parallel_trends_warning: boolean;
  };
  // flighting
  amplitude_pct?: number;
  block_weeks?: number;
  budget_neutral?: boolean;
  schedule?: SchedulePoint[];
  identification?: {
    historical_spend_cv: number;
    scheduled_window_cv: number;
    exogenous_share: number;
  };
}

// ── Service ───────────────────────────────────────────────────────────────────

export const measurementService = {
  async listExperiments(
    projectId?: string | null,
    status?: string,
    channel?: string,
  ): Promise<ExperimentRecord[]> {
    const { data } = await apiClient.get<{ experiments: ExperimentRecord[] }>('/experiments', {
      params: {
        ...(projectId ? { project_id: projectId } : {}),
        ...(status ? { status } : {}),
        ...(channel ? { channel } : {}),
      },
    });
    return data.experiments;
  },

  async getExperiment(id: string): Promise<ExperimentRecord> {
    const { data } = await apiClient.get<ExperimentRecord>(`/experiments/${id}`);
    return data;
  },

  async transitionExperiment(id: string, body: ExperimentTransition): Promise<ExperimentRecord> {
    const { data } = await apiClient.post<ExperimentRecord>(`/experiments/${id}/transition`, body);
    return data;
  },

  async getPriorities(projectId: string, asOf?: string): Promise<PrioritiesPayload> {
    const { data } = await apiClient.get<PrioritiesPayload>(
      `/projects/${projectId}/experiment-priorities`,
      { params: asOf ? { as_of: asOf } : {} },
    );
    return data;
  },

  async getHistory(projectId: string): Promise<HistoryPayload> {
    const { data } = await apiClient.get<HistoryPayload>(`/projects/${projectId}/history`);
    return data;
  },

  async getCoverage(projectId: string, asOf?: string): Promise<CoveragePayload> {
    const { data } = await apiClient.get<CoveragePayload>(
      `/projects/${projectId}/calibration-coverage`,
      { params: asOf ? { as_of: asOf } : {} },
    );
    return data;
  },

  async getDesignOptions(projectId: string, channel: string): Promise<DesignOptions> {
    const { data } = await apiClient.get<DesignOptions>(
      `/projects/${projectId}/experiment-design/options`,
      { params: { channel } },
    );
    return data;
  },

  async computeDesign(
    projectId: string,
    body: DesignRequest,
  ): Promise<ExperimentDesignPayload> {
    const { data } = await apiClient.post<ExperimentDesignPayload>(
      `/projects/${projectId}/experiment-design`,
      body,
    );
    return data;
  },
};
