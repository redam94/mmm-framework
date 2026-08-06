import { apiClient } from '../client';

// ── Result shapes (mirror the plan_budget / plan_scenario model-ops) ───────────

export interface AllocationRow {
  channel: string;
  current_spend?: number;
  current_share_pct?: number;
  optimal_spend: number;
  optimal_share_pct?: number;
  change_pct?: number;
  optimal_share_p5?: number;
  optimal_share_p95?: number;
  allocation_instability?: number;
  /** Decision-arm kind (#226): "media" | "promo". Absent = media. */
  arm_kind?: string;
  /** The arm's own units for optimal_level (e.g. "avg weekly depth (fraction)"). */
  level_units?: string;
  /** Recommended level in the arm's own units (a depth, not dollars, for promo). */
  optimal_level?: number;
}

// Budget optimizer v2 (#139).
export interface FrontierPoint {
  total_budget: number;
  expected_return: number;
  return_p5: number;
  return_p95: number;
  marginal_roi: number;
  allocation: Record<string, number>;
}

export interface FrontierResult {
  objective: string;
  objective_label: string;
  channels: string[];
  current_total: number;
  current_return: number;
  points: FrontierPoint[];
  notes: string[];
}

export interface GoalSeekResult {
  target_kpi: number;
  objective: string;
  objective_label: string;
  channels: string[];
  feasible: boolean;
  required_budget: number | null;
  allocation: Record<string, number> | null;
  expected_return: number | null;
  prob_hit_target: number | null;
  notes: string[];
}

export interface BudgetGroupConstraint {
  name?: string;
  channels: string[];
  min_share?: number;
  max_share?: number;
  min_spend?: number;
  max_spend?: number;
}

export interface GeoAllocationRow extends AllocationRow {
  geo: string;
}

export interface FlightingScheduleRow {
  period: string;
  total: number;
  [channel: string]: number | string;
}

export interface FlightingSchedule {
  pattern: string;
  n_periods: number;
  total_budget: number;
  periods: string[];
  channels: string[];
  schedule: FlightingScheduleRow[];
  by_channel: Record<string, number[]>;
}

export interface BudgetPlanResult {
  by_geo: boolean;
  total_budget: number;
  current_total: number;
  expected_uplift: number;
  uplift_hdi: [number, number];
  prob_positive_uplift: number;
  n_draws: number;
  allocation: AllocationRow[];
  geo_allocation?: GeoAllocationRow[];
  geos?: string[];
  flighting?: FlightingSchedule;
  notes: string[];
  // Budget optimizer v2 (#139).
  objective?: string;
  objective_label?: string;
  mode?: string;
  /** What one KPI unit was taken to be worth, and where that came from (#215,
   *  #221). Present only under mode='free' — the only mode whose recommendation
   *  depends on it. A dollar recommendation must name its exchange rate. */
  value_per_kpi?: number | null;
  value_source?: string | null;
  shadow_price?: number | null;
  marginal_roas?: Record<string, number> | null;
  frontier?: FrontierResult;
  goal_seek?: GoalSeekResult;
}

export interface ScenarioChannelDetail {
  original: number;
  scenario: number;
  change: number;
  change_pct: number;
}

export interface PlannerScenarioResult {
  spend_changes_applied: Record<string, number>;
  time_period: [number, number] | null;
  baseline_outcome: number;
  scenario_outcome: number;
  outcome_change: number;
  outcome_change_pct: number;
  channel_details: Record<string, ScenarioChannelDetail>;
  outcome_change_hdi?: [number, number];
  prob_positive?: number;
  n_draws?: number;
  hdi_prob?: number;
}

// ── Request shapes ─────────────────────────────────────────────────────────────

export interface FlightingRequest {
  pattern: string;
  n_periods: number;
  front_load?: number;
  pulse_on?: number;
  pulse_off?: number;
}

export interface PlannerOptimizeRequest {
  total_budget?: number | null;
  budget_change_pct?: number | null;
  min_multiplier?: number;
  max_multiplier?: number;
  channel_bounds?: Record<string, [number, number]> | null;
  by_geo?: boolean;
  flighting?: FlightingRequest | null;
  max_draws?: number;
  // Budget optimizer v2 (#139).
  abs_bounds?: Record<string, [number, number]> | null;
  groups?: BudgetGroupConstraint[] | null;
  min_channel_spend?: number | null;
  objective?: string;
  mode?: string;
  value_per_kpi?: number;
  frontier?: boolean | null;
  target_kpi?: number | null;
}

export interface PlannerScenarioRequest {
  spend_changes: Record<string, number>;
  time_period?: [number, number] | null;
  max_draws?: number;
}

export interface PlannerForecastRequest {
  channel_budgets?: Record<string, number> | null;
  future_media?: Record<string, number[]> | null;
  future_controls?: Record<string, number[]> | null;
  n_periods?: number;
  pattern?: string;
  interval?: number;
  include_noise?: boolean;
  start_date?: string | null;
  max_draws?: number;
}

/** A forecast is a counterfactual under a plan the model never observed, so the
 *  caveats are part of the payload rather than page furniture. `lower`/`upper`
 *  are null per period when the posterior had too few draws to form an interval
 *  (a MAP fit has one) — a collapsed band would read as extreme precision. */
export interface ForecastResultPayload {
  periods: string[];
  mean: number[];
  lower: (number | null)[];
  upper: (number | null)[];
  baseline: number[];
  by_channel: Record<string, number[]>;
  interval: number;
  caveats: string[];
  caveat_fields: {
    trend_extrapolation: { policy?: string; trend_type?: string; n_train_periods?: number };
    interval_widens_with_horizon: boolean;
    extrapolated_channels: { channel: string; multiple: number }[];
    residual_autocorrelation: { ljung_box_p: number | null; autocorrelated: boolean | null };
    interval_noun: string;
    inference_family: string;
    approximate: boolean;
    fit_method: string | null;
    interval_available: boolean;
  };
  headline: {
    total: number;
    total_lower: number | null;
    total_upper: number | null;
    interval_available: boolean;
    interval_noun: string;
  };
  n_draws: number;
  calendar: { start: string; n_periods: number; cadence: string } | null;
}

export interface PlannerJob<T> {
  status: 'pending' | 'running' | 'done' | 'error';
  project_id: string;
  result: T | null;
  error: string | null;
}

// ── Service (non-blocking jobs: start → poll, mirrors measurementService) ───────


/** A committed plan-of-record version (payload elided in listings) (#225). */
export interface PlanVersionSummary {
  id: string;
  plan_family: string;
  version: number;
  status: string;
  name?: string | null;
  committed_at?: string | null;
  committed_by?: string | null;
  run_id?: string | null;
}

export interface PlanOfRecordCommitResponse {
  committable?: boolean;
  assessment?: {
    committable: boolean;
    refusals: { gate: string; reason: string; overridable: boolean }[];
    missing_provenance?: string[];
  };
  id?: string;
  version?: number;
  plan_family?: string;
}

export const plannerService = {
  async startOptimize(
    projectId: string,
    body: PlannerOptimizeRequest,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/planner/optimize`,
      body,
    );
    return data;
  },
  async pollOptimize(
    projectId: string,
    jobId: string,
  ): Promise<PlannerJob<BudgetPlanResult>> {
    const { data } = await apiClient.get<PlannerJob<BudgetPlanResult>>(
      `/projects/${projectId}/planner/optimize/${jobId}`,
    );
    return data;
  },
  async startScenario(
    projectId: string,
    body: PlannerScenarioRequest,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/planner/scenario`,
      body,
    );
    return data;
  },
  async pollScenario(
    projectId: string,
    jobId: string,
  ): Promise<PlannerJob<PlannerScenarioResult>> {
    const { data } = await apiClient.get<PlannerJob<PlannerScenarioResult>>(
      `/projects/${projectId}/planner/scenario/${jobId}`,
    );
    return data;
  },
  async commitPlanOfRecord(
    projectId: string,
    body: {
      forecast: ForecastResultPayload;
      plan_family?: string;
      name?: string | null;
      overrides?: Record<string, string> | null;
      assess_only?: boolean;
    },
  ): Promise<PlanOfRecordCommitResponse> {
    const { data } = await apiClient.post<PlanOfRecordCommitResponse>(
      `/projects/${projectId}/plan-of-record`,
      body,
    );
    return data;
  },
  async planOfRecordHistory(
    projectId: string,
  ): Promise<{ versions: PlanVersionSummary[]; total: number }> {
    const { data } = await apiClient.get<{
      versions: PlanVersionSummary[];
      total: number;
    }>(`/projects/${projectId}/plan-of-record/history`);
    return data;
  },
  async startForecast(
    projectId: string,
    body: PlannerForecastRequest,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/planner/forecast`,
      body,
    );
    return data;
  },
  async pollForecast(
    projectId: string,
    jobId: string,
  ): Promise<PlannerJob<{ forecast: ForecastResultPayload }>> {
    const { data } = await apiClient.get<PlannerJob<{ forecast: ForecastResultPayload }>>(
      `/projects/${projectId}/planner/forecast/${jobId}`,
    );
    return data;
  },
};
