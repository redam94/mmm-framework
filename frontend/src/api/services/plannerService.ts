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

export interface PlannerJob<T> {
  status: 'pending' | 'running' | 'done' | 'error';
  project_id: string;
  result: T | null;
  error: string | null;
}

// ── Service (non-blocking jobs: start → poll, mirrors measurementService) ───────

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
};
