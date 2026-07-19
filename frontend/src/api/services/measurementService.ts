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
  /** creative/keyword/campaign arm within the channel (nullable) */
  subchannel?: string | null;
  recommending_run_id: string | null;
  calibrated_run_id: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic JSON blob; ExperimentDrawer renders fields like design.design_type/min_duration_periods directly as ReactNode, which `unknown` is not assignable to (narrowing here would break that consumer, which this cleanup must not edit)
  design: Record<string, any> | null;
  readout: Record<string, unknown> | null;
  priority: Record<string, unknown> | null;
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
  readout?: Record<string, unknown>;
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

/** A named analysis methodology from the planning.methods registry, with a
 * per-dataset supported flag + gate reason (geo count / pre-period length). */
export interface ExperimentMethodRow {
  key: string;
  name: string;
  family: 'geo' | 'national' | 'user' | 'switchback';
  min_geos: number;
  needs_panel: boolean;
  min_pre_weeks: number;
  references: string[];
  description: string;
  supported: boolean;
  reason: string;
}

export interface DesignOptions {
  n_geos: number;
  geos: string[];
  n_weeks: number;
  designs: DesignKey[];
  recommended: DesignKey;
  kpi: string;
  /** named methods (synthetic_control / tbr / gbr / did_mmt / …) */
  methods?: ExperimentMethodRow[];
}

export interface DesignRequest {
  channel: string;
  design_key?: DesignKey;
  /** named analysis methodology; selects the estimator and infers design_key */
  method?: string;
  design?: 'holdout' | 'scaling';
  intensity_pct?: number;
  n_pairs?: number;
  duration?: number;
  amplitude_pct?: number;
  block_weeks?: number;
  /** multi-level flighting spend multipliers (≥3 distinct levels trace the curve) */
  levels?: number[];
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
  /** model-anchored expected effect + verdict, attached when a fit is available */
  model_anchor?: Record<string, unknown>;
  se_source?: 'placebo_calibrated' | 'analytic';
  /** named analysis methodology (synthetic_control / tbr / gbr / did_mmt) */
  method?: string;
  method_name?: string;
  method_references?: string[];
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

// ── Ghost ads (user-level RCT power calculator) ───────────────────────────────

export interface GhostAdsPowerRequest {
  users_reached: number;
  baseline_rate?: number;
  treated_fraction?: number;
  outcome?: 'binary' | 'count' | 'revenue';
  baseline_mean?: number | null;
  baseline_dispersion?: number;
  value_sd?: number | null;
  alpha?: number;
  power_target?: number;
  two_sided?: boolean;
  exposure_rate?: number;
  cost_per_user?: number | null;
  value_per_conversion?: number | null;
  target_lift_abs?: number | null;
  simulate?: boolean;
  n_sims?: number;
  seed?: number;
}

export interface GhostAdsPowerPayload {
  outcome: string;
  users_reached: number;
  n_treated: number;
  n_ghost: number;
  baseline: number;
  se_null: number;
  mde_abs: number;
  mde_rel: number;
  itt_mde: number;
  tot_mde: number;
  exposure_rate: number;
  incremental_at_mde: number;
  alpha: number;
  power_target: number;
  two_sided: boolean;
  rare_event_regime: boolean;
  media_cost?: number;
  incremental_value_at_mde?: number;
  breakeven_lift_abs?: number;
  users_required_for_target?: number;
  power_at_target?: number;
  simulation?: {
    empirical_power: number;
    empirical_fpr: number;
    analytic_power: number;
    n_sims: number;
    true_lift_abs: number;
  };
}

// ── Model-anchored experiment economics (simulate) ────────────────────────────

export type ExperimentVerdict =
  | 'powered'
  | 'underpowered'
  | 'overpowered'
  | 'inconclusive';

/** Model's expected-effect anchor + powered-to-detect verdict (nullable). */
export interface ExperimentAnchor {
  roas_at_current_median: number;
  incremental_roas_median: number;
  incremental_roas_hdi: [number, number];
  expected_incremental_kpi_median: number;
  verdict: ExperimentVerdict;
  assurance: number | null;
  prob_detectable: number | null;
  recommended_duration: number | null;
  extrapolation_warning: boolean;
  /** present only when the EIG/EVOI loopback succeeds */
  eig?: number;
  evoi?: number;
  quadrant?: Quadrant;
  loopback_error?: string;
}

/** Short-term opportunity cost of deviating from BAU (nullable). */
export interface OpportunityCost {
  channel: string;
  kpi: string;
  design_key: string;
  design_type: string;
  duration_requested: number;
  duration_effective: number;
  n_treated_cells: number;
  n_test_rows: number;
  n_draws: number;
  carryover_basis: string;
  // KPI-unit risk (always present)
  expected_kpi_delta: number;
  kpi_delta_median: number;
  kpi_delta_p5: number;
  kpi_delta_p95: number;
  kpi_delta_with_carryover_median: number;
  forgone_kpi_median: number;
  forgone_kpi_p95: number;
  prob_kpi_loss: number;
  pct_of_window_kpi: number | null;
  // spend (deterministic, signed)
  spend_delta: number;
  abs_spend_change: number;
  spend_at_risk: number;
  // net-$ risk (null unless a margin resolves)
  margin_per_kpi: number | null;
  margin_source: string;
  kpi_kind: string;
  net_profit_impact_median: number | null;
  net_profit_impact_p5: number | null;
  net_profit_impact_p95: number | null;
  opportunity_cost_dollar_median: number | null;
  opportunity_cost_dollar_p95: number | null;
  prob_net_loss: number | null;
  prob_loss_over_threshold: number | null;
  loss_threshold: number | null;
  // learning-vs-cost
  evoi_kpi_units: number | null;
  evoi_per_week: number | null;
  cost_per_week: number | null;
  learning_to_cost_ratio: number | null;
  learning_to_cost_basis: string;
  response_horizon_weeks: number | null;
  // status / honesty
  low_information: boolean;
  extrapolation_warning: boolean;
  warnings: string[];
  notes: string[];
}

export interface PowerCurvePoint {
  effect: number;
  power: number;
  scale: number;
}

/** One estimator's A/A · A/B scorecard. */
export interface MethodologyRow {
  key: string;
  label: string;
  valid: boolean;
  fpr: number | null;
  fpr_tolerance: number;
  fpr_ci: [number, number];
  fpr_inflated: boolean;
  fpr_at_crit: number;
  n_eff_windows: number;
  null_method: string;
  null_sd: number;
  crit_value: number;
  empirical_mde: number | null;
  empirical_mde_roas: number | null;
  mde_method: string;
  power_at_expected_effect: number;
  powered: boolean;
  aa_status: string;
  ab_status: string;
  power_curve: PowerCurvePoint[];
}

/** A/A·A/B methodology leaderboard (nullable). */
export interface ExperimentSimulation {
  alpha: number;
  duration: number;
  kind: string;
  injection_basis: string;
  expected_effect: number;
  spend_delta_window: number;
  chosen_key: string | null;
  caveats: string[];
  methodologies: MethodologyRow[];
}

/** Full result of POST …/simulate → poll …/simulate/{jobId}. */
export interface ExperimentEconomicsPayload {
  channel: string;
  kpi: string;
  design_key: DesignKey | null;
  design_type: string | null;
  duration: number;
  randomized?: boolean;
  se_roas: number | null;
  mde_roas: number | null;
  se_source?: 'placebo_calibrated' | 'analytic' | null;
  model_anchored: boolean;
  anchor: ExperimentAnchor | null;
  opportunity_cost: OpportunityCost | null;
  simulation: ExperimentSimulation | null;
  /** Phase-3 headline: reallocation gain − test loss, netted */
  net_value?: ExperimentNetValuePayload | null;
  // surfaced when a sub-step fails but the overall job succeeds
  note?: string;
  anchor_error?: string;
  opportunity_cost_error?: string;
  simulation_error?: string;
  net_value_error?: string;
  design?: ExperimentDesignPayload;
}

/** Net experiment economics: E[reallocation gain] − E[test loss]. */
export interface ExperimentNetValuePayload {
  channel: string;
  unit: '$' | 'KPI units';
  basis: 'model_anchored' | 'evoi_bounded' | 'insufficient';
  test_loss: number | null;
  test_loss_p5: number | null;
  test_loss_p95: number | null;
  net_profit_during_test: number | null;
  evoi_raw: number | null;
  evpi_cap: number | null;
  decay_factor: number | null;
  reallocation_gain: number | null;
  net_value: number | null;
  net_value_p5: number | null;
  net_value_p95: number | null;
  prob_net_positive: number | null;
  breakeven_horizon_weeks: number | null;
  horizon_weeks: number;
  half_life_weeks: number | null;
  margin_per_kpi: number | null;
  warnings: string[];
}

export interface SimulateRequest {
  channel: string;
  design_key?: DesignKey;
  design?: 'holdout' | 'scaling';
  intensity_pct?: number;
  n_pairs?: number;
  duration?: number;
  amplitude_pct?: number;
  block_weeks?: number;
  /** multi-level flighting multipliers — keeps the sim on the same basis as a loaded candidate */
  levels?: number[];
  margin?: number;
  price?: number;
  kpi_kind?: string;
  seed?: number;
  max_draws?: number;
}

/** The polled job record from GET …/simulate/{jobId}. */
export interface SimulationJob {
  status: 'pending' | 'running' | 'done' | 'error';
  project_id: string;
  channel: string;
  result: ExperimentEconomicsPayload | null;
  error: string | null;
}

// ── PowerPoint slide-deck generation ──────────────────────────────────────────

export interface DeckRequest {
  client?: string | null;
  kpi_name?: string;
  currency?: string;
  break_even?: number;
  margin?: number | null;
  hdi_prob?: number;
}

export interface DeckJob {
  status: 'pending' | 'running' | 'done' | 'error';
  stage?: string;
  project_id: string;
  result: { n_slides: number; n_insights: number; filename?: string; download?: string } | null;
  error: string | null;
}

// ── Experiment-setup optimizer (Pareto front) ─────────────────────────────────

/** One evaluated design on the three Pareto objectives + its runnable setup. */
export interface CandidateEval {
  index: number;
  design_key: string;
  mode: 'holdout' | 'scaling' | 'flighting';
  footprint: 'full' | 'half' | 'national';
  n_pairs: number | null;
  intensity_pct: number;
  duration: number;
  // objectives (all "lower is better")
  mde_roas: number;
  power_shortfall: number;
  tradeoff: number;
  tradeoff_basis: 'net_dollar' | 'forgone_kpi' | 'spend_at_risk' | 'net_value';
  // net-value axis (present when the backend applied it): with basis
  // 'net_value', tradeoff === −net_value (lower-better Pareto convention).
  sigma_exp?: number | null;
  evoi_kpi?: number | null;
  reallocation_gain?: number | null;
  net_value?: number | null;
  net_value_basis?: string | null;
  // statistical power to detect the model's expected effect (null if unknown)
  power: number | null;
  // conservative power at the model's lower 95% effect bound (null if unknown)
  power_lower: number | null;
  power_target: number;
  // flighting only: power per estimand (ROAS, contribution, mROAS); the
  // `_lower` variants are the power at the model's lower 95% bound for each.
  power_breakdown: {
    roas: number | null;
    contribution: number | null;
    mroas: number | null;
    roas_lower: number | null;
    contribution_lower: number | null;
    mroas_lower: number | null;
    lower_quantile: number;
    mroas_identified: boolean;
    n_levels: number;
    min: number | null;
    min_lower: number | null;
    target: number;
  } | null;
  // supporting risk detail
  forgone_kpi_median: number;
  opportunity_cost_dollar_median: number | null;
  net_profit_impact_median: number | null;
  spend_at_risk: number;
  pct_of_window_kpi: number | null;
  duration_effective: number;
  // verdict
  powered: boolean;
  on_pareto: boolean;
  is_recommended: boolean;
  // runnable setup
  treatment_geos: string[];
  control_geos: string[];
  schedule: SchedulePoint[] | null;
  block_weeks: number | null;
  duration_requested: number | null;
  warnings: string[];
}

/** Adstock-derived washout period before treated cells are back to BAU. */
export interface Cooldown {
  cooldown_weeks: number;
  alpha: number | null;
  half_life: number | null;
  basis: string;
  threshold: number;
}

/** Full result of POST …/optimize → poll …/optimize/{jobId}. */
export interface ExperimentOptimizationPayload {
  channel: string;
  kpi: string;
  kind: 'geo' | 'national';
  cooldown: Cooldown;
  suggested_block_weeks: number;
  expected_incremental_roas: number | null;
  expected_incremental_kpi: number | null;
  margin_known: boolean;
  tradeoff_label: string;
  mixed_tradeoff_units: boolean;
  /** true when the cost objective is −net_value (gain − loss, $) */
  net_value_axis?: boolean;
  response_horizon_weeks?: number;
  /** provenance of the fitted EVOI surrogate: MC anchors + (k, delta) fit */
  evoi_anchor?: {
    anchors: [number, number][];
    evpi: number;
    tau: number;
    k: number;
    delta: number;
  } | null;
  power_target: number;
  design_space: {
    duration_min: number;
    duration_max: number;
    durations: number[];
    intensity_min: number;
    intensity_max: number;
    scaling_intensities: number[];
    include_holdout: boolean;
  };
  n_candidates: number;
  pareto_indices: number[];
  recommended_index: number | null;
  notes: string[];
  candidates: CandidateEval[];
  pareto: CandidateEval[];
  recommended: CandidateEval | null;
}

export interface OptimizeRequest {
  channel: string;
  margin?: number;
  price?: number;
  kpi_kind?: string;
  // design-space ranges (the optimizer auto-samples within each)
  duration_min?: number;
  duration_max?: number;
  intensity_min?: number;
  intensity_max?: number;
  include_holdout?: boolean;
  // explicit overrides (optional)
  durations?: number[];
  scaling_intensities?: number[];
  max_draws?: number;
  seed?: number;
}

/** The polled job record from GET …/optimize/{jobId}. */
export interface OptimizationJob {
  status: 'pending' | 'running' | 'done' | 'error';
  project_id: string;
  channel: string;
  result: ExperimentOptimizationPayload | null;
  error: string | null;
}

// ── Structural identification (multi-level flighting → parameter recovery) ────

/** One structural parameter's identification readout (β / α / ψ). */
export interface StructuralParamIdent {
  claimed: boolean;
  /** posterior-sd contraction vs the prior (the honest identification axis) */
  contraction: number | null;
  mde: number | null;
  mde_relative?: number | null;
  /** power to resolve the parameter from 0 (prior-heavy; UI-consistent) */
  power: number | null;
}

export interface StructuralIdentification {
  params: {
    beta: StructuralParamIdent;
    alpha: StructuralParamIdent;
    lam: StructuralParamIdent;
  };
  identifies_anything: boolean;
  binding_power: number | null;
  binding_contraction: number | null;
  power_target: number;
  n_clamped?: number;
}

/** Full result of POST …/identify → poll …/identify/{jobId}. An OPTIMISTIC
 *  Laplace upper bound on what the next refit identifies — never a guarantee. */
export interface StructuralIdentificationPayload {
  channel: string;
  kpi: string;
  design_key: string;
  block_weeks: number;
  duration: number;
  n_levels: number;
  schedule: { multiplier: number; [k: string]: unknown }[];
  cooldown_weeks: number | null;
  block_ge_cooldown: boolean;
  reduced_form: Record<string, unknown> | null;
  extrapolation_warning: boolean;
  structural: StructuralIdentification | null;
  /** false ⇒ reduced-form only (needs a parametric geometric+logistic national fit) */
  structural_gated: boolean;
  structural_gate_reason: string | null;
  note?: string;
}

export interface IdentifyRequest {
  channel: string;
  /** spend multipliers; ≥3 in-support levels trace the saturation curve */
  levels?: number[];
  /** default: the channel's adstock cool-down sets the block length */
  block_weeks?: number;
  duration?: number;
  max_draws?: number;
  seed?: number;
}

/** The polled job record from GET …/identify/{jobId}. */
export interface IdentificationJob {
  status: 'pending' | 'running' | 'done' | 'error';
  project_id: string;
  channel: string;
  result: StructuralIdentificationPayload | null;
  error: string | null;
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

  /** Stateless user-level ghost-ads power calculation (no dataset / model). */
  async ghostAdsPower(
    projectId: string,
    body: GhostAdsPowerRequest,
  ): Promise<GhostAdsPowerPayload> {
    const { data } = await apiClient.post<GhostAdsPowerPayload>(
      `/projects/${projectId}/ghost-ads/power`,
      body,
    );
    return data;
  },

  /** Kick off the non-blocking model-anchored economics + A/A·A/B job (HTTP 202). */
  async startSimulation(
    projectId: string,
    body: SimulateRequest,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/experiment-design/simulate`,
      body,
    );
    return data;
  },

  /** Poll a simulation job; resolves to {status, result|null, error|null}. */
  async pollSimulation(projectId: string, jobId: string): Promise<SimulationJob> {
    const { data } = await apiClient.get<SimulationJob>(
      `/projects/${projectId}/experiment-design/simulate/${jobId}`,
    );
    return data;
  },

  /** Kick off the non-blocking Pareto-front design optimizer (HTTP 202). */
  async startOptimization(
    projectId: string,
    body: OptimizeRequest,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/experiment-design/optimize`,
      body,
    );
    return data;
  },

  /** Poll an optimization job; resolves to {status, result|null, error|null}. */
  async pollOptimization(projectId: string, jobId: string): Promise<OptimizationJob> {
    const { data } = await apiClient.get<OptimizationJob>(
      `/projects/${projectId}/experiment-design/optimize/${jobId}`,
    );
    return data;
  },

  /** Kick off the non-blocking structural-identification analysis (HTTP 202). */
  async startIdentification(
    projectId: string,
    body: IdentifyRequest,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/experiment-design/identify`,
      body,
    );
    return data;
  },

  /** Poll an identification job; resolves to {status, result|null, error|null}. */
  async pollIdentification(projectId: string, jobId: string): Promise<IdentificationJob> {
    const { data } = await apiClient.get<IdentificationJob>(
      `/projects/${projectId}/experiment-design/identify/${jobId}`,
    );
    return data;
  },

  /** Kick off the non-blocking PowerPoint slide-deck build (HTTP 202). */
  async startDeckGeneration(
    projectId: string,
    body: DeckRequest,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/generate-deck`,
      body,
    );
    return data;
  },

  /** Poll a slide-deck job; resolves to {status, stage, result|null, error|null}. */
  async pollDeckJob(projectId: string, jobId: string): Promise<DeckJob> {
    const { data } = await apiClient.get<DeckJob>(
      `/projects/${projectId}/generate-deck/${jobId}`,
    );
    return data;
  },
};
