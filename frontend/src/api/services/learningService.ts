import { apiClient } from '../client';

// ── Continuous-learning programs (Sextant) ────────────────────────────────────
// Typed mirror of the REST surface pinned in
// technical-docs/continuous-learning-wiring.md §3.5, with the SNAPSHOT payload
// (§3.1) reproduced verbatim. All program money at this boundary is DOLLARS
// PER GEO-PERIOD (per geo, per period): a $2M/week national budget across
// 50 geos is $40k per geo-week.

/** Arm separator used by continuous_learning/arms.py (e.g. "Search │ Brand"). */
export const ARM_SEP = ' │ ';

export type LearningProgramStatus = 'active' | 'stopped' | 'archived';
export type LearningWaveStatus = 'designed' | 'ingested';
export type LearningWaveSource = 'wave' | 'experiment_import' | 'manual';
export type FundingVerdict = 'FUND' | 'HOLD' | 'CUT';

/** One channel's funding-line readout at the recommended allocation. */
export interface FundingRow {
  channel: string;
  /** value-inclusive marginal return per $1 of spend (break-even 1.0) */
  mroas_mean: number;
  /**
   * mROAS per margin-dollar (present when the program sets a margin);
   * verdicts are computed on this margin-adjusted value server-side.
   */
  mroas_margin_adjusted?: number;
  prob_above_line: number;
  funded: boolean;
  verdict: FundingVerdict;
}

/** One probed channel pair's interaction (γ) posterior. */
export interface GammaPair {
  pair: [string, string];
  mean: number;
  p5: number;
  p95: number;
  sign: string;
  prior_dominated: boolean;
}

/** 25-point response curve over [0, 2×center] with a 90% band (dollars). */
export interface LearningResponseCurve {
  spend_dollars: number[];
  mean: number[];
  lo: number[];
  hi: number[];
  current: number;
}

export interface LearningRegret {
  /** expected value-$ left on the table PER GEO-PERIOD (value-inclusive) */
  e_regret_kpi: number;
  e_regret_dollars: number;
  enbs: number;
  stop: boolean;
  margin: number;
  /** geo-periods the decision governs (n_geos × horizon periods) */
  population: number;
  wave_cost: number;
}

/** The pinned SNAPSHOT payload from fit_and_plan — THE UI payload (§3.1). */
export interface LearningSnapshot {
  schema_version: number;
  fitted_at: number;
  evidence: {
    n_rows: number;
    n_summaries: number;
    n_waves: number;
    shape_identified: Record<string, boolean>;
  };
  diagnostics: {
    max_rhat: number | null;
    min_ess: number | null;
    n_draws: number | null;
    flags: string[];
  };
  recommendation: Record<string, number>;
  recommendation_scaled: Record<string, number>;
  allocation_sd: Record<string, number>;
  funding: FundingRow[];
  regret: LearningRegret;
  gamma: GammaPair[];
  response_curves: Record<string, LearningResponseCurve>;
  warnings: string[];
}

/** Program config dict (stored as config_json; dollars per geo-period). */
export interface LearningProgramConfig {
  channels: string[];
  /** optional creative/keyword arms per parent channel; expands via arms.py */
  arms?: Record<string, string[]>;
  /** $/period PER GEO, per channel/arm */
  center?: Record<string, number>;
  /** $/period PER GEO */
  budget: number;
  /** $ per KPI unit (funding line) */
  value_per_unit: number;
  /** $ per scaled unit (default: center, floored) */
  spend_ref?: Record<string, number>;
  mode?: 'fixed' | 'free';
  cap?: number | null;
  /** response family: hill | logistic | monotone_spline | hill_mixture */
  activation?: string;
  /** observation model: normal (default) | studentt (heavy-tailed) |
   *  negbinomial (count KPI — y must be natural integer counts; CUPED off) */
  likelihood?: 'normal' | 'studentt' | 'negbinomial';
  /** national period effect τ_t: none (default) | national (needs a period
   *  column on every ingested wave) */
  time_effect?: 'none' | 'national';
  gamma_scale?: number;
  beta_scale?: number;
  pair_signs?: Record<string, string>;
  kpi?: string | null;
  cadence_weeks?: number;
  /** ENBS economics, in $ terms */
  margin?: number;
  /**
   * Decision horizon in periods — send THIS on create; the backend computes
   * population = n_geos × horizon_periods itself.
   */
  horizon_periods?: number;
  /**
   * DEPRECATED — geo-periods, read back from legacy stored programs only.
   * Do not send on create; use horizon_periods.
   */
  population?: number;
  wave_cost?: number;
}

/** A learning_programs row (JSON columns parsed server-side). */
export interface LearningProgram {
  id: string;
  project_id: string | null;
  thread_id: string | null;
  name: string;
  status: LearningProgramStatus;
  channels: string[];
  config: LearningProgramConfig;
  state_path: string | null;
  /** latest SNAPSHOT, null before the first fit */
  summary: LearningSnapshot | null;
  created_at: number;
  updated_at: number;
}

/** Per-candidate Laplace knowledge-gradient score (optimize=true). */
export interface KgScore {
  delta: number;
  probe_pairs: [number, number][];
  score: number;
}

export interface KgInfo {
  used: boolean;
  chosen_delta: number;
  chosen_probe_pairs: [number, number][];
  scores: KgScore[];
  sigma: number | null;
}

/** Result of POST …/design-wave (sync). */
export interface DesignWavePayload {
  cells_scaled: number[][];
  cells_dollars: number[][];
  cell_labels: string[];
  assignment?: Record<string, unknown> | null;
  n_cells: number;
  delta: number;
  probe_pairs: [number, number][];
  warnings: string[];
  /** present when the Laplace-KG design optimization actually ran */
  kg?: KgInfo;
}

/** A learning_waves row (JSON columns parsed server-side). */
export interface LearningWave {
  id: string;
  program_id: string;
  project_id: string | null;
  wave_index: number;
  status: LearningWaveStatus;
  source: LearningWaveSource | null;
  design: DesignWavePayload | null;
  observations: Record<string, unknown> | null;
  snapshot: LearningSnapshot | null;
  experiment_ids: string[] | null;
  created_at: number;
  updated_at: number;
}

// ── Requests ──────────────────────────────────────────────────────────────────

export interface CreateProgramRequest {
  name: string;
  /** §3.1 program config, dollars */
  config: LearningProgramConfig;
}

export interface DesignWaveRequest {
  delta: number;
  /**
   * Index pairs over the program's (arm-expanded) channel order. REQUIRED —
   * always send the selection explicitly: [] means a probe-free design,
   * whereas omitting the key makes the backend probe ALL program pairs.
   */
  probe_pairs: [number, number][];
  n_geo?: number;
  n_holdout?: number;
  seed?: number;
  /** stratify the geo→cell assignment on accumulated per-geo KPI (default true) */
  stratify?: boolean;
  /** score candidate deltas with the Laplace knowledge gradient and design
   *  the EVSI-best one (needs a fitted posterior; falls back to `delta`) */
  optimize?: boolean;
  /** candidates for optimize=true (server default 0.3/0.6/0.9; max 8) */
  candidate_deltas?: number[];
  /** fantasy outcomes per candidate for the KG (server default 32, 8–256) */
  kg_n_outcomes?: number;
}

/** Exactly one of rows / experiment_ids / csv_text should be provided. */
export interface IngestWaveRequest {
  rows?: Record<string, number | string>[];
  experiment_ids?: string[];
  csv_text?: string;
}

export interface FitRequest {
  margin?: number;
  /** geo-periods override (default: n_geos × horizon_periods, server-side) */
  population?: number;
  wave_cost?: number;
}

// ── Jobs ──────────────────────────────────────────────────────────────────────

export interface SkippedExperiment {
  id: string;
  reason: string;
}

/**
 * Job result: the SNAPSHOT, plus imported/skipped when the wave came from an
 * experiment import (§3.4 tool 2 mirrors the same report).
 */
export interface LearningJobResult extends LearningSnapshot {
  imported?: number;
  skipped?: SkippedExperiment[];
}

/** The polled job record from GET …/jobs/{jobId}. */
export interface LearningJob {
  status: 'pending' | 'running' | 'done' | 'error';
  result: LearningJobResult | null;
  error: string | null;
}

// ── Service ───────────────────────────────────────────────────────────────────

export const learningService = {
  async listPrograms(projectId: string): Promise<LearningProgram[]> {
    const { data } = await apiClient.get<{ programs: LearningProgram[] }>(
      `/projects/${projectId}/learning-programs`,
    );
    return data.programs;
  },

  async createProgram(
    projectId: string,
    body: CreateProgramRequest,
  ): Promise<LearningProgram> {
    // the 201 body is wrapped {program: {...}}, same envelope as the GETs
    const { data } = await apiClient.post<{ program: LearningProgram }>(
      `/projects/${projectId}/learning-programs`,
      body,
    );
    return data.program;
  },

  async getProgram(
    projectId: string,
    programId: string,
  ): Promise<{ program: LearningProgram; waves: LearningWave[] }> {
    const { data } = await apiClient.get<{ program: LearningProgram; waves: LearningWave[] }>(
      `/projects/${projectId}/learning-programs/${programId}`,
    );
    return data;
  },

  async deleteProgram(projectId: string, programId: string): Promise<void> {
    await apiClient.delete(`/projects/${projectId}/learning-programs/${programId}`);
  },

  /** Synchronous CCD wave design; also stores a `designed` wave row. */
  async designWave(
    projectId: string,
    programId: string,
    body: DesignWaveRequest,
  ): Promise<DesignWavePayload> {
    const { data } = await apiClient.post<DesignWavePayload>(
      `/projects/${projectId}/learning-programs/${programId}/design-wave`,
      body,
    );
    return data;
  },

  /** Ingest observations / import experiments; spawns a fit job (HTTP 202). */
  async ingestWave(
    projectId: string,
    programId: string,
    body: IngestWaveRequest,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/learning-programs/${programId}/waves`,
      body,
    );
    return data;
  },

  /** Refit on the current evidence, optionally overriding economics (HTTP 202). */
  async startFit(
    projectId: string,
    programId: string,
    body: FitRequest = {},
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/projects/${projectId}/learning-programs/${programId}/fit`,
      body,
    );
    return data;
  },

  /** Poll a learning fit job; resolves to {status, result|null, error|null}. */
  async pollJob(projectId: string, programId: string, jobId: string): Promise<LearningJob> {
    const { data } = await apiClient.get<LearningJob>(
      `/projects/${projectId}/learning-programs/${programId}/jobs/${jobId}`,
    );
    return data;
  },
};
