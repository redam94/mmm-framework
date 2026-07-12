import { apiClient } from '../client';

/**
 * Spec-curve / model-averaging robustness sweep (issue #118). A non-blocking
 * multi-fit job: start it, then poll the job id. Mirrors api/main.py
 * /projects/{id}/spec-curve[/{job_id}] and validation/spec_curve.py's to_dict().
 */

export type JobStatus = 'pending' | 'running' | 'done' | 'error';

export interface SpecCurveRobustness {
  min: number;
  max: number;
  range: number;
  spread_pct: number | null;
  sign_stable: boolean;
  profitable_weight: number;
  primary: number | null;
  n_specs: number;
}

export interface SpecCurveResult {
  channels: string[];
  specs: string[];
  primary: string | null;
  hdi_prob: number;
  weights: Record<string, number>;
  bma: Record<string, { mean: number; lower: number; upper: number }>;
  robustness: Record<string, SpecCurveRobustness>;
  per_spec: Record<
    string,
    {
      primary: boolean;
      roi: Record<string, { mean: number; lower: number; upper: number }>;
      loo: Record<string, number> | null;
      weight: number;
      error: string | null;
    }
  >;
}

export interface SpecCurveJob {
  status: JobStatus;
  n_specs?: number;
  result: SpecCurveResult | null;
  error: string | null;
  project_id?: string;
}

export interface StartSpecCurveBody {
  variants?: Array<Record<string, unknown>> | null;
  rationale?: string;
  max_draws?: number;
  compute_loo?: boolean;
}

export const specCurveService = {
  async start(
    projectId: string,
    body: StartSpecCurveBody = {},
  ): Promise<{ job_id: string; status: JobStatus; n_specs: number }> {
    const { data } = await apiClient.post<{ job_id: string; status: JobStatus; n_specs: number }>(
      `/projects/${projectId}/spec-curve`,
      body,
    );
    return data;
  },

  async poll(projectId: string, jobId: string): Promise<SpecCurveJob> {
    const { data } = await apiClient.get<SpecCurveJob>(
      `/projects/${projectId}/spec-curve/${jobId}`,
    );
    return data;
  },
};
