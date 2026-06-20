import { apiClient } from '../client';

/** A registered garden model version (mirrors sessions.py _garden_row_to_dict). */
export interface GardenModel {
  id: string;
  org_id: string;
  name: string;
  version: number;
  owner_user_id: string | null;
  status: 'draft' | 'tested' | 'published' | 'deprecated';
  docs: string | null;
  manifest: {
    contract_version?: string;
    class_name?: string;
    dataset_schema?: Record<string, unknown>;
    recommended_fit?: Record<string, unknown>;
    tags?: string[];
  };
  source_path: string | null;
  compat_report: CompatReport | null;
  base_run_id: string | null;
  reference_artifact_path: string | null;
  status_history: { status: string; at: number; note?: string }[];
  created_at: number;
  updated_at: number;
}

export interface CompatTier {
  name: string;
  passed: boolean;
  blocking: boolean;
  detail: string;
  skipped: boolean;
}

export interface CompatReport {
  contract_version: string;
  class_name: string;
  is_bayesian_mmm_subclass: boolean;
  scenario: string;
  fit_method: string;
  tiers: CompatTier[];
  blocking_passed: boolean;
  score: number | null;
  summary: string;
}

export interface GardenRegisterRequest {
  source_code: string;
  name: string;
  docs?: string;
  version?: number | null;
  tags?: string[] | null;
  dataset_schema?: Record<string, unknown> | null;
  recommended_fit?: Record<string, unknown> | null;
}

export interface GardenTestJob {
  status: 'pending' | 'running' | 'done' | 'error';
  result: {
    blocking_passed: boolean;
    score: number | null;
    promoted: boolean;
    summary: string | null;
    tiers: CompatTier[] | null;
  } | null;
  error: string | null;
}

export const modelGardenService = {
  /** GET /model-garden — latest version per name (org-scoped) by default. */
  async list(params?: {
    status?: string;
    name?: string;
    all_versions?: boolean;
  }): Promise<{ models: GardenModel[] }> {
    const { data } = await apiClient.get<{ models: GardenModel[] }>('/model-garden', {
      params,
    });
    return data;
  },

  /** GET /model-garden/{name}/versions — every version, newest first. */
  async listVersions(name: string): Promise<{ versions: GardenModel[] }> {
    const { data } = await apiClient.get<{ versions: GardenModel[] }>(
      `/model-garden/${encodeURIComponent(name)}/versions`,
    );
    return data;
  },

  /** GET /model-garden/{name}/{version} — one version (manifest + compat report). */
  async get(name: string, version: number): Promise<GardenModel> {
    const { data } = await apiClient.get<GardenModel>(
      `/model-garden/${encodeURIComponent(name)}/${version}`,
    );
    return data;
  },

  /** GET /model-garden/{name}/{version}/source — source text for the editor. */
  async getSource(name: string, version: number): Promise<{ source_code: string }> {
    const { data } = await apiClient.get<{ source_code: string }>(
      `/model-garden/${encodeURIComponent(name)}/${version}/source`,
    );
    return data;
  },

  /** POST /model-garden — register a new draft from editor source. */
  async register(req: GardenRegisterRequest): Promise<GardenModel> {
    const { data } = await apiClient.post<GardenModel>('/model-garden', req);
    return data;
  },

  /** POST /model-garden/{name}/{version}/test — kick off the compat job (202). */
  async startTest(
    name: string,
    version: number,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post<{ job_id: string; status: string }>(
      `/model-garden/${encodeURIComponent(name)}/${version}/test`,
    );
    return data;
  },

  /** GET /model-garden/{name}/{version}/test/{job_id} — poll the compat job. */
  async pollTest(name: string, version: number, jobId: string): Promise<GardenTestJob> {
    const { data } = await apiClient.get<GardenTestJob>(
      `/model-garden/${encodeURIComponent(name)}/${version}/test/${jobId}`,
    );
    return data;
  },

  /** POST /model-garden/{name}/{version}/promote — the human publish gate. */
  async promote(name: string, version: number, note = ''): Promise<GardenModel> {
    const { data } = await apiClient.post<GardenModel>(
      `/model-garden/${encodeURIComponent(name)}/${version}/promote`,
      { note },
    );
    return data;
  },

  /** DELETE /model-garden/{name}/{version} — draft/deprecated only. */
  async remove(name: string, version: number): Promise<{ deleted: boolean }> {
    const { data } = await apiClient.delete<{ deleted: boolean }>(
      `/model-garden/${encodeURIComponent(name)}/${version}`,
    );
    return data;
  },
};
