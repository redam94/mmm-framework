import { apiClient } from '../client';

/**
 * Manual (no-LLM) model-validation runner. Kicks off a background job that loads
 * the project's latest fitted model and runs a validation op (the same ops the
 * chat agent's validate_model / run_* tools use), then returns the markdown
 * verdict + content-addressed table/plot refs. Mirrors api/main.py
 * /projects/{id}/validate.
 */

export type ValidationCheck =
  | 'validate'
  | 'ppc'
  | 'residuals'
  | 'channels'
  | 'refutation'
  | 'cross_validation'
  | 'coverage';

export interface ValidationResult {
  content: string | null;
  tables: Array<{ id: string; title?: string; [k: string]: unknown }>;
  plots: Array<{ id: string; title?: string; [k: string]: unknown }>;
}

export interface ValidationJob {
  status: 'pending' | 'running' | 'done' | 'error';
  check: string;
  result: ValidationResult | null;
  error: string | null;
}

/** One row of the project's persistent validation history — UI-started jobs
 * AND chat-run checks (the agent's validate_model / run_* / SBC tools). */
export interface ValidationHistoryItem {
  job_id: string;
  check: string;
  status: 'pending' | 'running' | 'done' | 'error';
  source: 'job' | 'chat' | string;
  thread_id: string | null;
  error: string | null;
  created_at: number;
}

export const validationService = {
  async start(
    projectId: string,
    check: ValidationCheck,
    threadId?: string | null,
  ): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post(`/projects/${projectId}/validate`, {
      check,
      thread_id: threadId ?? null,
    });
    return data;
  },
  async poll(projectId: string, jobId: string): Promise<ValidationJob> {
    const { data } = await apiClient.get<ValidationJob>(
      `/projects/${projectId}/validate/${jobId}`,
    );
    return data;
  },
  async history(
    projectId: string,
    threadId?: string | null,
  ): Promise<ValidationHistoryItem[]> {
    const q = threadId ? `?thread_id=${encodeURIComponent(threadId)}` : '';
    const { data } = await apiClient.get<{ validations: ValidationHistoryItem[] }>(
      `/projects/${projectId}/validations${q}`,
    );
    return data.validations ?? [];
  },
};
