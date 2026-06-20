/**
 * Atelier IDE + modeling-copilot service.
 *
 * - `lint` / `format` are plain JSON round-trips (axios) backing the editor's
 *   Problems + Format buttons.
 * - `streamCopilot` opens the SSE stream from `POST /model-garden/copilot`
 *   (same `data: {...}\n\n` / `[DONE]` framing as `/chat`) and returns the raw
 *   `Response` so the caller can read tokens incrementally.
 */
import {
  apiClient,
  bearerHeader,
  expertHeaders,
  getStoredApiKey,
  getStoredBaseUrl,
  getStoredModelName,
  getStoredProvider,
} from '../client';

export interface LintProblem {
  severity: 'error' | 'warning' | 'info';
  message: string;
  line: number | null;
}

export interface LintResult {
  class_name: string | null;
  problems: LintProblem[];
  ok: boolean;
}

export interface FormatResult {
  formatted: string | null;
  error: string | null;
}

export interface CopilotTurn {
  role: 'user' | 'assistant';
  content: string;
}

/** Per-request auth/model headers — mirrors the axios interceptor so the SSE
 * fetch (which bypasses axios) still carries the user's key + model choice. */
function copilotHeaders(): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' };
  const key = getStoredApiKey();
  if (key) h['X-API-Key'] = key;
  const model = getStoredModelName();
  if (model) h['X-Model-Name'] = model;
  const baseUrl = getStoredBaseUrl();
  if (baseUrl) h['X-Base-Url'] = baseUrl;
  const provider = getStoredProvider();
  if (provider) h['X-Provider'] = provider;
  Object.assign(h, expertHeaders(), bearerHeader());
  return h;
}

export const copilotService = {
  async lint(sourceCode: string): Promise<LintResult> {
    const { data } = await apiClient.post<LintResult>('/model-garden/lint', {
      source_code: sourceCode,
    });
    return data;
  },

  async format(sourceCode: string): Promise<FormatResult> {
    const { data } = await apiClient.post<FormatResult>('/model-garden/format', {
      source_code: sourceCode,
    });
    return data;
  },

  streamCopilot(
    messages: CopilotTurn[],
    sourceCode: string,
    signal: AbortSignal,
  ): Promise<Response> {
    const base = apiClient.defaults.baseURL ?? '';
    return fetch(`${base}/model-garden/copilot`, {
      method: 'POST',
      headers: copilotHeaders(),
      body: JSON.stringify({ messages, source_code: sourceCode }),
      signal,
    });
  },
};
