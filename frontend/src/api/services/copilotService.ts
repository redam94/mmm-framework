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

/** Notebook context attached to a copilot turn so it can diagnose a failed cell.
 * Mirrors api/main.py NotebookCopilotContext. */
export interface NotebookCopilotContext {
  cell_code: string;
  traceback: string;
  dataset_preview?: string | null;
  other_cells: string[];
  is_error: boolean;
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
    notebook?: NotebookCopilotContext | null,
  ): Promise<Response> {
    const base = apiClient.defaults.baseURL ?? '';
    return fetch(`${base}/model-garden/copilot`, {
      method: 'POST',
      headers: copilotHeaders(),
      body: JSON.stringify({
        messages,
        source_code: sourceCode,
        notebook: notebook ?? null,
      }),
      signal,
    });
  },
};

/**
 * Read a copilot SSE stream (`data: {type:'token'|'error', content}` / `[DONE]`).
 * Calls `onToken` with the ACCUMULATED text on each token and `onError` with the
 * message on an in-stream error. Throws on a non-OK response (mapped to a 403 /
 * generic message) so callers can surface it as an error bubble.
 */
export async function readCopilotStream(
  res: Response,
  onToken: (acc: string) => void,
  onError: (msg: string) => void,
): Promise<void> {
  if (!res.ok || !res.body) {
    throw new Error(
      res.status === 403 ? 'Analyst role required.' : `Copilot error (${res.status}).`,
    );
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let acc = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = line.slice(6);
      if (payload === '[DONE]') continue;
      try {
        const data = JSON.parse(payload);
        if (data.type === 'token' && data.content) {
          acc += data.content;
          onToken(acc);
        } else if (data.type === 'error') {
          onError(data.content);
        }
      } catch {
        /* ignore partial / malformed frames */
      }
    }
  }
}
