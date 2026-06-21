import { apiClient } from '../client';

/** One captured cell run (mirrors api/main.py _notebook_cell_sync output). */
export interface NotebookOutput {
  stdout: string;
  plots: { id: string; title: string }[];
  tables: { id: string; title: string; source?: string; group?: string }[];
  is_error: boolean;
  setup_error?: boolean;
}

export interface NotebookCell {
  id: string;
  type: 'code' | 'markdown';
  source: string;
  outputs?: NotebookOutput | null;
}

export interface NotebookDataset {
  path: string;
  filename: string;
  preview?: string | null;
  kind?: string;
}

export interface NotebookDoc {
  cells: NotebookCell[];
  dataset?: NotebookDataset | null;
  name?: string;
  version?: number | null;
  seeded?: boolean;
}

export interface CellJob {
  status: 'pending' | 'running' | 'done' | 'error';
  result: NotebookOutput | null;
  error: string | null;
}

/** Cheap, stable string hash (djb2-xor) — identifies the live editor source so
 * the kernel only re-imports the model when the author actually edits it. */
export function hashSource(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h) ^ s.charCodeAt(i);
  return (h >>> 0).toString(36);
}

function nbParams(name: string, version?: number | null) {
  return version != null ? { name, version } : { name };
}

export const atelierNotebookService = {
  /** GET /model-garden/notebook — persisted doc or a seeded starter. */
  async getNotebook(name: string, version?: number | null): Promise<NotebookDoc> {
    const { data } = await apiClient.get<NotebookDoc>('/model-garden/notebook', {
      params: nbParams(name, version),
    });
    return data;
  },

  /** PUT /model-garden/notebook — upsert the doc (singleton per notebook). */
  async saveNotebook(req: {
    name: string;
    version?: number | null;
    cells: NotebookCell[];
    dataset?: NotebookDataset | null;
  }): Promise<{ saved: boolean; id: string }> {
    const { data } = await apiClient.put('/model-garden/notebook', req);
    return data;
  },

  /** POST /model-garden/notebook/dataset — stage a CSV into the notebook kernel. */
  async uploadDataset(
    name: string,
    file: File,
    version?: number | null,
  ): Promise<NotebookDataset & { size_bytes: number }> {
    const fd = new FormData();
    fd.append('file', file);
    const { data } = await apiClient.post('/model-garden/notebook/dataset', fd, {
      params: nbParams(name, version),
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  /** POST /model-garden/notebook/cell — start a cell run (202 → job_id). */
  async runCell(req: {
    name: string;
    version?: number | null;
    source_code: string | null;
    source_rev: string;
    code: string;
    dataset_path?: string | null;
  }): Promise<{ job_id: string; status: string }> {
    const { data } = await apiClient.post('/model-garden/notebook/cell', req);
    return data;
  },

  /** GET /model-garden/notebook/cell/{job_id} — poll a cell run. */
  async pollCell(jobId: string): Promise<CellJob> {
    const { data } = await apiClient.get<CellJob>(
      `/model-garden/notebook/cell/${jobId}`,
    );
    return data;
  },
};

/**
 * Run a cell to completion: POST, then poll until done/error. Cells share one
 * warm kernel, so callers run them SEQUENTIALLY (Run-all awaits each in order).
 * A code cell that fits a model is slow; the poll has a generous budget.
 */
export async function runCellToCompletion(
  req: Parameters<typeof atelierNotebookService.runCell>[0],
  onStatus?: (s: CellJob['status']) => void,
): Promise<NotebookOutput> {
  const { job_id } = await atelierNotebookService.runCell(req);
  // ~14 min budget (700ms × 1200) — a NUTS fit can be slow; MAP cells finish fast.
  for (let i = 0; i < 1200; i++) {
    const job = await atelierNotebookService.pollCell(job_id);
    onStatus?.(job.status);
    if (job.status === 'done') {
      return (
        job.result ?? { stdout: '', plots: [], tables: [], is_error: false }
      );
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'Cell execution failed.');
    }
    await new Promise((r) => setTimeout(r, 700));
  }
  throw new Error('Cell timed out.');
}
