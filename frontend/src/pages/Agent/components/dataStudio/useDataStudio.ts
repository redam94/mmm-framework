import { useCallback, useRef, useState } from 'react';
import { API_BASE, authHeaders } from '../../constants';
import type {
  DataStudioState, DatasetInfo, EdaFindings, ModelSpec, StudioEdaResult, StudioRole, TransformStep,
} from '../../types';

export interface CommitPayload {
  summary?: string;
  dataset_path?: string;
  dataset?: DatasetInfo;
  model_spec?: ModelSpec;
  eda?: EdaFindings;
  pending_spec_changes?: unknown[];
}

// ─── useDataStudio ────────────────────────────────────────────────────────────
// Raw fetch + authHeaders (matches the Agent page; NOT the orphaned axios
// dataService). The transform pipeline is a full-replace PUT; rapid edits are
// guarded by a monotonic request id so an out-of-order response can't clobber a
// newer one. `rev` bumps on every successful pipeline change so EDA panels know
// to refetch against the new frame.

export function useDataStudio(threadId: string | null, apiKey: string | null, modelName: string | null) {
  const [state, setState] = useState<DataStudioState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rev, setRev] = useState(0);
  const txnId = useRef(0);
  const edaCache = useRef<Map<string, StudioEdaResult>>(new Map());

  const base = threadId ? `${API_BASE}/data-studio/${encodeURIComponent(threadId)}` : null;
  const headers = useCallback(() => authHeaders(apiKey, modelName), [apiKey, modelName]);

  const mergeState = useCallback((patch: Partial<DataStudioState>) => {
    setState(prev => (prev ? { ...prev, ...patch } : (patch as DataStudioState)));
  }, []);

  const hydrate = useCallback(async (): Promise<DataStudioState | null> => {
    if (!base) return null;
    setLoading(true); setError(null);
    try {
      const r = await fetch(base, { headers: headers() });
      const data = await r.json();
      const s = data?.staging;
      if (!s || !s.staging_id) { setState(null); return null; }
      const next: DataStudioState = {
        staging_id: s.staging_id,
        filename: s.raw?.name ?? 'dataset',
        columns: s.columns ?? [],
        all_columns: s.all_columns ?? s.columns ?? [],
        dtypes: s.dtypes ?? {},
        roles: s.roles ?? {},
        date_col: s.date_col ?? null,
        is_long: !!s.is_long,
        n_rows: s.n_rows ?? 0,
        n_cols: s.n_cols ?? 0,
        preview_rows: s.preview_rows ?? [],
        steps: s.steps ?? [],
        diff: s.diff,
        warnings: s.warnings ?? [],
        committed: !!s.committed,
      };
      setState(next);
      edaCache.current.clear();
      return next;
    } catch {
      setError('Could not load the data studio.');
      return null;
    } finally { setLoading(false); }
  }, [base, headers]);

  const upload = useCallback(async (file: File): Promise<DataStudioState | null> => {
    if (!base) return null;
    setLoading(true); setError(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch(`${base}/upload`, { method: 'POST', headers: headers(), body: fd });
      const data = await r.json();
      if (!r.ok) { setError(data?.error || `Upload failed (${r.status})`); return null; }
      const next: DataStudioState = {
        staging_id: data.staging_id,
        filename: data.raw?.name ?? file.name,
        columns: data.columns ?? [],
        all_columns: data.all_columns ?? data.columns ?? [],
        dtypes: data.dtypes ?? {},
        roles: data.inferred_roles ?? data.roles ?? {},
        date_col: data.date_col ?? null,
        is_long: !!data.is_long,
        n_rows: data.n_rows ?? 0,
        n_cols: data.n_cols ?? 0,
        preview_rows: data.preview_rows ?? [],
        steps: [],
        diff: data.diff,
        warnings: data.warnings ?? [],
        committed: false,
      };
      setState(next);
      edaCache.current.clear();
      setRev(v => v + 1);
      return next;
    } catch {
      setError('Upload failed — is the API running?');
      return null;
    } finally { setLoading(false); }
  }, [base, headers]);

  const setSteps = useCallback(async (
    steps: TransformStep[], roles?: Record<string, StudioRole>,
  ): Promise<boolean> => {
    if (!base) return false;
    const myTxn = ++txnId.current;
    setError(null);
    // optimistic step list so the editor stays responsive while the preview loads
    mergeState({ steps, ...(roles ? { roles } : {}) });
    try {
      const r = await fetch(`${base}/pipeline`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', ...headers() },
        body: JSON.stringify({ steps, ...(roles ? { roles } : {}) }),
      });
      const data = await r.json();
      if (myTxn !== txnId.current) return false; // a newer edit superseded this one
      if (!r.ok) { setError(data?.error || `Pipeline failed (${r.status})`); return false; }
      mergeState({
        columns: data.columns ?? [],
        all_columns: data.all_columns ?? data.columns ?? [],
        dtypes: data.dtypes ?? {},
        roles: data.roles ?? roles ?? {},
        date_col: data.date_col ?? null,
        is_long: !!data.is_long,
        n_rows: data.n_rows ?? 0,
        n_cols: data.n_cols ?? 0,
        preview_rows: data.preview_rows ?? [],
        diff: data.diff,
        warnings: data.warnings ?? [],
        steps,
      });
      edaCache.current.clear();
      setRev(v => v + 1);
      return true;
    } catch {
      if (myTxn === txnId.current) setError('Pipeline update failed.');
      return false;
    }
  }, [base, headers, mergeState]);

  const runEda = useCallback(async (
    analyses: string[], opts?: { sensitivity?: string },
  ): Promise<StudioEdaResult | null> => {
    if (!base) return null;
    const key = `${rev}:${analyses.slice().sort().join(',')}:${opts?.sensitivity ?? ''}`;
    const cached = edaCache.current.get(key);
    if (cached) return cached;
    try {
      const r = await fetch(`${base}/eda`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers() },
        body: JSON.stringify({ analyses, sensitivity: opts?.sensitivity ?? 'default' }),
      });
      const data = await r.json();
      if (!r.ok) { setError(data?.error || `EDA failed (${r.status})`); return null; }
      edaCache.current.set(key, data);
      return data as StudioEdaResult;
    } catch {
      setError('EDA failed.');
      return null;
    }
  }, [base, headers, rev]);

  const commit = useCallback(async (): Promise<CommitPayload | null> => {
    if (!base) return null;
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${base}/commit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers() },
        body: JSON.stringify({}),
      });
      const data = await r.json();
      if (!r.ok) { setError(data?.error || `Commit failed (${r.status})`); return null; }
      return data as CommitPayload;
    } catch {
      setError('Commit failed.');
      return null;
    } finally { setLoading(false); }
  }, [base, headers]);

  const discard = useCallback(async (): Promise<void> => {
    if (!base) return;
    try {
      await fetch(`${base}/discard`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', ...headers() },
        body: JSON.stringify({}),
      });
    } catch { /* ignore */ }
    setState(null);
    edaCache.current.clear();
  }, [base, headers]);

  return { state, loading, error, rev, setError, hydrate, upload, setSteps, runEda, commit, discard };
}
