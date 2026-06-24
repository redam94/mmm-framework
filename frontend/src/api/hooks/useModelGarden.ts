import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { modelGardenService } from '../services/modelGardenService';
import type {
  GardenModel,
  GardenRegisterRequest,
} from '../services/modelGardenService';

export const gardenKeys = {
  all: ['modelGarden'] as const,
  lists: () => [...gardenKeys.all, 'list'] as const,
  list: (status?: string) => [...gardenKeys.lists(), status ?? 'any'] as const,
  versions: (name: string) => [...gardenKeys.all, 'versions', name] as const,
  version: (name: string, version: number) =>
    [...gardenKeys.all, 'version', name, version] as const,
  source: (name: string, version: number) =>
    [...gardenKeys.all, 'source', name, version] as const,
  test: (name: string, version: number, jobId: string) =>
    [...gardenKeys.version(name, version), 'test', jobId] as const,
};

/** Org's garden models — latest version per name (or filtered by status). */
export function useGardenModels(status?: string) {
  return useQuery({
    queryKey: gardenKeys.list(status),
    queryFn: () => modelGardenService.list(status ? { status } : undefined),
    staleTime: 30000,
  });
}

export function useGardenVersions(name: string | null) {
  return useQuery({
    queryKey: gardenKeys.versions(name!),
    queryFn: () => modelGardenService.listVersions(name!),
    enabled: !!name,
    staleTime: 30000,
  });
}

export function useGardenModel(name: string | null, version: number | null) {
  return useQuery({
    queryKey: gardenKeys.version(name!, version!),
    queryFn: () => modelGardenService.get(name!, version!),
    enabled: !!name && version != null,
    staleTime: 30000,
  });
}

export function useGardenSource(name: string | null, version: number | null) {
  return useQuery({
    queryKey: gardenKeys.source(name!, version!),
    queryFn: () => modelGardenService.getSource(name!, version!),
    enabled: !!name && version != null,
    staleTime: 60000,
  });
}

export function useRegisterGardenModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: GardenRegisterRequest) => modelGardenService.register(req),
    onSuccess: (row: GardenModel) => {
      qc.invalidateQueries({ queryKey: gardenKeys.lists() });
      qc.invalidateQueries({ queryKey: gardenKeys.versions(row.name) });
    },
  });
}

export function usePromoteGardenModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      version,
      note,
    }: {
      name: string;
      version: number;
      note?: string;
    }) => modelGardenService.promote(name, version, note),
    onSuccess: (_row, { name, version }) => {
      qc.invalidateQueries({ queryKey: gardenKeys.lists() });
      qc.invalidateQueries({ queryKey: gardenKeys.versions(name) });
      qc.invalidateQueries({ queryKey: gardenKeys.version(name, version) });
    },
  });
}

/** Edit a non-published version's docs in place (no new version). */
export function useUpdateGardenDocs() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      version,
      docs,
    }: {
      name: string;
      version: number;
      docs: string;
    }) => modelGardenService.updateDocs(name, version, docs),
    onSuccess: (_row, { name, version }) => {
      qc.invalidateQueries({ queryKey: gardenKeys.version(name, version) });
      qc.invalidateQueries({ queryKey: gardenKeys.versions(name) });
      qc.invalidateQueries({ queryKey: gardenKeys.lists() });
    },
  });
}

export function useDeleteGardenModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, version }: { name: string; version: number }) =>
      modelGardenService.remove(name, version),
    onSuccess: (_res, { name }) => {
      qc.invalidateQueries({ queryKey: gardenKeys.lists() });
      qc.invalidateQueries({ queryKey: gardenKeys.versions(name) });
    },
  });
}

/**
 * Non-blocking compatibility test: `start` POSTs and stores the job_id; `job`
 * polls every 2.5s until done/error. On a passing run the backend auto-promotes
 * draft→tested, so on a terminal status we refresh the listings + open detail
 * (action bar / status chip change) — done once, in the effect below.
 */
export function useGardenTest(name: string | null, version: number | null) {
  const qc = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  // Guards the one-shot post-test refresh to exactly one fire per terminal
  // transition (StrictMode double-invoke / re-render safe).
  const settledRef = useRef<string | null>(null);

  const start = useMutation({
    mutationFn: () => modelGardenService.startTest(name!, version!),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useQuery({
    queryKey: gardenKeys.test(name!, version!, jobId!),
    queryFn: () => modelGardenService.pollTest(name!, version!, jobId!),
    enabled: !!name && version != null && !!jobId,
    // PURE resolver — return ONLY the next interval. A side-effect here (e.g.
    // invalidateQueries) re-fires on every observer re-render, and because the
    // invalidated list/version queries re-render this component they retrigger
    // the resolver: an unbounded invalidate→refetch→re-render loop that exhausts
    // the browser connection pool (net::ERR_INSUFFICIENT_RESOURCES). The one-shot
    // refresh lives in the effect below instead.
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === 'done' || s === 'error' ? false : 2500;
    },
    // A poll must not retry-storm on a transient failure (a slow in-process fit
    // can starve the event loop and time the poll out) — the next interval tick
    // is the natural retry, so one in-flight poll at a time is enough.
    retry: false,
  });

  // Refresh listings AND the open detail once when the job terminates: a pass
  // may have flipped draft→tested, which changes the action bar (Publish
  // appears, Delete hides) and the status chip.
  const status = job.data?.status;
  useEffect(() => {
    if (!name || !jobId || (status !== 'done' && status !== 'error')) return;
    const settleKey = `${jobId}:${status}`;
    if (settledRef.current === settleKey) return;
    settledRef.current = settleKey;
    qc.invalidateQueries({ queryKey: gardenKeys.versions(name) });
    qc.invalidateQueries({ queryKey: gardenKeys.lists() });
    if (version != null) {
      qc.invalidateQueries({ queryKey: gardenKeys.version(name, version) });
    }
  }, [status, name, version, jobId, qc]);

  const reset = () => {
    setJobId(null);
    start.reset();
    settledRef.current = null;
  };

  return { start, job, reset, jobId };
}
