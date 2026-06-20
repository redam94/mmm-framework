import { useState } from 'react';
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
    onSuccess: (_row, { name }) => {
      qc.invalidateQueries({ queryKey: gardenKeys.lists() });
      qc.invalidateQueries({ queryKey: gardenKeys.versions(name) });
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
 * draft→tested, so callers should refresh the versions list when job.data.status
 * becomes 'done'.
 */
export function useGardenTest(name: string | null, version: number | null) {
  const qc = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: () => modelGardenService.startTest(name!, version!),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useQuery({
    queryKey: gardenKeys.test(name!, version!, jobId!),
    queryFn: () => modelGardenService.pollTest(name!, version!, jobId!),
    enabled: !!name && version != null && !!jobId,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      if (s === 'done' || s === 'error') {
        // Refresh listings once: a pass may have flipped draft→tested.
        if (name) {
          qc.invalidateQueries({ queryKey: gardenKeys.versions(name) });
          qc.invalidateQueries({ queryKey: gardenKeys.lists() });
        }
        return false;
      }
      return 2500;
    },
  });

  const reset = () => {
    setJobId(null);
    start.reset();
  };

  return { start, job, reset, jobId };
}
