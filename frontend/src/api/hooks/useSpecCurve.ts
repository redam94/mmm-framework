import { useMutation, useQuery } from '@tanstack/react-query';
import { specCurveService, type StartSpecCurveBody } from '../services/specCurveService';

export const specCurveKeys = {
  all: ['spec-curve'] as const,
  job: (projectId: string | null, jobId: string | null) =>
    [...specCurveKeys.all, projectId, jobId] as const,
};

/** Kick off a (non-blocking, multi-fit) spec-curve sweep; returns the job id. */
export function useStartSpecCurve(projectId: string | null) {
  return useMutation({
    mutationFn: (body: StartSpecCurveBody = {}) => specCurveService.start(projectId!, body),
  });
}

/** Poll a spec-curve job until it is done/errored (the sweep takes minutes). */
export function useSpecCurveJob(projectId: string | null, jobId: string | null) {
  return useQuery({
    queryKey: specCurveKeys.job(projectId, jobId),
    queryFn: () => specCurveService.poll(projectId!, jobId!),
    enabled: !!projectId && !!jobId,
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === 'pending' || s === 'running' ? 4000 : false;
    },
  });
}
