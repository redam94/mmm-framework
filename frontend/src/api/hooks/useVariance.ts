import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { varianceService } from '../services/varianceService';
import type { SuppliedLine } from '../services/varianceService';

export const varianceKeys = {
  all: ['variance'] as const,
  job: (projectId: string | null, jobId: string | null) =>
    [...varianceKeys.all, 'job', projectId, jobId] as const,
};

/**
 * Variance to plan (#227). `start` POSTs (a 409 carries the stated refusal —
 * no committed plan, missing delivery/actuals, changed dataset); `job` polls
 * until done/error. Same start/poll/reset shape as usePlannerOptimization.
 */
export function useVarianceBridge(projectId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: (supplied: SuppliedLine[] = []) =>
      varianceService.startVariance(projectId!, supplied),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useQuery({
    queryKey: varianceKeys.job(projectId, jobId),
    queryFn: () => varianceService.pollVariance(projectId!, jobId!),
    enabled: !!projectId && !!jobId,
    refetchInterval: (q) =>
      ['done', 'error'].includes(q.state.data?.status ?? '') ? false : 2000,
  });

  const reset = () => {
    setJobId(null);
    start.reset();
  };

  return { start, job, reset, jobId };
}
