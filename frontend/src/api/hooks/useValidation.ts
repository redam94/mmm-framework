import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { validationService } from '../services/validationService';
import type { ValidationCheck } from '../services/validationService';

/**
 * Run a model-validation check as a non-blocking background job. `start(check)`
 * POSTs and stores the returned job_id; `job` polls until done/error. Mirrors
 * useExperimentSimulation.
 *
 * Every run is persisted server-side (UI jobs and chat-run checks alike), so
 * `history` lists past validations across reloads and `load(item)` re-opens a
 * past run's full result.
 */
export function useValidation(projectId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [check, setCheck] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const start = useMutation({
    mutationFn: (c: ValidationCheck) => {
      setCheck(c);
      return validationService.start(projectId!, c);
    },
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useQuery({
    queryKey: ['validation', projectId, jobId],
    queryFn: () => validationService.poll(projectId!, jobId!),
    enabled: !!projectId && !!jobId,
    refetchInterval: (q) =>
      ['done', 'error'].includes(q.state.data?.status ?? '') ? false : 1500,
  });

  const history = useQuery({
    queryKey: ['validations', projectId],
    queryFn: () => validationService.history(projectId!),
    enabled: !!projectId,
  });

  // A job settling adds a row to the persistent history — refresh the list.
  const settled = ['done', 'error'].includes(job.data?.status ?? '');
  useEffect(() => {
    if (settled) queryClient.invalidateQueries({ queryKey: ['validations', projectId] });
  }, [settled, projectId, queryClient]);

  /** Re-open a past run (its result renders through the same `job` query). */
  const load = (item: { job_id: string; check: string }) => {
    setCheck(item.check);
    setJobId(item.job_id);
  };

  const reset = () => {
    setJobId(null);
    setCheck(null);
    start.reset();
  };

  return { start, job, history, load, reset, jobId, check };
}
