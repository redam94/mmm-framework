import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { validationService } from '../services/validationService';
import type { ValidationCheck } from '../services/validationService';

/**
 * Run a model-validation check as a non-blocking background job. `start(check)`
 * POSTs and stores the returned job_id; `job` polls until done/error. Mirrors
 * useExperimentSimulation.
 */
export function useValidation(projectId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [check, setCheck] = useState<ValidationCheck | null>(null);

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

  const reset = () => {
    setJobId(null);
    setCheck(null);
    start.reset();
  };

  return { start, job, reset, jobId, check };
}
