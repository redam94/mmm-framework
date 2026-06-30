import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { plannerService } from '../services/plannerService';
import type {
  PlannerOptimizeRequest,
  PlannerScenarioRequest,
} from '../services/plannerService';

export const plannerKeys = {
  all: ['planner'] as const,
  optimize: (projectId: string | null, jobId: string | null) =>
    [...plannerKeys.all, 'optimize', projectId, jobId] as const,
  scenario: (projectId: string | null, jobId: string | null) =>
    [...plannerKeys.all, 'scenario', projectId, jobId] as const,
};

/**
 * Budget-plan optimization. `start` POSTs the request and stores the returned
 * job_id; `job` polls until done/error. `reset` clears the in-flight job (call
 * it when inputs change). Mirrors useExperimentSimulation.
 */
export function usePlannerOptimization(projectId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: (body: PlannerOptimizeRequest) =>
      plannerService.startOptimize(projectId!, body),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useQuery({
    queryKey: plannerKeys.optimize(projectId, jobId),
    queryFn: () => plannerService.pollOptimize(projectId!, jobId!),
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

/** What-if scenario (uncertainty included). Same start/poll/reset shape. */
export function usePlannerScenario(projectId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: (body: PlannerScenarioRequest) =>
      plannerService.startScenario(projectId!, body),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useQuery({
    queryKey: plannerKeys.scenario(projectId, jobId),
    queryFn: () => plannerService.pollScenario(projectId!, jobId!),
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
