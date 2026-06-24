import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { measurementService } from '../services/measurementService';
import type {
  DesignRequest,
  ExperimentTransition,
  OptimizeRequest,
  SimulateRequest,
} from '../services/measurementService';
import { portfolioKeys } from './usePortfolio';

export const measurementKeys = {
  all: ['measurement'] as const,
  experiments: (projectId: string | null, status?: string) =>
    [...measurementKeys.all, 'experiments', projectId, status ?? 'any'] as const,
  experiment: (id: string) => [...measurementKeys.all, 'experiment', id] as const,
  priorities: (projectId: string | null) =>
    [...measurementKeys.all, 'priorities', projectId] as const,
  history: (projectId: string | null) => [...measurementKeys.all, 'history', projectId] as const,
  coverage: (projectId: string | null) => [...measurementKeys.all, 'coverage', projectId] as const,
  simulation: (projectId: string | null, jobId: string | null) =>
    [...measurementKeys.all, 'simulation', projectId, jobId] as const,
  optimization: (projectId: string | null, jobId: string | null) =>
    [...measurementKeys.all, 'optimization', projectId, jobId] as const,
};

export function useExperimentRegistry(projectId: string | null, status?: string) {
  return useQuery({
    queryKey: measurementKeys.experiments(projectId, status),
    queryFn: () => measurementService.listExperiments(projectId, status),
    staleTime: 15000,
  });
}

export function useExperimentRecord(id: string | null) {
  return useQuery({
    queryKey: measurementKeys.experiment(id ?? 'none'),
    queryFn: () => measurementService.getExperiment(id!),
    enabled: !!id,
    staleTime: 15000,
  });
}

export function useTransitionExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: ExperimentTransition }) =>
      measurementService.transitionExperiment(id, body),
    // a lifecycle move ripples into priorities (decay), coverage, and the
    // portfolio's next actions — invalidate the fan-out
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: measurementKeys.all });
      queryClient.invalidateQueries({ queryKey: portfolioKeys.all });
    },
  });
}

/** Latest EIG/EVOI grid; resolves to null when the project has no metrics. */
export function useExperimentPriorities(projectId: string | null) {
  return useQuery({
    queryKey: measurementKeys.priorities(projectId),
    queryFn: async () => {
      try {
        return await measurementService.getPriorities(projectId!);
      } catch (e: unknown) {
        const status = (e as { response?: { status?: number } } | null | undefined)?.response
          ?.status;
        if (status === 404) return null;
        throw e;
      }
    },
    enabled: !!projectId,
    staleTime: 30000,
  });
}

export function useProjectHistory(projectId: string | null) {
  return useQuery({
    queryKey: measurementKeys.history(projectId),
    queryFn: () => measurementService.getHistory(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}

export function useDesignOptions(projectId: string | null, channel: string | null) {
  return useQuery({
    queryKey: [...measurementKeys.all, 'design-options', projectId, channel],
    queryFn: () => measurementService.getDesignOptions(projectId!, channel!),
    enabled: !!projectId && !!channel,
    staleTime: 60000,
  });
}

export function useComputeDesign(projectId: string | null) {
  return useMutation({
    mutationFn: (body: DesignRequest) => measurementService.computeDesign(projectId!, body),
  });
}

/**
 * Model-anchored economics + A/A·A/B simulation. `start` POSTs the design and
 * stores the returned job_id; `job` polls until the job reaches done/error.
 * Call `reset` to clear the in-flight job (e.g. when studio inputs change).
 */
export function useExperimentSimulation(projectId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: (body: SimulateRequest) => measurementService.startSimulation(projectId!, body),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useQuery({
    queryKey: measurementKeys.simulation(projectId, jobId),
    queryFn: () => measurementService.pollSimulation(projectId!, jobId!),
    enabled: !!projectId && !!jobId,
    refetchInterval: (q) =>
      ['done', 'error'].includes(q.state.data?.status ?? '') ? false : 2500,
  });

  const reset = () => {
    setJobId(null);
    start.reset();
  };

  return { start, job, reset, jobId };
}

/**
 * Experiment-setup optimizer. `start` POSTs the channel and stores the returned
 * job_id; `job` polls until the Pareto-front job reaches done/error. Call
 * `reset` to clear the in-flight job (e.g. when studio inputs change).
 */
export function useExperimentOptimization(projectId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: (body: OptimizeRequest) =>
      measurementService.startOptimization(projectId!, body),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useQuery({
    queryKey: measurementKeys.optimization(projectId, jobId),
    queryFn: () => measurementService.pollOptimization(projectId!, jobId!),
    enabled: !!projectId && !!jobId,
    refetchInterval: (q) =>
      ['done', 'error'].includes(q.state.data?.status ?? '') ? false : 2500,
  });

  const reset = () => {
    setJobId(null);
    start.reset();
  };

  return { start, job, reset, jobId };
}

export function useCalibrationCoverage(projectId: string | null) {
  return useQuery({
    queryKey: measurementKeys.coverage(projectId),
    queryFn: () => measurementService.getCoverage(projectId!),
    enabled: !!projectId,
    staleTime: 30000,
  });
}
