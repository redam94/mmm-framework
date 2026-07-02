import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { learningService } from '../services/learningService';
import type {
  CreateProgramRequest,
  DesignWaveRequest,
  FitRequest,
  IngestWaveRequest,
} from '../services/learningService';

export const learningKeys = {
  all: ['learning'] as const,
  programs: (projectId: string | null) => [...learningKeys.all, 'programs', projectId] as const,
  program: (projectId: string | null, programId: string | null) =>
    [...learningKeys.all, 'program', projectId, programId] as const,
  job: (projectId: string | null, programId: string | null, jobId: string | null) =>
    [...learningKeys.all, 'job', projectId, programId, jobId] as const,
};

export const LEARNING_JOB_POLL_MS = 2500;

/** refetchInterval for learning fit jobs — stop polling once the job settles. */
export function learningJobPollInterval(status: string | undefined): number | false {
  return status === 'done' || status === 'error' ? false : LEARNING_JOB_POLL_MS;
}

export function useLearningPrograms(projectId: string | null) {
  return useQuery({
    queryKey: learningKeys.programs(projectId),
    queryFn: () => learningService.listPrograms(projectId!),
    enabled: !!projectId,
    staleTime: 15000,
  });
}

export function useLearningProgram(projectId: string | null, programId: string | null) {
  return useQuery({
    queryKey: learningKeys.program(projectId, programId),
    queryFn: () => learningService.getProgram(projectId!, programId!),
    enabled: !!projectId && !!programId,
    staleTime: 15000,
  });
}

export function useCreateProgram(projectId: string | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateProgramRequest) =>
      learningService.createProgram(projectId!, body),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: learningKeys.programs(projectId) }),
  });
}

export function useDeleteProgram(projectId: string | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (programId: string) => learningService.deleteProgram(projectId!, programId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: learningKeys.all }),
  });
}

/** Synchronous CCD wave design (also stores a `designed` wave row server-side). */
export function useDesignWave(projectId: string | null, programId: string | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: DesignWaveRequest) =>
      learningService.designWave(projectId!, programId!, body),
    // the endpoint persists a `designed` wave row — refresh the wave timeline
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: learningKeys.program(projectId, programId) }),
  });
}

/**
 * Poll a learning fit job until done/error. On done, the program detail and
 * list are invalidated so the new snapshot/waves refresh everywhere.
 */
export function useLearningFitJob(
  projectId: string | null,
  programId: string | null,
  jobId: string | null,
) {
  const queryClient = useQueryClient();
  const job = useQuery({
    queryKey: learningKeys.job(projectId, programId, jobId),
    queryFn: () => learningService.pollJob(projectId!, programId!, jobId!),
    enabled: !!projectId && !!programId && !!jobId,
    refetchInterval: (q) => learningJobPollInterval(q.state.data?.status),
  });

  const status = job.data?.status;
  useEffect(() => {
    if (status === 'done') {
      queryClient.invalidateQueries({ queryKey: learningKeys.program(projectId, programId) });
      queryClient.invalidateQueries({ queryKey: learningKeys.programs(projectId) });
    }
  }, [status, projectId, programId, queryClient]);

  return job;
}

/**
 * Ingest wave observations (rows / csv_text) or import past experiments
 * (experiment_ids). `start` POSTs and stores the returned job_id; `job` polls
 * until the refit reaches done/error (useExperimentSimulation pattern). Call
 * `reset` to clear the in-flight job when inputs change.
 */
export function useIngestWave(projectId: string | null, programId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: (body: IngestWaveRequest) =>
      learningService.ingestWave(projectId!, programId!, body),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useLearningFitJob(projectId, programId, jobId);

  const reset = () => {
    setJobId(null);
    start.reset();
  };

  return { start, job, reset, jobId };
}

/** Refit on the accumulated evidence, optionally overriding ENBS economics. */
export function useStartFit(projectId: string | null, programId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: (body: FitRequest = {}) =>
      learningService.startFit(projectId!, programId!, body),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const job = useLearningFitJob(projectId, programId, jobId);

  const reset = () => {
    setJobId(null);
    start.reset();
  };

  return { start, job, reset, jobId };
}
