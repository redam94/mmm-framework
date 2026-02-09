import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelService } from '../services/modelService';
import type {
  ModelFitRequest,
  ContributionRequest,
  ScenarioRequest,
  ReportRequest,
  PaginationParams,
} from '../types';

// Query keys
export const modelKeys = {
  all: ['models'] as const,
  lists: () => [...modelKeys.all, 'list'] as const,
  list: (params?: PaginationParams & { status?: string; project_id?: string }) =>
    [...modelKeys.lists(), params] as const,
  details: () => [...modelKeys.all, 'detail'] as const,
  detail: (id: string) => [...modelKeys.details(), id] as const,
  status: (id: string) => [...modelKeys.detail(id), 'status'] as const,
  results: (id: string) => [...modelKeys.detail(id), 'results'] as const,
  fit: (id: string) => [...modelKeys.detail(id), 'fit'] as const,
  posteriors: (id: string) => [...modelKeys.detail(id), 'posteriors'] as const,
  priorPosterior: (id: string) => [...modelKeys.detail(id), 'prior-posterior'] as const,
  responseCurves: (id: string) => [...modelKeys.detail(id), 'response-curves'] as const,
  decomposition: (id: string) => [...modelKeys.detail(id), 'decomposition'] as const,
  contributions: (id: string, contributionId: string) =>
    [...modelKeys.detail(id), 'contributions', contributionId] as const,
  scenarios: (id: string, scenarioId: string) =>
    [...modelKeys.detail(id), 'scenarios', scenarioId] as const,
  reports: (id: string) => [...modelKeys.detail(id), 'reports'] as const,
  report: (id: string, reportId: string) =>
    [...modelKeys.reports(id), reportId] as const,
};

// ============================================================================
// Model Fitting Hooks
// ============================================================================

/**
 * Hook to list all models
 */
export function useModels(
  params?: PaginationParams & { status?: string; project_id?: string }
) {
  return useQuery({
    queryKey: modelKeys.list(params),
    queryFn: () => modelService.listModels(params),
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to get a single model
 */
export function useModel(modelId: string | undefined) {
  return useQuery({
    queryKey: modelKeys.detail(modelId!),
    queryFn: () => modelService.getModel(modelId!),
    enabled: !!modelId,
    staleTime: 30000,
  });
}

/**
 * Hook to poll model status (for running jobs)
 */
export function useModelStatus(modelId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: modelKeys.status(modelId!),
    queryFn: () => modelService.getStatus(modelId!),
    enabled: enabled && !!modelId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Poll every 2 seconds while running/queued
      if (status === 'running' || status === 'queued' || status === 'pending') {
        return 2000;
      }
      return false;
    },
    refetchIntervalInBackground: false,
  });
}

/**
 * Hook to submit a model fit job
 */
export function useSubmitFitJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ModelFitRequest) => modelService.submitFitJob(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelKeys.lists() });
    },
  });
}

/**
 * Hook to delete a model
 */
export function useDeleteModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelId: string) => modelService.deleteModel(modelId),
    onSuccess: (_data, modelId) => {
      queryClient.removeQueries({ queryKey: modelKeys.detail(modelId) });
      queryClient.invalidateQueries({ queryKey: modelKeys.lists() });
    },
  });
}

// ============================================================================
// Results Hooks
// ============================================================================

/**
 * Hook to get model results (diagnostics + parameters)
 */
export function useModelResults(modelId: string | undefined) {
  return useQuery({
    queryKey: modelKeys.results(modelId!),
    queryFn: () => modelService.getResults(modelId!),
    enabled: !!modelId,
    staleTime: Infinity, // Results don't change
  });
}

/**
 * Hook to get model fit data (observed vs predicted)
 */
export function useModelFit(modelId: string | undefined) {
  return useQuery({
    queryKey: modelKeys.fit(modelId!),
    queryFn: () => modelService.getFitData(modelId!),
    enabled: !!modelId,
    staleTime: Infinity,
  });
}

/**
 * Hook to get posterior samples
 */
export function usePosteriors(modelId: string | undefined, nSamples?: number) {
  return useQuery({
    queryKey: modelKeys.posteriors(modelId!),
    queryFn: () => modelService.getPosteriors(modelId!, nSamples),
    enabled: !!modelId,
    staleTime: Infinity,
  });
}

/**
 * Hook to get prior vs posterior comparison
 */
export function usePriorPosterior(modelId: string | undefined, nSamples?: number) {
  return useQuery({
    queryKey: modelKeys.priorPosterior(modelId!),
    queryFn: () => modelService.getPriorPosterior(modelId!, nSamples),
    enabled: !!modelId,
    staleTime: Infinity,
  });
}

/**
 * Hook to get response curves
 */
export function useResponseCurves(modelId: string | undefined, nPoints?: number) {
  return useQuery({
    queryKey: modelKeys.responseCurves(modelId!),
    queryFn: () => modelService.getResponseCurves(modelId!, nPoints),
    enabled: !!modelId,
    staleTime: Infinity,
  });
}

/**
 * Hook to get decomposition data
 */
export function useDecomposition(modelId: string | undefined) {
  return useQuery({
    queryKey: modelKeys.decomposition(modelId!),
    queryFn: () => modelService.getDecomposition(modelId!),
    enabled: !!modelId,
    staleTime: Infinity,
  });
}

// ============================================================================
// Contributions & Scenarios Hooks
// ============================================================================

/**
 * Hook to compute contributions (async)
 */
export function useComputeContributions(modelId: string) {
  return useMutation({
    mutationFn: (request: ContributionRequest) =>
      modelService.computeContributions(modelId, request),
  });
}

/**
 * Hook to run a scenario (async)
 */
export function useRunScenario(modelId: string) {
  return useMutation({
    mutationFn: (request: ScenarioRequest) => modelService.runScenario(modelId, request),
  });
}

// ============================================================================
// Reports Hooks
// ============================================================================

/**
 * Hook to generate a report (async)
 */
export function useGenerateReport(modelId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ReportRequest) => modelService.generateReport(modelId, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelKeys.reports(modelId) });
    },
  });
}

/**
 * Hook to poll report status
 */
export function useReportStatus(
  modelId: string | undefined,
  reportId: string | undefined,
  enabled = true
) {
  return useQuery({
    queryKey: modelKeys.report(modelId!, reportId!),
    queryFn: () => modelService.getReportStatus(modelId!, reportId!),
    enabled: enabled && !!modelId && !!reportId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'pending' || status === 'running') {
        return 2000;
      }
      return false;
    },
  });
}

/**
 * Hook to list reports for a model
 */
export function useModelReports(modelId: string | undefined) {
  return useQuery({
    queryKey: modelKeys.reports(modelId!),
    queryFn: () => modelService.listReports(modelId!),
    enabled: !!modelId,
  });
}

/**
 * Hook to download a report
 */
export function useDownloadReport(modelId: string) {
  return useMutation({
    mutationFn: (reportId: string) => modelService.downloadReport(modelId, reportId),
    onSuccess: (blob, reportId) => {
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `mmm_report_${modelId}_${reportId}.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    },
  });
}
