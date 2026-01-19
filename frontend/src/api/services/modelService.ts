import { apiClient } from '../client';
import type {
  ModelFitRequest,
  ModelInfo,
  ModelStatusResponse,
  ModelListResponse,
  ModelResults,
  ModelFitData,
  PosteriorSamples,
  PriorPosteriorComparison,
  ResponseCurvesResponse,
  DecompositionResponse,
  ContributionRequest,
  ContributionResult,
  ScenarioRequest,
  ScenarioResult,
  ReportRequest,
  ReportInfo,
  PaginationParams,
} from '../types';

export const modelService = {
  // =========================================================================
  // Model Fitting
  // =========================================================================

  /**
   * Submit a model fitting job (async)
   */
  async submitFitJob(request: ModelFitRequest): Promise<ModelInfo> {
    const { data } = await apiClient.post<ModelInfo>('/models/fit', request);
    return data;
  },

  /**
   * List all models with pagination and filtering
   */
  async listModels(
    params?: PaginationParams & { status?: string; project_id?: string }
  ): Promise<ModelListResponse> {
    const { data } = await apiClient.get<ModelListResponse>('/models', { params });
    return data;
  },

  /**
   * Get model info by ID
   */
  async getModel(modelId: string): Promise<ModelInfo> {
    const { data } = await apiClient.get<ModelInfo>(`/models/${modelId}`);
    return data;
  },

  /**
   * Get real-time model status (for polling)
   */
  async getStatus(modelId: string): Promise<ModelStatusResponse> {
    const { data } = await apiClient.get<ModelStatusResponse>(`/models/${modelId}/status`);
    return data;
  },

  /**
   * Delete a model
   */
  async deleteModel(modelId: string): Promise<void> {
    await apiClient.delete(`/models/${modelId}`);
  },

  // =========================================================================
  // Results & Analysis
  // =========================================================================

  /**
   * Get model results summary (diagnostics + parameters)
   */
  async getResults(modelId: string): Promise<ModelResults> {
    const { data } = await apiClient.get<ModelResults>(`/models/${modelId}/results`);
    return data;
  },

  /**
   * Get model fit data (observed vs predicted)
   */
  async getFitData(modelId: string): Promise<ModelFitData> {
    const { data } = await apiClient.get<ModelFitData>(`/models/${modelId}/fit`);
    return data;
  },

  /**
   * Get posterior samples for all parameters
   */
  async getPosteriors(modelId: string, nSamples?: number): Promise<PosteriorSamples> {
    const { data } = await apiClient.get<PosteriorSamples>(`/models/${modelId}/posteriors`, {
      params: { n_samples: nSamples },
    });
    return data;
  },

  /**
   * Get prior vs posterior comparison
   */
  async getPriorPosterior(modelId: string, nSamples?: number): Promise<PriorPosteriorComparison> {
    const { data } = await apiClient.get<PriorPosteriorComparison>(
      `/models/${modelId}/prior-posterior`,
      { params: { n_samples: nSamples } }
    );
    return data;
  },

  /**
   * Get response curves for all channels
   */
  async getResponseCurves(modelId: string, nPoints?: number): Promise<ResponseCurvesResponse> {
    const { data } = await apiClient.get<ResponseCurvesResponse>(
      `/models/${modelId}/response-curves`,
      { params: { n_points: nPoints } }
    );
    return data;
  },

  /**
   * Get decomposition data (all components)
   */
  async getDecomposition(modelId: string): Promise<DecompositionResponse> {
    const { data } = await apiClient.get<DecompositionResponse>(`/models/${modelId}/decomposition`);
    return data;
  },

  // =========================================================================
  // Contributions & Scenarios
  // =========================================================================

  /**
   * Compute channel contributions (async)
   */
  async computeContributions(
    modelId: string,
    request: ContributionRequest
  ): Promise<ContributionResult> {
    const { data } = await apiClient.post<ContributionResult>(
      `/models/${modelId}/contributions`,
      request
    );
    return data;
  },

  /**
   * Get contribution results by ID
   */
  async getContributions(
    modelId: string,
    contributionId: string
  ): Promise<ContributionResult> {
    const { data } = await apiClient.get<ContributionResult>(
      `/models/${modelId}/contributions/${contributionId}`
    );
    return data;
  },

  /**
   * Run a scenario analysis (async)
   */
  async runScenario(modelId: string, request: ScenarioRequest): Promise<ScenarioResult> {
    const { data } = await apiClient.post<ScenarioResult>(`/models/${modelId}/scenario`, request);
    return data;
  },

  /**
   * Get scenario results by ID
   */
  async getScenario(modelId: string, scenarioId: string): Promise<ScenarioResult> {
    const { data } = await apiClient.get<ScenarioResult>(
      `/models/${modelId}/scenario/${scenarioId}`
    );
    return data;
  },

  /**
   * Get predictions
   */
  async getPredictions(
    modelId: string,
    params?: { media_spend?: Record<string, number>; n_periods?: number }
  ): Promise<{ periods: string[]; predictions: number[] }> {
    const { data } = await apiClient.get(`/models/${modelId}/prediction`, { params });
    return data;
  },

  // =========================================================================
  // Reports
  // =========================================================================

  /**
   * Generate an HTML report (async)
   */
  async generateReport(modelId: string, request: ReportRequest): Promise<ReportInfo> {
    const { data } = await apiClient.post<ReportInfo>(`/models/${modelId}/report`, request);
    return data;
  },

  /**
   * Get report generation status
   */
  async getReportStatus(modelId: string, reportId: string): Promise<ReportInfo> {
    const { data } = await apiClient.get<ReportInfo>(
      `/models/${modelId}/report/${reportId}/status`
    );
    return data;
  },

  /**
   * Download completed report
   */
  async downloadReport(modelId: string, reportId: string): Promise<Blob> {
    const { data } = await apiClient.get(`/models/${modelId}/report/${reportId}/download`, {
      responseType: 'blob',
    });
    return data;
  },

  /**
   * List all reports for a model
   */
  async listReports(modelId: string): Promise<ReportInfo[]> {
    const { data } = await apiClient.get<{ reports: ReportInfo[] }>(`/models/${modelId}/reports`);
    return data.reports;
  },
};
