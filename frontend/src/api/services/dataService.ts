import { apiClient } from '../client';
import type {
  DataUploadResponse,
  DatasetInfo,
  DatasetListResponse,
  VariableSummary,
  PaginationParams,
} from '../types';

export const dataService = {
  /**
   * Upload a data file (CSV, Parquet, Excel)
   */
  async uploadData(file: File): Promise<DataUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const { data } = await apiClient.post<DataUploadResponse>('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 120000, // 2 minutes for large files
    });
    return data;
  },

  /**
   * List all datasets with pagination
   */
  async listDatasets(params?: PaginationParams): Promise<DatasetListResponse> {
    const { data } = await apiClient.get<DatasetListResponse>('/data', { params });
    return data;
  },

  /**
   * Get dataset info by ID
   */
  async getDataset(
    dataId: string,
    options?: { includePreview?: boolean; previewRows?: number }
  ): Promise<DatasetInfo> {
    const { data } = await apiClient.get<DatasetInfo>(`/data/${dataId}`, {
      params: {
        include_preview: options?.includePreview,
        preview_rows: options?.previewRows,
      },
    });
    return data;
  },

  /**
   * Get variable summary statistics for a dataset
   */
  async getVariables(dataId: string): Promise<VariableSummary[]> {
    const { data } = await apiClient.get<{ variables: VariableSummary[] }>(
      `/data/${dataId}/variables`
    );
    return data.variables;
  },

  /**
   * Delete a dataset
   */
  async deleteDataset(dataId: string): Promise<void> {
    await apiClient.delete(`/data/${dataId}`);
  },
};
