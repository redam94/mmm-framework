import { apiClient } from '../client';
import type {
  ConfigCreateRequest,
  ConfigUpdateRequest,
  ConfigInfo,
  ConfigListResponse,
  ConfigValidationResult,
  PaginationParams,
} from '../types';

export const configService = {
  /**
   * Create a new model configuration
   */
  async createConfig(request: ConfigCreateRequest): Promise<ConfigInfo> {
    const { data } = await apiClient.post<ConfigInfo>('/configs', request);
    return data;
  },

  /**
   * List all configurations with pagination
   */
  async listConfigs(params?: PaginationParams): Promise<ConfigListResponse> {
    const { data } = await apiClient.get<ConfigListResponse>('/configs', { params });
    return data;
  },

  /**
   * Get configuration by ID
   */
  async getConfig(configId: string): Promise<ConfigInfo> {
    const { data } = await apiClient.get<ConfigInfo>(`/configs/${configId}`);
    return data;
  },

  /**
   * Update an existing configuration
   */
  async updateConfig(configId: string, request: ConfigUpdateRequest): Promise<ConfigInfo> {
    const { data } = await apiClient.put<ConfigInfo>(`/configs/${configId}`, request);
    return data;
  },

  /**
   * Delete a configuration
   */
  async deleteConfig(configId: string): Promise<void> {
    await apiClient.delete(`/configs/${configId}`);
  },

  /**
   * Duplicate a configuration
   */
  async duplicateConfig(configId: string, newName?: string): Promise<ConfigInfo> {
    const { data } = await apiClient.post<ConfigInfo>(
      `/configs/${configId}/duplicate`,
      null,
      { params: { new_name: newName } }
    );
    return data;
  },

  /**
   * Validate a configuration without saving
   */
  async validateConfig(request: ConfigCreateRequest): Promise<ConfigValidationResult> {
    const { data } = await apiClient.post<ConfigValidationResult>('/configs/validate', request);
    return data;
  },
};
