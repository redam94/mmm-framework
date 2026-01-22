import { apiClient } from '../client';
import type { HealthResponse, HealthDetailedResponse } from '../types';

export const healthService = {
  /**
   * Basic health check
   */
  async checkHealth(): Promise<HealthResponse> {
    const { data } = await apiClient.get<HealthResponse>('/health');
    return data;
  },

  /**
   * Detailed health check with stats
   */
  async checkHealthDetailed(): Promise<HealthDetailedResponse> {
    const { data } = await apiClient.get<HealthDetailedResponse>('/health/detailed');
    return data;
  },
};
