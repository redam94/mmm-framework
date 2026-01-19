import { useQuery } from '@tanstack/react-query';
import { healthService } from '../services/healthService';

// Query keys
export const healthKeys = {
  all: ['health'] as const,
  basic: () => [...healthKeys.all, 'basic'] as const,
  detailed: () => [...healthKeys.all, 'detailed'] as const,
};

/**
 * Hook for basic health check
 */
export function useHealth() {
  return useQuery({
    queryKey: healthKeys.basic(),
    queryFn: () => healthService.checkHealth(),
    staleTime: 30000, // 30 seconds
    refetchInterval: 60000, // Refresh every minute
  });
}

/**
 * Hook for detailed health check with stats
 */
export function useHealthDetailed() {
  return useQuery({
    queryKey: healthKeys.detailed(),
    queryFn: () => healthService.checkHealthDetailed(),
    staleTime: 30000,
  });
}
