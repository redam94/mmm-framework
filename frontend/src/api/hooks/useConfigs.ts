import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { configService } from '../services/configService';
import type { ConfigCreateRequest, ConfigUpdateRequest, PaginationParams } from '../types';

// Query keys
export const configKeys = {
  all: ['configs'] as const,
  lists: () => [...configKeys.all, 'list'] as const,
  list: (params?: PaginationParams) => [...configKeys.lists(), params] as const,
  details: () => [...configKeys.all, 'detail'] as const,
  detail: (id: string) => [...configKeys.details(), id] as const,
};

/**
 * Hook to list all configurations
 */
export function useConfigs(params?: PaginationParams) {
  return useQuery({
    queryKey: configKeys.list(params),
    queryFn: () => configService.listConfigs(params),
    staleTime: 60000, // 1 minute
  });
}

/**
 * Hook to get a single configuration
 */
export function useConfig(configId: string | undefined) {
  return useQuery({
    queryKey: configKeys.detail(configId!),
    queryFn: () => configService.getConfig(configId!),
    enabled: !!configId,
    staleTime: 60000,
  });
}

/**
 * Hook to create a new configuration
 */
export function useCreateConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ConfigCreateRequest) => configService.createConfig(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: configKeys.lists() });
    },
  });
}

/**
 * Hook to update an existing configuration
 */
export function useUpdateConfig(configId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ConfigUpdateRequest) => configService.updateConfig(configId, request),
    onSuccess: (data) => {
      // Update cache with new data
      queryClient.setQueryData(configKeys.detail(configId), data);
      // Invalidate list
      queryClient.invalidateQueries({ queryKey: configKeys.lists() });
    },
  });
}

/**
 * Hook to delete a configuration
 */
export function useDeleteConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (configId: string) => configService.deleteConfig(configId),
    onSuccess: (_data, configId) => {
      queryClient.removeQueries({ queryKey: configKeys.detail(configId) });
      queryClient.invalidateQueries({ queryKey: configKeys.lists() });
    },
  });
}

/**
 * Hook to duplicate a configuration
 */
export function useDuplicateConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ configId, newName }: { configId: string; newName?: string }) =>
      configService.duplicateConfig(configId, newName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: configKeys.lists() });
    },
  });
}

/**
 * Hook to validate a configuration
 */
export function useValidateConfig() {
  return useMutation({
    mutationFn: (request: ConfigCreateRequest) => configService.validateConfig(request),
  });
}
