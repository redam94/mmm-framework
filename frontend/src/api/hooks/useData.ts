import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { dataService } from '../services/dataService';
import type { PaginationParams } from '../types';

// Query keys
export const dataKeys = {
  all: ['data'] as const,
  lists: () => [...dataKeys.all, 'list'] as const,
  list: (params?: PaginationParams) => [...dataKeys.lists(), params] as const,
  details: () => [...dataKeys.all, 'detail'] as const,
  detail: (id: string) => [...dataKeys.details(), id] as const,
  variables: (id: string) => [...dataKeys.detail(id), 'variables'] as const,
};

/**
 * Hook to list all datasets
 */
export function useDatasets(params?: PaginationParams) {
  return useQuery({
    queryKey: dataKeys.list(params),
    queryFn: () => dataService.listDatasets(params),
    staleTime: 60000, // 1 minute
  });
}

/**
 * Hook to get a single dataset
 */
export function useDataset(
  dataId: string | undefined,
  options?: { includePreview?: boolean; previewRows?: number }
) {
  return useQuery({
    queryKey: dataKeys.detail(dataId!),
    queryFn: () => dataService.getDataset(dataId!, options),
    enabled: !!dataId,
    staleTime: 60000,
  });
}

/**
 * Hook to get variable statistics for a dataset
 */
export function useDatasetVariables(dataId: string | undefined) {
  return useQuery({
    queryKey: dataKeys.variables(dataId!),
    queryFn: () => dataService.getVariables(dataId!),
    enabled: !!dataId,
    staleTime: 60000,
  });
}

/**
 * Hook to upload a dataset
 */
export function useUploadData() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => dataService.uploadData(file),
    onSuccess: () => {
      // Invalidate datasets list
      queryClient.invalidateQueries({ queryKey: dataKeys.lists() });
    },
  });
}

/**
 * Hook to delete a dataset
 */
export function useDeleteDataset() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (dataId: string) => dataService.deleteDataset(dataId),
    onSuccess: (_data, dataId) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: dataKeys.detail(dataId) });
      // Invalidate list
      queryClient.invalidateQueries({ queryKey: dataKeys.lists() });
    },
  });
}
