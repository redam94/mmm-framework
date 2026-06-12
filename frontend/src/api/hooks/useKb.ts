import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { kbService } from '../services/kbService';

export const kbKeys = {
  all: ['kb'] as const,
  documents: (projectId: string) => [...kbKeys.all, 'documents', projectId] as const,
};

export function useKbDocuments(projectId: string | null) {
  return useQuery({
    queryKey: kbKeys.documents(projectId ?? ''),
    queryFn: () => kbService.listDocuments(projectId!),
    enabled: !!projectId,
    // Keep polling while any document is still chunking/embedding.
    refetchInterval: (query) =>
      query.state.data?.documents.some((d) => d.status === 'pending') ? 3000 : false,
  });
}

export function useUploadKbDocument(projectId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, template }: { file: File; template?: boolean }) =>
      kbService.uploadDocument(projectId!, file, template ?? false),
    onSuccess: () => qc.invalidateQueries({ queryKey: kbKeys.documents(projectId ?? '') }),
  });
}

export function useDeleteKbDocument(projectId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => kbService.deleteDocument(documentId),
    onSuccess: () => qc.invalidateQueries({ queryKey: kbKeys.documents(projectId ?? '') }),
  });
}

export function useKbSearch(projectId: string | null) {
  return useMutation({
    mutationFn: ({ query, k }: { query: string; k?: number }) =>
      kbService.search(projectId!, query, k ?? 6),
  });
}
