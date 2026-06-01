import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../client';
import { sessionService } from '../services/sessionService';
import type { SessionCreateRequest, SessionUpdateRequest } from '../services/sessionService';

export const sessionKeys = {
  all: ['sessions'] as const,
  lists: () => [...sessionKeys.all, 'list'] as const,
  list: (params?: { project_id?: string }) => [...sessionKeys.lists(), params] as const,
  detail: (id: string) => [...sessionKeys.all, 'detail', id] as const,
};

export function useSessions(params?: { project_id?: string; skip?: number; limit?: number }) {
  return useQuery({
    queryKey: sessionKeys.list({ project_id: params?.project_id }),
    queryFn: () => sessionService.listSessions(params),
    staleTime: 15000,
  });
}

export function useSession(threadId: string | undefined) {
  return useQuery({
    queryKey: sessionKeys.detail(threadId!),
    queryFn: () => sessionService.getSession(threadId!),
    enabled: !!threadId,
    staleTime: 15000,
  });
}

export function useCreateSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: SessionCreateRequest) => sessionService.createSession(req),
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionKeys.lists() }),
  });
}

export function useUpdateSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ threadId, ...req }: SessionUpdateRequest & { threadId: string }) =>
      sessionService.updateSession(threadId, req),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: sessionKeys.lists() });
      qc.invalidateQueries({ queryKey: sessionKeys.detail(vars.threadId) });
    },
  });
}

export function useDeleteSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (threadId: string) => sessionService.deleteSession(threadId),
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionKeys.lists() }),
  });
}

// ── Analysis Plans ─────────────────────────────────────────────────────────────

export interface AnalysisPlanInfo {
  id: string;
  thread_id: string;
  name: string;
  locked_at: number;
  payload: Record<string, unknown>;
}

export const analysisPlansKeys = {
  all: ['analysis-plans'] as const,
  list: (threadId?: string) => [...analysisPlansKeys.all, 'list', threadId] as const,
};

export function useAnalysisPlans(threadId?: string) {
  return useQuery({
    queryKey: analysisPlansKeys.list(threadId),
    queryFn: async () => {
      const { data } = await apiClient.get<{ plans: AnalysisPlanInfo[]; total: number }>(
        '/analysis-plans',
        { params: threadId ? { thread_id: threadId, limit: 20 } : { limit: 20 } }
      );
      return data;
    },
    staleTime: 30000,
  });
}
