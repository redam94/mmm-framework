import { useMutation, useQuery } from '@tanstack/react-query';
import { accountService } from '../services/accountService';
import { setStoredTokens } from '../client';
import { useAuthStore } from '../../stores/authStore';

export const accountKeys = {
  all: ['account'] as const,
  me: () => [...accountKeys.all, 'me'] as const,
  usage: () => [...accountKeys.all, 'usage'] as const,
  catalog: () => [...accountKeys.all, 'catalog'] as const,
};

/** Current principal (identity + org role). Quiet on 401 (not logged in via JWT). */
export function useMe() {
  return useQuery({
    queryKey: accountKeys.me(),
    queryFn: () => accountService.getMe(),
    staleTime: 60000,
    retry: false,
  });
}

export function useUsage() {
  return useQuery({
    queryKey: accountKeys.usage(),
    queryFn: () => accountService.getUsage(),
    staleTime: 30000,
    retry: false,
  });
}

export function useIntegrationsCatalog() {
  return useQuery({
    queryKey: accountKeys.catalog(),
    queryFn: () => accountService.getIntegrationsCatalog(),
    staleTime: 5 * 60000,
  });
}

/** Change password; the server re-issues tokens, which we persist to stay in. */
export function useChangePassword() {
  return useMutation({
    mutationFn: ({ current, next }: { current: string; next: string }) =>
      accountService.changePassword(current, next),
    onSuccess: (tokens) => {
      setStoredTokens(tokens.access_token, tokens.refresh_token);
      useAuthStore.setState({
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
        isAuthenticated: true,
      });
    },
  });
}
