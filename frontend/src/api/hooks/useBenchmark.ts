import { useQuery } from '@tanstack/react-query';
import { benchmarkService } from '../services/benchmarkService';

export const benchmarkKeys = {
  all: ['benchmark'] as const,
  portfolio: (staleAfterDays: number) => [...benchmarkKeys.all, 'portfolio', staleAfterDays] as const,
};

/** Cross-brand portfolio benchmark + governance signals (org-scoped). */
export function usePortfolioBenchmark(staleAfterDays = 90) {
  return useQuery({
    queryKey: benchmarkKeys.portfolio(staleAfterDays),
    queryFn: () => benchmarkService.getPortfolioBenchmark(staleAfterDays),
    staleTime: 60000,
  });
}
