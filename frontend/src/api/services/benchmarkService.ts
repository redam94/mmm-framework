import { apiClient } from '../client';

// ── Types ─────────────────────────────────────────────────────────────────────
// Mirrors mmm_framework.api.portfolio_benchmark.build_portfolio_benchmark.

/** A brand's standing for one channel relative to the whole portfolio. */
export interface VsPortfolio {
  roi_mean: number;
  /** Percentile rank (0–100) of this brand's ROI within the portfolio for the channel. */
  percentile: number | null;
}

export interface PortfolioBrand {
  project_id: string;
  name: string | null;
  n_runs: number;
  last_fit_at: number | null;
  age_days: number | null;
  n_channels: number;
  portfolio_marginal_roi: number | null;
  top_channel: { channel: string; roi_mean: number } | null;
  stale: boolean | null;
  n_calibrated: number;
  vs_portfolio: Record<string, VsPortfolio>;
}

export interface ChannelBenchmark {
  channel: string;
  n_brands: number;
  roi_median: number | null;
  roi_p25: number | null;
  roi_p75: number | null;
  roi_min: number;
  roi_max: number;
  marginal_roi_median: number | null;
}

export interface PortfolioGovernance {
  n_projects: number;
  n_with_fit: number;
  n_stale: number;
  n_fresh: number;
  n_calibrated_projects: number;
  median_model_age_days: number | null;
  stale_after_days: number;
}

export interface PortfolioBenchmark {
  projects: PortfolioBrand[];
  channels: ChannelBenchmark[];
  governance: PortfolioGovernance;
}

// ── Service ───────────────────────────────────────────────────────────────────

export const benchmarkService = {
  /** Org-scoped cross-brand benchmark + governance (latest run per project). */
  async getPortfolioBenchmark(staleAfterDays = 90): Promise<PortfolioBenchmark> {
    const { data } = await apiClient.get<PortfolioBenchmark>('/portfolio-benchmark', {
      params: { stale_after_days: staleAfterDays },
    });
    return data;
  },
};
