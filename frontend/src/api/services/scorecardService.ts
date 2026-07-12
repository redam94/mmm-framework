import { apiClient } from '../client';

/**
 * Recommendation scorecard (issue #109): each channel's realized experiment
 * readout joined to the ROI the model predicted for it, plus interval
 * calibration. Mirrors api/scorecard.py + GET /projects/{id}/scorecard.
 */

export interface ScorecardRow {
  channel: string;
  experiment_id: string | null;
  estimand: string | null;
  run_id: string | null;
  realized: number;
  realized_se: number | null;
  end_date: string | null;
  predicted: number | null;
  predicted_lower: number | null;
  predicted_upper: number | null;
  error: number | null;
  error_pct: number | null;
  /** true = realized landed inside the predicted interval; null = no interval. */
  in_interval: boolean | null;
}

export interface Scorecard {
  rows: ScorecardRow[];
  calibration: {
    n_with_interval: number;
    hits: number;
    coverage: number | null;
  };
  n_recommendations: number;
}

export const scorecardService = {
  async getScorecard(projectId: string): Promise<Scorecard> {
    const { data } = await apiClient.get<Scorecard>(`/projects/${projectId}/scorecard`);
    return data;
  },
};
