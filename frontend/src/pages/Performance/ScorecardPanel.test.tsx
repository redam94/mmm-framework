import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { Scorecard } from '../../api/services/scorecardService';

const useProjectScorecard = vi.fn();
vi.mock('../../api/hooks/useScorecard', () => ({
  useProjectScorecard: (id: string | null) => useProjectScorecard(id),
}));

import { ScorecardPanel } from './ScorecardPanel';

const DATA: Scorecard = {
  n_recommendations: 3,
  calibration: { n_with_interval: 2, hits: 1, coverage: 0.5 },
  rows: [
    {
      channel: 'Search',
      experiment_id: 'e2',
      estimand: 'roas',
      run_id: 'run_A',
      realized: 9.0,
      realized_se: 0.3,
      end_date: '2026-04-01',
      predicted: 4.0,
      predicted_lower: 3.0,
      predicted_upper: 5.0,
      error: 5.0,
      error_pct: 1.25,
      in_interval: false,
    },
    {
      channel: 'TV',
      experiment_id: 'e1',
      estimand: 'roas',
      run_id: 'run_A',
      realized: 2.2,
      realized_se: 0.2,
      end_date: '2026-03-01',
      predicted: 2.0,
      predicted_lower: 1.4,
      predicted_upper: 2.6,
      error: 0.2,
      error_pct: 0.1,
      in_interval: true,
    },
    {
      channel: 'Radio',
      experiment_id: 'e3',
      estimand: 'roas',
      run_id: null,
      realized: 1.5,
      realized_se: 0.1,
      end_date: '2026-02-01',
      predicted: null,
      predicted_lower: null,
      predicted_upper: null,
      error: null,
      error_pct: null,
      in_interval: null,
    },
  ],
};

beforeEach(() => useProjectScorecard.mockReset());

describe('ScorecardPanel', () => {
  it('renders the calibration summary and predicted-vs-realized rows', () => {
    useProjectScorecard.mockReturnValue({ data: DATA, isLoading: false, isError: false });
    render(<ScorecardPanel projectId="p" />);
    expect(screen.getByText('Recommendation scorecard')).toBeInTheDocument();
    expect(screen.getByText('Interval calibration')).toBeInTheDocument();
    // per-row hit/miss chips
    expect(screen.getByText('In interval')).toBeInTheDocument(); // TV
    expect(screen.getByText('Missed')).toBeInTheDocument(); // Search
    expect(screen.getByText('No prediction')).toBeInTheDocument(); // Radio
  });

  it('shows the empty state with no realized outcomes', () => {
    useProjectScorecard.mockReturnValue({
      data: { n_recommendations: 0, calibration: { n_with_interval: 0, hits: 0, coverage: null }, rows: [] },
      isLoading: false,
      isError: false,
    });
    render(<ScorecardPanel projectId="p" />);
    expect(screen.getByText('No realized outcomes yet')).toBeInTheDocument();
  });

  it('handles loading and error states', () => {
    useProjectScorecard.mockReturnValue({ data: undefined, isLoading: true, isError: false });
    const { rerender } = render(<ScorecardPanel projectId="p" />);
    expect(screen.getByText('Loading scorecard…')).toBeInTheDocument();
    useProjectScorecard.mockReturnValue({ data: undefined, isLoading: false, isError: true });
    rerender(<ScorecardPanel projectId="p" />);
    expect(screen.getByText('Failed to load the scorecard.')).toBeInTheDocument();
  });
});
