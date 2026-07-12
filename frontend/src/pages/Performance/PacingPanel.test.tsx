import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { Pacing } from '../../api/services/pacingService';

// Mock the data + mutation hooks so the panel test is purely presentational.
const useProjectPacing = vi.fn();
const mutateAsync = vi.fn();
vi.mock('../../api/hooks/usePacing', () => ({
  useProjectPacing: (id: string | null) => useProjectPacing(id),
  useUploadDelivery: () => ({ mutateAsync, isPending: false }),
}));

import { PacingPanel } from './PacingPanel';

const AVAILABLE: Pacing = {
  available: true,
  plan_basis: 'flighting',
  plan_name: 'Q3 Plan',
  threshold: 0.1,
  planned_total: 300,
  actual_total: 356,
  divergence_pct: 0.187,
  flagged: ['TV'],
  channels: [
    {
      channel: 'TV',
      planned: 200,
      actual: 250,
      divergence_pct: 0.25,
      status: 'over-pacing',
      planned_series: [100, 100],
      actual_series: [130, 120],
    },
    {
      channel: 'Search',
      planned: 100,
      actual: 106,
      divergence_pct: 0.06,
      status: 'on-track',
      planned_series: [50, 50],
      actual_series: [48, 58],
    },
  ],
  alert: {
    off_pace: true,
    n_flagged: 1,
    flagged: ['TV'],
    threshold: 0.1,
    worst: { channel: 'TV', divergence_pct: 0.25, abs_divergence: 0.25, status: 'over-pacing' },
    portfolio_divergence_pct: 0.187,
  },
};

beforeEach(() => {
  useProjectPacing.mockReset();
  mutateAsync.mockReset();
});

describe('PacingPanel', () => {
  it('renders the headline, per-channel table, and off-pace alert', () => {
    useProjectPacing.mockReturnValue({ data: AVAILABLE, isLoading: false, isError: false });
    render(<PacingPanel projectId="p" />);
    // headline stats
    expect(screen.getByText('Portfolio pacing vs plan')).toBeInTheDocument();
    expect(screen.getByText('Off-pace channels')).toBeInTheDocument();
    // per-channel rows + status chips ("TV" also appears in the alert text)
    expect(screen.getAllByText('TV').length).toBeGreaterThan(0);
    expect(screen.getByText('Over-pacing')).toBeInTheDocument();
    expect(screen.getByText('On track')).toBeInTheDocument();
    // the off-pace alert names the drifting channel
    expect(screen.getByText(/Off-pace: 1 channel/)).toBeInTheDocument();
    // the saved plan is named
    expect(screen.getByText('Q3 Plan')).toBeInTheDocument();
  });

  it('shows an on-track message when nothing is flagged', () => {
    const onTrack: Pacing = {
      ...AVAILABLE,
      flagged: [],
      channels: AVAILABLE.channels.map((c) => ({ ...c, status: 'on-track' as const })),
      alert: { ...AVAILABLE.alert!, off_pace: false, n_flagged: 0, flagged: [], worst: null },
    };
    useProjectPacing.mockReturnValue({ data: onTrack, isLoading: false, isError: false });
    render(<PacingPanel projectId="p" />);
    expect(screen.getByText(/All channels are pacing within/)).toBeInTheDocument();
  });

  it('prompts to save a plan when there is none', () => {
    useProjectPacing.mockReturnValue({
      data: { available: false, reason: 'no_plan', threshold: 0.1, flagged: [], channels: [] },
      isLoading: false,
      isError: false,
    });
    render(<PacingPanel projectId="p" />);
    expect(screen.getByText('No saved plan to pace against')).toBeInTheDocument();
  });

  it('prompts to upload delivery when a plan exists but no delivery', () => {
    useProjectPacing.mockReturnValue({
      data: {
        available: false,
        reason: 'no_delivery',
        plan_basis: 'flighting',
        threshold: 0.1,
        flagged: [],
        channels: [],
      },
      isLoading: false,
      isError: false,
    });
    render(<PacingPanel projectId="p" />);
    expect(screen.getByText('Upload actual delivery to see pacing')).toBeInTheDocument();
  });

  it('handles the loading and error states', () => {
    useProjectPacing.mockReturnValue({ data: undefined, isLoading: true, isError: false });
    const { rerender } = render(<PacingPanel projectId="p" />);
    expect(screen.getByText('Loading pacing…')).toBeInTheDocument();

    useProjectPacing.mockReturnValue({ data: undefined, isLoading: false, isError: true });
    rerender(<PacingPanel projectId="p" />);
    expect(screen.getByText('Failed to load pacing.')).toBeInTheDocument();
  });
});
