import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RunComparison } from './RunComparison';
import type {
  DeltaCell,
  RunComparison as RunComparisonData,
} from '../../api/services/runsService';

const cell = (a: number | null, b: number | null): DeltaCell => ({
  a,
  b,
  delta: a != null && b != null ? b - a : null,
});

const data: RunComparisonData = {
  run_a: { run_id: 'A', created_at: 1, project_id: 'p' },
  run_b: { run_id: 'B', created_at: 2, project_id: 'p' },
  channels: [
    {
      channel: 'TV',
      in_a: true,
      in_b: true,
      roi_mean: cell(2.1, 1.6),
      roi_hdi_low: cell(1.8, 1.3),
      roi_hdi_high: cell(2.4, 1.9),
      marginal_roi: cell(1.5, 1.2),
      spend: cell(100, 110),
      spend_share: cell(0.5, 0.5),
    },
    {
      channel: 'Radio',
      in_a: false,
      in_b: true,
      roi_mean: cell(null, 1.1),
      roi_hdi_low: cell(null, 0.9),
      roi_hdi_high: cell(null, 1.3),
      marginal_roi: cell(null, 0.8),
      spend: cell(null, 50),
      spend_share: cell(null, 0.25),
    },
  ],
  portfolio: { total_spend: cell(100, 160) },
};

describe('RunComparison', () => {
  it('renders per-channel ROI A and B', () => {
    render(<RunComparison data={data} />);
    expect(screen.getByText('TV')).toBeInTheDocument();
    expect(screen.getByText('2.10')).toBeInTheDocument(); // ROI A
    expect(screen.getByText('1.60')).toBeInTheDocument(); // ROI B
  });

  it('shows the absolute ROI delta', () => {
    render(<RunComparison data={data} />);
    expect(screen.getByText(/0\.50/)).toBeInTheDocument(); // |1.6 - 2.1|
  });

  it('tags a channel added in B', () => {
    render(<RunComparison data={data} />);
    expect(screen.getByText('Radio')).toBeInTheDocument();
    expect(screen.getByText(/new/)).toBeInTheDocument();
  });

  it('renders an empty state when there are no channels', () => {
    render(<RunComparison data={{ ...data, channels: [] }} />);
    expect(screen.getByText(/No channel metrics/)).toBeInTheDocument();
  });
});
