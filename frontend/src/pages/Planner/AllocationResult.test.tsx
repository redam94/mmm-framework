import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AllocationResult } from './AllocationResult';
import type { BudgetPlanResult } from '../../api/services/plannerService';

// Plotly can't render in jsdom — stub the calendar chart.
vi.mock('./FlightingCalendar', () => ({
  FlightingCalendar: () => <div data-testid="flighting-chart" />,
}));

const PLAN: BudgetPlanResult = {
  by_geo: true,
  total_budget: 1200,
  current_total: 1200,
  expected_uplift: 80,
  uplift_hdi: [20, 150],
  prob_positive_uplift: 0.92,
  n_draws: 120,
  allocation: [
    { channel: 'TV', current_spend: 600, optimal_spend: 700, change_pct: 16.7 },
    { channel: 'Search', current_spend: 600, optimal_spend: 500, change_pct: -16.7 },
  ],
  geo_allocation: [
    { geo: 'North', channel: 'TV', optimal_spend: 400, change_pct: 10 },
    { geo: 'South', channel: 'Search', optimal_spend: 300, change_pct: -5 },
  ],
  geos: ['North', 'South'],
  flighting: {
    pattern: 'even',
    n_periods: 2,
    total_budget: 1200,
    periods: ['P1', 'P2'],
    channels: ['TV', 'Search'],
    schedule: [
      { period: 'P1', TV: 350, Search: 250, total: 600 },
      { period: 'P2', TV: 350, Search: 250, total: 600 },
    ],
    by_channel: { TV: [350, 350], Search: [250, 250] },
  },
  notes: [],
};

describe('AllocationResult', () => {
  it('renders headline metrics, allocation, geo and flighting', () => {
    render(<AllocationResult plan={PLAN} />);
    expect(screen.getByText('Recommended allocation')).toBeInTheDocument();
    expect(screen.getAllByText('TV').length).toBeGreaterThan(0);
    expect(screen.getByText(/Allocation by geography/)).toBeInTheDocument();
    expect(screen.getByText('North')).toBeInTheDocument();
    expect(screen.getByText(/Flighting calendar/)).toBeInTheDocument();
    expect(screen.getByTestId('flighting-chart')).toBeInTheDocument();
    expect(screen.getByText('92%')).toBeInTheDocument(); // prob positive
  });

  it('omits geo + flighting for a national plan', () => {
    const national: BudgetPlanResult = {
      ...PLAN,
      by_geo: false,
      geo_allocation: undefined,
      geos: undefined,
      flighting: undefined,
    };
    render(<AllocationResult plan={national} />);
    expect(screen.getByText('Recommended allocation')).toBeInTheDocument();
    expect(screen.queryByText(/Allocation by geography/)).toBeNull();
    expect(screen.queryByText(/Flighting calendar/)).toBeNull();
  });
});
