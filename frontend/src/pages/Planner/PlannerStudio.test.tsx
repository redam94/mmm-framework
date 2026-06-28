import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { ReactNode } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PlannerStudio } from './PlannerStudio';
import { plannerService } from '../../api/services/plannerService';

// PlannerStudio transitively imports react-plotly.js (via AllocationResult →
// FlightingCalendar); plotly.js can't load in jsdom, so stub it out.
vi.mock('react-plotly.js', () => ({ default: () => null }));

vi.mock('../../api/services/plannerService', () => ({
  plannerService: {
    startOptimize: vi.fn(),
    pollOptimize: vi.fn(),
    startScenario: vi.fn(),
    pollScenario: vi.fn(),
  },
}));

const startOptimize = plannerService.startOptimize as unknown as ReturnType<typeof vi.fn>;
const pollOptimize = plannerService.pollOptimize as unknown as ReturnType<typeof vi.fn>;

function wrap() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  );
}

describe('PlannerStudio per-channel bounds', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    startOptimize.mockResolvedValue({ job_id: 'j1', status: 'pending' });
    pollOptimize.mockResolvedValue({
      status: 'pending',
      project_id: 'p1',
      result: null,
      error: null,
    });
  });

  it('sends channel_bounds only for an edited channel', async () => {
    render(<PlannerStudio projectId="p1" channels={['TV', 'Search']} />, { wrapper: wrap() });

    fireEvent.click(screen.getByText(/spend constraints/i));
    fireEvent.change(screen.getByLabelText('TV minimum spend multiplier'), {
      target: { value: '0.8' },
    });
    fireEvent.change(screen.getByLabelText('TV maximum spend multiplier'), {
      target: { value: '1.2' },
    });
    fireEvent.click(screen.getByRole('button', { name: /build plan/i }));

    await waitFor(() => expect(startOptimize).toHaveBeenCalled());
    const body = startOptimize.mock.calls[0][1];
    expect(body.channel_bounds).toEqual({ TV: [0.8, 1.2] });
    expect(body.channel_bounds.Search).toBeUndefined();
  });

  it('omits channel_bounds (null) when no per-channel edits', async () => {
    render(<PlannerStudio projectId="p1" channels={['TV']} />, { wrapper: wrap() });
    fireEvent.click(screen.getByRole('button', { name: /build plan/i }));
    await waitFor(() => expect(startOptimize).toHaveBeenCalled());
    expect(startOptimize.mock.calls[0][1].channel_bounds).toBeNull();
  });

  it('an unset side of an edited channel inherits the default', async () => {
    render(<PlannerStudio projectId="p1" channels={['TV']} />, { wrapper: wrap() });
    fireEvent.click(screen.getByText(/spend constraints/i));
    // edit only the max for TV; min should inherit the default (0)
    fireEvent.change(screen.getByLabelText('TV maximum spend multiplier'), {
      target: { value: '1.5' },
    });
    fireEvent.click(screen.getByRole('button', { name: /build plan/i }));
    await waitFor(() => expect(startOptimize).toHaveBeenCalled());
    expect(startOptimize.mock.calls[0][1].channel_bounds).toEqual({ TV: [0, 1.5] });
  });
});
