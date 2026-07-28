import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { ReactNode } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// plotly.js cannot load in jsdom.
vi.mock('react-plotly.js', () => ({ default: () => <div data-testid="forecast-chart" /> }));

vi.mock('../../api/services/plannerService', () => ({
  plannerService: {
    startForecast: vi.fn(),
    pollForecast: vi.fn(),
    startOptimize: vi.fn(),
    pollOptimize: vi.fn(),
    startScenario: vi.fn(),
    pollScenario: vi.fn(),
  },
}));

import { ForecastPanel } from './ForecastPanel';
import { plannerService } from '../../api/services/plannerService';

const startForecast = plannerService.startForecast as unknown as ReturnType<typeof vi.fn>;
const pollForecast = plannerService.pollForecast as unknown as ReturnType<typeof vi.fn>;

function wrap() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  );
}

function payload(over: Record<string, unknown> = {}) {
  return {
    periods: ['2025-01-06', '2025-01-13'],
    mean: [100, 110],
    lower: [90, 99],
    upper: [110, 121],
    baseline: [60, 62],
    by_channel: { TV: [40, 48] },
    interval: 0.9,
    caveats: ['Training residuals are autocorrelated, so this interval is TOO NARROW.'],
    caveat_fields: {
      trend_extrapolation: { policy: 'linear', trend_type: 'linear', n_train_periods: 100 },
      interval_widens_with_horizon: true,
      extrapolated_channels: [],
      residual_autocorrelation: { ljung_box_p: 0.001, autocorrelated: true },
      interval_noun: 'credible interval',
      inference_family: 'bayesian',
      approximate: false,
      fit_method: 'nuts',
      interval_available: true,
    },
    headline: {
      total: 210,
      total_lower: 189,
      total_upper: 231,
      interval_available: true,
      interval_noun: 'credible interval',
    },
    n_draws: 200,
    calendar: { start: '2025-01-06', n_periods: 2, cadence: 'weekly' },
    ...over,
  };
}

function done(p: ReturnType<typeof payload>) {
  return { status: 'done', project_id: 'p1', result: { forecast: p }, error: null };
}

describe('ForecastPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    startForecast.mockResolvedValue({ job_id: 'j1', status: 'pending' });
  });

  it('will not start without at least one channel budget', () => {
    render(<ForecastPanel projectId="p1" channels={['TV', 'Search']} />, { wrapper: wrap() });
    expect(screen.getByRole('button', { name: /forecast/i })).toBeDisabled();
  });

  it('sends only the channels given a positive budget', async () => {
    render(<ForecastPanel projectId="p1" channels={['TV', 'Search']} />, { wrapper: wrap() });
    fireEvent.change(screen.getByLabelText('TV forecast budget'), { target: { value: '5000' } });
    fireEvent.click(screen.getByRole('button', { name: /forecast/i }));
    await waitFor(() => expect(startForecast).toHaveBeenCalled());
    const body = startForecast.mock.calls[0][1];
    expect(body.channel_budgets).toEqual({ TV: 5000 });
    expect(body.n_periods).toBe(13);
  });

  // The panel's reason for existing: a mean line with a tight band reads as a
  // measurement, and this is a counterfactual under an unobserved plan.
  it('renders the caveats ABOVE the headline number', async () => {
    pollForecast.mockResolvedValue(done(payload()));
    const { container } = render(<ForecastPanel projectId="p1" channels={['TV']} />, {
      wrapper: wrap(),
    });
    fireEvent.change(screen.getByLabelText('TV forecast budget'), { target: { value: '100' } });
    fireEvent.click(screen.getByRole('button', { name: /forecast/i }));

    const caveat = await screen.findByText(/TOO NARROW/);
    const headline = screen.getByText('210');
    // Node.DOCUMENT_POSITION_FOLLOWING === 4: the headline comes after the caveat
    expect(caveat.compareDocumentPosition(headline) & 4).toBeTruthy();
    expect(container.querySelector('[data-testid="forecast-chart"]')).toBeTruthy();
  });

  it('states the interval when there is one', async () => {
    pollForecast.mockResolvedValue(done(payload()));
    render(<ForecastPanel projectId="p1" channels={['TV']} />, { wrapper: wrap() });
    fireEvent.change(screen.getByLabelText('TV forecast budget'), { target: { value: '100' } });
    fireEvent.click(screen.getByRole('button', { name: /forecast/i }));
    expect(await screen.findByText(/90% credible interval: 189 – 231/)).toBeInTheDocument();
  });

  it('says "no interval" rather than showing a collapsed band', async () => {
    // A MAP posterior has ONE draw; a zero-width band would read as extreme
    // precision, which is the opposite of what an approximate fit means.
    const p = payload({
      lower: [null, null],
      upper: [null, null],
      headline: {
        total: 210,
        total_lower: null,
        total_upper: null,
        interval_available: false,
        interval_noun: 'credible interval',
      },
    });
    p.caveat_fields.interval_available = false;
    p.caveats = ['No interval: this posterior has 1 draw(s).'];
    pollForecast.mockResolvedValue(done(p));

    render(<ForecastPanel projectId="p1" channels={['TV']} />, { wrapper: wrap() });
    fireEvent.change(screen.getByLabelText('TV forecast budget'), { target: { value: '100' } });
    fireEvent.click(screen.getByRole('button', { name: /forecast/i }));

    expect(await screen.findByText(/point estimate — no interval/)).toBeInTheDocument();
    expect(screen.queryByText(/credible interval: /)).not.toBeInTheDocument();
  });

  it('surfaces a job error instead of an empty panel', async () => {
    pollForecast.mockResolvedValue({
      status: 'error',
      project_id: 'p1',
      result: null,
      error: 'No fitted model for this project.',
    });
    render(<ForecastPanel projectId="p1" channels={['TV']} />, { wrapper: wrap() });
    fireEvent.change(screen.getByLabelText('TV forecast budget'), { target: { value: '100' } });
    fireEvent.click(screen.getByRole('button', { name: /forecast/i }));
    expect(await screen.findByText(/No fitted model/)).toBeInTheDocument();
  });
});
