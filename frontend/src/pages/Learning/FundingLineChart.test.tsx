import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { LearningSnapshot } from '../../api/services/learningService';

// Plotly can't render in jsdom — stub the chart surface.
vi.mock('react-plotly.js', () => ({
  default: () => <div data-testid="plot" />,
}));

import { FundingLineChart } from './FundingLineChart';

const SNAPSHOT: LearningSnapshot = {
  schema_version: 1,
  fitted_at: 1751470000,
  evidence: { n_rows: 640, n_summaries: 0, n_waves: 1, shape_identified: { Chatter: true } },
  diagnostics: { max_rhat: 1.01, min_ess: 400, n_draws: 1000, flags: [] },
  recommendation: { Chatter: 182000, Pulse: 98000 },
  recommendation_scaled: { Chatter: 0.91, Pulse: 0.49 },
  allocation_sd: { Chatter: 21000, Pulse: 14000 },
  funding: [
    { channel: 'Chatter', mroas_mean: 1.8, prob_above_line: 0.94, funded: true, verdict: 'FUND' },
    { channel: 'Pulse', mroas_mean: 0.6, prob_above_line: 0.12, funded: false, verdict: 'CUT' },
  ],
  regret: {
    e_regret_kpi: 3.2,
    e_regret_dollars: 41600,
    enbs: 16600,
    stop: false,
    margin: 1,
    population: 13,
    wave_cost: 25000,
  },
  gamma: [],
  response_curves: {
    Chatter: {
      spend_dollars: [0, 70000, 140000, 210000, 280000],
      mean: [0, 6, 10, 12, 13],
      lo: [0, 4, 8, 9, 10],
      hi: [0, 8, 12, 15, 16],
      current: 140000,
    },
  },
  warnings: [],
};

describe('FundingLineChart', () => {
  it('renders verdict chips and both chart panels from a snapshot', () => {
    render(<FundingLineChart snapshot={SNAPSHOT} />);
    // verdict chips
    expect(screen.getByText('FUND')).toBeInTheDocument();
    expect(screen.getByText('CUT')).toBeInTheDocument();
    // chip + channel <option> both carry the name
    expect(screen.getAllByText('Chatter').length).toBeGreaterThan(0);
    // bar panel + response-curve panel both mount a Plot
    expect(screen.getAllByTestId('plot')).toHaveLength(2);
    // the response-curve panel targets the first channel that has a curve
    expect(screen.getByText(/Response curve — Chatter/)).toBeInTheDocument();
  });

  it('labels the funding line P(mROAS > 1) — the served mROAS already includes value', () => {
    render(<FundingLineChart snapshot={SNAPSHOT} />);
    expect(screen.getByText(/P\(mROAS > 1\) at the recommended allocation/)).toBeInTheDocument();
    expect(screen.queryByText(/value · mROAS/)).not.toBeInTheDocument();
    // no margin-adjusted field served → plain mROAS in the chip tooltip
    expect(screen.getByTitle('mROAS 1.80 · P(above line) 0.94')).toBeInTheDocument();
  });

  it('prefers mroas_margin_adjusted with a margin-adjusted note when served', () => {
    render(
      <FundingLineChart
        snapshot={{
          ...SNAPSHOT,
          funding: [
            {
              channel: 'Chatter',
              mroas_mean: 1.8,
              mroas_margin_adjusted: 0.54,
              prob_above_line: 0.94,
              funded: true,
              verdict: 'FUND',
            },
          ],
        }}
      />,
    );
    expect(
      screen.getByTitle('mROAS 0.54 (margin-adjusted) · P(above line) 0.94'),
    ).toBeInTheDocument();
    expect(screen.getByText(/marginal return per \$1, margin-adjusted/)).toBeInTheDocument();
  });

  it('degrades gracefully when the snapshot has no funding or curves', () => {
    render(
      <FundingLineChart snapshot={{ ...SNAPSHOT, funding: [], response_curves: {} }} />,
    );
    expect(screen.getByText(/No funding readout yet/)).toBeInTheDocument();
    expect(screen.getByText(/No response curves in this snapshot yet/)).toBeInTheDocument();
    expect(screen.queryAllByTestId('plot')).toHaveLength(0);
  });
});
