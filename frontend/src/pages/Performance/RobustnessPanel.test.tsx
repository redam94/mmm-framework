import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { SpecCurveResult } from '../../api/services/specCurveService';

const useStartSpecCurve = vi.fn();
const useSpecCurveJob = vi.fn();
vi.mock('../../api/hooks/useSpecCurve', () => ({
  useStartSpecCurve: () => useStartSpecCurve(),
  useSpecCurveJob: (pid: string | null, jid: string | null) => useSpecCurveJob(pid, jid),
}));

import { RobustnessPanel } from './RobustnessPanel';

const RESULT: SpecCurveResult = {
  channels: ['TV', 'Search'],
  specs: ['geometric×hill', 'weibull×hill'],
  primary: 'geometric×hill',
  hdi_prob: 0.94,
  weights: { 'geometric×hill': 0.6, 'weibull×hill': 0.4 },
  bma: {
    TV: { mean: 2.1, lower: 1.5, upper: 2.7 },
    Search: { mean: 0.8, lower: 0.4, upper: 1.2 },
  },
  robustness: {
    TV: { min: 1.9, max: 2.3, range: 0.4, spread_pct: 18, sign_stable: true, profitable_weight: 1, primary: 2.1, n_specs: 2 },
    Search: { min: 0.6, max: 1.4, range: 0.8, spread_pct: 80, sign_stable: false, profitable_weight: 0.4, primary: 0.8, n_specs: 2 },
  },
  per_spec: {},
};

beforeEach(() => {
  useStartSpecCurve.mockReset();
  useSpecCurveJob.mockReset();
  useStartSpecCurve.mockReturnValue({ mutateAsync: vi.fn(), isPending: false });
});

describe('RobustnessPanel', () => {
  it('shows the empty state before a sweep is run', () => {
    useSpecCurveJob.mockReturnValue({ data: undefined });
    render(<RobustnessPanel projectId="p" />);
    expect(screen.getByText('Run a robustness check')).toBeInTheDocument();
  });

  it('shows a running message while the sweep is in flight', () => {
    useSpecCurveJob.mockReturnValue({ data: { status: 'running', n_specs: 4, result: null, error: null } });
    render(<RobustnessPanel projectId="p" />);
    expect(screen.getByText(/Fitting 4 specifications/)).toBeInTheDocument();
  });

  it('renders the result table with robust and spec-fragile verdicts', () => {
    useSpecCurveJob.mockReturnValue({ data: { status: 'done', result: RESULT, error: null } });
    render(<RobustnessPanel projectId="p" />);
    expect(screen.getByText('TV')).toBeInTheDocument();
    expect(screen.getByText('Robust')).toBeInTheDocument();
    // Search flips sign across specs → spec-fragile chip + callout
    expect(screen.getAllByText('Spec-fragile').length).toBeGreaterThan(0);
    expect(screen.getByText(/1 spec-fragile channel/)).toBeInTheDocument();
  });

  it('surfaces a job error', () => {
    useSpecCurveJob.mockReturnValue({
      data: { status: 'error', result: null, error: 'No fitted model run for this project.' },
    });
    render(<RobustnessPanel projectId="p" />);
    expect(screen.getByText(/No fitted model run/)).toBeInTheDocument();
  });
});
