import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { ExperimentRecord } from '../../api/services/measurementService';

// Plotly can't render in jsdom — stub the chart surface.
vi.mock('react-plotly.js', () => ({
  default: () => <div data-testid="plot" />,
}));

const mockNavigate = vi.fn();

vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
}));

const EXP: Partial<ExperimentRecord> = {
  id: 'exp_123',
  channel: 'TV',
  subchannel: 'Brand',
  status: 'completed',
  value: 1.4,
  se: 0.3,
  estimand: 'roas',
  start_date: '2026-01-05',
  end_date: '2026-03-01',
  created_at: 1767225600,
  updated_at: 1772496000,
  recommending_run_id: null,
  calibrated_run_id: null,
  design: null,
  readout: null,
  priority: null,
  preregistered_at: null,
  status_history: [],
};

vi.mock('../../api/hooks/useMeasurement', () => ({
  useExperimentRecord: () => ({ data: EXP, isLoading: false, isError: false }),
  useTransitionExperiment: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

// import AFTER the mocks so the drawer picks up the stubs
const { ExperimentDrawer } = await import('./ExperimentDrawer');

describe('ExperimentDrawer — completed → calibrated handoff', () => {
  beforeEach(() => mockNavigate.mockClear());

  it('offers a Calibrate in Workspace action for completed experiments', () => {
    render(<ExperimentDrawer experimentId="exp_123" onClose={() => {}} />);
    expect(screen.getByText('Calibrate in Workspace')).toBeInTheDocument();
  });

  it('hands off to the workspace with a staged calibration prompt', () => {
    render(<ExperimentDrawer experimentId="exp_123" onClose={() => {}} />);
    fireEvent.click(screen.getByText('Calibrate in Workspace'));
    expect(mockNavigate).toHaveBeenCalledTimes(1);
    const [path, opts] = mockNavigate.mock.calls[0];
    expect(path).toBe('/workspace');
    const prefill = (opts as { state: { prefill: string } }).state.prefill;
    // the prompt names the experiment, its channel/arm, and both tools
    expect(prefill).toContain('exp_123');
    expect(prefill).toContain('TV / Brand');
    expect(prefill).toContain('apply_experiment_calibration');
    expect(prefill).toContain('fit_mmm_model');
  });
});
