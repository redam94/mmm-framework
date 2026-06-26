import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

const mutate = vi.fn();
const useValidation = vi.fn();
vi.mock('../../../../api/hooks/useValidation', () => ({
  useValidation: () => useValidation(),
}));
// PlotCard/TableCard fetch by id; stub them so the tab test stays presentational.
vi.mock('../plots/PlotCard', () => ({
  PlotCard: ({ plot }: { plot: { title?: string } }) => <div>plot:{plot.title}</div>,
}));
vi.mock('../tables/TableCard', () => ({
  TableCard: ({ tableRef }: { tableRef: { id: string } }) => <div>table:{tableRef.id}</div>,
}));

import { ValidationTab } from './ValidationTab';

function mockHook(over: Record<string, unknown> = {}) {
  useValidation.mockReturnValue({
    start: { mutate, isPending: false, reset: vi.fn() },
    job: { data: undefined },
    check: null,
    reset: vi.fn(),
    jobId: null,
    ...over,
  });
}

beforeEach(() => {
  mutate.mockReset();
  useValidation.mockReset();
});

describe('ValidationTab', () => {
  it('renders the check buttons', () => {
    mockHook();
    render(<ValidationTab projectId="p" />);
    expect(screen.getByRole('button', { name: /Validate model/ })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Posterior predictive/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Cross-validation/ })).toBeInTheDocument();
  });

  it('starts a check when a button is clicked', () => {
    mockHook();
    render(<ValidationTab projectId="p" />);
    fireEvent.click(screen.getByRole('button', { name: /Validate model/ }));
    expect(mutate).toHaveBeenCalledWith('validate');
  });

  it('shows a running indicator', () => {
    mockHook({
      start: { mutate, isPending: false, reset: vi.fn() },
      check: 'ppc',
      job: { data: { status: 'running', check: 'ppc', result: null, error: null } },
    });
    render(<ValidationTab projectId="p" />);
    expect(screen.getByRole('status')).toHaveTextContent(/Running/);
  });

  it('renders content + plots + tables on done', () => {
    mockHook({
      check: 'validate',
      job: {
        data: {
          status: 'done',
          check: 'validate',
          error: null,
          result: {
            content: '### Model validation battery\n\nAll good.',
            plots: [{ id: 'pid1', title: 'PPC density overlay' }],
            tables: [{ id: 'tid1', source: 'validate_model' }],
          },
        },
      },
    });
    render(<ValidationTab projectId="p" />);
    expect(screen.getByText('Model validation battery')).toBeInTheDocument();
    expect(screen.getByText('plot:PPC density overlay')).toBeInTheDocument();
    expect(screen.getByText('table:tid1')).toBeInTheDocument();
  });

  it('shows the error state', () => {
    mockHook({
      check: 'validate',
      job: {
        data: { status: 'error', check: 'validate', result: null, error: 'No saved model run.' },
      },
    });
    render(<ValidationTab projectId="p" />);
    expect(screen.getByText('No saved model run.')).toBeInTheDocument();
  });
});
