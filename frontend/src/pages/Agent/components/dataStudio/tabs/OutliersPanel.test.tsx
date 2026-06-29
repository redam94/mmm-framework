import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Studio charts render via PlotCard (id-less inline); stub it presentational.
vi.mock('../../plots/PlotCard', () => ({
  PlotCard: ({ plot }: { plot: { title?: string } }) => <div>plot:{plot.title}</div>,
}));

import { OutliersPanel } from './OutliersPanel';
import type { StudioEdaResult } from '../../../types';

const RESULT: StudioEdaResult = {
  analyses: { outliers: { figures: [], tables: [], stats: {} } },
  issues: [],
  outlier_suggestions: [
    {
      action_id: 'winsorize:sales@2023-07-04', strategy: 'winsorize', variable: 'sales',
      rationale: 'isolated spike', step: { op: 'winsorize', column: 'sales', periods: ['2023-07-04'], cap_value: 200 },
      spec_change: null,
    },
  ],
  normalization_damaged: ['sales'],
  warnings: [],
};

describe('OutliersPanel', () => {
  it('renders a suggestion and accepts it as a transform step', async () => {
    const runEda = vi.fn().mockResolvedValue(RESULT);
    const onAccept = vi.fn().mockResolvedValue(true);
    render(<OutliersPanel runEda={runEda} rev={0} onAccept={onAccept} />);

    // suggestion row + damaged banner appear after the async EDA resolves
    await waitFor(() => expect(screen.getByText('winsorize')).toBeInTheDocument());
    expect(screen.getByText(/single point sets the saturation scale/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Confirm/ }));
    await waitFor(() =>
      expect(onAccept).toHaveBeenCalledWith({ op: 'winsorize', column: 'sales', periods: ['2023-07-04'], cap_value: 200 }),
    );
  });
});
