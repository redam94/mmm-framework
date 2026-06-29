import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TransformPanel } from './TransformPanel';
import type { DataStudioState } from '../../../types';

function makeState(over: Partial<DataStudioState> = {}): DataStudioState {
  return {
    staging_id: 's1', filename: 'x.csv', columns: ['sales', 'tv', 'notes'],
    all_columns: ['sales', 'tv', 'notes'], dtypes: {}, roles: { sales: 'kpi', tv: 'media' },
    date_col: 'week', is_long: false, n_rows: 10, n_cols: 3, preview_rows: [],
    steps: [], warnings: [], committed: false, ...over,
  };
}

describe('TransformPanel', () => {
  it('adds a drop_columns step via the builder', () => {
    const onSteps = vi.fn();
    render(<TransformPanel state={makeState()} busy={false} onSteps={onSteps} onRoles={vi.fn()} />);
    // default op is "Drop columns" — pick a column chip then Add step
    fireEvent.click(screen.getByRole('button', { name: 'notes' }));
    fireEvent.click(screen.getByRole('button', { name: /Add step/ }));
    expect(onSteps).toHaveBeenCalledWith([{ op: 'drop_columns', columns: ['notes'] }]);
  });

  it('removes and reorders existing steps', () => {
    const steps = [
      { op: 'drop_columns', columns: ['notes'] } as const,
      { op: 'fill_missing', strategy: 'zero' } as const,
    ];
    const onSteps = vi.fn();
    render(<TransformPanel state={makeState({ steps: [...steps] })} busy={false} onSteps={onSteps} onRoles={vi.fn()} />);

    fireEvent.click(screen.getAllByRole('button', { name: /Move down/ })[0]);  // first step's down
    expect(onSteps).toHaveBeenCalledWith([steps[1], steps[0]]);

    onSteps.mockClear();
    fireEvent.click(screen.getAllByRole('button', { name: /Remove step/ })[0]);
    expect(onSteps).toHaveBeenCalledWith([steps[1]]);
  });

  it('changes a column role', () => {
    const onRoles = vi.fn();
    render(<TransformPanel state={makeState()} busy={false} onSteps={vi.fn()} onRoles={onRoles} />);
    // the 'notes' column select → set to control
    const select = screen.getAllByRole('combobox').find(s =>
      (s.previousSibling as HTMLElement | null)?.textContent === 'notes' ||
      s.closest('div')?.textContent?.includes('notes'),
    )!;
    fireEvent.change(select, { target: { value: 'control' } });
    expect(onRoles).toHaveBeenCalledWith(expect.objectContaining({ notes: 'control' }));
  });
});
