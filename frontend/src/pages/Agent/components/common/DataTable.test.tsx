import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DataTable } from './DataTable';
import type { TableSpec } from '../../types';

// Regression: some legacy sessions persisted roi_metrics / decomposition table
// payloads concatenated 2–3× (a pre-2026-07 state-reducer bug), so reopening an
// old session rendered every channel repeated. DataTable now drops byte-identical
// duplicate rows so those baked-in payloads display correctly without a re-fit.
function spec(rows: TableSpec['rows']): TableSpec {
  return {
    title: 'ROI by Channel',
    columns: [
      { key: 'channel', label: 'Channel', type: 'string' },
      { key: 'roi_mean', label: 'Mean', type: 'number' },
    ],
    rows,
    source: 'get_roi_metrics',
    group: 'results',
  };
}

describe('DataTable duplicate-row collapse', () => {
  it('renders each byte-identical row once', () => {
    const unique = [
      { channel: 'TV', roi_mean: 0.18 },
      { channel: 'Search', roi_mean: 3.44 },
      { channel: 'Social', roi_mean: 3.08 },
    ];
    // The tripled payload the old checkpoints stored.
    render(<DataTable table={spec([...unique, ...unique, ...unique])} />);

    expect(screen.getAllByText('TV')).toHaveLength(1);
    expect(screen.getAllByText('Search')).toHaveLength(1);
    expect(screen.getAllByText('Social')).toHaveLength(1);
    expect(screen.getByText('3 rows')).toBeInTheDocument();
  });

  it('keeps genuinely distinct rows that merely share a channel value', () => {
    // Not byte-identical (different roi_mean) → both kept.
    render(
      <DataTable
        table={spec([
          { channel: 'TV', roi_mean: 0.18 },
          { channel: 'TV', roi_mean: 0.42 },
        ])}
      />,
    );
    expect(screen.getAllByText('TV')).toHaveLength(2);
    expect(screen.getByText('2 rows')).toBeInTheDocument();
  });
});
