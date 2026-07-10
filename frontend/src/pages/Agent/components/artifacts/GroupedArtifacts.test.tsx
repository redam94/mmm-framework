import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

// PlotCard/TableCard fetch by id; stub them so the timeline test stays presentational.
vi.mock('../plots/PlotCard', () => ({
  PlotCard: ({ plot }: { plot: { title?: string } }) => <div>plot:{plot.title}</div>,
}));
vi.mock('../tables/TableCard', () => ({
  TableCard: ({ tableRef }: { tableRef: { id: string } }) => <div>table:{tableRef.id}</div>,
}));

import { GroupedArtifacts } from './GroupedArtifacts';
import { buildArtifactGroups } from '../../utils/artifactGroups';
import type { ChatMessage, PythonOutput } from '../../types';

const messages: ChatMessage[] = [
  { id: 'h1', type: 'human', content: 'How does spend correlate with sales?' },
  { id: 'a1', type: 'ai', content: '', toolCalls: [{ id: 'call-1', name: 'execute_python', status: 'done' }] },
  { id: 'h2', type: 'human', content: 'Fit the model and show ROI.' },
  { id: 'a2', type: 'ai', content: '', toolCalls: [{ id: 'call-2', name: 'execute_python', status: 'done' }] },
];

// plotCount 0 so the code card's own "n plots" badge doesn't collide with
// group count-badge assertions below.
const py: PythonOutput = {
  id: 'call-2', code: 'import pandas as pd', output: 'ok', hasError: false, plotCount: 0,
};

function makeGroups() {
  return buildArtifactGroups(
    messages,
    [
      { id: 'p1', title: 'Spend vs sales scatter', call_id: 'call-1' },
      { id: 'p2', title: 'ROI forest', call_id: 'call-2' },
      { id: 'p-legacy', title: 'Old chart' }, // no call_id → Earlier work
    ],
    [{ id: 't1', title: 'ROI table', source: 'get_roi_metrics', group: 'results', call_id: 'call-2' }],
    [py],
  );
}

afterEach(() => vi.restoreAllMocks());

describe('GroupedArtifacts', () => {
  it('renders group titles newest question first with Earlier work last', () => {
    render(<GroupedArtifacts groups={makeGroups()} threadId="tid-1" />);
    const headings = [
      'Fit the model and show ROI.',
      'How does spend correlate with sales?',
      'Earlier work',
    ];
    const rendered = headings.map(h => screen.getByText(h));
    // DOM order matches newest-first with legacy appended last.
    for (let i = 0; i < rendered.length - 1; i++) {
      expect(
        rendered[i].compareDocumentPosition(rendered[i + 1]) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }
  });

  it('renders mixed plot/table/code items inside a group', () => {
    render(<GroupedArtifacts groups={makeGroups()} threadId="tid-1" />);
    // The two newest groups start expanded…
    expect(screen.getByText('plot:ROI forest')).toBeInTheDocument();
    expect(screen.getByText('table:t1')).toBeInTheDocument();
    expect(screen.getByText('import pandas as pd')).toBeInTheDocument(); // code header first line
    expect(screen.getByText('plot:Spend vs sales scatter')).toBeInTheDocument();
    // …while older groups (here: Earlier work, 3rd) start collapsed.
    expect(screen.queryByText('plot:Old chart')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Earlier work'));
    expect(screen.getByText('plot:Old chart')).toBeInTheDocument();
  });

  it('shows count badges per group', () => {
    render(<GroupedArtifacts groups={makeGroups()} threadId="tid-1" />);
    // Newest group: 1 plot + 1 table + 1 code; the older question group and
    // the legacy group each hold a single plot.
    expect(screen.getByText('1 plot · 1 table · 1 code')).toBeInTheDocument();
    expect(screen.getAllByText('1 plot')).toHaveLength(2);
  });

  it('filters by type and hides emptied groups', () => {
    render(<GroupedArtifacts groups={makeGroups()} threadId="tid-1" />);
    fireEvent.click(screen.getByRole('button', { name: 'Code' }));
    // Only the second question has a code segment.
    expect(screen.getByText('Fit the model and show ROI.')).toBeInTheDocument();
    expect(screen.queryByText('How does spend correlate with sales?')).not.toBeInTheDocument();
    expect(screen.queryByText('Earlier work')).not.toBeInTheDocument();
    expect(screen.queryByText('plot:ROI forest')).not.toBeInTheDocument();
    expect(screen.getByText('import pandas as pd')).toBeInTheDocument();
    // Back to All restores everything.
    fireEvent.click(screen.getByRole('button', { name: 'All' }));
    expect(screen.getByText('Earlier work')).toBeInTheDocument();
  });

  it('shows a hint when the filter empties every group', () => {
    const groups = buildArtifactGroups(messages, [{ id: 'p1', title: 'Only plot', call_id: 'call-1' }], [], []);
    render(<GroupedArtifacts groups={groups} threadId="tid-1" />);
    fireEvent.click(screen.getByRole('button', { name: 'Code' }));
    expect(screen.getByText(/Nothing matches this filter/)).toBeInTheDocument();
  });

  it('offers Export as Python per question group but not for Earlier work', () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    render(<GroupedArtifacts groups={makeGroups()} threadId="tid-1" />);
    const exports = screen.getAllByTitle('Export as Python');
    // Two question groups get the button; the legacy group does not.
    expect(exports).toHaveLength(2);
    fireEvent.click(exports[0]); // newest group = qIdx 1 → turn:2
    expect(open).toHaveBeenCalledWith(expect.stringContaining('/sessions/tid-1/export?scope=turn:2'), '_blank');
  });

  it('hides Export as Python when there is no session', () => {
    render(<GroupedArtifacts groups={makeGroups()} threadId={null} />);
    expect(screen.queryByTitle('Export as Python')).not.toBeInTheDocument();
  });

  it('renders the empty state when there are no artifacts at all', () => {
    render(<GroupedArtifacts groups={[]} threadId="tid-1" />);
    expect(screen.getByText('No analysis artifacts yet')).toBeInTheDocument();
  });
});
