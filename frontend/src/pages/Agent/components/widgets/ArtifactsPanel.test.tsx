import { describe, it, expect, vi } from 'vitest';
import { render, screen, within, fireEvent } from '@testing-library/react';

// SyntaxHighlighter is heavy and irrelevant here; stub it so the panel test
// stays presentational — same pattern as ValidationTab.test.tsx child stubs.
vi.mock('react-syntax-highlighter', () => ({
  Prism: ({ children }: { children: string }) => <pre>{children}</pre>,
}));
vi.mock('react-syntax-highlighter/dist/esm/styles/prism', () => ({ oneLight: {} }));

import { ArtifactsPanel } from './ArtifactsPanel';
import type { Artifact } from '../../types';

let seq = 0;
function art(over: Partial<Artifact> = {}): Artifact {
  seq += 1;
  return {
    id: `a${seq}`,
    thread_id: 't1',
    kind: 'code_snippet',
    payload: {},
    created_at: 1_700_000_000 + seq,
    ...over,
  };
}

function handlers() {
  return { onRerun: vi.fn(), onDelete: vi.fn(), onLoadRun: vi.fn() };
}

describe('ArtifactsPanel', () => {
  it('renders code snippets newest first', () => {
    // Seeded out of order (API returns created_at ASC — oldest first).
    const arts = [
      art({ kind: 'code_snippet', created_at: 100, payload: { code: 'print("oldest")' } }),
      art({ kind: 'code_snippet', created_at: 300, payload: { code: 'print("newest")' } }),
      art({ kind: 'code_snippet', created_at: 200, payload: { code: 'print("middle")' } }),
    ];
    const { container } = render(<ArtifactsPanel artifacts={arts} {...handlers()} />);
    const codes = Array.from(container.querySelectorAll('pre')).map(el => el.textContent);
    expect(codes).toEqual(['print("newest")', 'print("middle")', 'print("oldest")']);
  });

  it('renders model runs newest first', () => {
    const arts = [
      art({ kind: 'model_run', created_at: 10, payload: { run_name: 'run_old' } }),
      art({ kind: 'model_run', created_at: 30, payload: { run_name: 'run_new' } }),
      art({ kind: 'model_run', created_at: 20, payload: { run_name: 'run_mid' } }),
    ];
    const { container } = render(<ArtifactsPanel artifacts={arts} {...handlers()} />);
    const names = Array.from(container.querySelectorAll('tbody tr td:first-child')).map(el =>
      el.textContent?.trim(),
    );
    expect(names).toEqual(['run_new', 'run_mid', 'run_old']);
  });

  it('renders reports newest first', () => {
    const arts = [
      art({ kind: 'report', created_at: 1, payload: { path: '/tmp/report_old.html' } }),
      art({ kind: 'report', created_at: 3, payload: { path: '/tmp/report_new.html' } }),
      art({ kind: 'report', created_at: 2, payload: { path: '/tmp/report_mid.html' } }),
    ];
    render(<ArtifactsPanel artifacts={arts} {...handlers()} />);
    // getAllByText returns elements in document order.
    const paths = screen.getAllByText(/report_\w+\.html/).map(el => el.textContent);
    expect(paths).toEqual(['/tmp/report_new.html', '/tmp/report_mid.html', '/tmp/report_old.html']);
  });

  it('does not mutate the artifacts prop when sorting', () => {
    const arts = [
      art({ kind: 'code_snippet', created_at: 100, payload: { code: 'a' } }),
      art({ kind: 'model_run', created_at: 300, payload: { run_name: 'r' } }),
      art({ kind: 'code_snippet', created_at: 200, payload: { code: 'b' } }),
    ];
    const original = [...arts];
    render(<ArtifactsPanel artifacts={arts} {...handlers()} />);
    expect(arts).toEqual(original);
  });

  it('labels code snippets with the question, truncated with a full-text tooltip', () => {
    const q =
      'How does TV adstock decay affect the ROAS estimate for the Q3 campaign in the northeast region?';
    expect(q.length).toBeGreaterThan(80);
    const withQ = art({ kind: 'code_snippet', created_at: 200, payload: { code: 'x=1', question: q } });
    const withoutQ = art({ kind: 'code_snippet', created_at: 100, payload: { code: 'y=2' } });
    render(<ArtifactsPanel artifacts={[withoutQ, withQ]} {...handlers()} />);

    // Truncated label rendered; full question in the title tooltip.
    expect(screen.getByText(q.slice(0, 80) + '…')).toBeInTheDocument();
    const header = screen.getByTitle(q);
    expect(header).toBeInTheDocument();
    // Timestamp demoted to secondary text inside the labelled header.
    expect(
      within(header as HTMLElement).getByText(new Date(withQ.created_at * 1000).toLocaleString()),
    ).toBeInTheDocument();
    // No question → exactly today's fallback label (timestamp).
    expect(
      screen.getByText(new Date(withoutQ.created_at * 1000).toLocaleString()),
    ).toBeInTheDocument();
  });

  it('treats a whitespace-only question as absent (fallback label, no blank row)', () => {
    const arts = [
      art({ kind: 'report', payload: { path: '/tmp/report_ws.html', question: '   \n' } }),
    ];
    render(<ArtifactsPanel artifacts={arts} {...handlers()} />);
    expect(screen.getByText('/tmp/report_ws.html')).toBeInTheDocument();
  });

  it('labels report rows with the question when present, path otherwise', () => {
    const q = 'Which channels drove the holiday lift?';
    const arts = [
      art({ kind: 'report', created_at: 2, payload: { path: '/tmp/report_a.html', question: q } }),
      art({ kind: 'report', created_at: 1, payload: { path: '/tmp/report_b.html' } }),
    ];
    render(<ArtifactsPanel artifacts={arts} {...handlers()} />);
    expect(screen.getByText(q)).toBeInTheDocument();
    expect(screen.queryByText('/tmp/report_a.html')).not.toBeInTheDocument();
    expect(screen.getByText('/tmp/report_b.html')).toBeInTheDocument();
  });

  it('shows the question on model-run rows with the run name as secondary text', () => {
    const q = 'Fit the base MMM with weekly seasonality';
    const arts = [art({ kind: 'model_run', payload: { run_name: 'run_007', question: q } })];
    render(<ArtifactsPanel artifacts={arts} {...handlers()} />);
    expect(screen.getByText(q)).toBeInTheDocument();
    expect(screen.getByText('run_007')).toBeInTheDocument();
  });

  it('keeps the existing actions wired (rerun / delete / load run)', () => {
    const h = handlers();
    const arts = [
      art({ id: 'code1', kind: 'code_snippet', payload: { code: 'x' } }),
      art({ id: 'run1', kind: 'model_run', payload: { run_name: 'r1' } }),
    ];
    render(<ArtifactsPanel artifacts={arts} {...h} />);
    fireEvent.click(screen.getByTitle('Rerun'));
    expect(h.onRerun).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTitle('Delete'));
    expect(h.onDelete).toHaveBeenCalledWith('code1');
    fireEvent.click(screen.getByRole('button', { name: /Load/ }));
    expect(h.onLoadRun).toHaveBeenCalledWith('r1');
  });
});
