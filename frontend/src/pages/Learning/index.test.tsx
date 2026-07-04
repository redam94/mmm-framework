import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';

vi.mock('react-plotly.js', () => ({ default: () => <div data-testid="plot" /> }));

let projectId: string | null = 'p1';
vi.mock('../../stores/projectStore', () => ({
  useProjectStore: (sel: (s: { currentProjectId: string | null }) => unknown) =>
    sel({ currentProjectId: projectId }),
}));

const programs = { data: [] as unknown[], isLoading: false };
vi.mock('../../api/hooks/useLearning', async (orig) => ({
  ...(await orig<Record<string, unknown>>()),
  useLearningPrograms: () => programs,
  useLearningProgram: () => ({ data: null, isLoading: false }),
  useStartFit: () => ({
    start: { mutate: vi.fn(), isPending: false },
    job: { data: null },
    reset: vi.fn(),
  }),
  useDeleteProgram: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

const { LearningPage } = await import('./index');

describe('LearningPage (Sextant)', () => {
  beforeEach(() => {
    projectId = 'p1';
    programs.data = [];
  });

  it('shows the no-project empty state without a project', () => {
    projectId = null;
    render(<LearningPage />);
    expect(screen.getByText('No project selected')).toBeInTheDocument();
    expect(screen.getByText(/Start a program/).closest('button')).toBeDisabled();
  });

  it('shows the no-programs empty state with a start action', () => {
    render(<LearningPage />);
    expect(screen.getByText('No learning programs yet')).toBeInTheDocument();
    // one enabled CTA in the empty state + the header action
    const starts = screen.getAllByText(/Start a program/);
    expect(starts.length).toBeGreaterThanOrEqual(2);
  });
});
