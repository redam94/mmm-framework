import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Plotly can't render in jsdom (DesignStudio imports it at module level).
vi.mock('react-plotly.js', () => ({ default: () => <div data-testid="plot" /> }));

vi.mock('react-router-dom', () => ({
  useNavigate: () => vi.fn(),
  useParams: () => ({}),
}));

let projectId: string | null = 'p1';
vi.mock('../../stores/projectStore', () => ({
  useProjectStore: (sel: (s: { currentProjectId: string | null }) => unknown) =>
    sel({ currentProjectId: projectId }),
}));

const registry = { data: [] as unknown[], isLoading: false };
const priorities = { data: null as unknown, isLoading: false };
vi.mock('../../api/hooks/useMeasurement', async (orig) => ({
  ...(await orig<Record<string, unknown>>()),
  useExperimentRegistry: () => registry,
  useExperimentPriorities: () => priorities,
}));

const { ExperimentsPage } = await import('./index');

// Child components (drawer/modal/studio) use unmocked react-query hooks; give
// them a real (but network-less) client.
function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <ExperimentsPage />
    </QueryClientProvider>,
  );
}

describe('ExperimentsPage', () => {
  beforeEach(() => {
    projectId = 'p1';
    registry.data = [];
    priorities.data = null;
  });

  it('shows the no-project empty state without a project', () => {
    projectId = null;
    renderPage();
    expect(screen.getByText('No project selected')).toBeInTheDocument();
    // header actions are disabled without a project
    expect(screen.getByText(/Log experiment/).closest('button')).toBeDisabled();
  });

  it('shows the no-baseline empty state when nothing is logged or fitted', () => {
    renderPage();
    expect(screen.getByText('Nothing to prioritize yet')).toBeInTheDocument();
    expect(screen.getByText('Go to Workspace')).toBeInTheDocument();
  });

  it('renders the page frame with logged experiments and no priorities', () => {
    registry.data = [
      {
        id: 'e1',
        channel: 'TV',
        status: 'running',
        created_at: 1751000000,
        updated_at: 1751000000,
        status_history: [],
      },
    ];
    renderPage();
    expect(screen.getByText('Experiments')).toBeInTheDocument();
    // priorities missing → the explanatory empty state, not a crash
    expect(screen.getByText('No experiment priorities yet')).toBeInTheDocument();
  });
});
