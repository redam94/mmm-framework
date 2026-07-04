import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';

vi.mock('react-router-dom', () => ({ useNavigate: () => vi.fn() }));

let projectId: string | null = null;
vi.mock('../../stores/projectStore', () => ({
  useProjectStore: Object.assign(
    (sel?: (s: { currentProjectId: string | null }) => unknown) => {
      const state = { currentProjectId: projectId };
      return sel ? sel(state) : state;
    },
    {},
  ),
}));

const portfolio = { data: null as unknown, isLoading: false };
vi.mock('../../api/hooks/usePortfolio', () => ({
  usePortfolio: () => portfolio,
}));
vi.mock('../../api/hooks/useMeasurement', () => ({
  useExperimentPriorities: () => ({ data: null, isLoading: false }),
  useExperimentRegistry: () => ({ data: [], isLoading: false }),
  useCalibrationCoverage: () => ({ data: null, isLoading: false }),
}));

// The section components fetch their own data — stub them so this test pins
// the PAGE composition (which sections render, in what states), not their
// internals.
vi.mock('./CycleStageRing', () => ({
  CycleStageRing: () => <div data-testid="cycle-ring" />,
}));
vi.mock('./OnboardingChecklist', () => ({
  OnboardingChecklist: () => <div data-testid="onboarding" />,
}));
vi.mock('./HeadlineKPIs', () => ({ HeadlineKPIs: () => <div data-testid="kpis" /> }));
vi.mock('./NextBestActions', () => ({
  NextBestActions: () => <div data-testid="next-actions" />,
}));
vi.mock('./CoverageMap', () => ({ CoverageMap: () => <div data-testid="coverage" /> }));
vi.mock('./RecentActivity', () => ({
  RecentActivity: () => <div data-testid="activity" />,
}));
vi.mock('./IdentificationContract', () => ({
  IdentificationContract: () => <div data-testid="ident-contract" />,
}));

const { ProgramPage } = await import('./index');

describe('ProgramPage', () => {
  beforeEach(() => {
    projectId = null;
    portfolio.data = null;
  });

  it('asks for a project when none is selected', () => {
    render(<ProgramPage />);
    expect(
      screen.getByText('Pick a project to see its measurement program'),
    ).toBeInTheDocument();
  });

  it('points at T₀ when the project has no fitted runs yet', () => {
    projectId = 'p1';
    render(<ProgramPage />);
    expect(screen.getByText('Start at T₀ — fit a baseline model')).toBeInTheDocument();
    expect(screen.getByTestId('onboarding')).toBeInTheDocument();
  });

  it('composes the program sections once a fitted run exists', () => {
    projectId = 'p1';
    portfolio.data = { model_runs: [{ run_id: 'r1' }] };
    render(<ProgramPage />);
    for (const id of [
      'onboarding',
      'kpis',
      'next-actions',
      'coverage',
      'activity',
      'ident-contract',
    ]) {
      expect(screen.getByTestId(id)).toBeInTheDocument();
    }
  });
});
