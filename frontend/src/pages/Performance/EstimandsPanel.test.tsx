import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type {
  EstimandCell,
  EstimandGroup,
  EstimandModel,
  EstimandRunSummary,
  ProjectEstimands,
} from '../../api/services/estimandsService';

// The panel reads its data through the react-query hook; mock it so the test is
// purely presentational (mirrors the RunComparison pattern).
const useProjectEstimands = vi.fn();
vi.mock('../../api/hooks/useEstimands', () => ({
  useProjectEstimands: (id: string | null) => useProjectEstimands(id),
}));

import { EstimandsPanel } from './EstimandsPanel';

function cell(channel: string, mean: number | null, evidence: EstimandCell['evidence']): EstimandCell {
  return {
    channel,
    mean,
    lower: mean == null ? null : mean - 0.2,
    upper: mean == null ? null : mean + 0.2,
    units: 'ROI',
    status: mean == null ? 'unsupported' : 'ok',
    evidence,
    prob_positive: 0.9,
    prob_profitable: 0.8,
  };
}

function model(run_id: string, created_at: number, rows: EstimandCell[]): EstimandModel {
  return { run_id, label: run_id, model_kind: 'mmm', model_key: run_id, created_at, rows };
}

function runSummary(
  run_id: string,
  kpi: string,
  model_key: string,
  isLatest: boolean,
): EstimandRunSummary {
  return {
    run_id,
    label: run_id,
    model_kind: 'mmm',
    model_key,
    kpi,
    created_at: 1700000000,
    n_estimands: 1,
    is_latest_for_model: isLatest,
  };
}

const roiGroup: EstimandGroup = {
  key: 'contribution_roi|||revenue',
  estimand: 'contribution_roi',
  label: 'Contribution ROI',
  kpi: 'revenue',
  kind: 'roi',
  units: 'ROI',
  is_ratio: true,
  reference: 1.0,
  channels: ['TV'],
  models: [
    model('r1', 2000, [cell('TV', 2.1, 'strong')]),
    model('r2', 2000, [cell('TV', 1.8, 'strong')]),
    model('r0', 1000, [cell('TV', 9.99, 'strong')]), // older run of r1's model
  ],
  n_models: 3,
  n_models_with_data: 3,
};

const liftGroup: EstimandGroup = {
  key: 'awareness_lift|||awareness',
  estimand: 'awareness_lift',
  label: 'Awareness lift',
  kpi: 'awareness',
  kind: 'lift',
  units: 'KPI/period',
  is_ratio: false,
  reference: 0.0,
  channels: ['—'],
  models: [model('r3', 2000, [cell('—', 0.5, 'strong')])],
  n_models: 1,
  n_models_with_data: 1,
};

const payload: ProjectEstimands = {
  runs: [
    runSummary('r1', 'revenue', 'm1', true),
    runSummary('r2', 'revenue', 'm2', true),
    runSummary('r0', 'revenue', 'm1', false), // not latest -> hidden by default
    runSummary('r3', 'awareness', 'm3', true),
  ],
  kpis: ['awareness', 'revenue'],
  groups: [liftGroup, roiGroup],
};

beforeEach(() => {
  useProjectEstimands.mockReset();
});

describe('EstimandsPanel', () => {
  it('renders empty state when there are no runs', () => {
    useProjectEstimands.mockReturnValue({ data: { runs: [], kpis: [], groups: [] }, isLoading: false, isError: false });
    render(<EstimandsPanel projectId="p" />);
    expect(screen.getByText(/No estimands yet/)).toBeInTheDocument();
  });

  it('groups comparable estimands and separates KPIs', () => {
    useProjectEstimands.mockReturnValue({ data: payload, isLoading: false, isError: false });
    render(<EstimandsPanel projectId="p" />);
    // both metric clusters render
    expect(screen.getByText('Contribution ROI')).toBeInTheDocument();
    expect(screen.getByText('Awareness lift')).toBeInTheDocument();
    // KPI section headers keep different KPIs apart
    expect(screen.getByText('revenue')).toBeInTheDocument();
    expect(screen.getByText('awareness')).toBeInTheDocument();
  });

  it('shows the two latest models side by side as comparable', () => {
    useProjectEstimands.mockReturnValue({ data: payload, isLoading: false, isError: false });
    render(<EstimandsPanel projectId="p" />);
    // default selection = latest per model => r1 + r2 (revenue), not the older r0
    expect(screen.getByText('Comparable · 2 models')).toBeInTheDocument();
    expect(screen.getByText('2.10')).toBeInTheDocument();
    expect(screen.getByText('1.80')).toBeInTheDocument();
    // the non-latest run's value is hidden until selected
    expect(screen.queryByText('9.99')).not.toBeInTheDocument();
    // the single-model lift cluster is labelled as such
    expect(screen.getByText('Single model')).toBeInTheDocument();
  });

  it('reveals older runs when "Select all" is clicked', () => {
    useProjectEstimands.mockReturnValue({ data: payload, isLoading: false, isError: false });
    render(<EstimandsPanel projectId="p" />);
    fireEvent.click(screen.getByText('Select all'));
    expect(screen.getByText('9.99')).toBeInTheDocument();
    expect(screen.getByText('Comparable · 3 models')).toBeInTheDocument();
  });

  it('resets an explicit selection when the project changes', () => {
    // Project B has a single revenue model whose run_id collides with one in A.
    const projectB: ProjectEstimands = {
      runs: [runSummary('r1', 'revenue', 'mB', true)],
      kpis: ['revenue'],
      groups: [
        {
          ...roiGroup,
          models: [model('r1', 2000, [cell('TV', 4.2, 'strong')])],
          n_models: 1,
          n_models_with_data: 1,
        },
      ],
    };
    useProjectEstimands.mockReturnValue({ data: payload, isLoading: false, isError: false });
    const { rerender } = render(<EstimandsPanel projectId="A" />);
    // collapse to a single explicit selection in project A
    fireEvent.click(screen.getByText('Latest only'));
    expect(screen.getByText('2.10')).toBeInTheDocument();

    // switch to project B: the panel must fall back to B's default selection,
    // not carry A's explicit run_id set over.
    useProjectEstimands.mockReturnValue({ data: projectB, isLoading: false, isError: false });
    rerender(<EstimandsPanel projectId="B" />);
    expect(screen.getByText('4.20')).toBeInTheDocument();
    expect(screen.queryByText('2.10')).not.toBeInTheDocument();
  });
});
