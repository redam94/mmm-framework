import { describe, it, expect, vi, beforeEach } from 'vitest';
import { learningService } from './learningService';
import type { CreateProgramRequest, LearningSnapshot } from './learningService';
import { apiClient } from '../client';

vi.mock('../client', () => ({
  apiClient: { get: vi.fn(), post: vi.fn(), delete: vi.fn() },
}));

const get = apiClient.get as unknown as ReturnType<typeof vi.fn>;
const post = apiClient.post as unknown as ReturnType<typeof vi.fn>;
const del = apiClient.delete as unknown as ReturnType<typeof vi.fn>;

const SNAPSHOT: LearningSnapshot = {
  schema_version: 1,
  fitted_at: 1751470000,
  evidence: { n_rows: 1280, n_summaries: 4, n_waves: 2, shape_identified: { Chatter: true } },
  diagnostics: { max_rhat: 1.01, min_ess: 350, n_draws: 1000, flags: [] },
  recommendation: { Chatter: 182000 },
  recommendation_scaled: { Chatter: 0.91 },
  allocation_sd: { Chatter: 21000 },
  funding: [
    { channel: 'Chatter', mroas_mean: 1.8, prob_above_line: 0.94, funded: true, verdict: 'FUND' },
  ],
  regret: {
    e_regret_kpi: 3.2,
    e_regret_dollars: 41600,
    enbs: 16600,
    stop: false,
    margin: 1.0,
    population: 13,
    wave_cost: 25000,
  },
  gamma: [
    { pair: ['Chatter', 'Pulse'], mean: -0.42, p5: -0.7, p95: -0.1, sign: 'neg', prior_dominated: false },
  ],
  response_curves: {
    Chatter: { spend_dollars: [0, 140000], mean: [0, 10], lo: [0, 8], hi: [0, 12], current: 140000 },
  },
  warnings: [],
};

describe('learningService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('lists programs at the project-scoped endpoint and unwraps them', async () => {
    get.mockResolvedValue({ data: { programs: [{ id: 'prog1', name: 'P' }] } });
    const res = await learningService.listPrograms('p1');
    expect(get).toHaveBeenCalledWith('/projects/p1/learning-programs');
    expect(res).toHaveLength(1);
    expect(res[0].id).toBe('prog1');
  });

  it('creates a program with the §3.1 config body and unwraps the {program} envelope', async () => {
    // the 201 body is wrapped {program: {...}}, same envelope as the GETs
    post.mockResolvedValue({ data: { program: { id: 'prog1', name: 'FY27 Learning' } } });
    const body: CreateProgramRequest = {
      name: 'FY27 Learning',
      config: {
        channels: ['Chatter', 'Pulse'],
        // per-geo-period dollars: $/period for a single geo
        center: { Chatter: 14000, Pulse: 14000 },
        budget: 28000,
        value_per_unit: 5,
        mode: 'fixed',
        activation: 'hill',
        margin: 1.0,
        // horizon only — the backend computes population = n_geos × horizon
        horizon_periods: 13,
        wave_cost: 25000,
      },
    };
    const res = await learningService.createProgram('p1', body);
    expect(post).toHaveBeenCalledWith('/projects/p1/learning-programs', body);
    expect(res.id).toBe('prog1');
    expect(res.name).toBe('FY27 Learning');
  });

  it('fetches a program with its waves', async () => {
    get.mockResolvedValue({ data: { program: { id: 'prog1' }, waves: [{ id: 'w1' }] } });
    const res = await learningService.getProgram('p1', 'prog1');
    expect(get).toHaveBeenCalledWith('/projects/p1/learning-programs/prog1');
    expect(res.program.id).toBe('prog1');
    expect(res.waves).toHaveLength(1);
  });

  it('deletes a program', async () => {
    del.mockResolvedValue({ data: null });
    await learningService.deleteProgram('p1', 'prog1');
    expect(del).toHaveBeenCalledWith('/projects/p1/learning-programs/prog1');
  });

  it('designs a wave synchronously with delta/probe/holdout', async () => {
    post.mockResolvedValue({
      data: {
        cells_scaled: [[0.7, 0.7]],
        cells_dollars: [[140000, 140000]],
        cell_labels: ['center'],
        n_cells: 1,
        delta: 0.6,
        probe_pairs: [[0, 1]],
        warnings: [],
      },
    });
    const res = await learningService.designWave('p1', 'prog1', {
      delta: 0.6,
      probe_pairs: [[0, 1]],
      n_holdout: 2,
    });
    expect(post).toHaveBeenCalledWith('/projects/p1/learning-programs/prog1/design-wave', {
      delta: 0.6,
      probe_pairs: [[0, 1]],
      n_holdout: 2,
    });
    expect(res.n_cells).toBe(1);

    // probe_pairs is always sent explicitly — [] requests a probe-FREE design
    // (omitting the key would make the backend probe ALL program pairs)
    await learningService.designWave('p1', 'prog1', { delta: 0.6, probe_pairs: [] });
    expect(post).toHaveBeenLastCalledWith(
      '/projects/p1/learning-programs/prog1/design-wave',
      { delta: 0.6, probe_pairs: [] },
    );
  });

  it('ingests wave rows / csv / experiment ids as a 202 job', async () => {
    post.mockResolvedValue({ data: { job_id: 'j1', status: 'pending' } });
    const res = await learningService.ingestWave('p1', 'prog1', { csv_text: 'geo,TV,y\n' });
    expect(post).toHaveBeenCalledWith('/projects/p1/learning-programs/prog1/waves', {
      csv_text: 'geo,TV,y\n',
    });
    expect(res.job_id).toBe('j1');

    await learningService.ingestWave('p1', 'prog1', { experiment_ids: ['e1', 'e2'] });
    expect(post).toHaveBeenLastCalledWith('/projects/p1/learning-programs/prog1/waves', {
      experiment_ids: ['e1', 'e2'],
    });
  });

  it('starts a refit job with optional economics overrides', async () => {
    post.mockResolvedValue({ data: { job_id: 'j2', status: 'pending' } });
    const res = await learningService.startFit('p1', 'prog1', { wave_cost: 30000 });
    expect(post).toHaveBeenCalledWith('/projects/p1/learning-programs/prog1/fit', {
      wave_cost: 30000,
    });
    expect(res.job_id).toBe('j2');

    await learningService.startFit('p1', 'prog1');
    expect(post).toHaveBeenLastCalledWith('/projects/p1/learning-programs/prog1/fit', {});
  });

  it('polls a job and returns the pinned {status, result, error} shape', async () => {
    get.mockResolvedValue({ data: { status: 'done', result: SNAPSHOT, error: null } });
    const res = await learningService.pollJob('p1', 'prog1', 'j1');
    expect(get).toHaveBeenCalledWith('/projects/p1/learning-programs/prog1/jobs/j1');
    expect(res.status).toBe('done');
    expect(res.result?.regret.enbs).toBe(16600);
    expect(res.error).toBeNull();
  });
});
