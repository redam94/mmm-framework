import { describe, it, expect, vi, beforeEach } from 'vitest';
import { plannerService } from './plannerService';
import { apiClient } from '../client';

vi.mock('../client', () => ({
  apiClient: { get: vi.fn(), post: vi.fn() },
}));

const get = apiClient.get as unknown as ReturnType<typeof vi.fn>;
const post = apiClient.post as unknown as ReturnType<typeof vi.fn>;

describe('plannerService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('starts an optimize job at the project-scoped endpoint', async () => {
    post.mockResolvedValue({ data: { job_id: 'j1', status: 'pending' } });
    const res = await plannerService.startOptimize('p1', {
      by_geo: true,
      flighting: { pattern: 'even', n_periods: 13 },
    });
    expect(post).toHaveBeenCalledWith('/projects/p1/planner/optimize', {
      by_geo: true,
      flighting: { pattern: 'even', n_periods: 13 },
    });
    expect(res.job_id).toBe('j1');
  });

  it('polls an optimize job by id', async () => {
    get.mockResolvedValue({
      data: { status: 'done', project_id: 'p1', result: { total_budget: 100 }, error: null },
    });
    const res = await plannerService.pollOptimize('p1', 'j1');
    expect(get).toHaveBeenCalledWith('/projects/p1/planner/optimize/j1');
    expect(res.status).toBe('done');
    expect(res.result?.total_budget).toBe(100);
  });

  it('starts and polls a scenario job', async () => {
    post.mockResolvedValue({ data: { job_id: 'j2', status: 'pending' } });
    await plannerService.startScenario('p1', { spend_changes: { TV: 0.2 } });
    expect(post).toHaveBeenCalledWith('/projects/p1/planner/scenario', {
      spend_changes: { TV: 0.2 },
    });

    get.mockResolvedValue({
      data: { status: 'done', project_id: 'p1', result: { outcome_change: 50 }, error: null },
    });
    const poll = await plannerService.pollScenario('p1', 'j2');
    expect(get).toHaveBeenCalledWith('/projects/p1/planner/scenario/j2');
    expect(poll.result?.outcome_change).toBe(50);
  });
});
