import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { Mock } from 'vitest';
import type { ReactNode } from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('../services/learningService', () => ({
  learningService: {
    ingestWave: vi.fn(),
    pollJob: vi.fn(),
  },
}));

import { learningService } from '../services/learningService';
import {
  LEARNING_JOB_POLL_MS,
  learningJobPollInterval,
  learningKeys,
  useIngestWave,
} from './useLearning';

describe('learningJobPollInterval', () => {
  it('keeps polling while the job is pending/running (or unknown)', () => {
    expect(learningJobPollInterval(undefined)).toBe(LEARNING_JOB_POLL_MS);
    expect(learningJobPollInterval('pending')).toBe(LEARNING_JOB_POLL_MS);
    expect(learningJobPollInterval('running')).toBe(LEARNING_JOB_POLL_MS);
  });

  it('stops polling once the job settles', () => {
    expect(learningJobPollInterval('done')).toBe(false);
    expect(learningJobPollInterval('error')).toBe(false);
  });
});

describe('useIngestWave', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('starts the job, polls it, and invalidates the program on done', async () => {
    (learningService.ingestWave as Mock).mockResolvedValue({ job_id: 'j9', status: 'pending' });
    (learningService.pollJob as Mock).mockResolvedValue({
      status: 'done',
      result: null,
      error: null,
    });

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidate = vi.spyOn(qc, 'invalidateQueries');
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={qc}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(() => useIngestWave('p1', 'prog1'), { wrapper });

    act(() => result.current.start.mutate({ csv_text: 'geo,TV,y\ng1,1,2' }));

    await waitFor(() => expect(result.current.jobId).toBe('j9'));
    await waitFor(() => expect(result.current.job.data?.status).toBe('done'));

    expect(learningService.ingestWave).toHaveBeenCalledWith('p1', 'prog1', {
      csv_text: 'geo,TV,y\ng1,1,2',
    });
    expect(learningService.pollJob).toHaveBeenCalledWith('p1', 'prog1', 'j9');

    // done → the program detail + list refresh
    await waitFor(() =>
      expect(invalidate).toHaveBeenCalledWith({
        queryKey: learningKeys.program('p1', 'prog1'),
      }),
    );
    expect(invalidate).toHaveBeenCalledWith({ queryKey: learningKeys.programs('p1') });

    // reset clears the in-flight job
    act(() => result.current.reset());
    expect(result.current.jobId).toBeNull();
  });
});
