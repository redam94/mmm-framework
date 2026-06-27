import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useDataStudio } from './useDataStudio';

function jsonResp(body: unknown, ok = true) {
  return { ok, status: ok ? 200 : 400, json: async () => body } as Response;
}

beforeEach(() => { vi.restoreAllMocks(); });

describe('useDataStudio', () => {
  it('hydrate with no staging leaves state null', async () => {
    global.fetch = vi.fn().mockResolvedValue(jsonResp({ staging: null }));
    const { result } = renderHook(() => useDataStudio('t', 'k', 'm'));
    await act(async () => { await result.current.hydrate(); });
    expect(result.current.state).toBeNull();
  });

  it('upload populates state and bumps rev', async () => {
    global.fetch = vi.fn().mockResolvedValue(jsonResp({
      staging_id: 's1', raw: { name: 'sales.csv' }, columns: ['sales', 'tv'],
      inferred_roles: { sales: 'kpi', tv: 'media' }, n_rows: 12, n_cols: 2, preview_rows: [],
    }));
    const { result } = renderHook(() => useDataStudio('t', 'k', 'm'));
    await act(async () => {
      await result.current.upload(new File(['sales,tv\n1,2'], 'sales.csv', { type: 'text/csv' }));
    });
    expect(result.current.state?.filename).toBe('sales.csv');
    expect(result.current.state?.roles).toEqual({ sales: 'kpi', tv: 'media' });
    expect(result.current.rev).toBe(1);
  });

  it('drops an out-of-order pipeline response (newer edit wins)', async () => {
    // upload first
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResp({
      staging_id: 's1', raw: { name: 'x.csv' }, columns: ['a', 'b'], n_rows: 3, n_cols: 2, preview_rows: [],
    }));
    global.fetch = fetchMock;
    const { result } = renderHook(() => useDataStudio('t', 'k', 'm'));
    await act(async () => { await result.current.upload(new File(['a'], 'x.csv')); });

    // two pipeline PUTs; the FIRST resolves LAST (stale) and must be ignored.
    let resolveA: (v: Response) => void = () => {};
    let resolveB: (v: Response) => void = () => {};
    const pA = new Promise<Response>(r => { resolveA = r; });
    const pB = new Promise<Response>(r => { resolveB = r; });
    fetchMock.mockReturnValueOnce(pA).mockReturnValueOnce(pB);

    let callA: Promise<boolean>; let callB: Promise<boolean>;
    act(() => {
      callA = result.current.setSteps([{ op: 'drop_columns', columns: ['a'] }]);
      callB = result.current.setSteps([{ op: 'drop_columns', columns: ['b'] }]);
    });
    await act(async () => {
      resolveB(jsonResp({ columns: ['a'], n_rows: 3, n_cols: 1, preview_rows: [], roles: {} }));
      resolveA(jsonResp({ columns: ['ZZZ_STALE'], n_rows: 99, n_cols: 9, preview_rows: [], roles: {} }));
      await callA!; await callB!;
    });
    // The state must reflect B (the latest), not the stale A response.
    expect(result.current.state?.n_rows).toBe(3);
    expect(result.current.state?.columns).toEqual(['a']);
  });

  it('commit returns the merge payload', async () => {
    global.fetch = vi.fn().mockResolvedValue(jsonResp({
      summary: 'ok', dataset_path: '/x/data_studio_dataset.csv',
      dataset: { rows: 10, columns: ['Period'] }, model_spec: { kpi: 'sales' },
    }));
    const { result } = renderHook(() => useDataStudio('t', 'k', 'm'));
    let payload: unknown;
    await act(async () => { payload = await result.current.commit(); });
    await waitFor(() => expect((payload as { model_spec?: { kpi?: string } })?.model_spec?.kpi).toBe('sales'));
  });
});
