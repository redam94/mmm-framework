import { describe, expect, it } from 'vitest';
import { mergeDashboardData, unionRefs } from './dashboard';
import type { DashboardData, PlotRef } from '../types';

describe('unionRefs', () => {
  it('unions by id, existing refs first, new ones appended', () => {
    const a: PlotRef[] = [{ id: 'p1' }, { id: 'p2' }];
    const b: PlotRef[] = [{ id: 'p2' }, { id: 'p3' }];
    expect(unionRefs(a, b).map((r) => r.id)).toEqual(['p1', 'p2', 'p3']);
  });

  it('keeps legacy inline figures without an id', () => {
    const a = [{ data: [1] } as PlotRef];
    const b = [{ id: 'p1' }];
    expect(unionRefs(a, b)).toHaveLength(2);
  });

  it('tolerates null/undefined sides', () => {
    expect(unionRefs(null, [{ id: 'x' }])).toEqual([{ id: 'x' }]);
    expect(unionRefs([{ id: 'x' }], undefined)).toEqual([{ id: 'x' }]);
  });
});

describe('mergeDashboardData', () => {
  it('unions plots/tables instead of replacing — a subset payload must not wipe accumulated results', () => {
    const prev: DashboardData = {
      plots: [{ id: 'old1' }, { id: 'old2' }],
      tables: [{ id: 't1', title: 'T', source: 's' }],
    };
    // e.g. delegate_to_expert folds back ONLY the expert's plots
    const merged = mergeDashboardData(prev, { plots: [{ id: 'new1' }] });
    expect(merged.plots?.map((p) => p.id)).toEqual(['old1', 'old2', 'new1']);
    expect(merged.tables).toHaveLength(1); // untouched key survives
  });

  it('non-ref keys keep last-write-wins semantics', () => {
    const merged = mergeDashboardData(
      { model_spec: { kpi: 'a' } } as DashboardData,
      { model_spec: { kpi: 'b' } },
    );
    expect((merged.model_spec as { kpi: string }).kpi).toBe('b');
  });

  it('explicit null clears a ref list (backend escape hatch)', () => {
    const merged = mergeDashboardData({ plots: [{ id: 'p1' }] }, { plots: null });
    expect(merged.plots).toBeUndefined();
  });

  it('dedupes when the payload carries the full accumulated list (single-tool case)', () => {
    const prev: DashboardData = { plots: [{ id: 'a' }, { id: 'b' }] };
    const merged = mergeDashboardData(prev, { plots: [{ id: 'a' }, { id: 'b' }, { id: 'c' }] });
    expect(merged.plots?.map((p) => p.id)).toEqual(['a', 'b', 'c']);
  });
});
