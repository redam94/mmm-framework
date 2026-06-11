import { useEffect, useState } from 'react';
import { API_BASE } from '../constants';
import type { TableSpec } from '../types';

// Browser-side table cache, mirroring usePlotFigure. Tables are content-addressed
// on the backend and served immutable, so a given id never changes — we fetch its
// JSON at most once per session (and the browser HTTP-caches it across reloads).
// The cache is the source of truth read at render time; the state is only a
// re-render trigger bumped when an async fetch lands (keeps the effect free of
// synchronous setState).
const _tableCache = new Map<string, TableSpec>();

export function useTableSpec(id: string, enabled: boolean = true): TableSpec | null {
  const [, bump] = useState(0);
  const spec = _tableCache.get(id) ?? null;
  useEffect(() => {
    if (spec) return;
    if (!enabled) return; // defer the fetch until the card is near the viewport
    let alive = true;
    // No auth header → maximally cacheable; the agent API serves tables publicly.
    fetch(`${API_BASE}/tables/${id}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: TableSpec | null) => {
        if (j) {
          _tableCache.set(id, j);
          if (alive) bump((n) => n + 1);
        }
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [id, enabled, spec]);
  return spec;
}
