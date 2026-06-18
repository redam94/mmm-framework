import { useEffect, useState } from 'react';
import { API_BASE } from '../constants';

// Browser-side plot cache. Plots are content-addressed on the backend and served
// with an immutable cache header, so a given id never changes — we fetch its JSON
// at most once per session (and the browser HTTP-caches it across reloads). A
// plot can arrive either inline ({data, layout}, legacy) or as a ref ({id, title}).
const _plotCache = new Map<string, any>();

export function usePlotFigure(plot: any, enabled: boolean = true): any | null {
  const isRef = !!(plot && plot.id && !plot.data);
  const [fig, setFig] = useState<any | null>(
    isRef ? _plotCache.get(plot.id) ?? null : plot
  );
  useEffect(() => {
    if (!isRef) { setFig(plot); return; }
    const cached = _plotCache.get(plot.id);
    if (cached) { setFig(cached); return; }
    if (!enabled) return;  // defer the fetch until the card is near the viewport
    let alive = true;
    // No auth header → maximally cacheable; the agent API serves plots publicly.
    fetch(`${API_BASE}/plots/${plot.id}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => { if (j) { _plotCache.set(plot.id, j); if (alive) setFig(j); } })
      .catch(() => {});
    return () => { alive = false; };
  }, [isRef, plot, enabled]);
  return fig;
}
