import { useEffect, useState } from 'react';
import { API_BASE } from '../constants';

// A server-driven plot blob. It arrives either as a content-addressed ref
// ({ id, title }) or as a legacy inline figure ({ data, layout, title }). The
// shape is otherwise driven by the backend, so unknown extra keys are permitted
// to keep this at least as permissive as the loosely-typed call sites.
export interface PlotFigure {
  id?: string;
  title?: string;
  data?: unknown;
  layout?: unknown;
  [key: string]: unknown;
}

// Browser-side plot cache. Plots are content-addressed on the backend and served
// with an immutable cache header, so a given id never changes — we fetch its JSON
// at most once per session (and the browser HTTP-caches it across reloads). A
// plot can arrive either inline ({data, layout}, legacy) or as a ref ({id, title}).
const _plotCache = new Map<string, PlotFigure>();

export function usePlotFigure(plot: PlotFigure, enabled: boolean = true): PlotFigure | null {
  const isRef = !!(plot && plot.id && !plot.data);
  const [fig, setFig] = useState<PlotFigure | null>(
    isRef ? _plotCache.get(plot.id as string) ?? null : plot
  );
  useEffect(() => {
    // setFig in the synchronous branches mirrors the derived initial state when
    // the plot prop changes; the async branch resolves a fetched ref. Both must
    // stay in the effect to preserve the existing change-driven behavior.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (!isRef) { setFig(plot); return; }
    const cached = _plotCache.get(plot.id as string);
    if (cached) { setFig(cached); return; }
    if (!enabled) return;  // defer the fetch until the card is near the viewport
    let alive = true;
    // No auth header → maximally cacheable; the agent API serves plots publicly.
    fetch(`${API_BASE}/plots/${plot.id}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: PlotFigure | null) => { if (j) { _plotCache.set(plot.id as string, j); if (alive) setFig(j); } })
      .catch(() => {});
    return () => { alive = false; };
  }, [isRef, plot, enabled]);
  return fig;
}
