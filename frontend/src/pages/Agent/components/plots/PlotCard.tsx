import React, { useMemo, useState } from 'react';
import { Maximize2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import type { Data, Layout } from 'plotly.js';
import { Modal } from '../common/Modal';
import { useInView } from '../../hooks/useInView';
import { usePlotFigure } from '../../hooks/usePlotFigure';
import { applyLightModeLayout } from '../../utils/plotly';
import { stripHtml } from '../../utils/text';

// A server-driven dashboard plot blob. It arrives either as a content-addressed
// ref ({ id, title }) or as a legacy inline figure ({ data, layout, title }).
// The shape is otherwise driven by the backend, so unknown extra keys are
// permitted to keep this at least as permissive as the loosely-typed call sites.
interface DashboardPlot {
  id?: string;
  title?: string;
  data?: unknown;
  layout?: unknown;
  [key: string]: unknown;
}

// React.memo: a plot object keeps its identity once appended to dashboardData
// (we merge, never rebuild existing refs), so memoization stops every chart from
// re-rendering — and react-plotly from re-running Plotly.react() — on every SSE
// chunk while the agent streams. Combined with the viewport gate below, this is
// what fixes the "freezes as more outputs accumulate" symptom.
export const PlotCard = React.memo(function PlotCard({ plot, idx }: { plot: DashboardPlot; idx: number }) {
  const [fullscreen, setFullscreen] = useState(false);
  // The observed wrapper is ALWAYS in the DOM (even before reveal) so the
  // IntersectionObserver can fire; only the heavy <Plot> mounts once in view.
  const [wrapRef, inView] = useInView<HTMLDivElement>();
  const fig = usePlotFigure(plot, inView);

  // fig.data / fig.layout are server-driven JSON (typed `unknown`); cast at the
  // Plotly boundary. `title` may arrive as a string or a {text} object.
  const figLayout = fig?.layout as { title?: { text?: string } | string } | undefined;
  const rawTitle =
    (typeof figLayout?.title === 'object' ? figLayout?.title?.text : figLayout?.title) ??
    plot?.title ??
    `Chart ${idx + 1}`;
  const title = stripHtml(String(rawTitle || `Chart ${idx + 1}`));

  const fixedLayout = useMemo(
    () => applyLightModeLayout(fig?.layout as Partial<Layout> | undefined),
    [fig],
  );

  const plotEl = (height: string) =>
    fig ? (
      <Plot
        data={fig.data as Data[]}
        layout={{ ...fixedLayout, autosize: true }}
        useResizeHandler
        style={{ width: '100%', height }}
        config={{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['sendDataToCloud'] }}
      />
    ) : null;

  return (
    <>
      <div
        ref={wrapRef}
        className="rounded-xl overflow-hidden border border-line-200 bg-white relative group shadow-sm min-h-[400px]"
      >
        <button
          onClick={() => setFullscreen(true)}
          className="absolute top-2 right-2 z-10 p-1.5 rounded-lg bg-white/90 text-ink-300 hover:text-ink-700 hover:bg-cream-100 opacity-0 group-hover:opacity-100 transition-all border border-line-200"
          title="Expand chart"
        >
          <Maximize2 size={15} />
        </button>
        <p className="text-xs text-ink-400 px-4 pt-3 pb-0 font-semibold truncate">{title}</p>
        {inView && fig ? (
          plotEl('360px')
        ) : (
          <div className="h-[360px] flex items-center justify-center text-sm text-ink-300">
            {inView ? 'Loading chart…' : ''}
          </div>
        )}
      </div>
      {fullscreen && (
        <Modal title={title} onClose={() => setFullscreen(false)} fullWidth>
          {plotEl('calc(100vh - 120px)')}
        </Modal>
      )}
    </>
  );
}, (a, b) =>
  // Plots are content-addressed by `id`, but each streaming update re-parses the
  // dashboard_data JSON, giving every plot ref a NEW object identity. Compare by
  // id (falling back to reference for legacy inline figures) so existing charts
  // are not needlessly re-rendered when a new plot arrives.
  a.idx === b.idx && (a.plot?.id ?? a.plot) === (b.plot?.id ?? b.plot)
);
