import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { Card } from '../../components/ui';
import { COLORS } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import { useExperimentPriorities } from '../../api/hooks/useMeasurement';
import type { PriorityChannel } from '../../api/services/measurementService';

// ── Saturation & ROAS — why average ≠ marginal ────────────────────────────────
// The docs' headline distinction: average ROAS (total contribution / total
// spend) grades the past; marginal ROAS (∂contribution/∂spend at current
// spend) prices the next dollar. On a saturating channel they diverge — and
// reallocating on the average is the classic mistake this panel exists to
// prevent.

function fmt(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits);
}

function saturationNote(ch: PriorityChannel): { label: string; cls: string } | null {
  if (ch.roi_mean == null || ch.marginal_roi == null || ch.roi_mean <= 0) return null;
  const ratio = ch.marginal_roi / ch.roi_mean;
  if (ratio < 0.5) return { label: 'deep in saturation', cls: 'bg-rust-100 text-rust-700' };
  if (ratio < 0.8) return { label: 'saturating', cls: 'bg-gold-100 text-gold-700' };
  return { label: 'near-linear', cls: 'bg-sage-100 text-sage-800' };
}

export function ResponseCurvesPanel({ projectId }: { projectId: string }) {
  const { data: priorities, isLoading } = useExperimentPriorities(projectId);
  const curves = priorities?.response_curves ?? null;
  const channels = priorities?.channels ?? [];

  const [selected, setSelected] = useState<string | null>(null);
  const selectedChannel = selected ?? channels[0]?.channel ?? null;
  const curve = selectedChannel ? curves?.channels?.[selectedChannel] : null;
  const currentIdx = curves?.current_index ?? -1;

  const curveTraces = useMemo(() => {
    if (!curve) return [];
    return [
      {
        type: 'scatter',
        mode: 'lines',
        x: curve.spend,
        y: curve.p5,
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '5–95% interval',
        x: curve.spend,
        y: curve.p95,
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: 'rgba(109,138,74,0.16)', // sage-600 wash
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Posterior-mean contribution',
        x: curve.spend,
        y: curve.mean,
        line: { color: COLORS.sage700, width: 2 },
        marker: { size: 6 },
        showlegend: false,
      },
    ];
  }, [curve]);

  if (isLoading) return <p className="text-sm text-ink-400">Loading response curves…</p>;
  if (!priorities || channels.length === 0) {
    return (
      <Card padding="md">
        <h3 className="text-sm font-semibold text-ink-900">Saturation & ROAS</h3>
        <p className="mt-1 text-sm text-ink-400">
          No run metrics yet — fit a model in the Workspace and each channel's response curve and
          average-vs-marginal ROAS will appear here.
        </p>
      </Card>
    );
  }

  const currentSpend =
    curve && currentIdx >= 0 && currentIdx < curve.spend.length ? curve.spend[currentIdx] : null;

  return (
    <div className="space-y-6">
      <Card padding="md">
        <h3 className="text-sm font-semibold text-ink-900">Average vs marginal ROAS</h3>
        <p className="mt-0.5 text-xs text-ink-400">
          Average ROAS grades past spend (total contribution ÷ total spend); marginal ROAS prices
          the <em>next</em> dollar at the current operating point. On a saturating channel the
          marginal read is lower — reallocate on the marginal column, never the average.
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-line-200 text-[10px] uppercase tracking-wider text-ink-400">
                <th className="py-1.5 pr-3 font-semibold">Channel</th>
                <th className="py-1.5 pr-3 font-semibold">Avg ROAS (90% interval)</th>
                <th className="py-1.5 pr-3 font-semibold">Marginal ROAS</th>
                <th className="py-1.5 pr-3 font-semibold" title="marginal ÷ average — how far returns have bent">
                  Marginal / avg
                </th>
                <th className="py-1.5 font-semibold">Curve position</th>
              </tr>
            </thead>
            <tbody>
              {channels.map((ch) => {
                const note = saturationNote(ch);
                const ratio =
                  ch.roi_mean != null && ch.marginal_roi != null && ch.roi_mean > 0
                    ? ch.marginal_roi / ch.roi_mean
                    : null;
                return (
                  <tr
                    key={ch.channel}
                    onClick={() => setSelected(ch.channel)}
                    className={`cursor-pointer border-b border-line-200 last:border-0 hover:bg-cream-100/60 ${
                      ch.channel === selectedChannel ? 'bg-cream-100' : ''
                    }`}
                  >
                    <td className="py-1.5 pr-3 font-medium text-ink-900">{ch.channel}</td>
                    <td className="num py-1.5 pr-3 text-ink-700">
                      {fmt(ch.roi_mean)}{' '}
                      <span className="text-ink-400">
                        [{fmt(ch.roi_hdi_low)}, {fmt(ch.roi_hdi_high)}]
                      </span>
                    </td>
                    <td className="num py-1.5 pr-3 text-ink-700">{fmt(ch.marginal_roi)}</td>
                    <td className="num py-1.5 pr-3 text-ink-700">{fmt(ratio)}</td>
                    <td className="py-1.5">
                      {note ? (
                        <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${note.cls}`}>
                          {note.label}
                        </span>
                      ) : (
                        <span className="text-ink-300">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {channels.some((ch) => ch.marginal_roi == null) && (
          <p className="mt-2 text-[11px] text-ink-300">
            Channels showing “—” were fitted before marginal ROAS was snapshotted — refit to
            populate them.
          </p>
        )}
      </Card>

      <Card padding="md">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-ink-900">
              Response curve — {selectedChannel ?? '—'}
            </h3>
            <p className="mt-0.5 text-xs text-ink-400">
              Window-total contribution as spend scales from 0× to the top of the sampled grid;
              band is the 5–95% posterior interval. The dashed line marks current spend — the
              curve's slope there <em>is</em> the marginal ROAS.
            </p>
          </div>
          <label className="flex items-center gap-2 text-xs text-ink-600">
            Channel
            <select
              value={selectedChannel ?? ''}
              onChange={(e) => setSelected(e.target.value)}
              className="rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm text-ink-700 focus:outline-none focus:ring-1 focus:ring-sage-700"
            >
              {channels.map((ch) => (
                <option key={ch.channel} value={ch.channel}>
                  {ch.channel}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="mt-2">
          {curve ? (
            <Plot
              data={curveTraces as any}
              layout={mmmPlotlyLayout({
                height: 360,
                margin: { t: 30, l: 60, r: 30, b: 45 },
                showlegend: false,
                xaxis: { title: { text: 'Spend (window total)' }, rangemode: 'tozero' },
                yaxis: { title: { text: 'Contribution' }, rangemode: 'tozero' },
                ...(currentSpend != null
                  ? {
                      shapes: [
                        {
                          type: 'line',
                          x0: currentSpend,
                          x1: currentSpend,
                          yref: 'paper',
                          y0: 0,
                          y1: 1,
                          line: { color: COLORS.rust600, width: 1.5, dash: 'dash' },
                        },
                      ],
                      annotations: [
                        {
                          x: currentSpend,
                          yref: 'paper',
                          y: 1.02,
                          text: 'current spend',
                          showarrow: false,
                          font: { size: 11, color: COLORS.rust600 },
                        },
                      ],
                    }
                  : {}),
              })}
              config={PLOTLY_CONFIG as any}
              useResizeHandler
              style={{ width: '100%' }}
            />
          ) : (
            <p className="py-8 text-center text-sm text-ink-400">
              This run predates curve snapshots — refit in the Workspace to record the response
              curve for {selectedChannel ?? 'this channel'}.
            </p>
          )}
        </div>
        <p className="mt-1 text-[11px] text-ink-300">
          Beyond the sampled grid the curve is extrapolation, not evidence — the optimizer caps
          recommendations at the grid edge for the same reason.
        </p>
      </Card>
    </div>
  );
}

export default ResponseCurvesPanel;
