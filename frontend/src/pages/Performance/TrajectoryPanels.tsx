import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { Data } from 'plotly.js';
import Plot from 'react-plotly.js';
import { Card, StatHero } from '../../components/ui';
import { CHART_COLORWAY, COLORS } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import type { HistoryPayload, RoiPoint } from '../../api/services/measurementService';

// ── helpers ───────────────────────────────────────────────────────────────────

function shortDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/** Percent change last vs first; null when either side is missing or first is 0. */
function pctDelta(first: number | null | undefined, last: number | null | undefined): number | null {
  if (first == null || last == null || !Number.isFinite(first) || !Number.isFinite(last)) return null;
  if (first === 0) return null;
  return ((last - first) / Math.abs(first)) * 100;
}

const CHART_MARGIN = { t: 30, l: 50, r: 30, b: 40 };

function ChartCard({ title, caption, children }: { title: string; caption: string; children: ReactNode }) {
  return (
    <Card padding="md">
      <h3 className="text-sm font-semibold text-ink-900">{title}</h3>
      <p className="mt-0.5 text-xs text-ink-400">{caption}</p>
      <div className="mt-2">{children}</div>
    </Card>
  );
}

// ── component ─────────────────────────────────────────────────────────────────

export function TrajectoryPanels({ history }: { history: HistoryPayload }) {
  const { runs, channels, series, portfolio } = history;

  const [selectedChannel, setSelectedChannel] = useState(channels[0] ?? '');

  const xIdx = useMemo(() => runs.map((_, i) => i), [runs]);
  const tickText = useMemo(() => runs.map((r) => shortDate(r.timestamp_iso)), [runs]);
  const runAxis = { tickvals: xIdx, ticktext: tickText, tickangle: 0 };

  // ── hero stats: first vs last portfolio point ──
  const first = portfolio[0];
  const last = portfolio[portfolio.length - 1];

  // ── chart a: CI contraction per channel ──
  const ciTraces = useMemo<Data[]>(
    () =>
      channels.map((ch, i) => {
        const byRun = new Map((series.roi[ch] ?? []).map((p) => [p.run_id, p]));
        return {
          type: 'scatter',
          mode: 'lines+markers',
          name: ch,
          x: xIdx,
          y: runs.map((r) => byRun.get(r.run_id)?.ci_width ?? null),
          line: { color: CHART_COLORWAY[i % CHART_COLORWAY.length], width: 2 },
          marker: { size: 6 },
        } as Data;
      }),
    [channels, series.roi, runs, xIdx],
  );

  // ── chart b: budget-share migration (normalized stacked area) ──
  const shareTraces = useMemo<Data[]>(
    () =>
      channels.map((ch, i) => {
        const byRun = new Map((series.spend_share[ch] ?? []).map((p) => [p.run_id, p]));
        return {
          type: 'scatter',
          mode: 'lines',
          name: ch,
          x: xIdx,
          y: runs.map((r) => byRun.get(r.run_id)?.value ?? null),
          stackgroup: 'one',
          groupnorm: 'percent',
          line: { width: 0.5, color: CHART_COLORWAY[i % CHART_COLORWAY.length] },
        } as Data;
      }),
    [channels, series.spend_share, runs, xIdx],
  );

  // ── chart c: misallocation (expected uplift) ──
  const upliftTrace: Data = {
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Expected uplift',
    x: portfolio.map((p) => runs.findIndex((r) => r.run_id === p.run_id)),
    y: portfolio.map((p) => p.expected_uplift),
    line: { color: COLORS.rust600, width: 2 },
    marker: { size: 7, color: COLORS.rust600 },
  };

  // ── chart d: portfolio mROI (+ EVPI on a secondary axis when present) ──
  const portfolioX = portfolio.map((p) => runs.findIndex((r) => r.run_id === p.run_id));
  const hasEvpi = portfolio.some((p) => p.evpi != null);
  const mroiTraces: Data[] = [
    {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Marginal ROI',
      x: portfolioX,
      y: portfolio.map((p) => p.marginal_roi),
      line: { color: COLORS.sage600, width: 2 },
      marker: { size: 7, color: COLORS.sage600 },
    },
  ];
  if (hasEvpi) {
    mroiTraces.push({
      type: 'scatter',
      mode: 'lines',
      name: 'EVPI',
      x: portfolioX,
      y: portfolio.map((p) => p.evpi),
      yaxis: 'y2',
      line: { color: COLORS.steel300, width: 1.5, dash: 'dash' },
    } as Data);
  }

  // ── full-width ROI band chart for the selected channel ──
  const roiByRun = useMemo(() => {
    const m = new Map<string, RoiPoint>();
    for (const p of series.roi[selectedChannel] ?? []) m.set(p.run_id, p);
    return m;
  }, [series.roi, selectedChannel]);
  const calByRun = useMemo(() => {
    const m = new Map<string, string>();
    for (const p of series.calibration[selectedChannel] ?? []) m.set(p.run_id, p.status);
    return m;
  }, [series.calibration, selectedChannel]);

  const bandTraces: Data[] = [
    {
      type: 'scatter',
      mode: 'lines',
      x: xIdx,
      y: runs.map((r) => roiByRun.get(r.run_id)?.hdi_low ?? null),
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
    },
    {
      type: 'scatter',
      mode: 'lines',
      name: '90% interval',
      x: xIdx,
      y: runs.map((r) => roiByRun.get(r.run_id)?.hdi_high ?? null),
      line: { width: 0 },
      fill: 'tonexty',
      fillcolor: 'rgba(109,138,74,0.16)', // sage-600 wash
      hoverinfo: 'skip',
      showlegend: false,
    },
    {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Posterior mean',
      x: xIdx,
      y: runs.map((r) => roiByRun.get(r.run_id)?.mean ?? null),
      line: { color: COLORS.sage700, width: 2 },
      marker: {
        size: 9,
        color: runs.map((r) =>
          calByRun.get(r.run_id) === 'experiment_backed' ? COLORS.sage600 : COLORS.steel600,
        ),
        line: { color: '#ffffff', width: 1.5 },
      },
      showlegend: false,
    },
  ];

  return (
    <div className="space-y-6">
      {/* hero row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatHero
          label="Portfolio marginal ROI"
          value={last?.marginal_roi != null ? last.marginal_roi.toFixed(2) : '—'}
          delta={pctDelta(first?.marginal_roi, last?.marginal_roi)}
          hint="vs first cycle"
        />
        <StatHero
          label="Misallocation proxy"
          value={
            last?.expected_uplift != null ? Math.round(last.expected_uplift).toLocaleString() : '—'
          }
          delta={pctDelta(first?.expected_uplift, last?.expected_uplift)}
          increaseIsGood={false}
          hint="expected uplift left unclaimed"
        />
        <StatHero
          label="Mean ROI CI width"
          value={last?.mean_ci_width != null ? last.mean_ci_width.toFixed(2) : '—'}
          delta={pctDelta(first?.mean_ci_width, last?.mean_ci_width)}
          increaseIsGood={false}
          hint="narrower is sharper"
        />
      </div>

      {/* 2×2 chart grid */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <ChartCard
          title="CI contraction"
          caption="Width of each channel's 90% ROI interval per cycle — falling lines mean sharper estimates."
        >
          <Plot
            data={ciTraces}
            layout={mmmPlotlyLayout({
              height: 340,
              margin: CHART_MARGIN,
              showlegend: true,
              xaxis: runAxis,
              yaxis: { title: { text: 'CI width' }, rangemode: 'tozero' },
            })}
            config={PLOTLY_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </ChartCard>

        <ChartCard
          title="Budget-share migration"
          caption="How spend allocation shifted across channels as the model's recommendations evolved."
        >
          <Plot
            data={shareTraces}
            layout={mmmPlotlyLayout({
              height: 340,
              margin: CHART_MARGIN,
              showlegend: true,
              xaxis: runAxis,
              yaxis: { title: { text: 'Share of spend (%)' }, ticksuffix: '%' },
            })}
            config={PLOTLY_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </ChartCard>

        <ChartCard
          title="Misallocation"
          caption="KPI left on the table vs the optimal allocation — should fall as estimates sharpen."
        >
          <Plot
            data={[upliftTrace]}
            layout={mmmPlotlyLayout({
              height: 340,
              margin: CHART_MARGIN,
              showlegend: false,
              xaxis: runAxis,
              yaxis: { title: { text: 'Expected uplift' }, rangemode: 'tozero' },
            })}
            config={PLOTLY_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </ChartCard>

        <ChartCard
          title="Portfolio mROI"
          caption="Marginal ROI of the recommended allocation per cycle; dashed line is the expected value of perfect information."
        >
          <Plot
            data={mroiTraces}
            layout={mmmPlotlyLayout({
              height: 340,
              margin: { ...CHART_MARGIN, r: hasEvpi ? 50 : 30 },
              showlegend: hasEvpi,
              xaxis: runAxis,
              yaxis: { title: { text: 'Marginal ROI' } },
              ...(hasEvpi
                ? {
                    yaxis2: {
                      title: { text: 'EVPI' },
                      overlaying: 'y',
                      side: 'right',
                      showgrid: false,
                    },
                  }
                : {}),
            })}
            config={PLOTLY_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </ChartCard>
      </div>

      {/* full-width per-channel ROI band */}
      <Card padding="md">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-ink-900">
              ROI estimate trajectory — {selectedChannel || '—'}
            </h3>
            <p className="mt-0.5 text-xs text-ink-400">
              Band is the 90% interval around the posterior mean; sage dots mark experiment-backed
              cycles, steel dots are model-only.
            </p>
          </div>
          <label className="flex items-center gap-2 text-xs text-ink-600">
            Channel
            <select
              value={selectedChannel}
              onChange={(e) => setSelectedChannel(e.target.value)}
              className="rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm text-ink-700 focus:outline-none focus:ring-1 focus:ring-sage-700"
            >
              {channels.map((ch) => (
                <option key={ch} value={ch}>
                  {ch}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="mt-2">
          <Plot
            data={bandTraces}
            layout={mmmPlotlyLayout({
              height: 340,
              margin: CHART_MARGIN,
              showlegend: false,
              xaxis: runAxis,
              yaxis: { title: { text: 'ROI' } },
            })}
            config={PLOTLY_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </div>
      </Card>
    </div>
  );
}
