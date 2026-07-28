import type { ReactNode } from 'react';
import { useState } from 'react';
import Plot from 'react-plotly.js';
import { AlertTriangle } from 'lucide-react';
import { COLORS } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import { usePlannerForecast } from '../../api/hooks/usePlanner';
import type { ForecastResultPayload } from '../../api/services/plannerService';
import { fmtInt } from './format';

// ── Forward KPI forecast under a plan (#223) ──────────────────────────────────
//
// The chart is the small part. A single mean line with a tight band reads like a
// measurement, and this is a counterfactual under a plan the model has never
// observed — so the caveats render ABOVE the number, not in a footnote. They are
// computed server-side (trend extrapolation policy, Ljung-Box on the training
// residuals, per-channel spend beyond observed support), never templated here.

// Same classes PlannerStudio uses, so the panel reads as part of the page
// rather than a bolt-on.
const inputCls =
  'w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600';
const labelCls = 'mb-1 block text-xs font-medium text-ink-700';

function FLabel({ children }: { children: ReactNode }) {
  return <span className={labelCls}>{children}</span>;
}

function CaveatList({ caveats }: { caveats: string[] }) {
  if (!caveats.length) return null;
  return (
    <div className="space-y-1.5 rounded-lg border border-gold-200 bg-gold-50 px-3 py-2.5">
      {caveats.map((c) => (
        <div key={c} className="flex gap-2 text-xs leading-relaxed text-gold-800">
          <AlertTriangle size={13} className="mt-0.5 shrink-0" aria-hidden="true" />
          <span>{c}</span>
        </div>
      ))}
    </div>
  );
}

function ForecastChart({ fc }: { fc: ForecastResultPayload }) {
  const x = fc.periods;
  // Only draw a band when there IS one: a MAP posterior has a single draw, and
  // a collapsed band is the visual language of extreme precision.
  const hasBand = fc.caveat_fields.interval_available;
  const traces: Plotly.Data[] = [];
  if (hasBand) {
    traces.push(
      {
        type: 'scatter',
        mode: 'lines',
        x: [...x, ...[...x].reverse()],
        y: [
          ...(fc.upper as number[]),
          ...([...(fc.lower as number[])].reverse()),
        ],
        fill: 'toself',
        fillcolor: 'rgba(120,140,160,0.18)',
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false,
        name: 'interval',
      } as Plotly.Data,
    );
  }
  traces.push({
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Forecast',
    x,
    y: fc.mean,
    line: { color: COLORS.sage600, width: 2 },
    marker: { size: 5, color: COLORS.sage600 },
  } as Plotly.Data);
  traces.push({
    type: 'scatter',
    mode: 'lines',
    name: 'Baseline (non-media)',
    x,
    y: fc.baseline,
    line: { color: COLORS.ink400, width: 1, dash: 'dot' },
  } as Plotly.Data);

  return (
    <Plot
      data={traces}
      layout={mmmPlotlyLayout({
        height: 320,
        margin: { l: 56, r: 16, t: 8, b: 44 },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 },
        xaxis: { title: { text: 'Period' } },
        yaxis: { title: { text: 'KPI' } },
      })}
      config={PLOTLY_CONFIG}
      useResizeHandler
      style={{ width: '100%' }}
    />
  );
}

export function ForecastPanel({
  projectId,
  channels,
}: {
  projectId: string | null;
  channels: string[];
}) {
  const [nPeriods, setNPeriods] = useState(13);
  const [pattern, setPattern] = useState('even');
  const [budgets, setBudgets] = useState<Record<string, string>>({});
  const forecast = usePlannerForecast(projectId);

  const jobStatus = forecast.job.data?.status;
  const fc = forecast.job.data?.result?.forecast ?? null;
  const running = forecast.start.isPending || jobStatus === 'pending' || jobStatus === 'running';
  const error =
    jobStatus === 'error'
      ? forecast.job.data?.error
      : forecast.start.isError
        ? 'Could not start the forecast.'
        : null;

  const run = () => {
    const channel_budgets: Record<string, number> = {};
    for (const ch of channels) {
      const v = Number(budgets[ch]);
      if (Number.isFinite(v) && v > 0) channel_budgets[ch] = v;
    }
    if (!Object.keys(channel_budgets).length) return;
    forecast.reset();
    forecast.start.mutate({
      channel_budgets,
      n_periods: nPeriods,
      pattern,
    });
  };

  const anyBudget = channels.some((ch) => Number(budgets[ch]) > 0);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <FLabel>Periods to forecast</FLabel>
          <input
            type="number"
            min={1}
            value={nPeriods}
            onChange={(e) => setNPeriods(Number(e.target.value))}
            className={inputCls}
          />
        </label>
        <label className="block">
          <FLabel>Flighting pattern</FLabel>
          <select
            value={pattern}
            onChange={(e) => setPattern(e.target.value)}
            className={inputCls}
          >
            <option value="even">Even</option>
            <option value="front_loaded">Front-loaded</option>
            <option value="pulsed">Pulsed</option>
          </select>
        </label>
      </div>

      <div>
        <FLabel>Total budget per channel over the window</FLabel>
        <div className="mt-1 grid grid-cols-2 gap-2">
          {channels.map((ch) => (
            <label key={ch} className="block">
              <span className="text-xs text-ink-500">{ch}</span>
              <input
                type="number"
                min={0}
                value={budgets[ch] ?? ''}
                placeholder="0"
                onChange={(e) => setBudgets((b) => ({ ...b, [ch]: e.target.value }))}
                className={inputCls}
                aria-label={`${ch} forecast budget`}
              />
            </label>
          ))}
        </div>
      </div>

      <button
        onClick={run}
        disabled={running || !anyBudget || !projectId}
        className="rounded-md bg-sage-700 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
      >
        {running ? 'Forecasting…' : 'Forecast'}
      </button>

      {error && <p className="text-sm text-rust-600">{error}</p>}

      {fc && (
        <div className="space-y-3">
          {/* Caveats ABOVE the number: a reader who stops at the headline has
              still been told how the interval is optimistic. */}
          <CaveatList caveats={fc.caveats} />

          <div className="rounded-lg border border-line-200 bg-white px-3 py-2.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">
              Forecast KPI · {fc.periods.length} periods
            </div>
            <div className="mt-0.5 font-display text-2xl font-semibold text-ink-900 num">
              {fmtInt(fc.headline.total)}
            </div>
            <div className="num text-[11px] text-ink-400">
              {fc.headline.interval_available
                ? `${Math.round(fc.interval * 100)}% ${fc.headline.interval_noun}: ` +
                  `${fmtInt(fc.headline.total_lower)} – ${fmtInt(fc.headline.total_upper)}`
                : 'point estimate — no interval (see above)'}
            </div>
          </div>

          <ForecastChart fc={fc} />
        </div>
      )}
    </div>
  );
}
