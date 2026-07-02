import { useMemo, useState } from 'react';
import type { Config, Data } from 'plotly.js';
import Plot from 'react-plotly.js';
import { clsx } from 'clsx';
import { Card } from '../../components/ui';
import { COLORS } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import type {
  FundingRow,
  FundingVerdict,
  LearningSnapshot,
} from '../../api/services/learningService';
import { fmtDollars, fmtNum } from './format';

const VERDICT_COLOR: Record<FundingVerdict, string> = {
  FUND: COLORS.sage600,
  HOLD: COLORS.gold600,
  CUT: COLORS.rust600,
};

const VERDICT_CHIP: Record<FundingVerdict, string> = {
  FUND: 'bg-sage-100 text-sage-800',
  HOLD: 'bg-gold-100 text-gold-700',
  CUT: 'bg-rust-100 text-rust-700',
};

/**
 * The funding line, two stacked panels: (a) per-channel P(mROAS > 1)
 * horizontal bars against the 0.5 threshold, with FUND/HOLD/CUT verdicts —
 * the served mROAS is already the value-inclusive marginal return per $1
 * (margin-adjusted when the program sets a margin); (b) the fitted response
 * curve per channel with its 90% band and the current-spend marker
 * (ResponseCurvesPanel idiom).
 */
/** Displayed mROAS — prefer the margin-adjusted value when the server sends it. */
const mroasOf = (f: FundingRow): number => f.mroas_margin_adjusted ?? f.mroas_mean;

export function FundingLineChart({ snapshot }: { snapshot: LearningSnapshot }) {
  const funding = useMemo(() => snapshot.funding ?? [], [snapshot.funding]);
  const marginAdjusted = funding.some((f) => f.mroas_margin_adjusted != null);
  const curves = snapshot.response_curves ?? {};
  const curveChannels = Object.keys(curves);

  const [selected, setSelected] = useState<string | null>(null);
  const selectedChannel =
    selected && curves[selected]
      ? selected
      : funding.find((f) => curves[f.channel])?.channel ?? curveChannels[0] ?? null;
  const curve = selectedChannel ? curves[selectedChannel] ?? null : null;

  const barTraces = useMemo<Data[]>(() => {
    if (funding.length === 0) return [];
    // reverse so the first channel renders at the top of the bar chart
    const rows = [...funding].reverse();
    return [
      {
        type: 'bar',
        orientation: 'h',
        y: rows.map((f) => f.channel),
        x: rows.map((f) => f.prob_above_line),
        marker: { color: rows.map((f) => VERDICT_COLOR[f.verdict] ?? COLORS.gold600) },
        text: rows.map((f) => `${Math.round(f.prob_above_line * 100)}%`),
        textposition: 'auto',
        customdata: rows.map((f) => fmtNum(mroasOf(f))),
        hovertemplate:
          `%{y}: P(above line) = %{x:.2f} · mROAS %{customdata}${
            marginAdjusted ? ' (margin-adjusted)' : ''
          }<extra></extra>`,
        showlegend: false,
      } as unknown as Data,
    ];
  }, [funding, marginAdjusted]);

  const curveTraces = useMemo<Data[]>(() => {
    if (!curve) return [];
    return [
      {
        type: 'scatter',
        mode: 'lines',
        x: curve.spend_dollars,
        y: curve.lo,
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '90% interval',
        x: curve.spend_dollars,
        y: curve.hi,
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: 'rgba(109,138,74,0.16)', // sage-600 wash
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Posterior-mean incremental KPI',
        x: curve.spend_dollars,
        y: curve.mean,
        line: { color: COLORS.sage700, width: 2 },
        showlegend: false,
      },
    ] as Data[];
  }, [curve]);

  return (
    <div className="space-y-6">
      <Card padding="md">
        <h3 className="text-sm font-semibold text-ink-900">
          Funding line — which channels clear a dollar of marginal return
        </h3>
        <p className="mt-0.5 text-xs text-ink-400">
          P(mROAS &gt; 1) at the recommended allocation — mROAS is the marginal return per $1
          {marginAdjusted ? ', margin-adjusted' : ' (margin-adjusted when a margin is set)'}.
          Channels right of the dashed 0.5 line earn their next dollar; fund those, hold the
          uncertain middle, cut the rest.
        </p>
        {funding.length > 0 && (
          <div className="mt-2.5 flex flex-wrap gap-2">
            {funding.map((f) => (
              <span
                key={f.channel}
                className={clsx(
                  'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium',
                  VERDICT_CHIP[f.verdict] ?? 'bg-cream-200 text-ink-600',
                )}
                title={`mROAS ${fmtNum(mroasOf(f))}${
                  f.mroas_margin_adjusted != null ? ' (margin-adjusted)' : ''
                } · P(above line) ${fmtNum(f.prob_above_line)}`}
              >
                {f.channel}
                <span className="font-semibold">{f.verdict}</span>
              </span>
            ))}
          </div>
        )}
        <div className="mt-2">
          {funding.length > 0 ? (
            <Plot
              data={barTraces}
              layout={mmmPlotlyLayout({
                height: Math.max(180, 60 + funding.length * 44),
                margin: { t: 20, l: 110, r: 30, b: 40 },
                showlegend: false,
                xaxis: { title: { text: 'P(mROAS > 1)' }, range: [0, 1] },
                shapes: [
                  {
                    type: 'line',
                    x0: 0.5,
                    x1: 0.5,
                    yref: 'paper',
                    y0: 0,
                    y1: 1,
                    line: { color: COLORS.rust600, width: 1.5, dash: 'dash' },
                  },
                ],
                annotations: [
                  {
                    x: 0.5,
                    yref: 'paper',
                    y: 1.04,
                    text: 'funding line',
                    showarrow: false,
                    font: { size: 11, color: COLORS.rust600 },
                  },
                ],
              })}
              config={PLOTLY_CONFIG as Partial<Config>}
              useResizeHandler
              style={{ width: '100%' }}
            />
          ) : (
            <p className="py-6 text-center text-sm text-ink-400">
              No funding readout yet — refit after the first wave lands.
            </p>
          )}
        </div>
      </Card>

      <Card padding="md">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-ink-900">
              Response curve — {selectedChannel ?? '—'}
            </h3>
            <p className="mt-0.5 text-xs text-ink-400">
              Incremental KPI per geo-period as spend scales from 0 to 2× the current center;
              band is the 90% posterior interval. The dashed line marks current spend.
            </p>
          </div>
          {curveChannels.length > 0 && (
            <label className="flex items-center gap-2 text-xs text-ink-600">
              Channel
              <select
                value={selectedChannel ?? ''}
                onChange={(e) => setSelected(e.target.value)}
                className="rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm text-ink-700 focus:outline-none focus:ring-1 focus:ring-sage-700"
              >
                {curveChannels.map((ch) => (
                  <option key={ch} value={ch}>
                    {ch}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        <div className="mt-2">
          {curve ? (
            <Plot
              data={curveTraces}
              layout={mmmPlotlyLayout({
                height: 320,
                margin: { t: 30, l: 60, r: 30, b: 45 },
                showlegend: false,
                xaxis: { title: { text: 'Spend ($/geo-period)' }, rangemode: 'tozero' },
                yaxis: { title: { text: 'Incremental KPI' }, rangemode: 'tozero' },
                shapes: [
                  {
                    type: 'line',
                    x0: curve.current,
                    x1: curve.current,
                    yref: 'paper',
                    y0: 0,
                    y1: 1,
                    line: { color: COLORS.rust600, width: 1.5, dash: 'dash' },
                  },
                ],
                annotations: [
                  {
                    x: curve.current,
                    yref: 'paper',
                    y: 1.02,
                    text: `current ${fmtDollars(curve.current)}`,
                    showarrow: false,
                    font: { size: 11, color: COLORS.rust600 },
                  },
                ],
              })}
              config={PLOTLY_CONFIG as Partial<Config>}
              useResizeHandler
              style={{ width: '100%' }}
            />
          ) : (
            <p className="py-6 text-center text-sm text-ink-400">
              No response curves in this snapshot yet.
            </p>
          )}
        </div>
      </Card>
    </div>
  );
}
