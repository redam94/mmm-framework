import { useMemo, useState } from 'react';
import { TrendingUp } from 'lucide-react';
import Plot from 'react-plotly.js';
import { DashWidget } from '../common/DashWidget';
import { applyLightModeLayout } from '../../utils/plotly';
import { normalizeTrendType } from '../../utils/spec';

// ─── SeasonalityTrendWidget ───────────────────────────────────────────────────

function generateFourierTraces(order: number, period: number, label: string): any[] {
  const t = Array.from({ length: period }, (_, i) => i);
  const PALETTE = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];
  const traces: any[] = [];
  const combined = t.map(() => 0);

  for (let k = 1; k <= Math.min(order, 6); k++) {
    const phase = Math.PI / (2 * k); // deterministic phase offset
    const y = t.map(ti => Math.sin(2 * Math.PI * k * ti / period + phase));
    y.forEach((v, i) => { combined[i] += v; });
    traces.push({
      x: t, y, name: `${label} H${k}`,
      type: 'scatter', mode: 'lines',
      line: { color: PALETTE[(k - 1) % PALETTE.length], width: 1.5, dash: k > 1 ? 'dot' : 'solid' },
      opacity: 0.6,
    });
  }

  const maxAmp = Math.max(...combined.map(Math.abs)) || 1;
  traces.push({
    x: t, y: combined.map(v => v / maxAmp),
    name: `${label} Combined`,
    type: 'scatter', mode: 'lines',
    line: { color: '#111827', width: 2.5 },
    opacity: 1,
  });

  return traces;
}

function generateTrendTrace(type: string, spec: any): { traces: any[]; shapes: any[] } {
  const t = Array.from({ length: 100 }, (_, i) => i / 99);

  if (type === 'linear' || !type) {
    return {
      traces: [{ x: t, y: t.map(x => 0.15 + 0.7 * x), name: 'Linear Trend', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } }],
      shapes: [],
    };
  }

  if (type === 'piecewise') {
    const n = spec?.n_changepoints ?? 5;
    const range = spec?.changepoint_range ?? 0.8;
    // Evenly space n changepoints within (0, range) — mirrors backend linspace logic
    const cps = Array.from({ length: n }, (_, i) => range * (i + 1) / (n + 1));
    // Enough slope variety for up to 25 changepoints (26 segments)
    const slopes = [0.8, 0.3, 1.1, -0.2, 0.6, 0.9, -0.1, 0.7, 0.4, -0.3, 0.5, -0.4, 1.0, 0.2, -0.5, 0.8, 0.3, 1.1, -0.2, 0.6, 0.9, -0.1, 0.7, 0.4, -0.3, 0.5];
    // Walk through sorted t values, advancing segments at each changepoint
    const y: number[] = [];
    let segStart = 0, segBaseY = 0.1, segIdx = 0;
    for (const x of t) {
      while (segIdx < cps.length && x > cps[segIdx]) {
        segBaseY += slopes[segIdx % slopes.length] * (cps[segIdx] - segStart);
        segStart = cps[segIdx];
        segIdx++;
      }
      y.push(segBaseY + slopes[segIdx % slopes.length] * (x - segStart));
    }
    return {
      traces: [{ x: t, y, name: 'Piecewise Trend', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } }],
      shapes: cps.map(cp => ({
        type: 'line', x0: cp, x1: cp, y0: 0, y1: 1, yref: 'paper',
        line: { color: '#f59e0b', width: 1.5, dash: 'dash' },
      })),
    };
  }

  if (type === 'spline') {
    const nk = spec?.n_knots ?? 4;
    const knots = Array.from({ length: nk }, (_, i) => (i + 1) / (nk + 1));
    const y = t.map(x => {
      let val = 0.15 + 0.6 * x;
      knots.forEach(k => { val += 0.08 * Math.exp(-50 * Math.pow(x - k, 2)) * Math.sin(12 * (x - k)); });
      return val;
    });
    return {
      traces: [{ x: t, y, name: 'Spline Trend', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } }],
      shapes: knots.map(k => ({
        type: 'line', x0: k, x1: k, y0: 0, y1: 1, yref: 'paper',
        line: { color: '#10b981', width: 1, dash: 'dot' },
      })),
    };
  }

  if (type === 'gaussian_process') {
    // Smooth GP realisation using summed cosines
    const y = t.map(x => {
      let v = 0.35;
      for (let k = 1; k <= 6; k++) v += (0.08 / k) * Math.cos(k * Math.PI * x + k * 0.7);
      return v;
    });
    const y_upper = y.map(v => v + 0.12);
    const y_lower = y.map(v => v - 0.12);
    return {
      traces: [
        { x: t, y, name: 'GP Mean', type: 'scatter', mode: 'lines', line: { color: '#6366f1', width: 2.5 } },
        {
          x: [...t, ...t.slice().reverse()],
          y: [...y_upper, ...y_lower.slice().reverse()],
          name: '±1 σ', type: 'scatter', mode: 'lines', fill: 'toself',
          fillcolor: 'rgba(99,102,241,0.12)', line: { color: 'transparent' },
        },
      ],
      shapes: [],
    };
  }

  return { traces: [], shapes: [] };
}

interface SeasonalityTrendWidgetProps {
  spec: any;
  onQuickAction: (msg: string) => void;
  modelCompleted: boolean;
}

export function SeasonalityTrendWidget({ spec, onQuickAction, modelCompleted }: SeasonalityTrendWidgetProps) {
  const [tab, setTab] = useState<'trend' | 'yearly' | 'monthly' | 'weekly'>('trend');

  const trendType = normalizeTrendType(spec?.trend?.type);
  const seasonality = spec?.seasonality;
  const granularity = spec?.time_granularity ?? 'weekly';
  const period = granularity === 'daily' ? 365 : granularity === 'monthly' ? 12 : 52;

  const allTabs: { key: typeof tab; label: string; disabled?: boolean }[] = [
    { key: 'trend' as const, label: 'Trend' },
    { key: 'yearly' as const, label: 'Yearly Season', disabled: !(seasonality?.yearly > 0) },
    { key: 'monthly' as const, label: 'Monthly Season', disabled: !(seasonality?.monthly > 0) },
    { key: 'weekly' as const, label: 'Weekly Season', disabled: !(seasonality?.weekly > 0) },
  ];
  const tabs = allTabs.filter(t => !t.disabled || t.key === 'trend');

  const chartData = useMemo(() => {
    if (tab === 'trend') {
      const { traces, shapes } = generateTrendTrace(trendType, spec?.trend);
      return {
        data: traces,
        layout: applyLightModeLayout({
          shapes,
          xaxis: { title: { text: 'Relative Time' }, showticklabels: false },
          yaxis: { title: { text: 'Trend Value' } },
          legend: { orientation: 'h', y: -0.25 },
          title: `Trend Model: ${trendType.charAt(0).toUpperCase() + trendType.slice(1).replace('_', ' ')}${shapes.length ? ` (${shapes.length} changepoints)` : ''}`,
          margin: { t: 55, b: 55 },
        }),
      };
    }
    const orders: Record<string, number> = {
      yearly: seasonality?.yearly ?? 0,
      monthly: seasonality?.monthly ?? 0,
      weekly: seasonality?.weekly ?? 0,
    };
    const periods: Record<string, number> = { yearly: period, monthly: Math.round(period / 4), weekly: 7 };
    const order = orders[tab];
    const p = periods[tab];
    const traces = generateFourierTraces(order, p, tab.charAt(0).toUpperCase() + tab.slice(1));
    return {
      data: traces,
      layout: applyLightModeLayout({
        xaxis: { title: { text: tab === 'yearly' ? `Week of Year (period = ${p})` : tab === 'monthly' ? 'Week of Month' : 'Day of Week' } },
        yaxis: { title: { text: 'Normalised Amplitude' } },
        legend: { orientation: 'h', y: -0.3, font: { size: 10 } },
        title: `${tab.charAt(0).toUpperCase() + tab.slice(1)} Seasonality — ${order} Fourier Term${order !== 1 ? 's' : ''}`,
        margin: { t: 55, b: 70 },
      }),
    };
  }, [tab, trendType, seasonality, spec, period]);

  const quickActions = [
    { label: 'Trend over time', msg: 'Using execute_python, extract the fitted linear trend component from the `mmm` object and plot it as a Plotly time series. Show the posterior mean and 89% HDI band.' },
    { label: 'Channel contributions', msg: 'Using execute_python with the `mmm` and `results` objects, create a stacked area Plotly chart showing each media channel\'s contribution to the KPI over time.' },
    { label: 'Fitted vs Actual', msg: 'Using execute_python, plot the model\'s posterior predictive mean against the actual KPI values as a Plotly time series.' },
    { label: 'Saturation curves', msg: 'Using execute_python, plot the Hill saturation curves for each media channel showing diminishing returns.' },
  ];

  return (
    <DashWidget
      title="Trend & Seasonality"
      icon={<TrendingUp size={15} className="text-violet-500 shrink-0" />}
      color="violet"
      expandTitle="Trend & Seasonality Preview"
      expandContent={
        <div className="space-y-4">
          <div className="flex gap-2 flex-wrap">
            {tabs.map(t => (
              <button key={t.key} onClick={() => setTab(t.key)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${tab === t.key ? 'bg-violet-600 text-white border-violet-600' : 'bg-white text-ink-600 border-line-200 hover:bg-cream-100'}`}
              >{t.label}</button>
            ))}
          </div>
          <Plot data={chartData.data} layout={{ ...chartData.layout, autosize: true }}
            useResizeHandler style={{ width: '100%', height: '420px' }}
            config={{ responsive: true, displayModeBar: true, displaylogo: false }}
          />
          {modelCompleted && (
            <div className="border-t border-line-200 pt-4">
              <p className="text-xs font-semibold text-ink-600 mb-2">Generate from fitted model:</p>
              <div className="flex flex-wrap gap-2">
                {quickActions.map(qa => (
                  <button key={qa.label} onClick={() => onQuickAction(qa.msg)}
                    className="px-3 py-1.5 bg-violet-50 hover:bg-violet-100 text-violet-700 text-xs font-medium rounded-lg border border-violet-200 transition-colors">
                    {qa.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      }
    >
      {/* Mini preview */}
      <div className="space-y-3">
        <div className="flex gap-2 flex-wrap">
          {tabs.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${tab === t.key ? 'bg-violet-600 text-white border-violet-600' : 'bg-cream-50 text-ink-600 border-line-200 hover:bg-cream-100'}`}
            >{t.label}</button>
          ))}
        </div>
        <Plot data={chartData.data}
          layout={{ ...chartData.layout, autosize: true, margin: { l: 50, r: 20, t: 45, b: 55 } }}
          useResizeHandler style={{ width: '100%', height: '260px' }}
          config={{ responsive: true, displayModeBar: false }}
        />
        {modelCompleted && (
          <div>
            <p className="text-[10px] text-ink-300 uppercase tracking-widest mb-1.5">Quick Actions (from fitted model)</p>
            <div className="flex flex-wrap gap-1.5">
              {quickActions.map(qa => (
                <button key={qa.label} onClick={() => onQuickAction(qa.msg)}
                  className="px-2.5 py-1 bg-violet-50 hover:bg-violet-100 text-violet-700 text-xs rounded-lg border border-violet-200 transition-colors">
                  {qa.label}
                </button>
              ))}
            </div>
          </div>
        )}
        {!modelCompleted && (
          <p className="text-[11px] text-ink-300 italic">Fit the model to generate actual component plots.</p>
        )}
      </div>
    </DashWidget>
  );
}
