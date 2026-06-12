import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { AlertTriangle } from 'lucide-react';
import { Card, TierBadge } from '../../components/ui';
import { COLORS, EVIDENCE_TIER, type EvidenceTier } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import type { PrioritiesPayload, PriorityChannel } from '../../api/services/measurementService';

function tierOf(ch: PriorityChannel): EvidenceTier {
  if (ch.retest_due) return 'stale';
  return ch.calibration_status === 'experiment_backed' ? 'calibrated' : 'model_only';
}

const MIN_SIZE = 10;
const MAX_SIZE = 28;

export function PriorityMatrix({ priorities }: { priorities: PrioritiesPayload }) {
  const channels = useMemo(
    () => priorities.channels.filter((c) => c.eig != null && c.evoi != null),
    [priorities],
  );

  const { data, layout } = useMemo(() => {
    const xs = channels.map((c) => c.eig as number);
    const ys = channels.map((c) => c.evoi as number);

    const xt = priorities.portfolio.eig_threshold ?? 0;
    const yt = priorities.portfolio.evoi_threshold ?? 0;

    const xMax = Math.max(...xs, xt, 0.01) * 1.18;
    const yMax = Math.max(...ys, yt, 1) * 1.18;
    const xMin = -xMax * 0.04;
    const yMin = -yMax * 0.04;

    const maxShare = Math.max(...channels.map((c) => c.spend_share ?? 0), 1e-9);
    const sizes = channels.map((c) => {
      const share = c.spend_share ?? 0;
      return MIN_SIZE + (MAX_SIZE - MIN_SIZE) * Math.sqrt(share / maxShare);
    });

    const colors = channels.map((c) => EVIDENCE_TIER[tierOf(c)].fg);

    const hover = channels.map((c) => {
      const roi =
        c.roi_mean != null
          ? `ROI ${c.roi_mean.toFixed(2)} ± ${c.roi_sd != null ? c.roi_sd.toFixed(2) : '?'}`
          : 'ROI —';
      const prio = c.priority != null ? `priority ${c.priority.toFixed(2)}` : 'priority —';
      const share =
        c.spend_share != null ? `${(c.spend_share * 100).toFixed(1)}% of spend` : '';
      return `<b>${c.channel}</b><br>${roi}<br>${prio}${share ? `<br>${share}` : ''}`;
    });

    const trace = {
      type: 'scatter',
      mode: 'markers+text',
      x: xs,
      y: ys,
      text: channels.map((c) => c.channel),
      textposition: 'top center',
      textfont: { size: 11, color: COLORS.ink700 },
      hovertext: hover,
      hoverinfo: 'text',
      marker: {
        size: sizes,
        color: colors,
        opacity: 0.85,
        line: { color: '#ffffff', width: 1.5 },
      },
      showlegend: false,
    };

    const rect = (x0: number, x1: number, y0: number, y1: number, fill: string) => ({
      type: 'rect',
      xref: 'x',
      yref: 'y',
      x0,
      x1,
      y0,
      y1,
      fillcolor: fill,
      line: { width: 0 },
      layer: 'below',
    });

    const shapes = [
      // quadrant tints: test-now in sage, the rest in cream washes
      rect(xt, xMax, yt, yMax, 'rgba(238,242,231,0.65)'), // sage-100 — test now
      rect(xMin, xt, yt, yMax, 'rgba(243,240,230,0.45)'), // cream — monitor
      rect(xt, xMax, yMin, yt, 'rgba(243,240,230,0.45)'), // cream — learn cheaply
      rect(xMin, xt, yMin, yt, 'rgba(240,237,224,0.7)'), // cream-200 — deprioritize
      {
        type: 'line',
        x0: xt,
        x1: xt,
        yref: 'paper',
        y0: 0,
        y1: 1,
        line: { color: COLORS.line400, width: 1, dash: 'dash' },
      },
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: yt,
        y1: yt,
        line: { color: COLORS.line400, width: 1, dash: 'dash' },
      },
    ];

    const corner = (x: number, y: number, anchor: 'left' | 'right', text: string) => ({
      xref: 'paper',
      yref: 'paper',
      x,
      y,
      xanchor: anchor,
      yanchor: y > 0.5 ? 'top' : 'bottom',
      text,
      showarrow: false,
      font: { size: 11, color: COLORS.ink400 },
    });

    const layout = mmmPlotlyLayout({
      height: 480,
      xaxis: { title: { text: 'EIG — expected information gain (nats)' }, range: [xMin, xMax] },
      yaxis: { title: { text: 'EVOI — expected value of information (KPI units)' }, range: [yMin, yMax] },
      shapes,
      annotations: [
        corner(0.99, 0.98, 'right', 'Test now'),
        corner(0.01, 0.98, 'left', 'Monitor'),
        corner(0.99, 0.03, 'right', 'Learn cheaply'),
        corner(0.01, 0.03, 'left', 'Deprioritize'),
      ],
      margin: { l: 70, r: 30, t: 30, b: 60 },
      showlegend: false,
    });

    return { data: [trace as any], layout };
  }, [channels, priorities.portfolio]);

  return (
    <div className="space-y-3">
      {priorities.stale && (
        <div className="flex items-center gap-2 rounded-lg border border-gold-300 bg-gold-100 px-4 py-2.5 text-sm text-gold-700">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          Computed from an older run — refit to refresh.
        </div>
      )}

      <Card padding="sm">
        {channels.length === 0 ? (
          <p className="py-12 text-center text-sm text-ink-400">
            No channels with computed priorities in this run.
          </p>
        ) : (
          <Plot
            data={data}
            layout={layout}
            config={PLOTLY_CONFIG as any}
            useResizeHandler
            style={{ width: '100%', height: 480 }}
          />
        )}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line-200 px-2 pb-1 pt-3">
          <TierBadge tier="calibrated" />
          <TierBadge tier="model_only" />
          <TierBadge tier="stale" label="Re-test due" />
          <p className="ml-auto text-xs text-ink-400">
            EIG = how much a clean experiment would tighten the channel's ROI; EVOI = what that
            tightening is worth in expected KPI. Marker size tracks spend share.
          </p>
        </div>
      </Card>
    </div>
  );
}
