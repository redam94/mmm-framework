import { useMemo } from 'react';
import type { Config, Data } from 'plotly.js';
import Plot from 'react-plotly.js';
import { Card } from '../../components/ui';
import { COLORS } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import type { LearningSnapshot } from '../../api/services/learningService';

/**
 * The γ interaction matrix as a symmetric Plotly heatmap (first heatmap in the
 * app — self-contained). Diverging rust ↔ cream ↔ sage colorscale anchored in
 * COLORS; cell text is the posterior mean with a † suffix when the pair is
 * still prior-dominated; hover shows the 5–95% interval.
 */
export function SynergyHeatmap({
  snapshot,
  channels,
}: {
  snapshot: LearningSnapshot;
  /** display order; defaults to the union of probed pair names */
  channels?: string[];
}) {
  const gamma = useMemo(() => snapshot.gamma ?? [], [snapshot.gamma]);

  const names = useMemo(() => {
    if (channels && channels.length > 0) return channels;
    return Array.from(new Set(gamma.flatMap((g) => g.pair)));
  }, [channels, gamma]);

  const { z, text, hover, zmax, anyPriorDominated } = useMemo(() => {
    const k = names.length;
    const idx = new Map(names.map((n, i) => [n, i]));
    const zz: (number | null)[][] = Array.from({ length: k }, () =>
      Array.from({ length: k }, () => null),
    );
    const tt: string[][] = Array.from({ length: k }, () => Array.from({ length: k }, () => ''));
    const hh: string[][] = Array.from({ length: k }, () => Array.from({ length: k }, () => ''));
    let maxAbs = 0;
    let dagger = false;
    for (const g of gamma) {
      const i = idx.get(g.pair[0]);
      const j = idx.get(g.pair[1]);
      if (i == null || j == null) continue;
      const label = `${g.mean.toFixed(2)}${g.prior_dominated ? '†' : ''}`;
      const detail = `γ ${g.mean.toFixed(2)} [${g.p5.toFixed(2)}, ${g.p95.toFixed(2)}]${
        g.prior_dominated ? ' · prior-dominated' : ''
      }`;
      for (const [a, b] of [
        [i, j],
        [j, i],
      ]) {
        zz[a][b] = g.mean;
        tt[a][b] = label;
        hh[a][b] = detail;
      }
      maxAbs = Math.max(maxAbs, Math.abs(g.mean), Math.abs(g.p5), Math.abs(g.p95));
      if (g.prior_dominated) dagger = true;
    }
    return { z: zz, text: tt, hover: hh, zmax: maxAbs || 1, anyPriorDominated: dagger };
  }, [gamma, names]);

  if (gamma.length === 0) {
    return (
      <Card padding="md">
        <h3 className="text-sm font-semibold text-ink-900">Synergy map (γ)</h3>
        <p className="mt-1 text-sm text-ink-400">
          No probed pairs yet — design a wave with probe pairs to measure how channels
          cannibalize (rust) or reinforce (sage) each other.
        </p>
      </Card>
    );
  }

  const trace = {
    type: 'heatmap',
    x: names,
    y: names,
    z,
    text,
    customdata: hover,
    texttemplate: '%{text}',
    textfont: { size: 11, color: COLORS.ink900 },
    hovertemplate: '%{y} × %{x}<br>%{customdata}<extra></extra>',
    hoverongaps: false,
    zmin: -zmax,
    zmax,
    colorscale: [
      [0, COLORS.rust600],
      [0.5, COLORS.cream100],
      [1, COLORS.sage600],
    ],
    colorbar: {
      title: { text: 'γ', side: 'right' },
      thickness: 12,
      outlinewidth: 0,
      tickfont: { size: 10, color: COLORS.ink600 },
    },
  } as unknown as Data;

  return (
    <Card padding="md">
      <h3 className="text-sm font-semibold text-ink-900">Synergy map (γ)</h3>
      <p className="mt-0.5 text-xs text-ink-400">
        Pairwise interaction on the response surface: negative (rust) = the channels
        cannibalize a shared audience; positive (sage) = they reinforce each other.
      </p>
      <Plot
        data={[trace]}
        layout={mmmPlotlyLayout({
          height: Math.max(280, 120 + names.length * 56),
          margin: { t: 20, l: 110, r: 30, b: 70 },
          xaxis: { showgrid: false },
          yaxis: { showgrid: false, autorange: 'reversed' },
        })}
        config={PLOTLY_CONFIG as Partial<Config>}
        useResizeHandler
        style={{ width: '100%' }}
      />
      {anyPriorDominated && (
        <p className="mt-1 text-[11px] text-ink-300">
          † prior-dominated — this pair hasn't been probed with enough off-axis cells to move
          the prior; treat the sign as an assumption, not a finding.
        </p>
      )}
    </Card>
  );
}
