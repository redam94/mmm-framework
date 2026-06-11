import { Layers } from 'lucide-react';
import Plot from 'react-plotly.js';
import { DashWidget } from '../common/DashWidget';
import { applyLightModeLayout } from '../../utils/plotly';

export function DecompositionWidget({ decomposition }: { decomposition: Array<{ component: string; total_contribution: number; pct_of_total: number }> }) {
  const sorted = [...decomposition].sort((a, b) => b.pct_of_total - a.pct_of_total);
  const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#06b6d4', '#84cc16'];

  const barLayout = applyLightModeLayout({
    xaxis: { title: { text: '% of Total KPI' }, range: [0, 100], ticksuffix: '%' },
    yaxis: { autorange: 'reversed', tickfont: { size: 11 } },
    margin: { l: 120, r: 30, t: 30, b: 50 },
    showlegend: false,
  });

  return (
    <DashWidget
      title="Component Decomposition"
      icon={<Layers size={15} className="text-emerald-500 shrink-0" />}
      color="emerald"
      expandContent={
        <Plot
          data={[{
            type: 'bar', orientation: 'h',
            x: sorted.map(d => +(d.pct_of_total * 100).toFixed(1)),
            y: sorted.map(d => d.component),
            text: sorted.map(d => `${(d.pct_of_total * 100).toFixed(1)}%`),
            textposition: 'outside',
            textfont: { color: '#374151', size: 12 },
            marker: { color: sorted.map((_, i) => COLORS[i % COLORS.length]) },
          }]}
          layout={{ ...barLayout, autosize: true }}
          useResizeHandler style={{ width: '100%', height: '420px' }}
          config={{ responsive: true, displayModeBar: false }}
        />
      }
    >
      <div className="space-y-2">
        {sorted.map((d, i) => (
          <div key={d.component} className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
            <span className="text-xs text-gray-700 flex-1 font-medium">{d.component}</span>
            <div className="flex-1 max-w-[120px] bg-gray-100 rounded-full h-1.5 overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${(d.pct_of_total * 100).toFixed(1)}%`, backgroundColor: COLORS[i % COLORS.length] }} />
            </div>
            <span className="text-xs font-semibold text-gray-900 w-10 text-right">{(d.pct_of_total * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </DashWidget>
  );
}
