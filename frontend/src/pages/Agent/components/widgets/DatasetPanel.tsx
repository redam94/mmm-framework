import { useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import { DashWidget } from '../common/DashWidget';
import { API_BASE } from '../../constants';
import { applyLightModeLayout } from '../../utils/plotly';
import type { DatasetInfo } from '../../types';

export function DatasetPanel({ dataset, threadId }: { dataset: DatasetInfo; threadId: string | null }) {
  const [selectedVar, setSelectedVar] = useState<string | null>(null);
  const [dimFilters, setDimFilters] = useState<Record<string, string>>({});
  const [series, setSeries] = useState<{ date: string; value: number }[] | null>(null);
  const [loadingSeries, setLoadingSeries] = useState(false);

  const activeDims = dataset.active_dimensions ?? [];
  const variables = dataset.variable_names ?? [];

  useEffect(() => {
    if (!selectedVar || !threadId) return;
    setLoadingSeries(true);
    const params = new URLSearchParams({ variable: selectedVar });
    const activeDimFilters = activeDims.filter(d => dimFilters[d]);
    if (activeDimFilters.length > 0) {
      params.set('dim', activeDimFilters[0]);
      params.set('value', dimFilters[activeDimFilters[0]]);
    }
    fetch(`${API_BASE}/dataset/preview/${encodeURIComponent(threadId ?? "")}?${params}`)
      .then(r => r.json())
      .then(data => { setSeries(data.series ?? null); })
      .catch(() => setSeries(null))
      .finally(() => setLoadingSeries(false));
  }, [selectedVar, JSON.stringify(dimFilters), threadId]);

  const plotData = series ? [{
    x: series.map(p => p.date),
    y: series.map(p => p.value),
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { color: '#6366f1', width: 2 },
    name: selectedVar ?? '',
  }] : [];

  const plotLayout = applyLightModeLayout({
    title: selectedVar ? `${selectedVar}${dimFilters && Object.keys(dimFilters).length ? ` — ${Object.entries(dimFilters).map(([k, v]) => `${k}: ${v}`).join(', ')}` : ''}` : '',
    xaxis: { title: 'Date' },
    yaxis: { title: 'Value' },
    height: 280,
    margin: { l: 60, r: 20, t: 40, b: 60 },
  });

  return (
    <div className="space-y-4">
      {/* Summary */}
      <DashWidget title="Dataset Summary" dotColor="bg-indigo-500" color="indigo">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-gray-50 p-3 rounded-xl border border-gray-100">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Rows</p>
            <p className="text-2xl font-bold text-gray-900">{dataset.rows.toLocaleString()}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded-xl border border-gray-100">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Columns</p>
            <p className="text-2xl font-bold text-gray-900">{dataset.columns.length}</p>
          </div>
          {dataset.date_range && (
            <div className="bg-gray-50 p-3 rounded-xl border border-gray-100 col-span-2">
              <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-1">Date Range</p>
              <p className="text-sm font-medium text-gray-700">{dataset.date_range.min} → {dataset.date_range.max}</p>
            </div>
          )}
        </div>
      </DashWidget>

      {/* Variable names + preview */}
      {variables.length > 0 && (
        <DashWidget title={`Variables (${variables.length})`} dotColor="bg-violet-500" color="violet">
          {/* Dimension filters */}
          {activeDims.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {activeDims.map(dim => {
                const stat = dataset.column_stats?.[dim];
                const opts = stat?.top_values ?? [];
                return (
                  <div key={dim} className="flex items-center gap-1.5">
                    <span className="text-xs text-gray-500 font-medium">{dim}:</span>
                    <select
                      value={dimFilters[dim] ?? ''}
                      onChange={e => setDimFilters(prev => ({ ...prev, [dim]: e.target.value }))}
                      className="text-xs border border-gray-200 rounded-lg px-2 py-1 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-violet-400"
                    >
                      <option value="">All</option>
                      {opts.map(o => (
                        <option key={o.value} value={o.value}>{o.value}</option>
                      ))}
                    </select>
                  </div>
                );
              })}
            </div>
          )}
          {/* Variable chips */}
          <div className="flex flex-wrap gap-2 mb-3">
            {variables.map(v => (
              <button
                key={v}
                onClick={() => setSelectedVar(v === selectedVar ? null : v)}
                className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                  selectedVar === v
                    ? 'bg-violet-600 text-white border-violet-600'
                    : 'bg-white text-gray-600 border-gray-200 hover:border-violet-400 hover:text-violet-600'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
          {/* Chart */}
          {selectedVar && (
            <div className="rounded-xl overflow-hidden border border-gray-100 bg-gray-50">
              {loadingSeries ? (
                <div className="flex items-center justify-center h-[280px] text-gray-400">
                  <Loader2 size={22} className="animate-spin" />
                </div>
              ) : series ? (
                <Plot
                  data={plotData}
                  layout={{ ...plotLayout, autosize: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '280px' }}
                  config={{ responsive: true, displayModeBar: false }}
                />
              ) : (
                <div className="flex items-center justify-center h-[280px] text-gray-400 text-sm">No data</div>
              )}
            </div>
          )}
        </DashWidget>
      )}

      {/* Dimension value counts */}
      {Object.entries(dataset.column_stats ?? {}).map(([dim, stat]) => (
        <DashWidget
          key={dim}
          title={`${dim} (${stat.unique} unique${stat.truncated ? ', top 20 shown' : ''})`}
          dotColor="bg-sky-500"
          color="sky"
          defaultOpen={false}
        >
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-3 py-2 text-left font-semibold text-sky-600 border-b border-gray-200">Value</th>
                  <th className="px-3 py-2 text-right font-semibold text-sky-600 border-b border-gray-200">Count</th>
                  <th className="px-3 py-2 text-right font-semibold text-sky-600 border-b border-gray-200">%</th>
                </tr>
              </thead>
              <tbody>
                {stat.top_values.map((row, i) => {
                  const total = stat.top_values.reduce((s, r) => s + r.count, 0);
                  const pct = total > 0 ? ((row.count / total) * 100).toFixed(1) : '0.0';
                  return (
                    <tr key={i} className="even:bg-gray-50 hover:bg-sky-50 transition-colors">
                      <td className="px-3 py-1.5 text-gray-700 border-b border-gray-100 font-mono">{row.value}</td>
                      <td className="px-3 py-1.5 text-right text-gray-600 border-b border-gray-100">{row.count.toLocaleString()}</td>
                      <td className="px-3 py-1.5 text-right text-gray-400 border-b border-gray-100">{pct}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </DashWidget>
      ))}
    </div>
  );
}
