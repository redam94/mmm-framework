import { Card, Title, Text, Select, SelectItem, Tab, TabGroup, TabList, TabPanels, TabPanel, Button, Badge } from '@tremor/react';
import { DocumentArrowDownIcon } from '@heroicons/react/24/outline';
import { useQuery } from '@tanstack/react-query';
import { useModels } from '../../api/hooks';
import { apiClient } from '../../api/client';
import { useProjectStore } from '../../stores/projectStore';
import { LoadingPage, LoadingSpinner } from '../../components/common/LoadingSpinner';
import Plot from 'react-plotly.js';

interface ModelDashboard {
  model_id: string;
  thread_id: string;
  run_id: string | null;
  run_name: string | null;
  kpi: string | null;
  channels: string[];
  controls: string[];
  n_obs: number | null;
  n_channels: number | null;
  inference: Record<string, number>;
  trend: string | null;
  seasonality: Record<string, number>;
  summary: string | null;
  roi_metrics: Array<{
    channel: string;
    roi_mean: number;
    roi_hdi_lower?: number;
    roi_hdi_upper?: number;
    total_contribution?: number;
    pct_of_total?: number;
    [key: string]: unknown;
  }>;
  decomposition: Array<{
    component: string;
    total_contribution: number;
    pct_of_total: number;
  }>;
  report_path: string | null;
  model_path: string | null;
}

function useModelDashboard(modelId: string | undefined) {
  return useQuery({
    queryKey: ['model-dashboard', modelId],
    queryFn: async () => {
      const { data } = await apiClient.get<ModelDashboard>(`/models/${modelId}/dashboard`);
      return data;
    },
    enabled: !!modelId,
    staleTime: Infinity,
  });
}

function RoiChart({ data }: { data: ModelDashboard['roi_metrics'] }) {
  if (!data.length) return <Text className="text-gray-500">No ROI metrics available</Text>;

  const channels = data.map((r) => r.channel);
  const means = data.map((r) => r.roi_mean ?? 0);
  const lows = data.map((r) => r.roi_hdi_lower ?? r.roi_mean ?? 0);
  const highs = data.map((r) => r.roi_hdi_upper ?? r.roi_mean ?? 0);
  const errLow = means.map((m, i) => m - lows[i]);
  const errHigh = highs.map((m, i) => i - m + highs[i]);

  return (
    <Plot
      data={[{
        type: 'bar',
        x: channels,
        y: means,
        error_y: { type: 'data', symmetric: false, array: errHigh, arrayminus: errLow, visible: true },
        marker: { color: means.map((v) => v >= 1 ? '#10B981' : '#EF4444') },
        name: 'ROI',
      }]}
      layout={{
        height: 380,
        margin: { t: 20, r: 20, b: 80, l: 60 },
        yaxis: { title: { text: 'ROI (return per $ spent)' }, zeroline: true, zerolinecolor: '#6B7280' },
        xaxis: { tickangle: -30 },
        showlegend: false,
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function DecompositionChart({ data, kpi }: { data: ModelDashboard['decomposition']; kpi: string | null }) {
  if (!data.length) return <Text className="text-gray-500">No decomposition data available</Text>;

  const COLORS = ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6','#06B6D4','#F97316','#84CC16','#EC4899','#6B7280'];

  return (
    <div className="grid grid-cols-2 gap-6">
      <Plot
        data={[{
          type: 'pie',
          labels: data.map((d) => d.component),
          values: data.map((d) => d.pct_of_total),
          textinfo: 'label+percent',
          hovertemplate: '%{label}: %{value:.1f}%<extra></extra>',
          marker: { colors: COLORS },
        }]}
        layout={{ height: 360, margin: { t: 20, r: 20, b: 20, l: 20 }, showlegend: false }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
      <div className="space-y-2 self-center">
        {data.map((d, i) => (
          <div key={d.component} className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full flex-shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
              <span className="text-gray-700 truncate max-w-[160px]">{d.component}</span>
            </div>
            <div className="text-right ml-4">
              <span className="font-medium text-gray-900">{d.pct_of_total.toFixed(1)}%</span>
              <span className="text-gray-500 ml-2 text-xs">{d.total_contribution?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
            </div>
          </div>
        ))}
        {kpi && <p className="text-xs text-gray-400 mt-3">Contribution to {kpi}</p>}
      </div>
    </div>
  );
}

function SummaryPanel({ dash }: { dash: ModelDashboard }) {
  const inf = dash.inference || {};
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'KPI', value: dash.kpi || '—' },
          { label: 'Observations', value: dash.n_obs?.toString() || '—' },
          { label: 'Channels', value: dash.n_channels?.toString() || dash.channels.length.toString() },
          { label: 'Trend', value: dash.trend || '—' },
          { label: 'Chains', value: inf.chains?.toString() || '—' },
          { label: 'Draws', value: inf.draws?.toString() || '—' },
          { label: 'Tune', value: inf.tune?.toString() || '—' },
          { label: 'Target Accept', value: inf.target_accept?.toString() || '—' },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-500">{label}</p>
            <p className="text-sm font-semibold text-gray-900 mt-0.5">{value}</p>
          </div>
        ))}
      </div>
      {dash.channels.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Media Channels</p>
          <div className="flex flex-wrap gap-1.5">
            {dash.channels.map((ch) => <Badge key={ch} color="blue">{ch}</Badge>)}
          </div>
        </div>
      )}
      {dash.controls.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Controls</p>
          <div className="flex flex-wrap gap-1.5">
            {dash.controls.map((c) => <Badge key={c} color="gray">{c}</Badge>)}
          </div>
        </div>
      )}
      {dash.summary && (
        <div className="bg-blue-50 rounded-lg p-4">
          <p className="text-xs font-medium text-blue-700 uppercase tracking-wide mb-2">Fit Summary</p>
          <pre className="text-xs text-blue-900 whitespace-pre-wrap font-mono leading-relaxed">{dash.summary}</pre>
        </div>
      )}
    </div>
  );
}

export function ResultsPage() {
  const { data: modelsData, isLoading } = useModels({ status: 'completed' } as any);
  const { selectedModelId, setSelectedModel } = useProjectStore();
  const { data: dash, isLoading: dashLoading } = useModelDashboard(selectedModelId || undefined);

  if (isLoading) return <LoadingPage message="Loading models..." />;

  const models = modelsData?.models.filter((m) => m.status === 'completed') ?? [];

  const handleOpenReport = () => {
    window.open('/report', '_blank');
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <Title>Results & Export</Title>
          <Text>View model results and generate reports</Text>
        </div>
        {dash?.report_path && (
          <Button icon={DocumentArrowDownIcon} onClick={handleOpenReport}>
            View Report
          </Button>
        )}
      </div>

      <Card>
        <Title className="text-sm">Select Model</Title>
        <div className="mt-4">
          {models.length === 0 ? (
            <Text className="text-gray-400 text-sm">
              No completed models found. Fit a model via the Agent Copilot first.
            </Text>
          ) : (
            <Select
              value={selectedModelId || ''}
              onValueChange={(v) => setSelectedModel(v)}
              placeholder="Select a completed model..."
            >
              {models.map((m) => (
                <SelectItem key={m.model_id} value={m.model_id}>
                  {m.name || m.model_id}
                </SelectItem>
              ))}
            </Select>
          )}
        </div>
      </Card>

      {selectedModelId && (
        dashLoading ? (
          <Card><LoadingSpinner /></Card>
        ) : dash ? (
          <TabGroup>
            <TabList>
              <Tab>Summary</Tab>
              <Tab>ROI Metrics</Tab>
              <Tab>Decomposition</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <Card className="mt-4">
                  <SummaryPanel dash={dash} />
                </Card>
              </TabPanel>
              <TabPanel>
                <Card className="mt-4">
                  <Title className="text-sm mb-1">Return on Investment by Channel</Title>
                  <Text className="text-xs text-gray-500 mb-4">94% HDI error bars. Green = ROI &gt; 1.</Text>
                  <RoiChart data={dash.roi_metrics} />
                </Card>
              </TabPanel>
              <TabPanel>
                <Card className="mt-4">
                  <Title className="text-sm mb-4">Aggregate Component Decomposition</Title>
                  <DecompositionChart data={dash.decomposition} kpi={dash.kpi} />
                </Card>
              </TabPanel>
            </TabPanels>
          </TabGroup>
        ) : (
          <Card>
            <Text className="text-gray-500 text-center">Could not load model data.</Text>
          </Card>
        )
      )}

      {!selectedModelId && models.length > 0 && (
        <Card>
          <Text className="text-gray-500 text-center">Select a model above to view results.</Text>
        </Card>
      )}
    </div>
  );
}

export default ResultsPage;
