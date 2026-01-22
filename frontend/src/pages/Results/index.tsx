import { Card, Title, Text, Select, SelectItem, Metric, Tab, TabGroup, TabList, TabPanels, TabPanel, Button } from '@tremor/react';
import { DocumentArrowDownIcon } from '@heroicons/react/24/outline';
import { useModels, useModelFit, useDecomposition, useResponseCurves, useGenerateReport } from '../../api/hooks';
import { useProjectStore } from '../../stores/projectStore';
import { LoadingPage, LoadingSpinner } from '../../components/common/LoadingSpinner';
import Plot from 'react-plotly.js';

// Model fit chart component
function ModelFitChart({ modelId }: { modelId: string }) {
  const { data, isLoading } = useModelFit(modelId);

  if (isLoading) return <LoadingSpinner />;
  if (!data) return <Text className="text-gray-500">No fit data available</Text>;

  return (
    <div>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="p-3 bg-gray-50 rounded-lg">
          <Text className="text-xs text-gray-500">R²</Text>
          <Metric>{(data.r2 * 100).toFixed(1)}%</Metric>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <Text className="text-xs text-gray-500">RMSE</Text>
          <Metric>{data.rmse.toFixed(2)}</Metric>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <Text className="text-xs text-gray-500">MAPE</Text>
          <Metric>{(data.mape * 100).toFixed(1)}%</Metric>
        </div>
      </div>
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines',
            name: 'Observed',
            x: data.periods,
            y: data.observed,
            line: { color: '#6B7280' },
          },
          {
            type: 'scatter',
            mode: 'lines',
            name: 'Predicted',
            x: data.periods,
            y: data.predicted_mean,
            line: { color: '#3B82F6' },
          },
        ]}
        layout={{
          height: 400,
          margin: { t: 20, r: 20, b: 40, l: 60 },
          showlegend: true,
          legend: { orientation: 'h', y: -0.15 },
          xaxis: { title: { text: 'Period' } },
          yaxis: { title: { text: 'Value' } },
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    </div>
  );
}

// Decomposition chart component
function DecompositionChart({ modelId }: { modelId: string }) {
  const { data, isLoading } = useDecomposition(modelId);

  if (isLoading) return <LoadingSpinner />;
  if (!data) return <Text className="text-gray-500">No decomposition data available</Text>;

  const traces = Object.entries(data.components).map(([name, values], i) => ({
    type: 'scatter' as const,
    mode: 'none' as const,
    name,
    x: data.periods,
    y: values,
    stackgroup: 'one',
    fillcolor: `hsl(${(i * 50) % 360}, 70%, 50%)`,
  }));

  return (
    <Plot
      data={traces}
      layout={{
        height: 400,
        margin: { t: 20, r: 20, b: 40, l: 60 },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 },
        xaxis: { title: { text: 'Period' } },
        yaxis: { title: { text: 'Contribution' } },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

// Response curves component
function ResponseCurvesChart({ modelId }: { modelId: string }) {
  const { data, isLoading } = useResponseCurves(modelId);

  if (isLoading) return <LoadingSpinner />;
  if (!data) return <Text className="text-gray-500">No response curve data available</Text>;

  const channels = Object.keys(data.channels);
  const cols = Math.min(channels.length, 3);

  return (
    <div className={`grid grid-cols-1 md:grid-cols-${cols} gap-4`}>
      {channels.map((channel) => {
        const curve = data.channels[channel];
        return (
          <Card key={channel}>
            <Title className="text-sm">{channel}</Title>
            <Plot
              data={[
                {
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Response',
                  x: curve.spend,
                  y: curve.response,
                  line: { color: '#3B82F6' },
                },
                {
                  type: 'scatter',
                  mode: 'lines',
                  name: 'HDI',
                  x: [...curve.spend, ...curve.spend.slice().reverse()],
                  y: [...curve.response_hdi_high, ...curve.response_hdi_low.slice().reverse()],
                  fill: 'toself',
                  fillcolor: 'rgba(59, 130, 246, 0.2)',
                  line: { color: 'transparent' },
                  showlegend: false,
                },
                {
                  type: 'scatter',
                  mode: 'markers',
                  name: 'Current',
                  x: [curve.current_spend],
                  y: [curve.response[Math.floor(curve.spend.indexOf(curve.current_spend))] || 0],
                  marker: { color: '#EF4444', size: 10 },
                },
              ]}
              layout={{
                height: 200,
                margin: { t: 10, r: 10, b: 40, l: 50 },
                showlegend: false,
                xaxis: { title: { text: 'Spend' } },
                yaxis: { title: { text: 'Response' } },
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </Card>
        );
      })}
    </div>
  );
}

export function ResultsPage() {
  const { data: modelsData, isLoading } = useModels({ status: 'completed' });
  const { selectedModelId, setSelectedModel } = useProjectStore();
  const reportMutation = useGenerateReport(selectedModelId || '');

  if (isLoading) {
    return <LoadingPage message="Loading models..." />;
  }

  const models = modelsData?.models.filter((m) => m.status === 'completed') || [];

  const handleGenerateReport = () => {
    if (!selectedModelId) return;
    reportMutation.mutate({
      title: 'MMM Results Report',
      sections: {
        executive_summary: true,
        model_fit: true,
        channel_contributions: true,
        response_curves: true,
        diagnostics: true,
      },
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <Title>Results & Export</Title>
          <Text>View model results and generate reports</Text>
        </div>
        {selectedModelId && (
          <Button
            icon={DocumentArrowDownIcon}
            onClick={handleGenerateReport}
            loading={reportMutation.isPending}
          >
            Generate Report
          </Button>
        )}
      </div>

      {/* Model selector */}
      <Card>
        <Title className="text-sm">Select Model</Title>
        <div className="mt-4">
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
        </div>
      </Card>

      {/* Results tabs */}
      {selectedModelId ? (
        <TabGroup>
          <TabList>
            <Tab>Model Fit</Tab>
            <Tab>Decomposition</Tab>
            <Tab>Response Curves</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <Card className="mt-4">
                <Title className="text-sm">Observed vs Predicted</Title>
                <div className="mt-4">
                  <ModelFitChart modelId={selectedModelId} />
                </div>
              </Card>
            </TabPanel>
            <TabPanel>
              <Card className="mt-4">
                <Title className="text-sm">Component Decomposition</Title>
                <div className="mt-4">
                  <DecompositionChart modelId={selectedModelId} />
                </div>
              </Card>
            </TabPanel>
            <TabPanel>
              <div className="mt-4">
                <Title className="text-sm mb-4">Saturation Response Curves</Title>
                <ResponseCurvesChart modelId={selectedModelId} />
              </div>
            </TabPanel>
          </TabPanels>
        </TabGroup>
      ) : (
        <Card>
          <Text className="text-gray-500 text-center">
            Select a completed model to view results
          </Text>
        </Card>
      )}
    </div>
  );
}

export default ResultsPage;
