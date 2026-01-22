import { useState } from 'react';
import { Card, Title, Text, Button, Select, SelectItem, ProgressBar, Badge } from '@tremor/react';
import { PlayIcon } from '@heroicons/react/24/outline';
import { useDatasets, useConfigs, useModels, useSubmitFitJob, useModelStatus } from '../../api/hooks';
import { useProjectStore } from '../../stores/projectStore';
import { LoadingPage } from '../../components/common/LoadingSpinner';
import type { JobStatus } from '../../api/types';

// Status badge component
function JobStatusBadge({ status }: { status: JobStatus }) {
  const colorMap: Record<JobStatus, 'green' | 'yellow' | 'red' | 'blue' | 'gray'> = {
    completed: 'green',
    running: 'blue',
    queued: 'yellow',
    pending: 'gray',
    failed: 'red',
    cancelled: 'gray',
  };

  return (
    <Badge color={colorMap[status] || 'gray'} size="sm">
      {status}
    </Badge>
  );
}

// Job monitor component
function JobMonitor({ modelId }: { modelId: string }) {
  const { data: status } = useModelStatus(modelId);

  if (!status) return null;

  return (
    <Card className="mt-4">
      <div className="flex justify-between items-center">
        <Title className="text-sm">Job Progress</Title>
        <JobStatusBadge status={status.status} />
      </div>
      <div className="mt-4">
        <ProgressBar value={status.progress} color="blue" />
        <Text className="mt-2 text-sm text-gray-500">
          {status.progress_message || `${status.progress.toFixed(0)}% complete`}
        </Text>
      </div>
      {status.error_message && (
        <div className="mt-4 p-3 bg-red-50 rounded-lg">
          <Text className="text-red-600 text-sm">{status.error_message}</Text>
        </div>
      )}
    </Card>
  );
}

export function ModelFitPage() {
  const { data: datasetsData, isLoading: datasetsLoading } = useDatasets();
  const { data: configsData, isLoading: configsLoading } = useConfigs();
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const { selectedDataId, setSelectedData, selectedConfigId, setSelectedConfig } = useProjectStore();
  const submitMutation = useSubmitFitJob();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const isLoading = datasetsLoading || configsLoading || modelsLoading;

  if (isLoading) {
    return <LoadingPage message="Loading..." />;
  }

  const datasets = datasetsData?.datasets || [];
  const configs = configsData?.configs || [];
  const models = modelsData?.models || [];

  const handleSubmit = () => {
    if (!selectedDataId || !selectedConfigId) {
      alert('Please select both a dataset and configuration');
      return;
    }

    submitMutation.mutate(
      {
        data_id: selectedDataId,
        config_id: selectedConfigId,
        name: `Model ${new Date().toISOString().slice(0, 10)}`,
      },
      {
        onSuccess: (data) => {
          setActiveJobId(data.model_id);
        },
      }
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <Title>Model Fitting</Title>
        <Text>Submit model fitting jobs and monitor progress</Text>
      </div>

      {/* Submit form */}
      <Card>
        <Title className="text-sm">Submit New Job</Title>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dataset
            </label>
            <Select
              value={selectedDataId || ''}
              onValueChange={(v) => setSelectedData(v)}
              placeholder="Select dataset..."
            >
              {datasets.map((d) => (
                <SelectItem key={d.data_id} value={d.data_id}>
                  {d.filename}
                </SelectItem>
              ))}
            </Select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Configuration
            </label>
            <Select
              value={selectedConfigId || ''}
              onValueChange={(v) => setSelectedConfig(v)}
              placeholder="Select configuration..."
            >
              {configs.map((c) => (
                <SelectItem key={c.config_id} value={c.config_id}>
                  {c.name}
                </SelectItem>
              ))}
            </Select>
          </div>
        </div>
        <div className="mt-4">
          <Button
            icon={PlayIcon}
            onClick={handleSubmit}
            loading={submitMutation.isPending}
            disabled={!selectedDataId || !selectedConfigId}
          >
            Start Fitting
          </Button>
        </div>
      </Card>

      {/* Active job monitor */}
      {activeJobId && <JobMonitor modelId={activeJobId} />}

      {/* Recent jobs */}
      <Card>
        <Title className="text-sm">Recent Jobs</Title>
        {models.length === 0 ? (
          <Text className="mt-4 text-gray-500">No jobs submitted yet.</Text>
        ) : (
          <div className="mt-4 space-y-3">
            {models.slice(0, 10).map((model) => (
              <div
                key={model.model_id}
                className="p-4 bg-gray-50 rounded-lg flex justify-between items-center cursor-pointer hover:bg-gray-100"
                onClick={() => setActiveJobId(model.model_id)}
              >
                <div>
                  <Text className="font-medium">{model.name || model.model_id}</Text>
                  <Text className="text-xs text-gray-500">
                    Created {new Date(model.created_at).toLocaleString()}
                  </Text>
                </div>
                <div className="flex items-center gap-4">
                  {(model.status === 'running' || model.status === 'queued') && (
                    <div className="w-24">
                      <ProgressBar value={model.progress} color="blue" />
                    </div>
                  )}
                  <JobStatusBadge status={model.status} />
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

export default ModelFitPage;
