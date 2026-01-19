import { Card, Title, Text, Metric, Grid, Flex, Badge, ProgressBar } from '@tremor/react';
import {
  CircleStackIcon,
  Cog6ToothIcon,
  CubeIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import { useDatasets, useConfigs, useModels, useHealthDetailed } from '../../api/hooks';
import { LoadingPage } from '../../components/common/LoadingSpinner';

// Stat card component
function StatCard({
  title,
  value,
  icon: Icon,
  description,
}: {
  title: string;
  value: number | string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  description?: string;
}) {
  return (
    <Card decoration="top" decorationColor="blue">
      <Flex justifyContent="start" className="space-x-4">
        <Icon className="h-8 w-8 text-blue-500" />
        <div>
          <Text>{title}</Text>
          <Metric>{value}</Metric>
          {description && <Text className="text-xs text-gray-500">{description}</Text>}
        </div>
      </Flex>
    </Card>
  );
}

// Status badge for job status
function JobStatusBadge({ status }: { status: string }) {
  const colorMap: Record<string, 'green' | 'yellow' | 'red' | 'blue' | 'gray'> = {
    completed: 'green',
    running: 'blue',
    queued: 'yellow',
    pending: 'gray',
    failed: 'red',
  };

  return (
    <Badge color={colorMap[status] || 'gray'} size="sm">
      {status}
    </Badge>
  );
}

export function DashboardPage() {
  const { data: datasetsData, isLoading: datasetsLoading } = useDatasets();
  const { data: configsData, isLoading: configsLoading } = useConfigs();
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const { data: healthData } = useHealthDetailed();

  const isLoading = datasetsLoading || configsLoading || modelsLoading;

  if (isLoading) {
    return <LoadingPage message="Loading dashboard..." />;
  }

  const datasets = datasetsData?.datasets || [];
  const configs = configsData?.configs || [];
  const models = modelsData?.models || [];

  // Compute stats
  const runningModels = models.filter((m) => m.status === 'running' || m.status === 'queued');
  const completedModels = models.filter((m) => m.status === 'completed');
  const failedModels = models.filter((m) => m.status === 'failed');

  // Recent models (last 5)
  const recentModels = [...models]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Quick stats */}
      <Grid numItemsSm={2} numItemsLg={4} className="gap-6">
        <StatCard
          title="Datasets"
          value={datasets.length}
          icon={CircleStackIcon}
          description="Uploaded data files"
        />
        <StatCard
          title="Configurations"
          value={configs.length}
          icon={Cog6ToothIcon}
          description="Model configurations"
        />
        <StatCard
          title="Models"
          value={models.length}
          icon={CubeIcon}
          description={`${completedModels.length} completed`}
        />
        <StatCard
          title="Active Jobs"
          value={runningModels.length}
          icon={ClockIcon}
          description={failedModels.length > 0 ? `${failedModels.length} failed` : undefined}
        />
      </Grid>

      {/* Server status */}
      {healthData && (
        <Card>
          <Title>Server Status</Title>
          <div className="mt-4 space-y-3">
            <Flex>
              <Text>Redis</Text>
              <Badge color={healthData.redis_connected ? 'green' : 'red'}>
                {healthData.redis_connected ? 'Connected' : 'Disconnected'}
              </Badge>
            </Flex>
            <Flex>
              <Text>Worker</Text>
              <Badge color={healthData.worker_healthy ? 'green' : 'red'}>
                {healthData.worker_healthy ? 'Healthy' : 'Unhealthy'}
              </Badge>
            </Flex>
            {healthData.queue_stats && (
              <>
                <Flex>
                  <Text>Queue</Text>
                  <Text>
                    {healthData.queue_stats.pending} pending,{' '}
                    {healthData.queue_stats.running} running
                  </Text>
                </Flex>
              </>
            )}
          </div>
        </Card>
      )}

      {/* Recent models */}
      <Card>
        <Title>Recent Models</Title>
        {recentModels.length === 0 ? (
          <Text className="mt-4 text-gray-500">
            No models yet. Start by uploading data and creating a configuration.
          </Text>
        ) : (
          <div className="mt-4 space-y-4">
            {recentModels.map((model) => (
              <div
                key={model.model_id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div>
                  <Text className="font-medium">{model.name || model.model_id}</Text>
                  <Text className="text-xs text-gray-500">
                    Created {new Date(model.created_at).toLocaleDateString()}
                  </Text>
                </div>
                <div className="flex items-center gap-4">
                  {model.status === 'running' && (
                    <div className="w-32">
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

      {/* Quick actions */}
      <Card>
        <Title>Quick Actions</Title>
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-4">
          <a
            href="/data"
            className="p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors text-center"
          >
            <CircleStackIcon className="h-8 w-8 mx-auto text-blue-600" />
            <Text className="mt-2 font-medium">Upload Data</Text>
          </a>
          <a
            href="/config"
            className="p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors text-center"
          >
            <Cog6ToothIcon className="h-8 w-8 mx-auto text-green-600" />
            <Text className="mt-2 font-medium">Create Config</Text>
          </a>
          <a
            href="/fit"
            className="p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors text-center"
          >
            <CubeIcon className="h-8 w-8 mx-auto text-purple-600" />
            <Text className="mt-2 font-medium">Fit Model</Text>
          </a>
        </div>
      </Card>
    </div>
  );
}

export default DashboardPage;
