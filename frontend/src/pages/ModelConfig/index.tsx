import { Card, Title, Text, Button } from '@tremor/react';
import { Cog6ToothIcon } from '@heroicons/react/24/outline';
import { useConfigs } from '../../api/hooks';
import { LoadingPage } from '../../components/common/LoadingSpinner';

export function ModelConfigPage() {
  const { data, isLoading } = useConfigs();

  if (isLoading) {
    return <LoadingPage message="Loading configurations..." />;
  }

  const configs = data?.configs || [];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <Title>Model Configuration</Title>
          <Text>Create and manage model configurations</Text>
        </div>
        <Button icon={Cog6ToothIcon}>New Configuration</Button>
      </div>

      <Card>
        <Title className="text-sm">Existing Configurations</Title>
        {configs.length === 0 ? (
          <Text className="mt-4 text-gray-500">
            No configurations yet. Create a new configuration to get started.
          </Text>
        ) : (
          <div className="mt-4 space-y-3">
            {configs.map((config) => (
              <div
                key={config.config_id}
                className="p-4 bg-gray-50 rounded-lg flex justify-between items-center"
              >
                <div>
                  <Text className="font-medium">{config.name}</Text>
                  <Text className="text-xs text-gray-500">
                    {config.mff_config.media_channels.length} channels,{' '}
                    {config.mff_config.controls.length} controls
                  </Text>
                </div>
                <Button size="xs" variant="secondary">
                  Edit
                </Button>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <Title className="text-sm">Configuration Wizard</Title>
        <Text className="mt-2 text-gray-500">
          The full configuration wizard with KPI selection, media channel setup, control variables,
          and MCMC settings will be implemented here.
        </Text>
        <div className="mt-4 p-8 border-2 border-dashed border-gray-300 rounded-lg text-center text-gray-400">
          Configuration wizard coming soon...
        </div>
      </Card>
    </div>
  );
}

export default ModelConfigPage;
