import { Card, Title, Text, Button, Badge } from '@tremor/react';
import { Cog6ToothIcon, PlusIcon, PencilIcon, TrashIcon } from '@heroicons/react/24/outline';
import { useConfigs, useDeleteConfig } from '../../api/hooks';
import { useConfigWizardStore } from '../../stores/configWizardStore';
import { useProjectStore } from '../../stores/projectStore';
import { LoadingPage } from '../../components/common/LoadingSpinner';
import { ConfigWizard } from './components/ConfigWizard';
import { transformFromConfig } from './utils/transformers';
import type { ConfigInfo } from '../../api/types';

function ConfigCard({
  config,
  isSelected,
  onSelect,
  onEdit,
  onDelete,
}: {
  config: ConfigInfo;
  isSelected: boolean;
  onSelect: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const mediaChannelCount = config.mff_config?.media_channels?.length || 0;
  const controlCount = config.mff_config?.controls?.length || 0;
  const kpiName = config.mff_config?.kpi?.name || 'Not set';

  return (
    <div
      className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${
        isSelected
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-200 hover:border-gray-300'
      }`}
      onClick={onSelect}
    >
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <Text className="font-medium">{config.name}</Text>
          {config.description && (
            <Text className="text-xs text-gray-500 mt-1">{config.description}</Text>
          )}
        </div>
        <Badge color="blue">
          {config.model_settings?.inference_method === 'bayesian_numpyro'
            ? 'NumPyro'
            : 'PyMC'}
        </Badge>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <Badge color="gray">KPI: {kpiName}</Badge>
        <Badge color="green">{mediaChannelCount} channels</Badge>
        <Badge color="purple">{controlCount} controls</Badge>
      </div>

      <div className="mt-3 flex items-center justify-between">
        <Text className="text-xs text-gray-500">
          Created: {new Date(config.created_at).toLocaleDateString()}
        </Text>
        <div className="flex gap-2">
          <Button
            size="xs"
            variant="secondary"
            icon={PencilIcon}
            onClick={(e) => {
              e.stopPropagation();
              onEdit();
            }}
          >
            Edit
          </Button>
          <Button
            size="xs"
            variant="secondary"
            color="red"
            icon={TrashIcon}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            Delete
          </Button>
        </div>
      </div>
    </div>
  );
}

export function ModelConfigPage() {
  const { data, isLoading } = useConfigs();
  const deleteConfig = useDeleteConfig();
  const { openWizard } = useConfigWizardStore();
  const { selectedConfigId, setSelectedConfig } = useProjectStore();

  const configs = data?.configs || [];

  const handleNewConfig = () => {
    openWizard();
  };

  const handleEditConfig = (config: ConfigInfo) => {
    const draftData = transformFromConfig(config);
    openWizard(config.config_id, draftData);
  };

  const handleDeleteConfig = (configId: string) => {
    if (confirm('Are you sure you want to delete this configuration?')) {
      deleteConfig.mutate(configId, {
        onSuccess: () => {
          if (selectedConfigId === configId) {
            setSelectedConfig(null);
          }
        },
      });
    }
  };

  if (isLoading) {
    return <LoadingPage message="Loading configurations..." />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <Title>Model Configuration</Title>
          <Text>Create and manage model configurations</Text>
        </div>
        <Button icon={PlusIcon} onClick={handleNewConfig}>
          New Configuration
        </Button>
      </div>

      {/* Selected Config Banner */}
      {selectedConfigId && (
        <Card className="p-4 bg-blue-50 border-blue-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cog6ToothIcon className="h-5 w-5 text-blue-600" />
              <Text className="text-blue-900">
                Selected configuration:{' '}
                <strong>
                  {configs.find((c) => c.config_id === selectedConfigId)?.name || 'Unknown'}
                </strong>
              </Text>
            </div>
            <Button
              size="xs"
              variant="secondary"
              onClick={() => setSelectedConfig(null)}
            >
              Clear Selection
            </Button>
          </div>
        </Card>
      )}

      {/* Configurations Grid */}
      <Card className="p-4">
        <Title className="text-sm mb-4">
          Existing Configurations ({configs.length})
        </Title>

        {configs.length === 0 ? (
          <div className="text-center py-12">
            <Cog6ToothIcon className="mx-auto h-12 w-12 text-gray-400" />
            <Text className="mt-4 text-gray-500">
              No configurations yet. Create a new configuration to get started.
            </Text>
            <Button icon={PlusIcon} onClick={handleNewConfig} className="mt-4">
              Create First Configuration
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {configs.map((config) => (
              <ConfigCard
                key={config.config_id}
                config={config}
                isSelected={selectedConfigId === config.config_id}
                onSelect={() => setSelectedConfig(config.config_id)}
                onEdit={() => handleEditConfig(config)}
                onDelete={() => handleDeleteConfig(config.config_id)}
              />
            ))}
          </div>
        )}
      </Card>

      {/* Quick Actions */}
      {selectedConfigId && (
        <Card className="p-4">
          <Title className="text-sm mb-4">Quick Actions</Title>
          <div className="flex gap-3">
            <Button
              variant="secondary"
              onClick={() => {
                const config = configs.find((c) => c.config_id === selectedConfigId);
                if (config) handleEditConfig(config);
              }}
            >
              Edit Selected
            </Button>
            <Button
              variant="primary"
              onClick={() => {
                // Navigate to model fitting
                window.location.href = '/fit';
              }}
            >
              Use for Model Fitting
            </Button>
          </div>
        </Card>
      )}

      {/* Help Section */}
      <Card className="p-4 bg-gray-50">
        <Title className="text-sm">Configuration Guide</Title>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
          <div>
            <Text className="font-medium text-gray-900">1. KPI Setup</Text>
            <Text className="text-gray-500">
              Define your target variable and its dimensions (geography, product, etc.)
            </Text>
          </div>
          <div>
            <Text className="font-medium text-gray-900">2. Media Channels</Text>
            <Text className="text-gray-500">
              Configure marketing channels with adstock and saturation settings
            </Text>
          </div>
          <div>
            <Text className="font-medium text-gray-900">3. Model Settings</Text>
            <Text className="text-gray-500">
              Set MCMC parameters, trends, seasonality, and hierarchical options
            </Text>
          </div>
        </div>
      </Card>

      {/* Wizard Modal */}
      <ConfigWizard />
    </div>
  );
}

export default ModelConfigPage;
