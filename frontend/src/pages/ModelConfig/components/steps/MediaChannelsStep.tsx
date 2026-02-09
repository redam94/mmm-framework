import { useEffect } from 'react';
import { Card, Title, Text, Button } from '@tremor/react';
import { PlusIcon } from '@heroicons/react/24/outline';
import { useConfigWizardStore } from '../../../../stores/configWizardStore';
import { useDatasetVariables } from '../../../../api/hooks';
import { useProjectStore } from '../../../../stores/projectStore';
import { MediaChannelCard } from '../shared';
import { LoadingSpinner } from '../../../../components/common/LoadingSpinner';
import { DEFAULT_MEDIA_CHANNEL } from '../../types/wizardSchema';
import type { MediaChannelConfig } from '../../../../api/types';

export function MediaChannelsStep() {
  const { draftConfig, updateDraft, setStepValidation } = useConfigWizardStore();
  const { selectedDataId } = useProjectStore();
  const { data: variablesData, isLoading: variablesLoading } = useDatasetVariables(selectedDataId || undefined);

  const variables = variablesData?.map((v) => v.name) || [];
  const mediaChannels = draftConfig.media_channels || [];

  // Validate step on changes
  useEffect(() => {
    const hasChannels = mediaChannels.length > 0;
    const allChannelsNamed = mediaChannels.every((ch) => ch.name && ch.name.trim() !== '');
    const isValid = hasChannels && allChannelsNamed;

    const errors: string[] = [];
    if (!hasChannels) {
      errors.push('At least one media channel is required');
    } else if (!allChannelsNamed) {
      errors.push('All channels must have a variable selected');
    }

    setStepValidation('media_channels', isValid, errors);
  }, [mediaChannels, setStepValidation]);

  const handleAddChannel = () => {
    const newChannel: MediaChannelConfig = {
      ...DEFAULT_MEDIA_CHANNEL,
      name: '',
    };
    updateDraft({ media_channels: [...mediaChannels, newChannel] });
  };

  const handleUpdateChannel = (index: number, updated: MediaChannelConfig) => {
    const newChannels = [...mediaChannels];
    newChannels[index] = updated;
    updateDraft({ media_channels: newChannels });
  };

  const handleRemoveChannel = (index: number) => {
    const newChannels = mediaChannels.filter((_, i) => i !== index);
    updateDraft({ media_channels: newChannels });
  };

  // Get other channel names for parent selection
  const getOtherChannelNames = (currentIndex: number) => {
    return mediaChannels
      .filter((_, i) => i !== currentIndex && mediaChannels[i].name)
      .map((ch) => ch.name);
  };

  if (variablesLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner />
        <Text className="ml-2">Loading variables...</Text>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <Title className="text-sm">Media Channels</Title>
            <Text className="text-gray-600">
              Configure the media/marketing channels to include in your model.
            </Text>
          </div>
          <Button icon={PlusIcon} onClick={handleAddChannel}>
            Add Channel
          </Button>
        </div>
      </Card>

      {/* Channel List */}
      {mediaChannels.length === 0 ? (
        <Card className="p-8 text-center border-2 border-dashed">
          <Title className="text-gray-500">No channels added</Title>
          <Text className="text-gray-400 mt-2">
            Click "Add Channel" to add your first media channel.
          </Text>
          <Button icon={PlusIcon} onClick={handleAddChannel} className="mt-4">
            Add Channel
          </Button>
        </Card>
      ) : (
        <div className="space-y-4">
          {mediaChannels.map((channel, index) => (
            <MediaChannelCard
              key={index}
              value={channel}
              onChange={(updated) => handleUpdateChannel(index, updated)}
              onRemove={() => handleRemoveChannel(index)}
              variables={variables}
              otherChannels={getOtherChannelNames(index)}
              index={index}
            />
          ))}
        </div>
      )}

      {/* Add More Button */}
      {mediaChannels.length > 0 && (
        <Button variant="secondary" icon={PlusIcon} onClick={handleAddChannel}>
          Add Another Channel
        </Button>
      )}

      {/* Help Section */}
      <Card className="p-4 bg-blue-50 border-blue-200">
        <Title className="text-sm text-blue-900">About Media Channels</Title>
        <div className="mt-2 text-sm text-blue-800 space-y-2">
          <p>
            <strong>Adstock:</strong> Models the carryover effect of advertising. Geometric decay is
            most common - higher alpha means faster decay.
          </p>
          <p>
            <strong>Saturation:</strong> Models diminishing returns. The Hill function is most
            flexible, allowing for various S-curve shapes.
          </p>
          <p>
            <strong>Parent Channel:</strong> Use this to group related channels (e.g., group
            "TV_National" and "TV_Regional" under a "TV" parent).
          </p>
        </div>
      </Card>
    </div>
  );
}

export default MediaChannelsStep;
