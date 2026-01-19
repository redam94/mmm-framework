import { Card, Title, TextInput, Text, Button, Select, SelectItem } from '@tremor/react';
import { TrashIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { useState } from 'react';
import type { MediaChannelConfig, ControlVariableConfig, DimensionType } from '../../../../api/types';
import { VariableSelector } from './VariableSelector';
import { DimensionSelector } from './DimensionSelector';
import { AdstockConfigEditor } from './AdstockConfigEditor';
import { SaturationConfigEditor } from './SaturationConfigEditor';
import { PriorConfigEditor } from './PriorConfigEditor';

// ============================================================================
// Media Channel Card
// ============================================================================

interface MediaChannelCardProps {
  value: MediaChannelConfig;
  onChange: (value: MediaChannelConfig) => void;
  onRemove: () => void;
  variables: string[];
  otherChannels: string[];
  index: number;
}

export function MediaChannelCard({
  value,
  onChange,
  onRemove,
  variables,
  otherChannels,
  index,
}: MediaChannelCardProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <Card className="p-4 border-2 border-gray-200">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2 text-left flex-1"
        >
          <Title className="text-sm">
            Channel {index + 1}: {value.name || 'Unnamed'}
          </Title>
          {isExpanded ? (
            <ChevronUpIcon className="h-4 w-4 text-gray-500" />
          ) : (
            <ChevronDownIcon className="h-4 w-4 text-gray-500" />
          )}
        </button>
        <Button
          size="xs"
          variant="secondary"
          color="red"
          icon={TrashIcon}
          onClick={onRemove}
        >
          Remove
        </Button>
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {/* Basic Info */}
          <div className="grid grid-cols-2 gap-4">
            <VariableSelector
              label="Variable Name"
              value={value.name}
              onChange={(name) => onChange({ ...value, name })}
              variables={variables}
              required
            />
            <div>
              <label className="text-sm font-medium text-gray-700">Display Name</label>
              <TextInput
                value={value.display_name || ''}
                onChange={(e) => onChange({ ...value, display_name: e.target.value })}
                placeholder="Optional display name"
                className="mt-1"
              />
            </div>
          </div>

          {/* Dimensions */}
          <DimensionSelector
            label="Dimensions"
            value={value.dimensions as DimensionType[]}
            onChange={(dimensions) => onChange({ ...value, dimensions })}
            required
          />

          {/* Parent Channel (optional) */}
          {otherChannels.length > 0 && (
            <div>
              <label className="text-sm font-medium text-gray-700">Parent Channel (optional)</label>
              <Select
                value={value.parent_channel || ''}
                onValueChange={(parent) =>
                  onChange({ ...value, parent_channel: parent || undefined })
                }
                placeholder="None"
                className="mt-1"
              >
                <SelectItem value="">None</SelectItem>
                {otherChannels.map((ch) => (
                  <SelectItem key={ch} value={ch}>
                    {ch}
                  </SelectItem>
                ))}
              </Select>
              <Text className="text-xs text-gray-500 mt-1">
                Group this channel under a parent for hierarchical reporting
              </Text>
            </div>
          )}

          {/* Adstock Configuration */}
          <AdstockConfigEditor
            value={value.adstock}
            onChange={(adstock) => onChange({ ...value, adstock })}
            defaultCollapsed={true}
          />

          {/* Saturation Configuration */}
          <SaturationConfigEditor
            value={value.saturation}
            onChange={(saturation) => onChange({ ...value, saturation })}
            defaultCollapsed={true}
          />

          {/* Coefficient Prior */}
          {value.coefficient_prior && (
            <PriorConfigEditor
              label="Coefficient Prior"
              value={value.coefficient_prior}
              onChange={(prior) => onChange({ ...value, coefficient_prior: prior })}
              defaultCollapsed={true}
            />
          )}
        </div>
      )}
    </Card>
  );
}

// ============================================================================
// Control Variable Card
// ============================================================================

interface ControlVariableCardProps {
  value: ControlVariableConfig;
  onChange: (value: ControlVariableConfig) => void;
  onRemove: () => void;
  variables: string[];
  index: number;
}

export function ControlVariableCard({
  value,
  onChange,
  onRemove,
  variables,
  index,
}: ControlVariableCardProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <Card className="p-4 border-2 border-gray-200">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2 text-left flex-1"
        >
          <Title className="text-sm">
            Control {index + 1}: {value.name || 'Unnamed'}
          </Title>
          {isExpanded ? (
            <ChevronUpIcon className="h-4 w-4 text-gray-500" />
          ) : (
            <ChevronDownIcon className="h-4 w-4 text-gray-500" />
          )}
        </button>
        <Button
          size="xs"
          variant="secondary"
          color="red"
          icon={TrashIcon}
          onClick={onRemove}
        >
          Remove
        </Button>
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {/* Basic Info */}
          <div className="grid grid-cols-2 gap-4">
            <VariableSelector
              label="Variable Name"
              value={value.name}
              onChange={(name) => onChange({ ...value, name })}
              variables={variables}
              required
            />
            <div>
              <label className="text-sm font-medium text-gray-700">Display Name</label>
              <TextInput
                value={value.display_name || ''}
                onChange={(e) => onChange({ ...value, display_name: e.target.value })}
                placeholder="Optional display name"
                className="mt-1"
              />
            </div>
          </div>

          {/* Dimensions */}
          <DimensionSelector
            label="Dimensions"
            value={value.dimensions as DimensionType[]}
            onChange={(dimensions) => onChange({ ...value, dimensions })}
            required
          />

          {/* Options */}
          <div className="flex gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={value.allow_negative}
                onChange={(e) => onChange({ ...value, allow_negative: e.target.checked })}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Allow negative effect</span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={value.use_shrinkage}
                onChange={(e) => onChange({ ...value, use_shrinkage: e.target.checked })}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Use shrinkage prior</span>
            </label>
          </div>

          {/* Coefficient Prior */}
          {value.coefficient_prior && (
            <PriorConfigEditor
              label="Coefficient Prior"
              value={value.coefficient_prior}
              onChange={(prior) => onChange({ ...value, coefficient_prior: prior })}
              defaultCollapsed={true}
            />
          )}
        </div>
      )}
    </Card>
  );
}

export default MediaChannelCard;
