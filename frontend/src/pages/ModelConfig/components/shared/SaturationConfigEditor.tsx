import { useState } from 'react';
import { Card, Title, Select, SelectItem, Text } from '@tremor/react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import type { SaturationConfig, SaturationType } from '../../../../api/types';
import { PriorConfigEditor } from './PriorConfigEditor';

const SATURATION_TYPES: SaturationType[] = ['hill', 'logistic', 'michaelis_menten', 'tanh', 'none'];

const SATURATION_TYPE_LABELS: Record<SaturationType, string> = {
  hill: 'Hill',
  logistic: 'Logistic',
  michaelis_menten: 'Michaelis-Menten',
  tanh: 'Tanh',
  none: 'None (Linear)',
};

const SATURATION_TYPE_DESCRIPTIONS: Record<SaturationType, string> = {
  hill: 'S-shaped curve with half-saturation point (most flexible)',
  logistic: 'Classic S-curve with upper bound',
  michaelis_menten: 'Simpler saturation curve (special case of Hill)',
  tanh: 'Hyperbolic tangent saturation',
  none: 'No saturation - linear relationship',
};

interface SaturationConfigEditorProps {
  value: SaturationConfig;
  onChange: (value: SaturationConfig) => void;
  label?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

export function SaturationConfigEditor({
  value,
  onChange,
  label = 'Saturation Configuration',
  collapsible = true,
  defaultCollapsed = true,
}: SaturationConfigEditorProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  const handleTypeChange = (newType: string) => {
    const type = newType as SaturationType;
    const updated: SaturationConfig = {
      ...value,
      type,
    };

    // Set default priors based on type
    if (type === 'none') {
      delete updated.kappa_prior;
      delete updated.slope_prior;
      delete updated.beta_prior;
    } else if (type === 'hill') {
      updated.kappa_prior = { type: 'gamma', params: { alpha: 2, beta: 1 } };
      updated.slope_prior = { type: 'gamma', params: { alpha: 2, beta: 1 } };
      delete updated.beta_prior;
    } else if (type === 'michaelis_menten') {
      updated.kappa_prior = { type: 'gamma', params: { alpha: 2, beta: 1 } };
      delete updated.slope_prior;
      delete updated.beta_prior;
    } else {
      updated.beta_prior = { type: 'gamma', params: { alpha: 2, beta: 1 } };
      delete updated.kappa_prior;
      delete updated.slope_prior;
    }

    onChange(updated);
  };

  const content = (
    <div className="space-y-4">
      {/* Type selector */}
      <div>
        <label className="text-xs font-medium text-gray-600">Saturation Type</label>
        <Select value={value.type} onValueChange={handleTypeChange} className="mt-1">
          {SATURATION_TYPES.map((type) => (
            <SelectItem key={type} value={type}>
              {SATURATION_TYPE_LABELS[type]}
            </SelectItem>
          ))}
        </Select>
        <Text className="text-xs text-gray-500 mt-1">
          {SATURATION_TYPE_DESCRIPTIONS[value.type]}
        </Text>
      </div>

      {/* Prior configurations based on type */}
      {value.type === 'hill' && (
        <>
          {value.kappa_prior && (
            <PriorConfigEditor
              label="Half-Saturation Prior (kappa)"
              value={value.kappa_prior}
              onChange={(prior) => onChange({ ...value, kappa_prior: prior })}
              defaultCollapsed={true}
            />
          )}
          {value.slope_prior && (
            <PriorConfigEditor
              label="Slope Prior"
              value={value.slope_prior}
              onChange={(prior) => onChange({ ...value, slope_prior: prior })}
              defaultCollapsed={true}
            />
          )}
        </>
      )}

      {value.type === 'michaelis_menten' && value.kappa_prior && (
        <PriorConfigEditor
          label="Half-Saturation Prior (kappa)"
          value={value.kappa_prior}
          onChange={(prior) => onChange({ ...value, kappa_prior: prior })}
          defaultCollapsed={true}
        />
      )}

      {(value.type === 'logistic' || value.type === 'tanh') && value.beta_prior && (
        <PriorConfigEditor
          label="Scale Prior (beta)"
          value={value.beta_prior}
          onChange={(prior) => onChange({ ...value, beta_prior: prior })}
          defaultCollapsed={true}
        />
      )}
    </div>
  );

  if (!collapsible) {
    return (
      <div className="space-y-2">
        <Text className="text-sm font-medium text-gray-700">{label}</Text>
        {content}
      </div>
    );
  }

  return (
    <Card className="p-3 bg-gray-50">
      <button
        type="button"
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between text-left"
      >
        <Title className="text-sm">{label}</Title>
        <div className="flex items-center gap-2">
          <Text className="text-xs text-gray-500">{SATURATION_TYPE_LABELS[value.type]}</Text>
          {isCollapsed ? (
            <ChevronDownIcon className="h-4 w-4 text-gray-500" />
          ) : (
            <ChevronUpIcon className="h-4 w-4 text-gray-500" />
          )}
        </div>
      </button>

      {!isCollapsed && <div className="mt-3 pt-3 border-t">{content}</div>}
    </Card>
  );
}

export default SaturationConfigEditor;
