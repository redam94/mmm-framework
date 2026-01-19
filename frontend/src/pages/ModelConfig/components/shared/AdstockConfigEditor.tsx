import { useState } from 'react';
import { Card, Title, Select, SelectItem, NumberInput, Switch, Text } from '@tremor/react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import type { AdstockConfig, AdstockType } from '../../../../api/types';
import { PriorConfigEditor } from './PriorConfigEditor';

const ADSTOCK_TYPES: AdstockType[] = ['geometric', 'weibull', 'delayed', 'none'];

const ADSTOCK_TYPE_LABELS: Record<AdstockType, string> = {
  geometric: 'Geometric',
  weibull: 'Weibull',
  delayed: 'Delayed',
  none: 'None',
};

const ADSTOCK_TYPE_DESCRIPTIONS: Record<AdstockType, string> = {
  geometric: 'Simple exponential decay - most common choice',
  weibull: 'Flexible decay with shape and scale parameters',
  delayed: 'Peak effect occurs after a delay',
  none: 'No carryover effect',
};

interface AdstockConfigEditorProps {
  value: AdstockConfig;
  onChange: (value: AdstockConfig) => void;
  label?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

export function AdstockConfigEditor({
  value,
  onChange,
  label = 'Adstock Configuration',
  collapsible = true,
  defaultCollapsed = true,
}: AdstockConfigEditorProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  const handleTypeChange = (newType: string) => {
    const type = newType as AdstockType;
    const updated: AdstockConfig = {
      ...value,
      type,
    };

    // Set default priors based on type
    if (type === 'geometric') {
      updated.alpha_prior = { type: 'beta', params: { alpha: 2, beta: 2 } };
      delete updated.theta_prior;
    } else if (type === 'weibull') {
      updated.alpha_prior = { type: 'gamma', params: { alpha: 2, beta: 1 } };
      updated.theta_prior = { type: 'gamma', params: { alpha: 2, beta: 1 } };
    } else if (type === 'delayed') {
      updated.alpha_prior = { type: 'beta', params: { alpha: 2, beta: 2 } };
      updated.theta_prior = { type: 'gamma', params: { alpha: 2, beta: 1 } };
    } else {
      delete updated.alpha_prior;
      delete updated.theta_prior;
    }

    onChange(updated);
  };

  const content = (
    <div className="space-y-4">
      {/* Type selector */}
      <div>
        <label className="text-xs font-medium text-gray-600">Adstock Type</label>
        <Select value={value.type} onValueChange={handleTypeChange} className="mt-1">
          {ADSTOCK_TYPES.map((type) => (
            <SelectItem key={type} value={type}>
              {ADSTOCK_TYPE_LABELS[type]}
            </SelectItem>
          ))}
        </Select>
        <Text className="text-xs text-gray-500 mt-1">
          {ADSTOCK_TYPE_DESCRIPTIONS[value.type]}
        </Text>
      </div>

      {value.type !== 'none' && (
        <>
          {/* Max Lag */}
          <div>
            <label className="text-xs font-medium text-gray-600">Max Lag (weeks)</label>
            <NumberInput
              value={value.l_max}
              onValueChange={(val) => onChange({ ...value, l_max: val ?? 8 })}
              min={1}
              max={52}
              step={1}
              className="mt-1"
            />
            <Text className="text-xs text-gray-500 mt-1">
              Maximum carryover effect duration
            </Text>
          </div>

          {/* Normalize switch */}
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Normalize</label>
              <Text className="text-xs text-gray-500">
                Scale adstock weights to sum to 1
              </Text>
            </div>
            <Switch
              checked={value.normalize ?? true}
              onChange={(checked) => onChange({ ...value, normalize: checked })}
            />
          </div>

          {/* Prior configurations based on type */}
          {value.type === 'geometric' && value.alpha_prior && (
            <PriorConfigEditor
              label="Decay Rate Prior (alpha)"
              value={value.alpha_prior}
              onChange={(prior) => onChange({ ...value, alpha_prior: prior })}
              defaultCollapsed={true}
            />
          )}

          {value.type === 'weibull' && (
            <>
              {value.alpha_prior && (
                <PriorConfigEditor
                  label="Shape Prior (alpha)"
                  value={value.alpha_prior}
                  onChange={(prior) => onChange({ ...value, alpha_prior: prior })}
                  defaultCollapsed={true}
                />
              )}
              {value.theta_prior && (
                <PriorConfigEditor
                  label="Scale Prior (theta)"
                  value={value.theta_prior}
                  onChange={(prior) => onChange({ ...value, theta_prior: prior })}
                  defaultCollapsed={true}
                />
              )}
            </>
          )}

          {value.type === 'delayed' && (
            <>
              {value.alpha_prior && (
                <PriorConfigEditor
                  label="Decay Prior (alpha)"
                  value={value.alpha_prior}
                  onChange={(prior) => onChange({ ...value, alpha_prior: prior })}
                  defaultCollapsed={true}
                />
              )}
              {value.theta_prior && (
                <PriorConfigEditor
                  label="Delay Prior (theta)"
                  value={value.theta_prior}
                  onChange={(prior) => onChange({ ...value, theta_prior: prior })}
                  defaultCollapsed={true}
                />
              )}
            </>
          )}
        </>
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
          <Text className="text-xs text-gray-500">{ADSTOCK_TYPE_LABELS[value.type]}</Text>
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

export default AdstockConfigEditor;
