import { useState } from 'react';
import { Card, Title, Select, SelectItem, NumberInput, Text } from '@tremor/react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import type { PriorConfig, PriorType } from '../../../../api/types';

const PRIOR_TYPES: PriorType[] = [
  'half_normal',
  'normal',
  'log_normal',
  'gamma',
  'beta',
  'truncated_normal',
  'half_student_t',
];

const PRIOR_TYPE_LABELS: Record<PriorType, string> = {
  half_normal: 'Half Normal',
  normal: 'Normal',
  log_normal: 'Log Normal',
  gamma: 'Gamma',
  beta: 'Beta',
  truncated_normal: 'Truncated Normal',
  half_student_t: 'Half Student-t',
};

interface PriorParam {
  name: string;
  label: string;
  default: number;
  min?: number;
  max?: number;
  step?: number;
}

const PRIOR_PARAMS: Record<PriorType, PriorParam[]> = {
  half_normal: [{ name: 'sigma', label: 'Sigma', default: 1, min: 0.001, step: 0.1 }],
  normal: [
    { name: 'mu', label: 'Mean (mu)', default: 0, step: 0.1 },
    { name: 'sigma', label: 'Sigma', default: 1, min: 0.001, step: 0.1 },
  ],
  log_normal: [
    { name: 'mu', label: 'Mu', default: 0, step: 0.1 },
    { name: 'sigma', label: 'Sigma', default: 1, min: 0.001, step: 0.1 },
  ],
  gamma: [
    { name: 'alpha', label: 'Alpha (shape)', default: 2, min: 0.001, step: 0.1 },
    { name: 'beta', label: 'Beta (rate)', default: 1, min: 0.001, step: 0.1 },
  ],
  beta: [
    { name: 'alpha', label: 'Alpha', default: 2, min: 0.001, step: 0.1 },
    { name: 'beta', label: 'Beta', default: 2, min: 0.001, step: 0.1 },
  ],
  truncated_normal: [
    { name: 'mu', label: 'Mean (mu)', default: 0, step: 0.1 },
    { name: 'sigma', label: 'Sigma', default: 1, min: 0.001, step: 0.1 },
    { name: 'lower', label: 'Lower bound', default: 0, step: 0.1 },
    { name: 'upper', label: 'Upper bound', default: 10, step: 0.1 },
  ],
  half_student_t: [
    { name: 'nu', label: 'Nu (df)', default: 3, min: 1, step: 1 },
    { name: 'sigma', label: 'Sigma', default: 1, min: 0.001, step: 0.1 },
  ],
};

interface PriorConfigEditorProps {
  value: PriorConfig;
  onChange: (value: PriorConfig) => void;
  label: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

export function PriorConfigEditor({
  value,
  onChange,
  label,
  collapsible = true,
  defaultCollapsed = true,
}: PriorConfigEditorProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  const handleTypeChange = (newType: string) => {
    const type = newType as PriorType;
    const params: Record<string, number> = {};
    PRIOR_PARAMS[type].forEach((param) => {
      params[param.name] = param.default;
    });
    onChange({ ...value, type, params });
  };

  const handleParamChange = (paramName: string, paramValue: number) => {
    onChange({
      ...value,
      params: { ...value.params, [paramName]: paramValue },
    });
  };

  const currentParams = PRIOR_PARAMS[value.type] || [];

  const content = (
    <div className="space-y-3">
      <div>
        <label className="text-xs font-medium text-gray-600">Distribution</label>
        <Select value={value.type} onValueChange={handleTypeChange} className="mt-1">
          {PRIOR_TYPES.map((type) => (
            <SelectItem key={type} value={type}>
              {PRIOR_TYPE_LABELS[type]}
            </SelectItem>
          ))}
        </Select>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {currentParams.map((param) => (
          <div key={param.name}>
            <label className="text-xs font-medium text-gray-600">{param.label}</label>
            <NumberInput
              value={value.params[param.name] ?? param.default}
              onValueChange={(val) => handleParamChange(param.name, val ?? param.default)}
              min={param.min}
              max={param.max}
              step={param.step}
              className="mt-1"
            />
          </div>
        ))}
      </div>
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
    <Card className="p-3">
      <button
        type="button"
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between text-left"
      >
        <Title className="text-sm">{label}</Title>
        {isCollapsed ? (
          <ChevronDownIcon className="h-4 w-4 text-gray-500" />
        ) : (
          <ChevronUpIcon className="h-4 w-4 text-gray-500" />
        )}
      </button>

      {!isCollapsed && <div className="mt-3 pt-3 border-t">{content}</div>}
    </Card>
  );
}

export default PriorConfigEditor;
