import { Select, SelectItem, Text } from '@tremor/react';

interface VariableSelectorProps {
  value: string;
  onChange: (value: string) => void;
  variables: string[];
  label?: string;
  placeholder?: string;
  error?: string;
  disabled?: boolean;
  required?: boolean;
}

export function VariableSelector({
  value,
  onChange,
  variables,
  label,
  placeholder = 'Select variable...',
  error,
  disabled = false,
  required = false,
}: VariableSelectorProps) {
  return (
    <div className="space-y-1">
      {label && (
        <label className="text-sm font-medium text-gray-700">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <Select
        value={value}
        onValueChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        className={error ? 'border-red-500' : ''}
      >
        {variables.map((variable) => (
          <SelectItem key={variable} value={variable}>
            {variable}
          </SelectItem>
        ))}
      </Select>
      {error && <Text className="text-red-500 text-xs">{error}</Text>}
    </div>
  );
}

export default VariableSelector;
