import { Text } from '@tremor/react';
import type { DimensionType } from '../../../../api/types';

const ALL_DIMENSIONS: DimensionType[] = [
  'Period',
  'Geography',
  'Product',
  'Campaign',
  'Outlet',
  'Creative',
];

interface DimensionSelectorProps {
  value: DimensionType[];
  onChange: (value: DimensionType[]) => void;
  availableDimensions?: DimensionType[];
  label?: string;
  error?: string;
  disabled?: boolean;
  required?: boolean;
}

export function DimensionSelector({
  value,
  onChange,
  availableDimensions = ALL_DIMENSIONS,
  label,
  error,
  disabled = false,
  required = false,
}: DimensionSelectorProps) {
  const toggleDimension = (dim: DimensionType) => {
    if (disabled) return;

    if (value.includes(dim)) {
      // Don't allow removing Period if it's required
      if (dim === 'Period' && required) return;
      onChange(value.filter((d) => d !== dim));
    } else {
      onChange([...value, dim]);
    }
  };

  return (
    <div className="space-y-2">
      {label && (
        <label className="text-sm font-medium text-gray-700">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <div className="flex flex-wrap gap-2">
        {availableDimensions.map((dim) => {
          const isSelected = value.includes(dim);
          const isPeriodRequired = dim === 'Period' && required;

          return (
            <button
              key={dim}
              type="button"
              onClick={() => toggleDimension(dim)}
              disabled={disabled || isPeriodRequired}
              className={`
                px-3 py-1.5 text-sm rounded-full border transition-colors
                ${
                  isSelected
                    ? 'bg-blue-100 border-blue-500 text-blue-700'
                    : 'bg-gray-50 border-gray-300 text-gray-600 hover:border-gray-400'
                }
                ${disabled || isPeriodRequired ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              {dim}
              {isPeriodRequired && ' (required)'}
            </button>
          );
        })}
      </div>
      {error && <Text className="text-red-500 text-xs">{error}</Text>}
    </div>
  );
}

export default DimensionSelector;
