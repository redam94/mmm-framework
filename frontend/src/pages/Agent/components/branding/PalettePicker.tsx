import { Check } from 'lucide-react';
import { PALETTE_PRESETS, matchPresetId, type PalettePreset } from './lib';

/** Preset palette rows (sage / corporate / warm / monochrome) rendered as
 * swatch strips. Clicking a row applies the preset's colors to the form;
 * "Custom" is implicit — any manual edit just stops matching a preset. */
export function PalettePicker({ activePrimary, onApply, disabled = false }: {
  activePrimary: string;
  onApply: (preset: PalettePreset) => void;
  disabled?: boolean;
}) {
  const activeId = matchPresetId(activePrimary);
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {PALETTE_PRESETS.map(p => {
        const active = activeId === p.id;
        return (
          <button
            key={p.id}
            type="button"
            disabled={disabled}
            onClick={() => onApply(p)}
            className={`flex items-center gap-3 px-3 py-2 rounded-xl border text-left transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
              active
                ? 'border-indigo-400 bg-indigo-50/60 ring-1 ring-indigo-300'
                : 'border-gray-200 bg-white hover:border-indigo-300 hover:bg-gray-50'
            }`}
          >
            <span className="flex rounded-md overflow-hidden border border-gray-200 shrink-0">
              {p.palette.map(c => (
                <span key={c} className="w-5 h-5" style={{ backgroundColor: c }} />
              ))}
            </span>
            <span className="text-xs font-medium text-gray-700 flex-1">{p.label}</span>
            {active && <Check size={14} className="text-indigo-600 shrink-0" />}
          </button>
        );
      })}
    </div>
  );
}
