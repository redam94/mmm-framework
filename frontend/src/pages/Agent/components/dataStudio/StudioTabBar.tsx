import { Activity, AlertTriangle, BarChart3, Grid3x3, Droplets, Wand2 } from 'lucide-react';

export const STUDIO_TABS = [
  { id: 'overview', label: 'Overview', icon: BarChart3 },
  { id: 'distributions', label: 'Distributions', icon: Activity },
  { id: 'correlation', label: 'Correlation', icon: Grid3x3 },
  { id: 'missingness', label: 'Missingness', icon: Droplets },
  { id: 'outliers', label: 'Outliers', icon: AlertTriangle },
  { id: 'transform', label: 'Transform', icon: Wand2 },
] as const;

export type StudioTab = (typeof STUDIO_TABS)[number]['id'];

export function StudioTabBar({
  active, onChange, dots,
}: {
  active: StudioTab;
  onChange: (t: StudioTab) => void;
  dots?: Partial<Record<StudioTab, boolean>>;
}) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto px-5 border-b border-line-200 bg-cream-50">
      {STUDIO_TABS.map(t => {
        const Icon = t.icon;
        const on = active === t.id;
        return (
          <button
            key={t.id}
            onClick={() => onChange(t.id)}
            className={`flex items-center gap-2 px-3.5 py-2.5 text-sm font-medium rounded-t-lg border-b-2 transition-colors shrink-0 ${
              on ? 'border-sage-700 text-sage-800 bg-white' : 'border-transparent text-ink-400 hover:text-ink-900 hover:bg-cream-100'
            }`}
          >
            <Icon size={14} className={on ? 'text-sage-700' : 'text-ink-300'} />
            <span>{t.label}</span>
            {dots?.[t.id] && <span className={`w-1.5 h-1.5 rounded-full ${on ? 'bg-sage-600' : 'bg-amber-400'}`} />}
          </button>
        );
      })}
    </div>
  );
}
