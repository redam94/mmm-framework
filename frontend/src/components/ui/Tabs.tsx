import { clsx } from 'clsx';

export interface TabDef {
  id: string;
  label: string;
  badge?: number | string;
}

interface TabsProps {
  tabs: TabDef[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}

/** Underline tabs with sage active state. */
export function Tabs({ tabs, active, onChange, className }: TabsProps) {
  return (
    <div className={clsx('border-b border-line-200', className)}>
      <nav className="-mb-px flex gap-6" aria-label="Tabs">
        {tabs.map((t) => {
          const isActive = t.id === active;
          return (
            <button
              key={t.id}
              onClick={() => onChange(t.id)}
              className={clsx(
                'flex items-center gap-1.5 border-b-2 px-1 pb-2.5 pt-1 text-sm font-medium transition-colors',
                isActive
                  ? 'border-sage-700 text-sage-800'
                  : 'border-transparent text-ink-400 hover:border-line-300 hover:text-ink-700',
              )}
            >
              {t.label}
              {t.badge !== undefined && (
                <span
                  className={clsx(
                    'rounded-full px-1.5 py-0.5 text-[10px] font-semibold num',
                    isActive ? 'bg-sage-100 text-sage-800' : 'bg-cream-200 text-ink-600',
                  )}
                >
                  {t.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>
    </div>
  );
}
