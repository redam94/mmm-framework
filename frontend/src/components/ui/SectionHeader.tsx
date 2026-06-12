import type { ReactNode } from 'react';
import { clsx } from 'clsx';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  /** Right-aligned actions (buttons, toggles) */
  actions?: ReactNode;
  /** h1 for page titles, h2 for sections */
  level?: 1 | 2;
  className?: string;
}

/** Editorial section header: Fraunces display title + quiet sans subtitle. */
export function SectionHeader({ title, subtitle, actions, level = 2, className }: SectionHeaderProps) {
  const Tag = level === 1 ? 'h1' : 'h2';
  return (
    <div className={clsx('flex items-end justify-between gap-4', className)}>
      <div className="min-w-0">
        <Tag
          className={clsx(
            'font-display text-ink-900 tracking-tight',
            level === 1 ? 'text-3xl font-semibold' : 'text-xl font-semibold',
          )}
        >
          {title}
        </Tag>
        {subtitle && <p className="mt-1 text-sm text-ink-400">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
