import type { ReactNode } from 'react';
import { clsx } from 'clsx';
import type { LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  /** Primary call to action (Button or Link) */
  action?: ReactNode;
  /** Secondary line, e.g. "Explore the demo project" link */
  secondary?: ReactNode;
  className?: string;
}

/** Cream callout used by every page's no-data tier. */
export function EmptyState({ icon: Icon, title, description, action, secondary, className }: EmptyStateProps) {
  return (
    <div
      className={clsx(
        'flex flex-col items-center justify-center rounded-lg border border-dashed border-line-300 bg-cream-200/60 px-8 py-12 text-center',
        className,
      )}
    >
      {Icon && <Icon className="mb-3 h-8 w-8 text-ink-300" strokeWidth={1.5} />}
      <h3 className="font-display text-lg font-semibold text-ink-900">{title}</h3>
      {description && <p className="mt-1.5 max-w-md text-sm text-ink-400">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
      {secondary && <div className="mt-2 text-sm">{secondary}</div>}
    </div>
  );
}
