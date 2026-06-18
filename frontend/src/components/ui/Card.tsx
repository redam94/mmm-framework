import { clsx } from 'clsx';
import type { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  /** cream = inset well (sidebars, callouts); white = default content card */
  tone?: 'white' | 'cream';
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

const PADDING: Record<NonNullable<CardProps['padding']>, string> = {
  none: '',
  sm: 'p-3',
  md: 'p-5',
  lg: 'p-6',
};

export function Card({ children, className, tone = 'white', padding = 'md' }: CardProps) {
  return (
    <div
      className={clsx(
        'rounded-lg border',
        tone === 'white' ? 'bg-white border-line-200 shadow-sm' : 'bg-cream-100 border-line-200',
        PADDING[padding],
        className,
      )}
    >
      {children}
    </div>
  );
}
