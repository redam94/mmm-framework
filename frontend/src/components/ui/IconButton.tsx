import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { clsx } from 'clsx';

interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'aria-label'> {
  /** REQUIRED accessible name — icon-only buttons must have one (a11y / H2). */
  label: string;
  children: ReactNode;
}

/**
 * Accessible icon-only button: the `label` prop is required and becomes both
 * `aria-label` (screen readers) and `title` (tooltip). The review found 227
 * `title=`-only icon buttons with ~2 `aria-label`s total; use this so new
 * icon-only controls have an accessible name by construction.
 */
export function IconButton({ label, children, className, ...rest }: IconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={clsx('inline-flex items-center justify-center', className)}
      {...rest}
    >
      {children}
    </button>
  );
}
