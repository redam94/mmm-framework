/**
 * Static class maps for components that take a color *name* prop.
 * Tailwind 4 scans source statically — `text-${color}-600` templates produce
 * classes the build never emits. Every color name used anywhere in the app
 * must appear here as a full literal class.
 */

export const TITLE_COLOR_CLASS: Record<string, string> = {
  amber: 'text-amber-600',
  blue: 'text-blue-600',
  emerald: 'text-emerald-600',
  fuchsia: 'text-fuchsia-600',
  gray: 'text-gray-600',
  green: 'text-green-600',
  indigo: 'text-indigo-600',
  red: 'text-red-600',
  sky: 'text-sky-600',
  teal: 'text-teal-600',
  violet: 'text-violet-600',
  // brand tokens
  sage: 'text-sage-700',
  steel: 'text-steel-600',
  gold: 'text-gold-600',
  rust: 'text-rust-600',
  ink: 'text-ink-700',
};

export function titleColorClass(color: string | undefined): string {
  return TITLE_COLOR_CLASS[color ?? 'indigo'] ?? TITLE_COLOR_CLASS.indigo;
}
