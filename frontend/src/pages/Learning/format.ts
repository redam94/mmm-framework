/** Shared helpers for the Sextant (continuous learning) page. */

import { ARM_SEP } from '../../api/services/learningService';
import type { LearningProgram } from '../../api/services/learningService';

/**
 * The program's surface dimensions in display order: parent channels, with
 * split channels expanded to their arms ("Search │ Brand") the same way
 * continuous_learning/arms.py flattens them.
 */
export function effectiveChannels(program: LearningProgram): string[] {
  const parents = program.channels ?? program.config?.channels ?? [];
  const arms = program.config?.arms ?? {};
  const out: string[] = [];
  for (const ch of parents) {
    const a = arms[ch];
    if (a && a.length > 0) a.forEach((arm) => out.push(`${ch}${ARM_SEP}${arm}`));
    else out.push(ch);
  }
  return out;
}

export function errorDetail(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
  return detail ?? String(e);
}

export const fmtDollars = (x: number | null | undefined): string =>
  x == null || !Number.isFinite(x)
    ? '—'
    : `${x < 0 ? '−' : ''}$${Math.round(Math.abs(x)).toLocaleString()}`;

export const fmtSignedDollars = (x: number | null | undefined): string =>
  x == null || !Number.isFinite(x)
    ? '—'
    : `${x >= 0 ? '+' : '−'}$${Math.round(Math.abs(x)).toLocaleString()}`;

export const fmtNum = (x: number | null | undefined, digits = 2): string =>
  x == null || !Number.isFinite(x) ? '—' : x.toFixed(digits);

export const fmtPct = (x: number | null | undefined, digits = 0): string =>
  x == null || !Number.isFinite(x) ? '—' : `${(x * 100).toFixed(digits)}%`;
