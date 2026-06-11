import type { TableColumn } from '../types';

// ─── Cell formatting for structured tables ───────────────────────────────────

const NUMBER_FMT = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 4 });
const CURRENCY_FMT = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });

/** Format a single table cell according to its column type.
 *  number   → Intl.NumberFormat (≤4 significant digits, thousands separators)
 *  percent  → |v| ≤ 1 treated as a fraction → "12.3%", else already-scaled → "12.3%"
 *  currency → Intl USD
 *  date/string/unknown → String(value); null/undefined → "—". */
export function formatCell(value: unknown, type?: TableColumn['type']): string {
  if (value == null) return '—';
  switch (type) {
    case 'number': {
      const n = Number(value);
      return Number.isFinite(n) ? NUMBER_FMT.format(n) : String(value);
    }
    case 'percent': {
      const n = Number(value);
      if (!Number.isFinite(n)) return String(value);
      return Math.abs(n) <= 1 ? `${(n * 100).toFixed(1)}%` : `${n.toFixed(1)}%`;
    }
    case 'currency': {
      const n = Number(value);
      return Number.isFinite(n) ? CURRENCY_FMT.format(n) : String(value);
    }
    default:
      return String(value);
  }
}
