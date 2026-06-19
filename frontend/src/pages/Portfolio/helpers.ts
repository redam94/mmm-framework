import type { PortfolioBrand } from '../../api/services/benchmarkService';

/** ROI multiple, e.g. 2.34×. */
export function fmtRoi(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${v.toFixed(2)}×`;
}

/** Compact age, e.g. "12d" / "4mo". */
export function fmtAge(days: number | null | undefined): string {
  if (days == null || !Number.isFinite(days)) return '—';
  if (days < 1) return '<1d';
  if (days < 60) return `${Math.round(days)}d`;
  return `${Math.round(days / 30)}mo`;
}

/** Number of channels where this brand ranks at or above the 75th portfolio percentile. */
export function leaderCount(brand: PortfolioBrand): number {
  return Object.values(brand.vs_portfolio).filter(
    (v) => v.percentile != null && v.percentile >= 75,
  ).length;
}

/** Number of channels where this brand ranks at or below the 25th portfolio percentile. */
export function laggardCount(brand: PortfolioBrand): number {
  return Object.values(brand.vs_portfolio).filter(
    (v) => v.percentile != null && v.percentile <= 25,
  ).length;
}

/** Per-channel scatter of each brand's ROI, keyed by channel name. */
export function brandDotsByChannel(
  brands: PortfolioBrand[],
): Record<string, { name: string; roi: number; percentile: number | null }[]> {
  const out: Record<string, { name: string; roi: number; percentile: number | null }[]> = {};
  for (const b of brands) {
    for (const [channel, v] of Object.entries(b.vs_portfolio)) {
      (out[channel] ??= []).push({
        name: b.name ?? b.project_id,
        roi: v.roi_mean,
        percentile: v.percentile,
      });
    }
  }
  return out;
}
