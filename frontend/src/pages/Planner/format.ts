export const fmtInt = (x: number | null | undefined): string =>
  x == null || !Number.isFinite(x) ? '—' : Math.round(x).toLocaleString();

export const fmtSignedInt = (x: number | null | undefined): string =>
  x == null || !Number.isFinite(x)
    ? '—'
    : `${x >= 0 ? '+' : ''}${Math.round(x).toLocaleString()}`;

export const fmtSignedPct = (x: number | null | undefined): string =>
  x == null || !Number.isFinite(x) ? '—' : `${x >= 0 ? '+' : ''}${Math.round(x)}%`;

export const fmtPct = (x: number | null | undefined, digits = 0): string =>
  x == null || !Number.isFinite(x) ? '—' : `${(x * 100).toFixed(digits)}%`;
