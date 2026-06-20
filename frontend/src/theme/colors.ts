/**
 * TS mirror of the design tokens in tokens.css — Plotly layouts, XYFlow node
 * styles, and inline styles need literal hex strings, not CSS variables.
 * KEEP IN SYNC with tokens.css.
 */

export const COLORS = {
  cream50: '#faf8f3',
  cream100: '#f3f0e6',
  cream200: '#f0ede0',
  cream300: '#e9e5d4',
  ink900: '#2a3528',
  ink700: '#3a4838',
  ink600: '#4a5a48',
  ink400: '#7a8a78',
  ink300: '#9aa498',
  sage100: '#eef2e7',
  sage200: '#dde7cf',
  sage300: '#a8c485',
  sage600: '#6d8a4a',
  sage700: '#5a7a3a',
  sage800: '#4a6d2a',
  sage900: '#3a5d1a',
  gold100: '#f5ecd8',
  gold300: '#d4a86a',
  gold600: '#b8860b',
  gold700: '#8a6408',
  steel100: '#e7eef2',
  steel300: '#9db8c9',
  steel600: '#4a6d8a',
  steel700: '#3a5a75',
  rust100: '#f5e7e2',
  rust600: '#a04535',
  rust700: '#7a3525',
  line200: '#e8e4d5',
  line300: '#d8d4c5',
  line400: '#b8b4a5',
} as const;

export const FONTS = {
  display: '"Fraunces", Georgia, "Times New Roman", serif',
  sans: '"IBM Plex Sans", system-ui, -apple-system, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, "SF Mono", monospace',
} as const;

/** How well a channel's ROI is causally anchored. */
export type EvidenceTier = 'calibrated' | 'running' | 'model_only' | 'stale';

export const EVIDENCE_TIER: Record<
  EvidenceTier,
  { fg: string; bg: string; border: string; label: string }
> = {
  calibrated: {
    fg: COLORS.sage800,
    bg: COLORS.sage100,
    border: COLORS.sage300,
    label: 'Experiment-backed',
  },
  running: {
    fg: COLORS.gold700,
    bg: COLORS.gold100,
    border: COLORS.gold300,
    label: 'Experiment running',
  },
  model_only: {
    fg: COLORS.steel700,
    bg: COLORS.steel100,
    border: COLORS.steel300,
    label: 'Model-only',
  },
  stale: {
    fg: COLORS.rust700,
    bg: COLORS.rust100,
    border: COLORS.rust600,
    label: 'Evidence stale',
  },
};

/** Experiment lifecycle statuses → chip styling. */
export type ExperimentStatus =
  | 'draft'
  | 'planned'
  | 'running'
  | 'completed'
  | 'calibrated'
  | 'abandoned'
  | 'cancelled';

export const EXPERIMENT_STATUS: Record<
  ExperimentStatus,
  { fg: string; bg: string; label: string }
> = {
  draft: { fg: COLORS.ink600, bg: COLORS.cream200, label: 'Draft' },
  planned: { fg: COLORS.steel700, bg: COLORS.steel100, label: 'Planned' },
  running: { fg: COLORS.gold700, bg: COLORS.gold100, label: 'Running' },
  completed: { fg: COLORS.gold700, bg: COLORS.gold100, label: 'Needs calibration' },
  calibrated: { fg: COLORS.sage800, bg: COLORS.sage100, label: 'Calibrated' },
  abandoned: { fg: COLORS.ink400, bg: COLORS.cream200, label: 'Abandoned' },
  cancelled: { fg: COLORS.ink400, bg: COLORS.cream200, label: 'Cancelled' },
};

/** Model Garden lifecycle statuses → chip styling (mirrors EXPERIMENT_STATUS). */
export type GardenStatus = 'draft' | 'tested' | 'published' | 'deprecated';

export const GARDEN_STATUS: Record<
  GardenStatus,
  { fg: string; bg: string; label: string }
> = {
  draft: { fg: COLORS.ink600, bg: COLORS.cream200, label: 'Draft' },
  tested: { fg: COLORS.steel700, bg: COLORS.steel100, label: 'Tested' },
  published: { fg: COLORS.sage800, bg: COLORS.sage100, label: 'Published' },
  deprecated: { fg: COLORS.rust700, bg: COLORS.rust100, label: 'Deprecated' },
};

/**
 * Categorical chart colorway: ≥8 hue-distinct entries anchored in the brand
 * system (sage/steel/gold/rust first), staying readable on cream surfaces.
 */
export const CHART_COLORWAY = [
  COLORS.sage600,
  COLORS.steel600,
  COLORS.gold600,
  COLORS.rust600,
  '#7a6a9a', // muted violet
  '#3a8a7a', // teal
  COLORS.sage300,
  '#a85a7a', // muted plum
  COLORS.steel300,
  COLORS.gold300,
] as const;
