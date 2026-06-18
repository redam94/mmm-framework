// ─── Plotly layout helper ─────────────────────────────────────────────────────
// The actual theming lives in src/theme/plotlyTheme.ts (the app-wide editorial
// chart theme). These re-exports keep the original names so PlotCard's
// memoization path and existing imports are untouched.

export { applyMmmTheme as applyLightModeLayout, isLightOnWhite } from '../../../theme/plotlyTheme';
