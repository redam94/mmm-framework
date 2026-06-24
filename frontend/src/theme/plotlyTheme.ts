/**
 * Plotly theming for the editorial identity — the single choke point for
 * every chart in the app (agent-generated figures included).
 *
 * MERGE, never replace: agent/backend figures own their traces and may carry
 * meaningful layout (own colorscales, annotations, subplot axes). The theme
 * only defaults what the figure didn't specify, exactly like the previous
 * applyLightModeLayout (which re-exports from here).
 */

import type { Annotations, Layout, ModeBarDefaultButtons } from 'plotly.js';

import { CHART_COLORWAY, COLORS, FONTS } from './colors';

/**
 * Loose layout blob: agent/backend figures carry arbitrary extra keys (subplot
 * axes, custom annotations, etc.) on top of the typed Plotly layout, and we
 * index/merge them dynamically — so we treat the working copy as an open record.
 */
type LayoutBlob = Partial<Layout> & Record<string, unknown>;

export function isLightOnWhite(hex: string): boolean {
  const clean = hex.replace('#', '');
  if (clean.length !== 6) return false;
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  // relative luminance; light colours are bad on light surfaces
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.7;
}

/** Theme an existing (possibly agent-authored) layout. Caller values win. */
export function applyMmmTheme(rawLayout?: Partial<Layout> | null): Partial<Layout> {
  const layout: LayoutBlob = { ...(rawLayout || {}) };

  layout.paper_bgcolor = 'rgba(0,0,0,0)';
  layout.plot_bgcolor = COLORS.cream50;
  layout.font = {
    family: FONTS.sans,
    size: 12,
    ...(layout.font || {}),
    color: COLORS.ink700,
  };

  if (layout.title) {
    layout.title =
      typeof layout.title === 'string'
        ? {
            text: layout.title,
            font: { family: FONTS.display, color: COLORS.ink900, size: 16 },
          }
        : {
            ...layout.title,
            font: {
              family: FONTS.display,
              size: 16,
              ...(layout.title.font || {}),
              color: COLORS.ink900,
            },
          };
  }

  const axisBase = {
    automargin: true,
    gridcolor: COLORS.line200,
    linecolor: COLORS.line300,
    zerolinecolor: COLORS.line300,
    zerolinewidth: 1,
    tickfont: { color: COLORS.ink600, size: 11 },
    titlefont: { color: COLORS.ink600, size: 12 },
  };

  Object.keys(layout).forEach((key) => {
    if (/^[xy]axis\d*$/.test(key)) {
      const existing = (layout[key] || {}) as Record<string, unknown>;
      layout[key] = {
        ...axisBase,
        ...existing,
        automargin: true,
        gridcolor: COLORS.line200,
        linecolor: COLORS.line300,
        zerolinecolor: COLORS.line300,
        tickfont: { color: COLORS.ink600, size: 11, ...((existing.tickfont as object) || {}) },
        titlefont: { color: COLORS.ink600, size: 12, ...((existing.titlefont as object) || {}) },
      };
    }
  });

  if (!layout.xaxis) layout.xaxis = { ...axisBase };
  if (!layout.yaxis) layout.yaxis = { ...axisBase };

  layout.legend = {
    bgcolor: 'rgba(250,248,243,0.95)',
    bordercolor: COLORS.line200,
    borderwidth: 1,
    ...(layout.legend || {}),
    font: { color: COLORS.ink600, size: 11 },
  };

  layout.hoverlabel = {
    bgcolor: '#ffffff',
    bordercolor: COLORS.line300,
    font: { color: COLORS.ink900, size: 12, family: FONTS.sans },
    ...(layout.hoverlabel || {}),
  };

  if (Array.isArray(layout.annotations)) {
    layout.annotations = layout.annotations.map((a: Partial<Annotations>) => {
      const fontColor = a.font?.color;
      return {
        ...a,
        font: {
          size: 11,
          ...(a.font || {}),
          color:
            typeof fontColor === 'string' && !isLightOnWhite(fontColor)
              ? fontColor
              : COLORS.ink600,
        },
      };
    });
  }

  if (!layout.colorway) {
    layout.colorway = [...CHART_COLORWAY];
  }

  layout.margin = { l: 70, r: 40, t: 90, b: 80, ...(layout.margin || {}) };

  return layout;
}

/** Fresh layout for charts the app authors itself (Plotly `layout` prop). */
export function mmmPlotlyLayout(overrides: Partial<Layout> = {}): Partial<Layout> {
  return applyMmmTheme(overrides);
}

const MODE_BAR_BUTTONS_TO_REMOVE: ModeBarDefaultButtons[] = [
  'lasso2d',
  'select2d',
  'autoScale2d',
];

export const PLOTLY_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: MODE_BAR_BUTTONS_TO_REMOVE,
};
