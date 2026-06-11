// ─── Plotly light-mode layout helper ─────────────────────────────────────────

export function applyLightModeLayout(rawLayout: any): any {
  const layout = { ...(rawLayout || {}) };

  layout.paper_bgcolor = 'rgba(0,0,0,0)';
  layout.plot_bgcolor = '#f9fafb';
  layout.font = { family: 'Inter, system-ui, sans-serif', size: 12, ...(layout.font || {}), color: '#1f2937' };

  if (layout.title) {
    layout.title = typeof layout.title === 'string'
      ? { text: layout.title, font: { color: '#111827', size: 15 } }
      : { ...layout.title, font: { size: 15, ...(layout.title.font || {}), color: '#111827' } };
  }

  const axisBase = {
    automargin: true,
    gridcolor: '#f3f4f6',
    linecolor: '#e5e7eb',
    zerolinecolor: '#e5e7eb',
    zerolinewidth: 1,
    tickfont: { color: '#374151', size: 11 },
    titlefont: { color: '#4b5563', size: 12 },
  };

  Object.keys(layout).forEach(key => {
    if (/^[xy]axis\d*$/.test(key)) {
      const existing = layout[key] || {};
      layout[key] = {
        ...axisBase,
        ...existing,
        automargin: true,
        gridcolor: '#f3f4f6',
        linecolor: '#e5e7eb',
        zerolinecolor: '#e5e7eb',
        tickfont: { color: '#374151', size: 11, ...(existing.tickfont || {}) },
        titlefont: { color: '#4b5563', size: 12, ...(existing.titlefont || {}) },
      };
    }
  });

  if (!layout.xaxis) layout.xaxis = { ...axisBase };
  if (!layout.yaxis) layout.yaxis = { ...axisBase };

  layout.legend = {
    bgcolor: 'rgba(255,255,255,0.95)',
    bordercolor: '#e5e7eb',
    borderwidth: 1,
    ...(layout.legend || {}),
    font: { color: '#374151', size: 11 },
  };

  layout.hoverlabel = {
    bgcolor: '#ffffff',
    bordercolor: '#d1d5db',
    font: { color: '#1f2937', size: 12 },
    ...(layout.hoverlabel || {}),
  };

  if (Array.isArray(layout.annotations)) {
    layout.annotations = layout.annotations.map((a: any) => ({
      ...a,
      font: { size: 11, ...(a.font || {}), color: a.font?.color && !isLightOnWhite(a.font.color) ? a.font.color : '#374151' },
    }));
  }

  if (!layout.colorway) {
    layout.colorway = ['#4f46e5', '#0d9488', '#f59e0b', '#e11d48', '#059669', '#7c3aed', '#0284c7', '#b45309', '#6366f1', '#0f766e'];
  }

  layout.margin = { l: 70, r: 40, t: 90, b: 80, ...(layout.margin || {}) };

  return layout;
}

export function isLightOnWhite(hex: string): boolean {
  const clean = hex.replace('#', '');
  if (clean.length !== 6) return false;
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  // relative luminance; light colours are bad on white
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.7;
}
