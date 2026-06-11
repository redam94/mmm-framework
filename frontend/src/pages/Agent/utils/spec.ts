// Canonical trend-type names (api/types.ts TrendType). The agent LLM sometimes
// writes aliases like "piecewise_linear" or "gp" into the spec; map them back
// so widget type-switches (and the trend preview plot) don't fall through.
const TREND_TYPE_ALIASES: Record<string, string> = {
  piecewise_linear: 'piecewise',
  gp: 'gaussian_process',
};

export function normalizeTrendType(raw: any): string {
  const t = String(raw ?? 'linear').toLowerCase().replace(/-/g, '_');
  return TREND_TYPE_ALIASES[t] ?? t;
}

// Normalize an incoming (possibly minimal) spec into a full editable form
export function specWithDefaults(raw: any) {
  return {
    kpi: raw?.kpi ?? '',
    kpi_level: raw?.kpi_level ?? 'national',
    time_granularity: raw?.time_granularity ?? 'weekly',
    inference: {
      chains: raw?.inference?.chains ?? 4,
      draws: raw?.inference?.draws ?? 1000,
      tune: raw?.inference?.tune ?? 1000,
      target_accept: raw?.inference?.target_accept ?? 0.85,
      random_seed: raw?.inference?.random_seed ?? 42,
    },
    trend: {
      type: normalizeTrendType(raw?.trend?.type),
      n_changepoints: raw?.trend?.n_changepoints ?? 5,
      changepoint_range: raw?.trend?.changepoint_range ?? 0.8,
      n_knots: raw?.trend?.n_knots ?? 5,
      spline_degree: raw?.trend?.spline_degree ?? 3,
    },
    seasonality: {
      yearly: raw?.seasonality?.yearly ?? 0,
      monthly: raw?.seasonality?.monthly ?? 0,
      weekly: raw?.seasonality?.weekly ?? 0,
    },
    media_channels: (raw?.media_channels ?? []).map((ch: any) => ({
      name: ch.name,
      adstock: { type: ch.adstock?.type ?? 'geometric', l_max: ch.adstock?.l_max ?? 8 },
      saturation: { type: ch.saturation?.type ?? 'hill' },
    })),
    control_variables: (raw?.control_variables ?? []).map((c: any) =>
      typeof c === 'string' ? { name: c } : c
    ),
  };
}

// Flatten a spec into {dot_path: leaf}, mirroring the server's spec_locks
// semantics (named lists keyed by item name). Used to lock ONLY the leaves the
// user actually changed in the editor, not every materialized default.
export function flattenLeaves(obj: any, prefix = ''): Record<string, any> {
  const out: Record<string, any> = {};
  const isNamedList = Array.isArray(obj) && obj.length > 0 &&
    obj.every((x: any) => x && typeof x === 'object' && 'name' in x);
  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    const keys = Object.keys(obj);
    if (keys.length === 0 && prefix) { out[prefix] = obj; return out; }
    for (const k of keys) Object.assign(out, flattenLeaves(obj[k], prefix ? `${prefix}.${k}` : k));
  } else if (isNamedList) {
    for (const item of obj) {
      const p = prefix ? `${prefix}.${item.name}` : String(item.name);
      for (const k of Object.keys(item)) Object.assign(out, flattenLeaves(item[k], `${p}.${k}`));
    }
  } else {
    out[prefix] = obj;
  }
  return out;
}

// Leaf paths that differ between a baseline spec and an edited spec.
export function specLeafDiff(baseline: any, edited: any): string[] {
  const a = flattenLeaves(baseline ?? {});
  const b = flattenLeaves(edited ?? {});
  const changed: string[] = [];
  for (const [path, val] of Object.entries(b)) {
    if (!(path in a) || JSON.stringify(a[path]) !== JSON.stringify(val)) changed.push(path);
  }
  return changed;
}

// Human-friendly label for a locked dot-path, e.g.
// "media_channels.TV.adstock.l_max" → "TV · adstock · l_max"
export function lockPathLabel(path: string): string {
  const parts = path.split('.');
  if (parts[0] === 'media_channels' || parts[0] === 'control_variables') {
    return parts.slice(1).join(' · ');
  }
  return parts.join(' · ');
}

export const fmtVal = (v: any): string =>
  v === null || v === undefined ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v);
