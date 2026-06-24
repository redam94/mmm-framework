// A normalized spec variable: a `{ name, ... }` object whose remaining fields
// (adstock, saturation, role, …) are spec-driven and only loosely typed.
export interface SpecVar {
  name: string;
  [key: string]: unknown;
}

// Canonical trend-type names (api/types.ts TrendType). The agent LLM sometimes
// writes aliases like "piecewise_linear" or "gp" into the spec; map them back
// so widget type-switches (and the trend preview plot) don't fall through.
const TREND_TYPE_ALIASES: Record<string, string> = {
  piecewise_linear: 'piecewise',
  gp: 'gaussian_process',
};

export function normalizeTrendType(raw: unknown): string {
  const t = String(raw ?? 'linear').toLowerCase().replace(/-/g, '_');
  return TREND_TYPE_ALIASES[t] ?? t;
}

/**
 * Normalize a spec variable collection to an array of `{ name, ... }` objects.
 * The agent / backend may serialize media_channels / control_variables as an
 * array of objects, an array of bare-string names, OR a dict keyed by name —
 * calling `.map` / `for...of` on the dict form throws ("not a function" /
 * "object is not iterable"). This coerces all three shapes to one array.
 */
export function asVarArray(v: unknown): SpecVar[] {
  if (Array.isArray(v)) {
    return v
      .map((item) => (typeof item === 'string' ? { name: item } : item))
      .filter((x): x is SpecVar => !!x && typeof x === 'object' && 'name' in x && !!x.name);
  }
  if (v && typeof v === 'object') {
    return Object.entries(v as Record<string, unknown>).map(([name, val]) => ({
      name,
      ...(val && typeof val === 'object' ? (val as object) : {}),
    }));
  }
  return [];
}

// The accessed surface of an incoming spec. Leaves are optional with their
// expected primitive type (the agent/backend may serialize a minimal or partial
// spec); typing them concretely — rather than `unknown` — lets the `?? default`
// fallbacks below produce a concretely-typed DraftSpec instead of `{} | T`.
interface RawSpec {
  kpi?: string;
  kpi_level?: string;
  time_granularity?: string;
  inference?: {
    chains?: number; draws?: number; tune?: number;
    target_accept?: number; random_seed?: number;
  };
  trend?: {
    type?: unknown; n_changepoints?: number; changepoint_range?: number;
    n_knots?: number; spline_degree?: number;
  };
  seasonality?: { yearly?: number; monthly?: number; weekly?: number };
  media_channels?: unknown;
  control_variables?: unknown;
}

// Normalize an incoming (possibly minimal) spec into a full editable form
export function specWithDefaults(rawSpec: unknown) {
  // The incoming spec is genuinely dynamic (agent/backend-serialized); narrow
  // to a loose record once at the boundary and read leaves defensively.
  const raw = (rawSpec ?? {}) as RawSpec;
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
    media_channels: asVarArray(raw?.media_channels).map((ch) => {
      const adstock = (ch.adstock ?? {}) as { type?: string; l_max?: number };
      const saturation = (ch.saturation ?? {}) as { type?: string };
      return {
        name: ch.name,
        adstock: { type: adstock.type ?? 'geometric', l_max: adstock.l_max ?? 8 },
        saturation: { type: saturation.type ?? 'hill' },
      };
    }),
    control_variables: asVarArray(raw?.control_variables),
  };
}

// Flatten a spec into {dot_path: leaf}, mirroring the server's spec_locks
// semantics (named lists keyed by item name). Used to lock ONLY the leaves the
// user actually changed in the editor, not every materialized default.
export function flattenLeaves(obj: unknown, prefix = ''): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const isNamedList = Array.isArray(obj) && obj.length > 0 &&
    obj.every((x: unknown) => !!x && typeof x === 'object' && 'name' in x);
  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    const rec = obj as Record<string, unknown>;
    const keys = Object.keys(rec);
    if (keys.length === 0 && prefix) { out[prefix] = obj; return out; }
    for (const k of keys) Object.assign(out, flattenLeaves(rec[k], prefix ? `${prefix}.${k}` : k));
  } else if (isNamedList) {
    for (const item of obj as SpecVar[]) {
      const p = prefix ? `${prefix}.${item.name}` : String(item.name);
      for (const k of Object.keys(item)) Object.assign(out, flattenLeaves(item[k], `${p}.${k}`));
    }
  } else {
    out[prefix] = obj;
  }
  return out;
}

// Leaf paths that differ between a baseline spec and an edited spec.
export function specLeafDiff(baseline: unknown, edited: unknown): string[] {
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

export const fmtVal = (v: unknown): string =>
  v === null || v === undefined ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v);
