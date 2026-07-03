// ─── Prior Configuration ──────────────────────────────────────────────────────

import { normalizeTrendType, asVarArray } from './spec';

// Re-exported for widgets that already import it from this module.
export { asVarArray } from './spec';

// --- Math helpers ---

export function linspace(a: number, b: number, n: number): number[] {
  if (n <= 1) return [a];
  const step = (b - a) / (n - 1);
  return Array.from({ length: n }, (_, i) => a + i * step);
}

export function normalizeDensity(x: number[], y: number[]): number[] {
  if (x.length < 2) return y;
  const dx = (x[x.length - 1] - x[0]) / (x.length - 1);
  const total = y.reduce((s, v) => s + v * dx, 0);
  return total > 1e-12 ? y.map(v => v / total) : y;
}

export function computeDensity(dist: string, params: Record<string, number>): { x: number[]; y: number[] } {
  const N = 160;
  const normalPDF = (xi: number, mu: number, sig: number) => {
    const z = (xi - mu) / sig;
    return Math.exp(-0.5 * z * z) / (sig * Math.sqrt(2 * Math.PI));
  };

  if (dist === 'normal') {
    const mu = params.mu ?? 0; const sigma = Math.max(params.sigma ?? 1, 1e-6);
    const x = linspace(mu - 4 * sigma, mu + 4 * sigma, N);
    return { x, y: x.map(xi => normalPDF(xi, mu, sigma)) };
  }
  if (dist === 'half_normal') {
    const sigma = Math.max(params.sigma ?? 1, 1e-6);
    const x = linspace(0, 4.5 * sigma, N);
    return { x, y: x.map(xi => 2 * normalPDF(xi, 0, sigma)) };
  }
  if (dist === 'log_normal') {
    const mu = params.mu ?? 0; const sigma = Math.max(params.sigma ?? 1, 1e-6);
    const xMax = Math.exp(mu + 3.5 * sigma);
    const x = linspace(1e-6, xMax, N);
    const y = x.map(xi => Math.exp(-0.5 * ((Math.log(xi) - mu) / sigma) ** 2) / (xi * sigma * Math.sqrt(2 * Math.PI)));
    return { x, y };
  }
  if (dist === 'beta') {
    const alpha = Math.max(params.alpha ?? 2, 1e-3); const beta = Math.max(params.beta ?? 2, 1e-3);
    const x = linspace(1e-4, 1 - 1e-4, N);
    const unnorm = x.map(xi => Math.exp((alpha - 1) * Math.log(xi) + (beta - 1) * Math.log(1 - xi)));
    return { x, y: normalizeDensity(x, unnorm) };
  }
  if (dist === 'gamma') {
    const alpha = Math.max(params.alpha ?? 2, 1e-3); const rate = Math.max(params.beta ?? 1, 1e-6);
    const xMax = (alpha + 4 * Math.sqrt(alpha)) / rate;
    const x = linspace(1e-6, xMax, N);
    const unnorm = x.map(xi => xi <= 0 ? 0 : Math.exp((alpha - 1) * Math.log(xi) - rate * xi));
    return { x, y: normalizeDensity(x, unnorm) };
  }
  if (dist === 'truncated_normal') {
    const mu = params.mu ?? 0; const sigma = Math.max(params.sigma ?? 1, 1e-6); const lower = params.lower ?? 0;
    const xMax = Math.max(mu + 4 * sigma, lower + 0.1);
    const x = linspace(lower, xMax, N);
    return { x, y: normalizeDensity(x, x.map(xi => normalPDF(xi, mu, sigma))) };
  }
  if (dist === 'half_student_t') {
    const nu = Math.max(params.nu ?? 3, 0.5); const sigma = Math.max(params.sigma ?? 1, 1e-6);
    const x = linspace(0, sigma * 7, N);
    const unnorm = x.map(xi => Math.pow(1 + (xi / sigma) ** 2 / nu, -(nu + 1) / 2));
    return { x, y: normalizeDensity(x, unnorm) };
  }
  return { x: [], y: [] };
}

// --- Distribution parameter definitions ---

export type DistKey = 'normal' | 'half_normal' | 'log_normal' | 'gamma' | 'beta' | 'truncated_normal' | 'half_student_t';

export interface ParamDef { key: string; label: string; min: number; max: number; step: number; default: number }
export interface DistDef { label: string; params: ParamDef[] }

export const DIST_DEFS: Record<DistKey, DistDef> = {
  normal:            { label: 'Normal',           params: [{ key: 'mu', label: 'μ', min: -10, max: 10, step: 0.1, default: 0 }, { key: 'sigma', label: 'σ', min: 0.01, max: 10, step: 0.1, default: 1 }] },
  half_normal:       { label: 'Half-Normal',       params: [{ key: 'sigma', label: 'σ', min: 0.01, max: 10, step: 0.1, default: 1 }] },
  log_normal:        { label: 'Log-Normal',        params: [{ key: 'mu', label: 'μ (log)', min: -5, max: 5, step: 0.1, default: 0 }, { key: 'sigma', label: 'σ (log)', min: 0.01, max: 5, step: 0.1, default: 1 }] },
  gamma:             { label: 'Gamma',             params: [{ key: 'alpha', label: 'α (shape)', min: 0.1, max: 20, step: 0.1, default: 2 }, { key: 'beta', label: 'β (rate)', min: 0.01, max: 10, step: 0.1, default: 1 }] },
  beta:              { label: 'Beta',              params: [{ key: 'alpha', label: 'α', min: 0.1, max: 20, step: 0.1, default: 2 }, { key: 'beta', label: 'β', min: 0.1, max: 20, step: 0.1, default: 2 }] },
  truncated_normal:  { label: 'Truncated Normal',  params: [{ key: 'mu', label: 'μ', min: -10, max: 10, step: 0.1, default: 0 }, { key: 'sigma', label: 'σ', min: 0.01, max: 10, step: 0.1, default: 1 }, { key: 'lower', label: 'Lower bound', min: -10, max: 10, step: 0.1, default: 0 }] },
  half_student_t:    { label: 'Half-Student t',    params: [{ key: 'nu', label: 'ν (df)', min: 1, max: 30, step: 0.5, default: 3 }, { key: 'sigma', label: 'σ', min: 0.01, max: 10, step: 0.1, default: 1 }] },
};

// Allowed distributions by prior type
export const POSITIVE_DISTS: DistKey[] = ['half_normal', 'half_student_t', 'log_normal', 'gamma', 'truncated_normal'];
export const UNIT_DISTS: DistKey[]     = ['beta', 'truncated_normal'];
export const ANY_DISTS: DistKey[]      = ['normal', 'half_normal', 'log_normal', 'gamma', 'beta', 'truncated_normal', 'half_student_t'];

export interface PriorValue { distribution: string; params: Record<string, number> }

// ROI-scale media prior: roi_<ch> ~ LogNormal(ln(median), sigma). `median` is
// in ROI units (1.0 = break-even); `sigma` is the log-scale spread. Mirrors
// the backend's `priors.media.<ch>.roi` spec key.
export interface RoiPriorValue { median: number; sigma: number }

// How a channel's effect prior is stated: directly on the ROI (decision)
// scale, or on the standardized coefficient scale (advanced).
export type MediaPriorMode = 'roi' | 'coefficient';

// Standard normal CDF (Abramowitz–Stegun 7.1.26 erf approximation; max abs
// error ~1.5e-7 — plenty for a UI readout).
export function normalCdf(x: number): number {
  const z = Math.abs(x) / Math.SQRT2; // Φ(x) = (1 + erf(x/√2)) / 2
  const t = 1 / (1 + 0.3275911 * z);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return 0.5 * (1 + Math.sign(x) * erf);
}

// Closed-form implications of an ROI-scale prior — the instant feedback the
// widget shows as the user types ("what does this prior say about ROI?").
export function roiPriorStats(median: number, sigma: number): { lower: number; upper: number; pAbove1: number } {
  const m = Math.max(median, 1e-9);
  const s = Math.max(sigma, 1e-9);
  const z90 = 1.6448536269514722; // 95th percentile of N(0,1) → central 90%
  return {
    lower: m * Math.exp(-z90 * s),
    upper: m * Math.exp(z90 * s),
    pAbove1: normalCdf(Math.log(m) / s),
  };
}

// --- Prior defaults per context ---

export const PRIOR_DEFAULTS = {
  media_coefficient: { distribution: 'half_normal', params: { sigma: 2.0 } },
  // Matches the backend ROI-mode default: LogNormal(0, 1) — median 1.0
  // (break-even), 90% within ~[0.19x, 5.2x], P(ROI>1) = 50%.
  media_roi:         { median: 1.0, sigma: 1.0 },
  adstock_alpha:     { distribution: 'beta',        params: { alpha: 1.0, beta: 3.0 } },
  sat_kappa:         { distribution: 'beta',        params: { alpha: 2.0, beta: 2.0 } },
  sat_slope:         { distribution: 'half_normal', params: { sigma: 1.5 } },
  control_coef:      { distribution: 'normal',      params: { mu: 0.0, sigma: 1.0 } },
};

// Shape of the priors object produced by initPriors. Each prior is a normalized
// {distribution, params} value; trend/seasonality are concrete numeric configs.
export interface MediaPriors {
  // Which scale the effect prior is stated on. 'roi' (the default, matching
  // the backend's media_prior_mode="roi") writes `roi` and omits
  // `coefficient` on apply; 'coefficient' does the reverse.
  mode: MediaPriorMode;
  roi: RoiPriorValue;
  coefficient: PriorValue;
  adstock_alpha: PriorValue;
  saturation_kappa: PriorValue;
  saturation_slope: PriorValue;
}
export interface ControlPriors { coefficient: PriorValue; allow_negative: boolean }
export interface TrendPriors {
  growth_prior_mu: number;
  growth_prior_sigma: number;
  changepoint_prior_scale: number;
  spline_prior_sigma: number;
  gp_lengthscale_prior_mu: number;
  gp_lengthscale_prior_sigma: number;
  gp_amplitude_prior_sigma: number;
  _type: string;
}
export interface SeasonalityPriors {
  prior_sigma: number;
  yearly_prior_sigma: number | null;
  monthly_prior_sigma: number | null;
  weekly_prior_sigma: number | null;
  // Components are read by dynamic key (`${c}_prior_sigma`); all are number|null.
  [key: string]: number | null;
}
export interface InitializedPriors {
  media: Record<string, MediaPriors>;
  controls: Record<string, ControlPriors>;
  trend: TrendPriors;
  seasonality: SeasonalityPriors;
}

// `spec` is a dynamic, partially-typed config blob (array / dict / string forms);
// accept `unknown` (at least as permissive as the prior `any`) and narrow internally.
export function initPriors(spec: unknown): InitializedPriors {
  const specObj = (spec ?? {}) as Record<string, unknown>;
  const specPriors = (specObj.priors ?? {}) as Record<string, unknown>;
  const media: Record<string, MediaPriors> = {};
  const mediaPriors = (specPriors.media ?? {}) as Record<
    string, Partial<MediaPriors> & { roi?: { median?: number; mu?: number; sigma?: number } }
  >;
  for (const ch of asVarArray(specObj.media_channels)) {
    const name = ch.name;
    const existing = mediaPriors[name] ?? {};
    // The spec's roi dict may carry `mu` (log scale) instead of `median`.
    const roiRaw = existing.roi;
    const roi: RoiPriorValue = {
      median: roiRaw?.median ?? (roiRaw?.mu != null ? Math.exp(roiRaw.mu) : PRIOR_DEFAULTS.media_roi.median),
      sigma:  roiRaw?.sigma  ?? PRIOR_DEFAULTS.media_roi.sigma,
    };
    media[name] = {
      // A channel with an explicit coefficient prior in the spec stays on the
      // coefficient scale; one with an roi entry stays on the ROI scale; the
      // rest follow the spec's media_prior_mode (agent default: "roi") so an
      // untouched Apply never flips a coefficient-mode model onto ROI priors.
      mode: existing.coefficient
        ? 'coefficient'
        : roiRaw || specObj.media_prior_mode !== 'coefficient' ? 'roi' : 'coefficient',
      roi,
      coefficient:      existing.coefficient      ?? { ...PRIOR_DEFAULTS.media_coefficient },
      adstock_alpha:    existing.adstock_alpha    ?? { ...PRIOR_DEFAULTS.adstock_alpha },
      saturation_kappa: existing.saturation_kappa ?? { ...PRIOR_DEFAULTS.sat_kappa },
      saturation_slope: existing.saturation_slope ?? { ...PRIOR_DEFAULTS.sat_slope },
    };
  }

  const controls: Record<string, ControlPriors> = {};
  const controlPriors = (specPriors.controls ?? {}) as Record<string, Partial<ControlPriors>>;
  for (const cv of asVarArray(specObj.control_variables)) {
    const name = cv.name;
    const existing = controlPriors[name] ?? {};
    controls[name] = {
      coefficient:    existing.coefficient  ?? { ...PRIOR_DEFAULTS.control_coef },
      allow_negative: existing.allow_negative ?? true,
    };
  }

  const specTrend = (specObj.trend ?? {}) as Record<string, unknown>;
  const trendType = normalizeTrendType(specTrend.type);
  const existingTrend = (specPriors.trend ?? {}) as Partial<TrendPriors>;
  const trend = {
    growth_prior_mu:            existingTrend.growth_prior_mu            ?? 0.0,
    growth_prior_sigma:         existingTrend.growth_prior_sigma         ?? 0.1,
    changepoint_prior_scale:    existingTrend.changepoint_prior_scale    ?? 0.05,
    spline_prior_sigma:         existingTrend.spline_prior_sigma         ?? 1.0,
    gp_lengthscale_prior_mu:    existingTrend.gp_lengthscale_prior_mu    ?? 0.3,
    gp_lengthscale_prior_sigma: existingTrend.gp_lengthscale_prior_sigma ?? 0.2,
    gp_amplitude_prior_sigma:   existingTrend.gp_amplitude_prior_sigma   ?? 0.5,
    _type: trendType,
  };

  // Seasonal amplitude prior: sigma of the Normal prior on Fourier coefficients
  // (standardized-y scale). null override = inherit the shared prior_sigma.
  const existingSeas = (specPriors.seasonality ?? {}) as Partial<SeasonalityPriors>;
  const seasonality = {
    prior_sigma:         existingSeas.prior_sigma         ?? 0.3,
    yearly_prior_sigma:  existingSeas.yearly_prior_sigma  ?? null,
    monthly_prior_sigma: existingSeas.monthly_prior_sigma ?? null,
    weekly_prior_sigma:  existingSeas.weekly_prior_sigma  ?? null,
  };

  return { media, controls, trend, seasonality };
}

// What actually gets written to spec.priors.media on Apply. The `mode` marker
// is internal; per channel exactly ONE of `roi` / `coefficient` is emitted —
// the backend treats an explicit coefficient prior as overriding the ROI
// prior, so writing both would silently disable the ROI entry.
export function serializeMediaPriors(
  media: Record<string, MediaPriors>,
): Record<string, Record<string, unknown>> {
  const out: Record<string, Record<string, unknown>> = {};
  for (const [name, p] of Object.entries(media)) {
    const { mode, roi, coefficient, ...rest } = p;
    out[name] = mode === 'roi'
      ? { roi: { median: roi.median, sigma: roi.sigma }, ...rest }
      : { coefficient, ...rest };
  }
  return out;
}
