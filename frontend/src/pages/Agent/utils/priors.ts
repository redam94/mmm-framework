// ─── Prior Configuration ──────────────────────────────────────────────────────

import { normalizeTrendType } from './spec';

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

// --- Prior defaults per context ---

export const PRIOR_DEFAULTS = {
  media_coefficient: { distribution: 'half_normal', params: { sigma: 2.0 } },
  adstock_alpha:     { distribution: 'beta',        params: { alpha: 1.0, beta: 3.0 } },
  sat_kappa:         { distribution: 'beta',        params: { alpha: 2.0, beta: 2.0 } },
  sat_slope:         { distribution: 'half_normal', params: { sigma: 1.5 } },
  control_coef:      { distribution: 'normal',      params: { mu: 0.0, sigma: 1.0 } },
};

export function initPriors(spec: any): any {
  const media: Record<string, any> = {};
  for (const ch of (spec?.media_channels ?? [])) {
    const name = ch.name;
    const existing = spec?.priors?.media?.[name] ?? {};
    media[name] = {
      coefficient:      existing.coefficient      ?? { ...PRIOR_DEFAULTS.media_coefficient },
      adstock_alpha:    existing.adstock_alpha    ?? { ...PRIOR_DEFAULTS.adstock_alpha },
      saturation_kappa: existing.saturation_kappa ?? { ...PRIOR_DEFAULTS.sat_kappa },
      saturation_slope: existing.saturation_slope ?? { ...PRIOR_DEFAULTS.sat_slope },
    };
  }

  const controls: Record<string, any> = {};
  for (const cv of (spec?.control_variables ?? [])) {
    const name = cv.name;
    const existing = spec?.priors?.controls?.[name] ?? {};
    controls[name] = {
      coefficient:    existing.coefficient  ?? { ...PRIOR_DEFAULTS.control_coef },
      allow_negative: existing.allow_negative ?? true,
    };
  }

  const trendType = normalizeTrendType(spec?.trend?.type);
  const existingTrend = spec?.priors?.trend ?? {};
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
  const existingSeas = spec?.priors?.seasonality ?? {};
  const seasonality = {
    prior_sigma:         existingSeas.prior_sigma         ?? 0.3,
    yearly_prior_sigma:  existingSeas.yearly_prior_sigma  ?? null,
    monthly_prior_sigma: existingSeas.monthly_prior_sigma ?? null,
    weekly_prior_sigma:  existingSeas.weekly_prior_sigma  ?? null,
  };

  return { media, controls, trend, seasonality };
}
