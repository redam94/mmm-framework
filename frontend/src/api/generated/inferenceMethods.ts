// GENERATED — do not edit by hand.
//
// Source: src/mmm_framework/config/inference_methods.py
// Regenerate: uv run python scripts/gen_fe_enums.py
// Gated by: tests/test_fe_enum_mirror.py
//
// `inference.method` accepts a union of the Bayesian FitMethod members and the
// two frequentist InferenceMethod members. `approximate` is FALSE for the
// frequentist estimators: a penalized point estimate with bootstrap confidence
// intervals is not an approximation of a posterior, and labelling it one told
// users to "re-fit with NUTS" about a fit that never had a posterior to
// approximate.

export interface InferenceMethodInfo {
  value: string;
  label: string;
  paradigm: 'bayesian' | 'frequentist';
  approximate: boolean;
  intervalKind: 'credible' | 'confidence';
  caveat: string | null;
}

export const INFERENCE_METHODS: ReadonlyArray<InferenceMethodInfo> = [
  {
    "value": "nuts",
    "label": "NUTS (full MCMC)",
    "paradigm": "bayesian",
    "approximate": false,
    "intervalKind": "credible",
    "caveat": null
  },
  {
    "value": "smc",
    "label": "SMC (Sequential Monte Carlo)",
    "paradigm": "bayesian",
    "approximate": false,
    "intervalKind": "credible",
    "caveat": "SMC is an exact sampler for multimodal posteriors and yields a log marginal likelihood for model comparison. It is not a speedup."
  },
  {
    "value": "map",
    "label": "MAP (point estimate)",
    "paradigm": "bayesian",
    "approximate": true,
    "intervalKind": "credible",
    "caveat": "Approximate fits run in seconds for model checking, but their uncertainty is not calibrated — re-fit with NUTS before trusting intervals or making spend decisions."
  },
  {
    "value": "laplace",
    "label": "Laplace (MAP + Gaussian)",
    "paradigm": "bayesian",
    "approximate": true,
    "intervalKind": "credible",
    "caveat": "Approximate fits run in seconds for model checking, but their uncertainty is not calibrated — re-fit with NUTS before trusting intervals or making spend decisions."
  },
  {
    "value": "advi",
    "label": "ADVI (variational)",
    "paradigm": "bayesian",
    "approximate": true,
    "intervalKind": "credible",
    "caveat": "Approximate fits run in seconds for model checking, but their uncertainty is not calibrated — re-fit with NUTS before trusting intervals or making spend decisions."
  },
  {
    "value": "fullrank_advi",
    "label": "Full-rank ADVI (variational)",
    "paradigm": "bayesian",
    "approximate": true,
    "intervalKind": "credible",
    "caveat": "Approximate fits run in seconds for model checking, but their uncertainty is not calibrated — re-fit with NUTS before trusting intervals or making spend decisions."
  },
  {
    "value": "pathfinder",
    "label": "Pathfinder",
    "paradigm": "bayesian",
    "approximate": true,
    "intervalKind": "credible",
    "caveat": "Approximate fits run in seconds for model checking, but their uncertainty is not calibrated — re-fit with NUTS before trusting intervals or making spend decisions."
  },
  {
    "value": "frequentist_ridge",
    "label": "Ridge (penalized, bootstrap CIs)",
    "paradigm": "frequentist",
    "approximate": false,
    "intervalKind": "confidence",
    "caveat": "A penalized point estimate with bootstrap CONFIDENCE intervals — not a posterior. Convergence diagnostics, posterior-predictive checks and prior-based views do not apply; they are reported as not applicable rather than passing."
  },
  {
    "value": "frequentist_cvxpy",
    "label": "Constrained LS (convex, bootstrap CIs)",
    "paradigm": "frequentist",
    "approximate": false,
    "intervalKind": "confidence",
    "caveat": "A penalized point estimate with bootstrap CONFIDENCE intervals — not a posterior. Convergence diagnostics, posterior-predictive checks and prior-based views do not apply; they are reported as not applicable rather than passing."
  }
];

const BY_VALUE: Record<string, InferenceMethodInfo> = Object.fromEntries(
  INFERENCE_METHODS.map((m) => [m.value, m]),
);

/** Descriptor for a method value, or `undefined` when it is not recognized.
 *
 *  Returns `undefined` rather than guessing — the guess is what broke: a
 *  `!(nuts|smc)` fallback classified every unknown value as approximate. */
export function methodInfo(value: string | null | undefined): InferenceMethodInfo | undefined {
  if (!value) return undefined;
  return BY_VALUE[String(value).trim().toLowerCase()];
}

/** Human label, falling back to the raw value for an unrecognized method. */
export function methodLabel(value: string | null | undefined): string {
  return methodInfo(value)?.label ?? String(value ?? '');
}
