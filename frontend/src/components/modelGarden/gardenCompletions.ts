import type { BeforeMount, OnMount } from '@monaco-editor/react';
import type { LintProblem } from '../../api/services/copilotService';

// ─────────────────────────────────────────────────────────────────────────────
// Atelier editor intelligence
// ─────────────────────────────────────────────────────────────────────────────
// Framework-aware IntelliSense for authoring garden models: snippet + symbol
// completions (member-aware after `self.` / `pm.` / `pt.` / `np.`), signature
// help, hover docs, a refined cream/sage theme, and a lint→marker bridge. Shared
// by the main Atelier editor AND the notebook code cells from a SINGLE
// registration (Monaco providers are global per monaco instance).
//
// The knowledge here is derived from the real CustomMMM authoring contract (see
// agents/garden_authoring.py) so completions teach the actual conventions.

type SymKind = 'method' | 'property' | 'class' | 'function' | 'snippet';

interface SymbolDef {
  label: string;
  kind: SymKind;
  /** Inserted text; defaults to `label`. Use ${n} placeholders when snippet. */
  insert?: string;
  snippet?: boolean;
  /** Short type/signature shown to the right of the label. */
  detail: string;
  /** Markdown documentation (completion details + hover). */
  doc: string;
}

// ── Snippets (multi-line scaffolds) ─────────────────────────────────────────
export const GARDEN_SNIPPETS: SymbolDef[] = [
  {
    label: 'custommmm',
    kind: 'snippet',
    snippet: true,
    detail: 'CustomMMM subclass skeleton',
    doc: 'Minimal garden model: subclass CustomMMM and override _build_model.',
    insert: [
      'from mmm_framework.garden import CustomMMM',
      'import pymc as pm',
      'import pytensor.tensor as pt',
      '',
      '',
      'class ${1:MyMMM}(CustomMMM):',
      '    """${2:What makes this model bespoke and when to use it.}"""',
      '',
      '    def _build_model(self) -> pm.Model:',
      '        coords = self._build_coords()',
      '        x_media = self._prepare_raw_media_for_model()',
      '        with pm.Model(coords=coords) as model:',
      '            $0',
      '        return model',
      '',
      '',
      'GARDEN_MODEL = ${1:MyMMM}',
    ].join('\n'),
  },
  {
    label: 'build_model',
    kind: 'snippet',
    snippet: true,
    detail: 'override _build_model (contract-complete)',
    doc: 'A _build_model registering every deterministic the read-ops consume.',
    insert: [
      'def _build_model(self) -> pm.Model:',
      '    coords = self._build_coords()',
      '    x_media_norm = self._prepare_raw_media_for_model()',
      '    n_obs = self.n_obs',
      '    with pm.Model(coords=coords) as model:',
      '        x_media = pm.Data("X_media_raw", x_media_norm, dims=("obs", "channel"))',
      '        intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)',
      '        pm.Deterministic("intercept_component", intercept + pt.zeros(n_obs), dims="obs")',
      '        contribs = []',
      '        for c, ch in enumerate(self.channel_names):',
      '            sat_kind, sat_params = self._build_channel_saturation(ch)',
      '            x_sat = _apply_saturation_pt(x_media[:, c], sat_kind, sat_params)',
      '            beta = pm.Gamma(f"beta_{ch}", mu=1.5, sigma=1.0)',
      '            contribs.append(beta * x_sat)',
      '        channels = pt.stack(contribs, axis=1)',
      '        pm.Deterministic("channel_contributions", channels, dims=("obs", "channel"))',
      '        media_total = channels.sum(axis=1)',
      '        pm.Deterministic("media_total", media_total)',
      '        sigma = pm.HalfNormal("sigma", sigma=0.5)',
      '        y_obs = pm.Normal("y_obs", mu=intercept + media_total, sigma=sigma, observed=self.y, dims="obs")',
      '        pm.Deterministic("y_obs_scaled", y_obs * self.y_std + self.y_mean, dims="obs")',
      '    return model',
    ].join('\n'),
  },
  {
    label: 'vectorized_adstock',
    kind: 'snippet',
    snippet: true,
    detail: 'geometric carryover WITHOUT pytensor.scan',
    doc: 'Lower-triangular Toeplitz matmul: Sₜ = Σ ρ^(t-τ)·xₜ. Compiles instantly.',
    insert: [
      'import numpy as np',
      '',
      't = np.arange(n_obs)',
      'lag = t[:, None] - t[None, :]',
      'decay = pt.where(',
      '    pt.as_tensor_variable(lag >= 0),',
      '    ${1:rho} ** pt.as_tensor_variable(np.maximum(lag, 0)),',
      '    0.0,',
      ')',
      '${2:carryover} = decay @ ${3:media_inflow}  # (n_obs, n_channel), no scan',
    ].join('\n'),
  },
  {
    label: 'deterministic_channels',
    kind: 'snippet',
    snippet: true,
    detail: 'register channel_contributions + media_total',
    doc: 'The two deterministics ROI/decomposition reporting needs.',
    insert: [
      'pm.Deterministic("channel_contributions", ${1:channels}, dims=("obs", "channel"))',
      'pm.Deterministic("media_total", ${1:channels}.sum(axis=1))',
    ].join('\n'),
  },
  {
    label: 'config_schema',
    kind: 'snippet',
    snippet: true,
    detail: 'declare settable params (CONFIG_SCHEMA)',
    doc: 'A pydantic CONFIG_SCHEMA → settable/validated/serialized params via self.model_params.',
    insert: [
      'from pydantic import BaseModel, Field',
      '',
      '',
      'class ${1:MyParams}(BaseModel):',
      '    ${2:retention}: float = Field(default=${3:0.75}, ge=0.0, le=1.0)',
      '',
      '',
      '# In your class body:',
      '#     CONFIG_SCHEMA = ${1:MyParams}',
      '# In _build_model:  n = self.model_params.${2:retention}',
    ].join('\n'),
  },
  {
    label: 'prior_normal',
    kind: 'snippet',
    snippet: true,
    detail: 'pm.Normal prior',
    doc: 'Normal prior.',
    insert: 'pm.Normal("${1:name}", mu=${2:0.0}, sigma=${3:1.0})',
  },
  {
    label: 'prior_retention',
    kind: 'snippet',
    snippet: true,
    detail: 'pm.Beta carryover/retention prior',
    doc: 'Beta(6, 2) ≈ mean 0.75 — a sticky carryover rate on (0, 1).',
    insert: 'pm.Beta("${1:adstock_alpha_channel}", alpha=${2:6.0}, beta=${3:2.0})',
  },
];

// ── `self.` members — base-class helpers + instance attributes to REUSE ──────
const SELF_MEMBERS: SymbolDef[] = [
  {
    label: '_build_coords',
    kind: 'method',
    insert: '_build_coords()',
    detail: '() -> dict',
    doc: 'PyMC coords for the model (obs / channel / control / fourier …). Pass to `pm.Model(coords=...)`.',
  },
  {
    label: '_prepare_raw_media_for_model',
    kind: 'method',
    insert: '_prepare_raw_media_for_model()',
    detail: '() -> np.ndarray  # (n_obs, n_channel)',
    doc: 'Normalized [0, 1] spend matrix. Build the `X_media_raw` pm.Data from this — the parametric path predict()/sample_channel_contributions() swap it.',
  },
  {
    label: '_build_channel_saturation',
    kind: 'method',
    insert: '_build_channel_saturation(${1:ch})',
    snippet: true,
    detail: '(ch) -> (kind, params)  # creates sat_*_<ch> RVs',
    doc: 'Create the saturation RVs (`sat_*_<ch>`) for one channel; pair the return with `_apply_saturation_pt(x, kind, params)` from `mmm_framework.model.base`.',
  },
  {
    label: '_build_trend_component',
    kind: 'method',
    insert: '_build_trend_component(${1:model}, ${2:time_idx})',
    snippet: true,
    detail: '(model, time_idx)',
    doc: 'Build the configured trend (linear / piecewise / spline / GP) and register `trend_component`.',
  },
  {
    label: '_build_control_betas',
    kind: 'method',
    insert: '_build_control_betas(${1:sigma})',
    snippet: true,
    detail: '(sigma) -> beta_controls',
    doc: 'Control coefficients honoring causal roles / variable selection.',
  },
  {
    label: '_sample_from_prior_config',
    kind: 'method',
    insert: '_sample_from_prior_config(${1:name}, ${2:prior}, ${3:default})',
    snippet: true,
    detail: '(name, prior, default)',
    doc: 'Sample an RV from a PriorConfig (honors per-channel ROI priors).',
  },
  { label: 'channel_names', kind: 'property', detail: 'list[str]', doc: 'Media channel names, in column order of the media matrix.' },
  { label: 'y', kind: 'property', detail: 'np.ndarray  # standardized KPI', doc: 'Standardized KPI: `(y_raw - y_mean) / y_std`. Pass as `observed=` for a Gaussian likelihood.' },
  { label: 'y_mean', kind: 'property', detail: 'float', doc: 'KPI mean used for standardization. Add it back (× nothing) to read the level in original units.' },
  { label: 'y_std', kind: 'property', detail: 'float', doc: 'KPI std used for standardization. Multiply a standardized component by this to read original units.' },
  { label: '_media_raw_max', kind: 'property', detail: 'dict[str, float]', doc: 'Per-channel max raw spend used to normalize media to [0, 1].' },
  { label: 'panel', kind: 'property', detail: 'PanelDataset', doc: 'The fitted panel (raw data + metadata).' },
  { label: 'model_config', kind: 'property', detail: 'ModelConfig', doc: 'The resolved model configuration (priors, likelihood, fit method, …).' },
  { label: 'model_params', kind: 'property', detail: 'CONFIG_SCHEMA instance', doc: 'Your validated bespoke params (when you declare a `CONFIG_SCHEMA`). e.g. `self.model_params.number_of_trials`.' },
  { label: 'trend_config', kind: 'property', detail: 'TrendConfig | None', doc: 'Trend configuration (None → no trend).' },
  { label: 'has_geo', kind: 'property', detail: 'bool', doc: 'True for a geo panel. Run any recursion INDEPENDENTLY per cell when set.' },
  { label: 'has_product', kind: 'property', detail: 'bool', doc: 'True for a product panel.' },
  { label: 'n_obs', kind: 'property', detail: 'int', doc: 'Number of observation rows.' },
  { label: 'n_channels', kind: 'property', detail: 'int', doc: 'Number of media channels.' },
  { label: 'n_controls', kind: 'property', detail: 'int', doc: 'Number of control variables.' },
  { label: 'n_periods', kind: 'property', detail: 'int', doc: 'Number of distinct time periods.' },
  { label: 'time_idx', kind: 'property', detail: 'np.ndarray  # (n_obs,)', doc: 'Period index per observation. Register/swap as the `time_idx` pm.Data.' },
  { label: 'geo_idx', kind: 'property', detail: 'np.ndarray  # (n_obs,)', doc: 'Geo cell index per observation (geo panels).' },
  { label: 'product_idx', kind: 'property', detail: 'np.ndarray  # (n_obs,)', doc: 'Product cell index per observation (product panels).' },
  { label: 'X_controls', kind: 'property', detail: 'np.ndarray  # (n_obs, n_control)', doc: 'Standardized control matrix.' },
  { label: 'seasonality_features', kind: 'property', detail: 'dict', doc: 'Fourier seasonality feature arrays.' },
];

// ── `pm.` symbols — PyMC distributions + model primitives ────────────────────
const PM_SYMBOLS: SymbolDef[] = [
  { label: 'Normal', kind: 'class', insert: 'Normal("${1:name}", mu=${2:0.0}, sigma=${3:1.0})', snippet: true, detail: '(name, mu, sigma, *, dims, observed)', doc: 'Normal (Gaussian) distribution.' },
  { label: 'HalfNormal', kind: 'class', insert: 'HalfNormal("${1:name}", sigma=${2:1.0})', snippet: true, detail: '(name, sigma)', doc: 'Half-Normal — a positive scale prior (e.g. for sigma).' },
  { label: 'LogNormal', kind: 'class', insert: 'LogNormal("${1:name}", mu=${2:0.0}, sigma=${3:1.0})', snippet: true, detail: '(name, mu, sigma)', doc: 'Log-Normal — positive, right-skewed.' },
  { label: 'StudentT', kind: 'class', insert: 'StudentT("${1:name}", nu=${2:4.0}, mu=${3:0.0}, sigma=${4:1.0})', snippet: true, detail: '(name, nu, mu, sigma)', doc: "Student-T — heavy-tailed; a robust likelihood." },
  { label: 'HalfCauchy', kind: 'class', insert: 'HalfCauchy("${1:name}", beta=${2:1.0})', snippet: true, detail: '(name, beta)', doc: 'Half-Cauchy — a weakly-informative positive scale prior.' },
  { label: 'Exponential', kind: 'class', insert: 'Exponential("${1:name}", lam=${2:1.0})', snippet: true, detail: '(name, lam)', doc: 'Exponential — positive support.' },
  { label: 'Gamma', kind: 'class', insert: 'Gamma("${1:name}", mu=${2:1.5}, sigma=${3:1.0})', snippet: true, detail: '(name, alpha, beta | mu, sigma)', doc: 'Gamma — positive support; common for a media `beta_<ch>` so the contribution stays ≥ 0.' },
  { label: 'InverseGamma', kind: 'class', insert: 'InverseGamma("${1:name}", alpha=${2:3.0}, beta=${3:1.0})', snippet: true, detail: '(name, alpha, beta)', doc: 'Inverse-Gamma — a positive scale prior.' },
  { label: 'Beta', kind: 'class', insert: 'Beta("${1:name}", alpha=${2:6.0}, beta=${3:2.0})', snippet: true, detail: '(name, alpha, beta)', doc: 'Beta on (0, 1) — a carryover/retention `adstock_alpha_<ch>` rate.' },
  { label: 'Uniform', kind: 'class', insert: 'Uniform("${1:name}", lower=${2:0.0}, upper=${3:1.0})', snippet: true, detail: '(name, lower, upper)', doc: 'Uniform on [lower, upper].' },
  { label: 'Bernoulli', kind: 'class', insert: 'Bernoulli("${1:name}", p=${2:0.5})', snippet: true, detail: '(name, p)', doc: 'Bernoulli — binary outcomes.' },
  { label: 'Binomial', kind: 'class', insert: 'Binomial("${1:y_obs}", n=${2:self.model_params.number_of_trials}, p=${3:p}, observed=self.y, dims="obs")', snippet: true, detail: '(name, n, p, *, observed, dims)', doc: 'Binomial — a survey/awareness count KPI. Map a standardized mean to (0, 1) with `pm.math.sigmoid` first; binomial KPIs are NOT standardized.' },
  { label: 'Poisson', kind: 'class', insert: 'Poisson("${1:name}", mu=${2:mu})', snippet: true, detail: '(name, mu)', doc: 'Poisson — counts. Keep `mu` positive (e.g. via `pm.math.exp`).' },
  { label: 'NegativeBinomial', kind: 'class', insert: 'NegativeBinomial("${1:name}", mu=${2:mu}, alpha=${3:alpha})', snippet: true, detail: '(name, mu, alpha)', doc: 'Negative-Binomial — over-dispersed counts.' },
  { label: 'Dirichlet', kind: 'class', insert: 'Dirichlet("${1:name}", a=${2:a})', snippet: true, detail: '(name, a)', doc: 'Dirichlet — simplex (e.g. mixing weights).' },
  { label: 'MvNormal', kind: 'class', insert: 'MvNormal("${1:name}", mu=${2:mu}, cov=${3:cov})', snippet: true, detail: '(name, mu, cov | chol)', doc: 'Multivariate Normal.' },
  { label: 'Categorical', kind: 'class', insert: 'Categorical("${1:name}", p=${2:p})', snippet: true, detail: '(name, p)', doc: 'Categorical — discrete class labels (integrate out for NUTS).' },
  { label: 'ZeroSumNormal', kind: 'class', insert: 'ZeroSumNormal("${1:name}", sigma=${2:1.0}, dims=${3:"geo"})', snippet: true, detail: '(name, sigma, *, dims)', doc: 'Sum-to-zero Normal — identified random effects (geo/product offsets).' },
  { label: 'Deterministic', kind: 'function', insert: 'Deterministic("${1:name}", ${2:expr}, dims=${3:"obs"})', snippet: true, detail: '(name, var, *, dims)', doc: 'Register a named deterministic the read-ops consume — `channel_contributions`, `media_total`, `y_obs_scaled`, `beta_<ch>`, …' },
  { label: 'Potential', kind: 'function', insert: 'Potential("${1:name}", ${2:logp})', snippet: true, detail: '(name, var)', doc: 'Add an arbitrary term to the model log-density (e.g. a marginalized mixture logsumexp).' },
  { label: 'Data', kind: 'function', insert: 'Data("${1:X_media_raw}", ${2:value}, dims=${3:("obs", "channel")})', snippet: true, detail: '(name, value, *, dims)', doc: 'A mutable data container predict()/set_data swap. Name it `X_media_raw` / `X_controls` / `time_idx` to match the swap contract.' },
  { label: 'ConstantData', kind: 'function', insert: 'ConstantData("${1:name}", ${2:value}, dims=${3:"obs"})', snippet: true, detail: '(name, value, *, dims)', doc: 'An immutable data container recorded in the trace.' },
  { label: 'Model', kind: 'class', insert: 'Model(coords=${1:coords})', snippet: true, detail: '(*, coords)', doc: 'The model context: `with pm.Model(coords=self._build_coords()) as model:`.' },
  { label: 'math', kind: 'property', detail: 'module', doc: 'PyMC math ops: `pm.math.sigmoid`, `pm.math.exp`, `pm.math.log`, `pm.math.dot`, `pm.math.stack`, `pm.math.clip`, …' },
  { label: 'sample', kind: 'function', insert: 'sample(draws=${1:1000}, tune=${2:1000}, chains=${3:4})', snippet: true, detail: '(draws, tune, chains, ...)', doc: 'Run NUTS. In the Atelier prefer `mmm.fit(method="map")` for a fast structural check first.' },
  { label: 'sample_prior_predictive', kind: 'function', insert: 'sample_prior_predictive(draws=${1:500})', snippet: true, detail: '(draws)', doc: 'Draw from the prior — sanity-check priors before fitting.' },
  { label: 'sample_posterior_predictive', kind: 'function', insert: 'sample_posterior_predictive(${1:idata}, var_names=[${2:"y_obs"}])', snippet: true, detail: '(trace, *, var_names)', doc: 'Posterior predictive draws (re-evaluating named deterministics).' },
  { label: 'set_data', kind: 'function', insert: 'set_data({${1:"X_media_raw"}: ${2:value}})', snippet: true, detail: '(new_data)', doc: 'Swap a pm.Data container (counterfactual / prediction).' },
];

// ── `pt.` symbols — pytensor.tensor ops ─────────────────────────────────────
const PT_SYMBOLS: SymbolDef[] = [
  { label: 'stack', kind: 'function', insert: 'stack(${1:tensors}, axis=${2:1})', snippet: true, detail: '(tensors, axis)', doc: 'Stack a list of tensors along a new axis — build `(n_obs, n_channel)` from per-channel contributions.' },
  { label: 'concatenate', kind: 'function', insert: 'concatenate(${1:tensors}, axis=${2:0})', snippet: true, detail: '(tensors, axis)', doc: 'Join tensors along an existing axis.' },
  { label: 'zeros', kind: 'function', insert: 'zeros(${1:shape})', snippet: true, detail: '(shape)', doc: 'A tensor of zeros (e.g. `pt.zeros(n_obs)` to broadcast a scalar over obs).' },
  { label: 'ones', kind: 'function', insert: 'ones(${1:shape})', snippet: true, detail: '(shape)', doc: 'A tensor of ones.' },
  { label: 'exp', kind: 'function', insert: 'exp(${1:x})', snippet: true, detail: '(x)', doc: 'Elementwise exp — a positive inverse link.' },
  { label: 'log', kind: 'function', insert: 'log(${1:x})', snippet: true, detail: '(x)', doc: 'Elementwise natural log (guard against log(≤0)).' },
  { label: 'log1p', kind: 'function', insert: 'log1p(${1:x})', snippet: true, detail: '(x)', doc: 'log(1 + x), stable for small x.' },
  { label: 'sqrt', kind: 'function', insert: 'sqrt(${1:x})', snippet: true, detail: '(x)', doc: 'Elementwise square root.' },
  { label: 'sum', kind: 'function', insert: 'sum(${1:x}, axis=${2:1})', snippet: true, detail: '(x, axis)', doc: 'Reduce-sum (e.g. `media_total = channels.sum(axis=1)`).' },
  { label: 'dot', kind: 'function', insert: 'dot(${1:a}, ${2:b})', snippet: true, detail: '(a, b)', doc: 'Matrix/vector product. The vectorized adstock is `decay @ media_inflow`.' },
  { label: 'where', kind: 'function', insert: 'where(${1:cond}, ${2:a}, ${3:b})', snippet: true, detail: '(cond, a, b)', doc: 'Elementwise select — used to build the lower-triangular adstock decay matrix.' },
  { label: 'switch', kind: 'function', insert: 'switch(${1:cond}, ${2:a}, ${3:b})', snippet: true, detail: '(cond, a, b)', doc: 'Elementwise conditional (like where).' },
  { label: 'clip', kind: 'function', insert: 'clip(${1:x}, ${2:lo}, ${3:hi})', snippet: true, detail: '(x, min, max)', doc: 'Clamp into [min, max].' },
  { label: 'maximum', kind: 'function', insert: 'maximum(${1:a}, ${2:b})', snippet: true, detail: '(a, b)', doc: 'Elementwise max.' },
  { label: 'minimum', kind: 'function', insert: 'minimum(${1:a}, ${2:b})', snippet: true, detail: '(a, b)', doc: 'Elementwise min.' },
  { label: 'cumsum', kind: 'function', insert: 'cumsum(${1:x}, axis=${2:0})', snippet: true, detail: '(x, axis)', doc: 'Cumulative sum — cold-start adstock ramp via cumulative weights.' },
  { label: 'as_tensor_variable', kind: 'function', insert: 'as_tensor_variable(${1:x})', snippet: true, detail: '(x)', doc: 'Wrap a numpy array/scalar as a pytensor tensor (e.g. the integer lag matrix).' },
  { label: 'sigmoid', kind: 'function', insert: 'sigmoid(${1:x})', snippet: true, detail: '(x)', doc: 'Logistic sigmoid → (0, 1). Use for a bounded/probability mean.' },
  { label: 'softplus', kind: 'function', insert: 'softplus(${1:x})', snippet: true, detail: '(x)', doc: 'log(1 + exp(x)) → positive; a smooth positive inverse link.' },
  { label: 'set_subtensor', kind: 'function', insert: 'set_subtensor(${1:x}[${2:idx}], ${3:value})', snippet: true, detail: '(x[idx], value)', doc: 'Functional in-place update of a tensor slice.' },
];

// ── `np.` symbols — a light dusting for the host-side prep ───────────────────
const NP_SYMBOLS: SymbolDef[] = [
  { label: 'arange', kind: 'function', insert: 'arange(${1:n})', snippet: true, detail: '(n)', doc: 'Integer range — `t = np.arange(n_obs)` for the adstock lag matrix.' },
  { label: 'maximum', kind: 'function', insert: 'maximum(${1:a}, ${2:b})', snippet: true, detail: '(a, b)', doc: 'Elementwise max (e.g. `np.maximum(lag, 0)`).' },
  { label: 'zeros', kind: 'function', insert: 'zeros(${1:shape})', snippet: true, detail: '(shape)', doc: 'Zeros array.' },
  { label: 'ones', kind: 'function', insert: 'ones(${1:shape})', snippet: true, detail: '(shape)', doc: 'Ones array.' },
  { label: 'array', kind: 'function', insert: 'array(${1:obj})', snippet: true, detail: '(obj)', doc: 'Make an ndarray.' },
  { label: 'newaxis', kind: 'property', detail: 'None', doc: 'Insert a new axis: `t[:, None]` ≡ `t[:, np.newaxis]`.' },
  { label: 'linspace', kind: 'function', insert: 'linspace(${1:lo}, ${2:hi}, ${3:num})', snippet: true, detail: '(start, stop, num)', doc: 'Evenly spaced samples.' },
];

// Member catalogs keyed by trigger owner.
const MEMBERS: Record<string, SymbolDef[]> = {
  self: SELF_MEMBERS,
  pm: PM_SYMBOLS,
  pt: PT_SYMBOLS,
  np: NP_SYMBOLS,
};

// ── Signature help — parameter hints on `pm.X(` / `self._helper(` ───────────
interface SigDef {
  label: string;
  params: { label: string; doc?: string }[];
  doc?: string;
}
const SIGNATURES: Record<string, SigDef> = {
  'pm.Normal': {
    label: 'pm.Normal(name, mu=0.0, sigma=1.0, *, dims=None, observed=None)',
    params: [
      { label: 'name', doc: 'RV name (string).' },
      { label: 'mu=0.0', doc: 'Mean.' },
      { label: 'sigma=1.0', doc: 'Standard deviation (> 0).' },
      { label: 'dims=None', doc: 'Coord dims, e.g. "obs" or ("obs","channel").' },
      { label: 'observed=None', doc: 'Observed data → makes this a likelihood.' },
    ],
    doc: 'Normal (Gaussian) distribution.',
  },
  'pm.HalfNormal': {
    label: 'pm.HalfNormal(name, sigma=1.0, *, dims=None)',
    params: [
      { label: 'name' },
      { label: 'sigma=1.0', doc: 'Scale (> 0).' },
      { label: 'dims=None' },
    ],
    doc: 'Positive half-Normal scale prior.',
  },
  'pm.Gamma': {
    label: 'pm.Gamma(name, alpha=None, beta=None, *, mu=None, sigma=None, dims=None)',
    params: [
      { label: 'name' },
      { label: 'alpha', doc: 'Shape (or use mu/sigma).' },
      { label: 'beta', doc: 'Rate.' },
      { label: 'mu', doc: 'Mean parameterization.' },
      { label: 'sigma', doc: 'Std parameterization.' },
      { label: 'dims=None' },
    ],
    doc: 'Positive-support Gamma — common for a media beta so contribution ≥ 0.',
  },
  'pm.Beta': {
    label: 'pm.Beta(name, alpha=1.0, beta=1.0, *, dims=None)',
    params: [
      { label: 'name' },
      { label: 'alpha', doc: 'Beta(6, 2) ≈ mean 0.75.' },
      { label: 'beta' },
      { label: 'dims=None' },
    ],
    doc: 'Beta on (0, 1) — a carryover/retention rate.',
  },
  'pm.Binomial': {
    label: 'pm.Binomial(name, n, p, *, observed=None, dims=None)',
    params: [
      { label: 'name' },
      { label: 'n', doc: 'Number of trials (e.g. self.model_params.number_of_trials).' },
      { label: 'p', doc: 'Success probability in (0, 1) — map a mean via pm.math.sigmoid.' },
      { label: 'observed=None', doc: 'Observed success counts (NOT standardized).' },
      { label: 'dims=None' },
    ],
    doc: 'Binomial likelihood for a survey/awareness count KPI.',
  },
  'pm.Deterministic': {
    label: 'pm.Deterministic(name, var, *, dims=None)',
    params: [
      { label: 'name', doc: 'Contract name: channel_contributions, media_total, y_obs_scaled, beta_<ch>, …' },
      { label: 'var', doc: 'The tensor expression to record.' },
      { label: 'dims=None', doc: '"obs", ("obs","channel"), …' },
    ],
    doc: 'Register a named deterministic the read-ops consume.',
  },
  'pm.Data': {
    label: 'pm.Data(name, value, *, dims=None)',
    params: [
      { label: 'name', doc: 'Swap-contract name: X_media_raw / X_controls / time_idx.' },
      { label: 'value', doc: 'Initial array.' },
      { label: 'dims=None' },
    ],
    doc: 'Mutable data container that predict()/set_data swap.',
  },
  'pm.StudentT': {
    label: 'pm.StudentT(name, nu, mu=0.0, sigma=1.0, *, observed=None, dims=None)',
    params: [
      { label: 'name' },
      { label: 'nu', doc: 'Degrees of freedom (tail heaviness).' },
      { label: 'mu=0.0' },
      { label: 'sigma=1.0' },
      { label: 'observed=None' },
      { label: 'dims=None' },
    ],
    doc: 'Heavy-tailed Student-T — a robust likelihood.',
  },
  'self._build_channel_saturation': {
    label: 'self._build_channel_saturation(ch) -> (kind, params)',
    params: [{ label: 'ch', doc: 'Channel name from self.channel_names.' }],
    doc: 'Create the saturation RVs for one channel; pair with _apply_saturation_pt(x, kind, params).',
  },
  'self._sample_from_prior_config': {
    label: 'self._sample_from_prior_config(name, prior, default)',
    params: [
      { label: 'name', doc: 'RV name.' },
      { label: 'prior', doc: 'A PriorConfig (or None).' },
      { label: 'default', doc: 'Fallback when prior is None.' },
    ],
    doc: 'Sample an RV from a PriorConfig (honors per-channel ROI priors).',
  },
};

// ── Monaco wiring ───────────────────────────────────────────────────────────

function memberOwner(textUntil: string): keyof typeof MEMBERS | null {
  const m = /(^|[^\w.])(self|pm|pt|np)\.\w*$/.exec(textUntil);
  return m ? (m[2] as keyof typeof MEMBERS) : null;
}

/** Replace the interior of Python string literals (', ", and triple-quoted) and
 * `#` comments with spaces, preserving length + newlines, so a paren/comma scan
 * over the result ignores characters inside literals. Structural-only — not a
 * real tokenizer, but enough for the signature-help heuristic. */
function maskStringsAndComments(src: string): string {
  const out = src.split('');
  const n = src.length;
  let i = 0;
  let quote: string | null = null; // active delimiter: ', ", ''' or """
  let inComment = false;
  while (i < n) {
    const c = src[i];
    if (inComment) {
      if (c === '\n') inComment = false;
      else out[i] = ' ';
      i++;
      continue;
    }
    if (quote) {
      if (c === '\\') {
        out[i] = ' ';
        if (i + 1 < n) out[i + 1] = ' ';
        i += 2;
        continue;
      }
      if (src.startsWith(quote, i)) {
        i += quote.length; // closing delimiter (left intact)
        quote = null;
        continue;
      }
      out[i] = ' ';
      i++;
      continue;
    }
    if (c === '#') {
      inComment = true;
      out[i] = ' ';
      i++;
      continue;
    }
    if (c === '"' || c === "'") {
      const triple = c.repeat(3);
      quote = src.startsWith(triple, i) ? triple : c;
      i += quote.length; // opening delimiter (left intact)
      continue;
    }
    i++;
  }
  return out.join('');
}

type Monaco = typeof import('monaco-editor');
type CodeEditor = import('monaco-editor').editor.IStandaloneCodeEditor;
type ITextModel = import('monaco-editor').editor.ITextModel;
type IRange = import('monaco-editor').IRange;
type Position = import('monaco-editor').Position;

function kindOf(monaco: Monaco, d: SymbolDef) {
  const K = monaco.languages.CompletionItemKind;
  switch (d.kind) {
    case 'method':
      return K.Method;
    case 'property':
      return K.Property;
    case 'class':
      return K.Class;
    case 'function':
      return K.Function;
    default:
      return K.Snippet;
  }
}

function toItem(monaco: Monaco, d: SymbolDef, range: IRange) {
  const insert = d.insert ?? d.label;
  return {
    label: d.label,
    kind: kindOf(monaco, d),
    insertText: insert,
    insertTextRules: d.snippet
      ? monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
      : undefined,
    detail: d.detail,
    documentation: { value: d.doc },
    range,
  };
}

// Hover index: `${owner}:${label}` → markdown.
const HOVER = new Map<string, string>();
for (const [owner, defs] of Object.entries(MEMBERS)) {
  for (const d of defs) HOVER.set(`${owner}:${d.label}`, `**\`${d.detail}\`**\n\n${d.doc}`);
}

let registered = false;

/** Register the Atelier's completion / signature-help / hover providers once
 * (Monaco providers are global per monaco instance). Safe to call from every
 * editor's onMount. */
export const registerGardenCompletions: OnMount = (_editor, monaco) => {
  if (registered) return;
  registered = true;

  monaco.languages.registerCompletionItemProvider('python', {
    triggerCharacters: ['.'],
    provideCompletionItems(model: ITextModel, position: Position) {
      const word = model.getWordUntilPosition(position);
      const range: IRange = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      const textUntil = model.getValueInRange({
        startLineNumber: position.lineNumber,
        startColumn: 1,
        endLineNumber: position.lineNumber,
        endColumn: position.column,
      });
      const owner = memberOwner(textUntil);
      if (owner) {
        return { suggestions: MEMBERS[owner].map((d) => toItem(monaco, d, range)) };
      }
      // General context: snippets first, then bare distribution names so typing
      // `Norm…` still surfaces pm.Normal even without the `pm.` prefix.
      const bare = PM_SYMBOLS.filter((d) => d.kind === 'class').map((d) => ({
        ...d,
        insert: `pm.${d.insert ?? d.label}`,
      }));
      return {
        suggestions: [...GARDEN_SNIPPETS, ...bare].map((d) => toItem(monaco, d, range)),
      };
    },
  });

  monaco.languages.registerSignatureHelpProvider('python', {
    signatureHelpTriggerCharacters: ['(', ','],
    signatureHelpRetriggerCharacters: [','],
    provideSignatureHelp(model: ITextModel, position: Position) {
      // Look back far enough to cover verbose multi-line calls (one arg / line).
      const from = Math.max(1, position.lineNumber - 16);
      const code = model.getValueInRange({
        startLineNumber: from,
        startColumn: 1,
        endLineNumber: position.lineNumber,
        endColumn: position.column,
      });
      // Mask string / comment spans so parens & commas inside literals (e.g.
      // pm.Deterministic("media )total", x) don't corrupt the scan. Same length
      // as `code`, so indices map 1:1 back to the original for token extraction.
      const masked = maskStringsAndComments(code);
      // Walk back to the nearest unmatched '('.
      let depth = 0;
      let open = -1;
      for (let i = masked.length - 1; i >= 0; i--) {
        const ch = masked[i];
        if (ch === ')') depth++;
        else if (ch === '(') {
          if (depth === 0) {
            open = i;
            break;
          }
          depth--;
        }
      }
      if (open < 0) return null;
      const tok = /([A-Za-z_][\w.]*)\s*$/.exec(code.slice(0, open));
      if (!tok) return null;
      const name = tok[1];
      const sig =
        SIGNATURES[name] ?? SIGNATURES[`pm.${name.split('.').pop()}`] ?? null;
      if (!sig) return null;
      // Active parameter = top-level commas after the open paren (masked).
      let d = 0;
      let commas = 0;
      for (const ch of masked.slice(open + 1)) {
        if (ch === '(' || ch === '[' || ch === '{') d++;
        else if (ch === ')' || ch === ']' || ch === '}') d--;
        else if (ch === ',' && d === 0) commas++;
      }
      return {
        value: {
          signatures: [
            {
              label: sig.label,
              documentation: sig.doc ? { value: sig.doc } : undefined,
              parameters: sig.params.map((p) => ({
                label: p.label,
                documentation: p.doc ? { value: p.doc } : undefined,
              })),
            },
          ],
          activeSignature: 0,
          activeParameter: Math.min(commas, sig.params.length - 1),
        },
        dispose() {},
      };
    },
  });

  monaco.languages.registerHoverProvider('python', {
    provideHover(model: ITextModel, position: Position) {
      const word = model.getWordAtPosition(position);
      if (!word) return null;
      const before = model
        .getLineContent(position.lineNumber)
        .slice(0, word.startColumn - 1);
      const owner = /(self|pm|pt|np)\.$/.exec(before)?.[1] ?? null;
      const md = owner ? HOVER.get(`${owner}:${word.word}`) : null;
      if (!md) return null;
      return {
        range: {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn,
        },
        contents: [{ value: md }],
      };
    },
  });
};

/** Define the Atelier's cream/sage editor theme. Call from `beforeMount` and set
 * `theme="atelier-light"`. */
export const defineAtelierTheme: BeforeMount = (monaco) => {
  monaco.editor.defineTheme('atelier-light', {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '8a9384', fontStyle: 'italic' },
      { token: 'keyword', foreground: '3f7a4e' },
      { token: 'keyword.flow', foreground: 'b1632b' },
      { token: 'string', foreground: 'a76d2c' },
      { token: 'string.escape', foreground: 'b1632b' },
      { token: 'number', foreground: '9c5a1e' },
      { token: 'type', foreground: '6a5acd' },
      { token: 'type.identifier', foreground: '6a5acd' },
      { token: 'identifier', foreground: '283326' },
      { token: 'delimiter', foreground: '6b7568' },
      { token: 'function', foreground: '2f6f57' },
    ],
    colors: {
      'editor.background': '#fcfbf6',
      'editor.foreground': '#283326',
      'editorLineNumber.foreground': '#b8c0b2',
      'editorLineNumber.activeForeground': '#5a6a52',
      'editor.selectionBackground': '#d8e6d2',
      'editor.inactiveSelectionBackground': '#e9efe3',
      'editor.lineHighlightBackground': '#f1efe4',
      'editorCursor.foreground': '#3f7a4e',
      'editorIndentGuide.background': '#e8e4d6',
      'editorIndentGuide.activeBackground': '#cfd8c7',
      'editorBracketMatch.background': '#d8e6d2',
      'editorBracketMatch.border': '#7fa67f',
      'editorWhitespace.foreground': '#e0ddcf',
      'editorGutter.background': '#fcfbf6',
    },
  });
};

/** Bridge backend lint problems to inline Monaco markers (squiggles). Problems
 * without a line are panel-only (markers need a position). */
export function applyLintMarkers(
  monaco: Monaco,
  editor: CodeEditor,
  problems: LintProblem[],
): void {
  const model = editor.getModel();
  if (!model) return;
  const markers = problems
    .filter((p) => p.line != null)
    .map((p) => {
      const line = p.line as number;
      const startColumn = p.column ?? 1;
      let endColumn = p.end_column ?? null;
      const endLineNumber = p.end_line ?? line;
      if (endColumn == null) {
        try {
          endColumn = model.getLineMaxColumn(line);
        } catch {
          endColumn = startColumn + 1;
        }
      }
      const severity =
        p.severity === 'error'
          ? monaco.MarkerSeverity.Error
          : p.severity === 'warning'
            ? monaco.MarkerSeverity.Warning
            : monaco.MarkerSeverity.Info;
      return {
        severity,
        message: p.message,
        startLineNumber: line,
        startColumn,
        endLineNumber,
        endColumn: Math.max(endColumn, startColumn + 1),
        source: 'atelier',
      };
    });
  monaco.editor.setModelMarkers(model, 'atelier-lint', markers);
}
