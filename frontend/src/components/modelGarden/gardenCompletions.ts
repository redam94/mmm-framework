import type { OnMount } from '@monaco-editor/react';

// ── IDE: framework-aware autocomplete snippets ──────────────────────────────
// Shared by the Atelier editor AND the notebook code cells so both get the same
// garden-model completions from a SINGLE registration (Monaco completion
// providers are global per monaco instance — register once, app-wide).
export const GARDEN_SNIPPETS: {
  label: string;
  detail: string;
  doc: string;
  insert: string;
}[] = [
  {
    label: 'custommmm',
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
    detail: 'register channel_contributions + media_total',
    doc: 'The two deterministics ROI/decomposition reporting needs.',
    insert: [
      'pm.Deterministic("channel_contributions", ${1:channels}, dims=("obs", "channel"))',
      'pm.Deterministic("media_total", ${1:channels}.sum(axis=1))',
    ].join('\n'),
  },
  {
    label: 'prior_normal',
    detail: 'pm.Normal prior',
    doc: 'Normal prior.',
    insert: 'pm.Normal("${1:name}", mu=${2:0.0}, sigma=${3:1.0})',
  },
  {
    label: 'prior_retention',
    detail: 'pm.Beta carryover/retention prior',
    doc: 'Beta(6, 2) ≈ mean 0.75 — a sticky carryover rate on (0, 1).',
    insert: 'pm.Beta("${1:adstock_alpha_channel}", alpha=${2:6.0}, beta=${3:2.0})',
  },
];

// Register framework completions once for the whole app lifetime.
let gardenCompletionsRegistered = false;
export const registerGardenCompletions: OnMount = (_editor, monaco) => {
  if (gardenCompletionsRegistered) return;
  gardenCompletionsRegistered = true;
  monaco.languages.registerCompletionItemProvider('python', {
    provideCompletionItems(
      model: import('monaco-editor').editor.ITextModel,
      position: import('monaco-editor').Position,
    ) {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      return {
        suggestions: GARDEN_SNIPPETS.map((s) => ({
          label: s.label,
          kind: monaco.languages.CompletionItemKind.Snippet,
          insertText: s.insert,
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: s.detail,
          documentation: s.doc,
          range,
        })),
      };
    },
  });
};
