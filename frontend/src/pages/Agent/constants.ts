import { API_BASE_URL, expertHeaders, bearerHeader } from '../../api/client';
import type { TableRef } from './types';

// Mapping: tool name → tab it produced an artifact in. Only tools that mutate
// some visible workspace state appear here; pure reads (list_*, get_session_status)
// don't get a navigation link.
// Tab ids here must be REAL WorkspaceTabs ids (plan/data/model/results/
// validation/experiments/library) — legacy ids (workflow/causal/eda/plots/
// artifacts) only worked via the index.tsx alias map.
export const TOOL_TO_TAB: Record<string, { tab: string; label: string }> = {
  // Workflow + causal planning (DAG + identification + assumptions log)
  mark_workflow_step:               { tab: 'plan',      label: 'Plan' },
  propose_dag:                      { tab: 'plan',      label: 'Plan' },
  validate_causal_identification:   { tab: 'plan',      label: 'Plan' },
  define_research_question:         { tab: 'plan',      label: 'Plan' },
  record_assumption:                { tab: 'plan',      label: 'Plan' },
  define_analysis_plan:             { tab: 'plan',      label: 'Plan' },
  prior_predictive_check:           { tab: 'plan',      label: 'Plan' },
  leave_one_out_decomposition:      { tab: 'plan',      label: 'Plan' },
  // Data (incl. EDA: data quality, outliers)
  generate_synthetic_data:          { tab: 'data',      label: 'Data' },
  inspect_dataset:                  { tab: 'data',      label: 'Data' },
  validate_data:                    { tab: 'data',      label: 'Data' },
  run_eda:                          { tab: 'data',      label: 'Data' },
  detect_outliers:                  { tab: 'data',      label: 'Data' },
  apply_outlier_treatment:          { tab: 'data',      label: 'Data' },
  // Model
  configure_model:                  { tab: 'model',     label: 'Model' },
  update_model_setting:             { tab: 'model',     label: 'Model' },
  load_config:                      { tab: 'model',     label: 'Model' },
  // Results (fit + analysis + REPL plots/tables/code)
  fit_mmm_model:                    { tab: 'results',   label: 'Results' },
  load_fitted_model:                { tab: 'results',   label: 'Results' },
  get_roi_metrics:                  { tab: 'results',   label: 'Results' },
  get_component_decomposition:      { tab: 'results',   label: 'Results' },
  get_model_diagnostics:            { tab: 'results',   label: 'Results' },
  get_adstock_weights:              { tab: 'results',   label: 'Results' },
  get_saturation_curves:            { tab: 'results',   label: 'Results' },
  execute_python:                   { tab: 'results',   label: 'Results' },
  // Reporting (project docs live in the Library tab)
  generate_project_report:          { tab: 'library',   label: 'Library' },
};

// Single source of truth from the API client: relative "/api" in dev (proxied),
// or VITE_API_URL when set. Keeps all fetch() calls on the same origin as the app.
export const API_BASE = API_BASE_URL;

export function authHeaders(apiKey: string | null, modelName: string | null): HeadersInit {
  const h: Record<string, string> = { 'X-API-Key': apiKey || '', 'X-Model-Name': modelName || '' };
  // Optional provider/base-url overrides; harmless for non-chat routes.
  const baseUrl = (typeof localStorage !== 'undefined' && localStorage.getItem('mmm_base_url')) || '';
  const provider = (typeof localStorage !== 'undefined' && localStorage.getItem('mmm_provider')) || '';
  if (baseUrl) h['X-Base-Url'] = baseUrl;
  if (provider) h['X-Provider'] = provider;
  // Expert (strong) tier selection for delegate_to_expert.
  Object.assign(h, expertHeaders());
  // JWT bearer (optional, additive); {} when no token is stored.
  Object.assign(h, bearerHeader());
  return h;
}

// Groups the backend stamps on table refs. Tables with a missing/unknown group
// fall into the "repl" bucket so per-group selection stays total. (The Results
// tab now renders ALL non-eda tables via the grouped analysis timeline; this
// selector remains for group-scoped surfaces like the Data tab's EDA tables.)
const KNOWN_TABLE_GROUPS = new Set(['eda', 'results', 'repl', 'causal', 'validation', 'garden']);

export function selectTables(tables: TableRef[] | undefined, group: string): TableRef[] {
  if (!tables || tables.length === 0) return [];
  return tables.filter(t => {
    const g = t.group && KNOWN_TABLE_GROUPS.has(t.group) ? t.group : 'repl';
    return g === group;
  });
}

export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
