import { API_BASE_URL, expertHeaders, bearerHeader } from '../../api/client';
import type { TableRef } from './types';

// Mapping: tool name → tab it produced an artifact in. Only tools that mutate
// some visible workspace state appear here; pure reads (list_*, get_session_status)
// don't get a navigation link.
export const TOOL_TO_TAB: Record<string, { tab: string; label: string }> = {
  // Workflow
  mark_workflow_step:               { tab: 'workflow',  label: 'Workflow' },
  // Causal (DAG + identification + assumptions log)
  propose_dag:                      { tab: 'causal',    label: 'Causal' },
  validate_causal_identification:   { tab: 'causal',    label: 'Causal' },
  define_research_question:         { tab: 'causal',    label: 'Causal' },
  record_assumption:                { tab: 'causal',    label: 'Causal' },
  define_analysis_plan:             { tab: 'causal',    label: 'Causal' },
  prior_predictive_check:           { tab: 'causal',    label: 'Causal' },
  leave_one_out_decomposition:      { tab: 'causal',    label: 'Causal' },
  // Data
  generate_synthetic_data:          { tab: 'data',      label: 'Data' },
  inspect_dataset:                  { tab: 'data',      label: 'Data' },
  // EDA (data quality, outliers)
  validate_data:                    { tab: 'eda',       label: 'EDA' },
  run_eda:                          { tab: 'eda',       label: 'EDA' },
  detect_outliers:                  { tab: 'eda',       label: 'EDA' },
  apply_outlier_treatment:          { tab: 'eda',       label: 'EDA' },
  // Model
  configure_model:                  { tab: 'model',     label: 'Model' },
  update_model_setting:             { tab: 'model',     label: 'Model' },
  load_config:                      { tab: 'model',     label: 'Model' },
  // Results (fit + analysis)
  fit_mmm_model:                    { tab: 'results',   label: 'Results' },
  load_fitted_model:                { tab: 'results',   label: 'Results' },
  get_roi_metrics:                  { tab: 'results',   label: 'Results' },
  get_component_decomposition:      { tab: 'results',   label: 'Results' },
  get_model_diagnostics:            { tab: 'results',   label: 'Results' },
  get_adstock_weights:              { tab: 'results',   label: 'Results' },
  get_saturation_curves:            { tab: 'results',   label: 'Results' },
  // Artifacts (code snippets, plots, REPL output)
  execute_python:                   { tab: 'plots',     label: 'Plots' },
  // Reporting
  generate_project_report:          { tab: 'artifacts', label: 'Artifacts' },
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
// fall into the "repl" bucket so they still surface somewhere (the Plots tab).
const KNOWN_TABLE_GROUPS = new Set(['eda', 'results', 'repl', 'causal']);

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
