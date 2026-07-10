// ─── Types ───────────────────────────────────────────────────────────────────

export interface ToolCall {
  id: string;
  name: string;
  // Tool args are a server-driven JSON blob; consumers (e.g. useChatStream) read
  // arbitrary keys (args.code) straight into string positions — keep `any` so this
  // exported shape stays as permissive as before for un-editable callers.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  args?: Record<string, any>;
  result?: string;
  status: 'running' | 'done' | 'error';
}

export interface ChatMessage {
  id: string;
  type: 'human' | 'ai' | 'error';
  content: string;
  toolCalls?: ToolCall[];
}

// ─── Python output ───────────────────────────────────────────────────────────

export interface PythonOutput {
  id: string;
  code: string;
  output: string;
  hasError: boolean;
  plotCount: number;
}

// ─── Session + Artifact types ────────────────────────────────────────────────

export interface Session {
  thread_id: string;
  name: string;
  created_at: number;
  updated_at: number;
}

export interface Artifact {
  id: string;
  thread_id: string;
  kind: 'code_snippet' | 'report' | 'model_run' | 'project_report' | 'project_slides' | 'text_output' | string;
  // Server-driven JSON blob whose shape varies by `kind`; consumers (ArtifactsPanel,
  // index, useChatStream) read arbitrary nested keys (payload.code, payload.inference?.draws,
  // payload.call_id as an index) into typed positions — keep `any` so this exported shape
  // stays as permissive as before for un-editable callers.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  payload: any;
  created_at: number;
}

// ─── Project + KB + Workspace types ──────────────────────────────────────────

export interface Project {
  project_id: string;
  name: string;
  description?: string | null;
  session_count?: number;
  doc_count?: number;
  created_at: number;
  updated_at: number;
}

export interface KbDocument {
  id: string;
  name: string;
  kind: string;
  size_bytes: number | null;
  n_chunks: number;
  status: 'pending' | 'ready' | 'error' | string;
  created_at: number;
}

export interface KbSearchResult {
  document: string;
  chunk_index: number;
  text: string;
  score: number;
}

export interface WorkspaceFile {
  id: string;
  name: string;
  path: string;
  kind: string;
  size_bytes: number | null;
  created_at: number;
}

// ─── Dataset ─────────────────────────────────────────────────────────────────

export interface DatasetInfo {
  rows: number;
  columns: string[];
  date_range?: { min: string; max: string } | null;
  variable_names?: string[];
  geographies?: string[];
  column_stats?: Record<string, { unique: number; top_values: { value: string; count: number }[]; truncated: boolean }>;
  active_dimensions?: string[];
}

// ─── Model spec + flexible dataset roles ─────────────────────────────────────

export type DatasetRole =
  | 'target' | 'predictor' | 'control' | 'indicator'
  | 'group' | 'time' | 'offset' | 'weight' | 'trials' | 'auxiliary';

export interface RoleBinding {
  name: string;
  role: DatasetRole;
  dimensions?: string[];
}

export interface DatasetSchemaSpec {
  bindings?: RoleBinding[];
  time_col?: string;
  group_cols?: string[];
}

export interface GardenRef {
  name?: string;
  version?: number;
  class_name?: string;
  contract_version?: string;
  source_path?: string;
}

/** The agent's evolving model spec (a superset; most fields are optional). */
export interface ModelSpec {
  kpi?: string;
  media_channels?: { name: string }[];
  control_variables?: { name: string; role?: string }[];
  garden_ref?: GardenRef;
  model_params?: Record<string, unknown>;
  likelihood?: { family?: string; link?: string; params?: Record<string, unknown> };
  dataset?: DatasetSchemaSpec;
  inference?: Record<string, unknown>;
  [key: string]: unknown;
}

// ─── Plots ───────────────────────────────────────────────────────────────────

/** A dashboard plot ref (dashboard_data.plots): content-addressed ({id,title})
 *  or a legacy inline figure ({data,layout,…} via the index signature).
 *  call_id/ts/source are optional provenance stamps — newer backends emit
 *  them; refs from older backends/checkpoints lack them. */
export interface PlotRef {
  id?: string;
  title?: string;
  /** Producing tool call id (joins the plot to the question that asked for it). */
  call_id?: string;
  /** Creation time, epoch seconds. */
  ts?: number;
  /** Producing tool name. */
  source?: string;
  [key: string]: unknown;
}

// ─── Tables + EDA (additive — not wired into the UI yet) ─────────────────────

export interface TableColumn { key: string; label: string; type?: 'number' | 'string' | 'percent' | 'currency' | 'date' }
export interface TableRef {
  id: string;
  title: string;
  source: string;
  group?: string;
  /** Producing tool call id (optional provenance; older refs lack it). */
  call_id?: string;
  /** Creation time, epoch seconds (optional provenance; older refs lack it). */
  ts?: number;
}
export interface TableSpec { title: string; columns: TableColumn[]; rows: Record<string, unknown>[]; total_rows?: number; truncated?: boolean; source: string; group?: string }
export interface EdaIssue { severity: string; check: string; variable?: string; message: string }
export interface OutlierAction { action_id: string; strategy: string; variable?: string; rationale?: string; status: string }
export interface EdaFindings { issues?: EdaIssue[]; outlier_actions?: OutlierAction[]; normalization_damaged?: string[]; updated_at?: number }

// ─── Data Studio (staged upload → interactive EDA → clean → convert) ─────────

export type StudioRole = 'kpi' | 'media' | 'control' | 'date' | 'group' | 'ignore';

/** One step in the replayable cleaning pipeline. `op` discriminates the union. */
export type TransformStep =
  | { op: 'rename'; from: string; to: string }
  | { op: 'drop_columns'; columns: string[] }
  | { op: 'cast'; column: string; dtype: 'number' | 'integer' | 'string' | 'category' | 'boolean' | 'datetime' }
  | { op: 'parse_date'; column?: string; format?: string; dayfirst?: boolean }
  | { op: 'fill_missing'; columns?: string[]; strategy: 'mean' | 'median' | 'zero' | 'ffill' | 'bfill' | 'interpolate' | 'constant'; value?: number | string }
  | { op: 'drop_duplicates'; subset?: string[]; keep?: 'first' | 'last' }
  | { op: 'filter_rows'; column: string; operator: string; value?: unknown; value2?: unknown }
  | { op: 'date_range'; column?: string; start?: string; end?: string }
  | { op: 'winsorize'; column: string; periods?: string[]; cap_value: number }
  | { op: 'impute'; column: string; periods?: string[]; value: number }
  | { op: 'event_dummy'; name: string; periods: string[] };

export interface TransformDiff { rows_before: number; rows_after: number; cols_before: number; cols_after: number }

/** An inline figure blob (PlotCard renders it directly — no /plots fetch). */
export interface StudioChart { key: string; title: string; data: unknown; layout: unknown }

export interface StudioTableSpec { title: string; columns: string[]; rows: unknown[][] }

export interface StudioOutlierSuggestion {
  action_id: string;
  strategy: string;
  variable: string;
  rationale: string;
  step: TransformStep | null;
  spec_change?: Record<string, unknown> | null;
}

export interface StudioEdaResult {
  analyses: Record<string, { figures: StudioChart[]; tables: StudioTableSpec[]; stats: Record<string, unknown> }>;
  issues: EdaIssue[];
  outlier_suggestions: StudioOutlierSuggestion[];
  normalization_damaged: string[];
  warnings: string[];
  meta?: Record<string, unknown>;
}

/** The current staged dataset + pipeline (hydrated from GET /data-studio). */
export interface DataStudioState {
  staging_id: string;
  filename: string;
  columns: string[];
  all_columns: string[];
  dtypes: Record<string, string>;
  roles: Record<string, StudioRole>;
  date_col: string | null;
  is_long: boolean;
  n_rows: number;
  n_cols: number;
  preview_rows: Record<string, unknown>[];
  steps: TransformStep[];
  diff?: TransformDiff;
  warnings: string[];
  committed: boolean;
}

/** Lightweight pointer the backend rides on dashboard_data so the Data tab can
 *  badge an in-progress staging without holding the full preview. */
export interface DataStudioPointer {
  active?: boolean;
  committed?: boolean;
  staging_id?: string;
  filename?: string;
  n_rows?: number;
  n_cols?: number;
}

// ─── Dashboard data (streamed via dashboard_update; shape is server-driven) ──

export interface DashboardData {
  plots?: PlotRef[];
  tables?: TableRef[];
  eda?: EdaFindings;
  data_studio?: DataStudioPointer | null;
  // dataset / model_spec / decomposition / roi_metrics / … remain server-driven
  // and untyped via this index signature (callers cast where they need shape).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}
