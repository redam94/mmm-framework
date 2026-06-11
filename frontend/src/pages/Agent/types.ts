// ─── Types ───────────────────────────────────────────────────────────────────

export interface ToolCall {
  id: string;
  name: string;
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

// ─── Tables + EDA (additive — not wired into the UI yet) ─────────────────────

export interface TableColumn { key: string; label: string; type?: 'number' | 'string' | 'percent' | 'currency' | 'date' }
export interface TableRef { id: string; title: string; source: string; group?: string }
export interface TableSpec { title: string; columns: TableColumn[]; rows: Record<string, unknown>[]; total_rows?: number; truncated?: boolean; source: string; group?: string }
export interface EdaIssue { severity: string; check: string; variable?: string; message: string }
export interface OutlierAction { action_id: string; strategy: string; variable?: string; rationale?: string; status: string }
export interface EdaFindings { issues?: EdaIssue[]; outlier_actions?: OutlierAction[]; normalization_damaged?: string[]; updated_at?: number }

// ─── Dashboard data (streamed via dashboard_update; shape is server-driven) ──

export interface DashboardData {
  tables?: TableRef[];
  eda?: EdaFindings;
  // The rest of the payload is server-driven and untyped (was `any` in AgentPage.tsx).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}
