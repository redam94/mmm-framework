// ─── Python output helpers ────────────────────────────────────────────────────

import type { Artifact, PythonOutput } from '../types';

export function extractPythonOutput(raw: string): string {
  const m = raw.match(/```(?:text|python)?\n([\s\S]*?)\n?```/);
  return m ? m[1] : raw.replace(/^###[^\n]*\n/, '').trim();
}

/** Rebuild the REPL/code-segment list from persisted session artifacts,
 * pairing each `text_output` with its `code_snippet` by call_id (code may be
 * unavailable for some). This is the source of truth after a reload AND after
 * each turn settles — code the EXPERT sub-agent ran never streams an
 * `execute_python` tool event to this client, so it only exists as artifacts. */
export function pythonOutputsFromArtifacts(arts: Artifact[]): PythonOutput[] {
  const codeByCall: Record<string, string> = {};
  for (const a of arts) {
    if (a.kind === 'code_snippet' && a.payload?.call_id) {
      codeByCall[a.payload.call_id] = String(a.payload.code ?? '');
    }
  }
  return arts
    .filter(a => a.kind === 'text_output')
    .sort((x, y) => x.created_at - y.created_at)
    .map(a => {
      const callId = String(a.payload?.call_id ?? a.id);
      return {
        id: callId,
        code: codeByCall[callId] ?? '',
        output: String(a.payload?.stdout ?? ''),
        hasError: !!a.payload?.is_error,
        plotCount: Number(a.payload?.plot_count ?? 0),
        // Expert-run cells carry the delegation's call_id for question grouping.
        callId: a.payload?.parent_call_id ? String(a.payload.parent_call_id) : undefined,
      };
    });
}

export function pyOutputKind(text: string): 'error' | 'table' | 'text' {
  if (/Traceback \(most recent call last\)|^\w+Error:|^\w+Exception:/m.test(text)) return 'error';
  const lines = text.split('\n').filter(l => l.trim());
  if (lines.length >= 3) {
    const dataLines = lines.slice(1);
    const indexed = dataLines.filter(l => /^\s*\d+\s/.test(l)).length;
    if (indexed > dataLines.length * 0.5) return 'table';
  }
  return 'text';
}

export function parseTextTable(text: string): { headers: string[]; rows: string[][] } | null {
  const lines = text.split('\n').filter(l => l.trim());
  if (lines.length < 2) return null;
  // Try to split by 2+ spaces (common pandas formatting)
  const splitLine = (l: string) => l.trim().split(/\s{2,}/).map(s => s.trim()).filter(Boolean);
  const headers = splitLine(lines[0]);
  if (headers.length < 2) return null;
  const rows = lines.slice(1).map(l => splitLine(l));
  if (rows.some(r => r.length === 0)) return null;
  return { headers, rows };
}
