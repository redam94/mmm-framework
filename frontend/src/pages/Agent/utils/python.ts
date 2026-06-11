// ─── Python output helpers ────────────────────────────────────────────────────

export function extractPythonOutput(raw: string): string {
  const m = raw.match(/```(?:text|python)?\n([\s\S]*?)\n?```/);
  return m ? m[1] : raw.replace(/^###[^\n]*\n/, '').trim();
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
