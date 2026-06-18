// ─── Helpers ─────────────────────────────────────────────────────────────────

export function normalizeContent(content: any): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map(b => (typeof b === 'string' ? b : b?.text ?? ''))
      .filter(Boolean)
      .join('\n');
  }
  if (content && typeof content === 'object') return JSON.stringify(content);
  return String(content ?? '');
}

export function formatToolName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export function truncate(str: string, n = 300): string {
  return str.length > n ? str.slice(0, n) + '…' : str;
}

export function stripHtml(s: string): string {
  return s.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
}
