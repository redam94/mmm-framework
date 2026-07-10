import type { ChatMessage, PlotRef, PythonOutput, TableRef } from '../types';

// ─── Grouped-by-question artifact timeline ───────────────────────────────────
//
// Pure helpers (no React) that fold the session's mixed artifacts — plots,
// tables, and executed-code segments — into groups keyed by the user question
// that produced them, newest question first. Resolution rides the tool
// call_id: each AI message's tool calls belong to the latest preceding human
// message, and a ref stamped with that call_id joins the question's group.
//
// Graceful degradation is mandatory: refs from older backends/checkpoints have
// no call_id and fall into a single trailing "Earlier work" group that
// preserves today's flat ordering. Code segments always resolve (their id IS
// the producing call_id).

export interface ArtifactItem {
  kind: 'plot' | 'table' | 'code';
  /** Stable identity within its kind (content-addressed ref id / call_id). */
  id: string;
  /** Producing-time epoch seconds (newer backends stamp it; legacy refs lack it). */
  ts?: number;
  /** Producing tool call id — the join key to a question. */
  callId?: string;
  plot?: PlotRef;
  tableRef?: TableRef;
  python?: PythonOutput;
}

export interface ArtifactGroup {
  key: string;
  /** Question text truncated for the header (or "Earlier work" for legacy). */
  title: string;
  /** Untruncated question text (tooltip); undefined for the legacy group. */
  question?: string;
  /** Index into the session's question list; -1 for the legacy bucket. */
  qIdx: number;
  /** Newest item timestamp in the group (epoch seconds), when known. */
  ts?: number;
  counts: { plots: number; tables: number; code: number };
  items: ArtifactItem[];
}

const TITLE_MAX = 80;

function truncateTitle(q: string): string {
  const one = q.replace(/\s+/g, ' ').trim();
  return one.length <= TITLE_MAX ? one : `${one.slice(0, TITLE_MAX - 1).trimEnd()}…`;
}

/** Walk the chat transcript mapping each tool call id to the index of the
 *  latest preceding human question. Tool calls arriving before any question
 *  stay unmapped (→ legacy bucket). */
export function buildCallToQuestion(messages: ChatMessage[]): {
  map: Record<string, number>;
  questions: string[];
} {
  const map: Record<string, number> = {};
  const questions: string[] = [];
  for (const m of messages) {
    if (m.type === 'human') {
      const content = m.content.trim();
      if (content) questions.push(content);
    } else if (m.type === 'ai' || m.type === 'error') {
      // 'error' included: a stream failure rewrites the live AI bubble to
      // type 'error' but keeps its toolCalls — artifacts already produced by
      // that turn must stay under their question, not fall to legacy.
      const qIdx = questions.length - 1;
      if (qIdx < 0) continue;
      for (const tc of m.toolCalls ?? []) {
        if (tc.id) map[tc.id] = qIdx;
      }
    }
  }
  return { map, questions };
}

/** Collect plots + non-EDA tables + code segments, dedupe, resolve each to a
 *  question via its call_id, and emit groups newest-question-first with an
 *  "Earlier work" bucket LAST for anything unresolvable. */
export function buildArtifactGroups(
  messages: ChatMessage[],
  plots: PlotRef[] | undefined,
  tables: TableRef[] | undefined,
  pythonOutputs: PythonOutput[] | undefined,
): ArtifactGroup[] {
  const { map, questions } = buildCallToQuestion(messages);

  // Deterministic collection order: plots, then tables, then code — each in
  // array order. Legacy (all-undefined ts) items keep this order within their
  // group because the per-group sort below is stable.
  const items: ArtifactItem[] = [];
  const seen = new Set<string>();
  const push = (item: ArtifactItem) => {
    const k = `${item.kind}:${item.id}`;
    if (seen.has(k)) return;
    seen.add(k);
    items.push(item);
  };

  (plots ?? []).forEach((p, i) => {
    push({
      kind: 'plot',
      id: p?.id ?? `inline-${i}`,
      ts: typeof p?.ts === 'number' ? p.ts : undefined,
      callId: p?.call_id,
      plot: p,
    });
  });
  (tables ?? []).forEach((t) => {
    if (t.group === 'eda') return; // the Data tab owns EDA tables
    push({
      kind: 'table',
      id: t.id,
      ts: typeof t.ts === 'number' ? t.ts : undefined,
      callId: t.call_id,
      tableRef: t,
    });
  });
  (pythonOutputs ?? []).forEach((po) => {
    // A python output's id IS the producing tool call_id.
    push({ kind: 'code', id: po.id, callId: po.id, python: po });
  });

  // Bucket by resolved question index (-1 = unresolved → legacy).
  const buckets = new Map<number, ArtifactItem[]>();
  for (const item of items) {
    const qIdx = item.callId != null && map[item.callId] != null ? map[item.callId] : -1;
    const bucket = buckets.get(qIdx);
    if (bucket) bucket.push(item);
    else buckets.set(qIdx, [item]);
  }

  const toGroup = (qIdx: number, bucket: ArtifactItem[]): ArtifactGroup => {
    // Stable sort (ES2019+) ascending on (ts ?? 0): legacy all-0 preserves
    // collection order; timestamped items read chronologically.
    const sorted = [...bucket].sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0));
    const counts = { plots: 0, tables: 0, code: 0 };
    for (const it of sorted) {
      if (it.kind === 'plot') counts.plots += 1;
      else if (it.kind === 'table') counts.tables += 1;
      else counts.code += 1;
    }
    const tsVals = sorted
      .map((i) => i.ts)
      .filter((t): t is number => typeof t === 'number');
    const ts = tsVals.length > 0 ? Math.max(...tsVals) : undefined;
    if (qIdx === -1) {
      return { key: 'legacy', title: 'Earlier work', qIdx: -1, ts, counts, items: sorted };
    }
    const question = questions[qIdx] ?? `Question ${qIdx + 1}`;
    return { key: `q-${qIdx}`, title: truncateTitle(question), question, qIdx, ts, counts, items: sorted };
  };

  const groups: ArtifactGroup[] = [...buckets.keys()]
    .filter((q) => q >= 0)
    .sort((a, b) => b - a) // newest question first
    .map((q) => toGroup(q, buckets.get(q) as ArtifactItem[]));

  const legacy = buckets.get(-1);
  if (legacy && legacy.length > 0) groups.push(toGroup(-1, legacy));

  return groups;
}
