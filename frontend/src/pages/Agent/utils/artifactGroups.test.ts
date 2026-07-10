import { describe, it, expect } from 'vitest';
import { buildArtifactGroups, buildCallToQuestion } from './artifactGroups';
import type { ChatMessage, PlotRef, PythonOutput, TableRef } from '../types';

// ─── fixtures ────────────────────────────────────────────────────────────────

function human(id: string, content: string): ChatMessage {
  return { id, type: 'human', content };
}

function ai(id: string, callIds: string[]): ChatMessage {
  return {
    id,
    type: 'ai',
    content: '',
    toolCalls: callIds.map(cid => ({ id: cid, name: 'execute_python', status: 'done' as const })),
  };
}

function plot(id: string, extra: Partial<PlotRef> = {}): PlotRef {
  return { id, title: `Plot ${id}`, ...extra };
}

function table(id: string, extra: Partial<TableRef> = {}): TableRef {
  return { id, title: `Table ${id}`, source: 'execute_python', ...extra };
}

function code(id: string, extra: Partial<PythonOutput> = {}): PythonOutput {
  return { id, code: `print("${id}")`, output: '', hasError: false, plotCount: 0, ...extra };
}

// ─── buildCallToQuestion ─────────────────────────────────────────────────────

describe('buildCallToQuestion', () => {
  it('maps each tool call to the latest preceding question', () => {
    const { map, questions } = buildCallToQuestion([
      human('h1', 'First question?'),
      ai('a1', ['c1', 'c2']),
      human('h2', 'Second question?'),
      ai('a2', ['c3']),
    ]);
    expect(questions).toEqual(['First question?', 'Second question?']);
    expect(map).toEqual({ c1: 0, c2: 0, c3: 1 });
  });

  it('maps tool calls on error-converted AI bubbles (stream failure keeps toolCalls)', () => {
    const { map } = buildCallToQuestion([
      human('h1', 'Question'),
      {
        id: 'e1', type: 'error', content: 'Connection error',
        toolCalls: [{ id: 'c1', name: 'execute_python', status: 'error' as const }],
      },
    ]);
    expect(map).toEqual({ c1: 0 });
  });

  it('skips empty human messages and pre-question tool calls', () => {
    const { map, questions } = buildCallToQuestion([
      ai('a0', ['orphan']),
      human('h0', '   '),
      human('h1', 'Real question'),
      ai('a1', ['c1']),
    ]);
    expect(questions).toEqual(['Real question']);
    expect(map).toEqual({ c1: 0 });
  });
});

// ─── buildArtifactGroups ─────────────────────────────────────────────────────

describe('buildArtifactGroups', () => {
  const messages: ChatMessage[] = [
    human('h1', 'How does spend correlate with sales?'),
    ai('a1', ['call-1']),
    human('h2', 'Fit the model and show ROI.'),
    ai('a2', ['call-2', 'call-3']),
  ];

  it('orders groups newest question first', () => {
    const groups = buildArtifactGroups(
      messages,
      [plot('p1', { call_id: 'call-1' }), plot('p2', { call_id: 'call-2' })],
      [table('t1', { call_id: 'call-3' })],
      [],
    );
    expect(groups.map(g => g.qIdx)).toEqual([1, 0]);
    expect(groups[0].title).toBe('Fit the model and show ROI.');
    expect(groups[0].counts).toEqual({ plots: 1, tables: 1, code: 0 });
    expect(groups[1].title).toBe('How does spend correlate with sales?');
  });

  it('falls back to a single trailing "Earlier work" group when refs lack call_ids', () => {
    const groups = buildArtifactGroups(
      messages,
      [plot('p1'), plot('p2')],
      [table('t1')],
      [],
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].key).toBe('legacy');
    expect(groups[0].title).toBe('Earlier work');
    expect(groups[0].qIdx).toBe(-1);
    // Preserves today's flat ordering: plots in array order, then tables.
    expect(groups[0].items.map(i => i.id)).toEqual(['p1', 'p2', 't1']);
  });

  it('routes unresolvable call_ids to the legacy group, appended LAST', () => {
    const groups = buildArtifactGroups(
      messages,
      [plot('p1', { call_id: 'call-1' }), plot('p2', { call_id: 'unknown-call' })],
      [],
      [],
    );
    expect(groups.map(g => g.key)).toEqual(['q-0', 'legacy']);
    expect(groups[1].items.map(i => i.id)).toEqual(['p2']);
  });

  it('dedupes items by kind:id keeping the first occurrence', () => {
    const groups = buildArtifactGroups(
      messages,
      [plot('p1', { call_id: 'call-1' }), plot('p1', { call_id: 'call-2' })],
      [],
      [],
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].qIdx).toBe(0); // first occurrence won
    expect(groups[0].items).toHaveLength(1);
  });

  it('allows a plot and a table to share an id (dedupe is per kind)', () => {
    const groups = buildArtifactGroups(
      [],
      [plot('x1')],
      [table('x1')],
      [],
    );
    expect(groups[0].items.map(i => `${i.kind}:${i.id}`)).toEqual(['plot:x1', 'table:x1']);
  });

  it('excludes EDA tables (the Data tab owns them)', () => {
    const groups = buildArtifactGroups(
      messages,
      [],
      [table('t-eda', { group: 'eda', call_id: 'call-1' }), table('t-res', { group: 'results', call_id: 'call-1' })],
      [],
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].items.map(i => i.id)).toEqual(['t-res']);
  });

  it('keys code segments by po.id (the call_id), so they group without backend ref changes', () => {
    const groups = buildArtifactGroups(
      messages,
      [],
      [],
      [code('call-2'), code('call-1')],
    );
    expect(groups.map(g => g.qIdx)).toEqual([1, 0]);
    expect(groups[0].items[0].kind).toBe('code');
    expect(groups[0].items[0].id).toBe('call-2');
    expect(groups[0].counts.code).toBe(1);
  });

  it('assigns legacy refs synthetic inline ids when the plot ref has no id', () => {
    const groups = buildArtifactGroups([], [{ title: 'inline fig' }, { title: 'another' }], [], []);
    expect(groups[0].items.map(i => i.id)).toEqual(['inline-0', 'inline-1']);
  });

  it('sorts within a group by ts ascending, stable for missing ts', () => {
    const groups = buildArtifactGroups(
      messages,
      [
        plot('p-late', { call_id: 'call-2', ts: 200 }),
        plot('p-early', { call_id: 'call-2', ts: 100 }),
        plot('p-a', { call_id: 'call-2' }), // no ts → 0, insertion order kept
        plot('p-b', { call_id: 'call-2' }),
      ],
      [],
      [],
    );
    expect(groups[0].items.map(i => i.id)).toEqual(['p-a', 'p-b', 'p-early', 'p-late']);
    // Group ts = max item ts.
    expect(groups[0].ts).toBe(200);
  });

  it('mixes types within one question group in chronological order', () => {
    const groups = buildArtifactGroups(
      messages,
      [plot('p1', { call_id: 'call-2', ts: 30 })],
      [table('t1', { call_id: 'call-3', ts: 10 })],
      [{ ...code('call-2'), }],
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].qIdx).toBe(1);
    // code has no ts (0) → first; then table (10); then plot (30).
    expect(groups[0].items.map(i => i.kind)).toEqual(['code', 'table', 'plot']);
    expect(groups[0].counts).toEqual({ plots: 1, tables: 1, code: 1 });
  });

  it('truncates long question titles to ~80 chars and keeps the full question', () => {
    const long = 'Q '.repeat(100).trim();
    const groups = buildArtifactGroups(
      [human('h1', long), ai('a1', ['c1'])],
      [plot('p1', { call_id: 'c1' })],
      [],
      [],
    );
    expect(groups[0].title.length).toBeLessThanOrEqual(81);
    expect(groups[0].title.endsWith('…')).toBe(true);
    expect(groups[0].question).toBe(long);
  });

  it('returns no groups when there are no artifacts', () => {
    expect(buildArtifactGroups(messages, [], [], [])).toEqual([]);
    expect(buildArtifactGroups(messages, undefined, undefined, undefined)).toEqual([]);
  });
});
