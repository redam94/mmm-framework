import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import Editor from '@monaco-editor/react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ChevronDown,
  ChevronUp,
  Loader2,
  Pencil,
  Play,
  Plus,
  Sparkles,
  Trash2,
} from 'lucide-react';
import { PlotCard } from '../../pages/Agent/components/plots/PlotCard';
import { TableCard } from '../../pages/Agent/components/tables/TableCard';
import type { TableRef } from '../../pages/Agent/types';
import type { NotebookCell as Cell } from '../../api/services/atelierNotebookService';
import { registerGardenCompletions } from './gardenCompletions';

export type CellStatus = 'idle' | 'running' | 'done' | 'error';

// Compact markdown map matching the Atelier docs renderer.
const CELL_MD: Components = {
  h1: ({ children }) => <h3 className="mb-1.5 mt-2 first:mt-0 font-display text-lg font-semibold text-ink-900">{children}</h3>,
  h2: ({ children }) => <h4 className="mb-1 mt-3 font-display text-base font-semibold text-ink-900">{children}</h4>,
  h3: ({ children }) => <h5 className="mb-1 mt-2 text-sm font-semibold text-ink-800">{children}</h5>,
  p: ({ children }) => <p className="mb-2 leading-relaxed text-ink-700">{children}</p>,
  ul: ({ children }) => <ul className="mb-2 ml-4 list-disc space-y-1 text-ink-700">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 ml-4 list-decimal space-y-1 text-ink-700">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-sage-700 underline">{children}</a>
  ),
  strong: ({ children }) => <strong className="font-semibold text-ink-900">{children}</strong>,
  code: ({ children }) => (
    <code className="rounded bg-cream-100 px-1 py-0.5 font-mono text-[0.85em] text-ink-800">{children}</code>
  ),
};

// Auto-indenting code/markdown textarea: Tab inserts 2 spaces, Cmd/Ctrl+Enter runs.
function CellEditor({
  value,
  onChange,
  onRun,
  language,
}: {
  value: string;
  onChange: (v: string) => void;
  onRun?: () => void;
  language: 'python' | 'markdown';
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const rows = Math.min(24, Math.max(3, value.split('\n').length + 1));
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && onRun) {
      e.preventDefault();
      onRun();
      return;
    }
    if (e.key === 'Tab') {
      e.preventDefault();
      const el = e.currentTarget;
      const s = el.selectionStart;
      const next = value.slice(0, s) + '  ' + value.slice(el.selectionEnd);
      onChange(next);
      requestAnimationFrame(() => {
        el.selectionStart = el.selectionEnd = s + 2;
      });
    }
  };
  return (
    <textarea
      ref={ref}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKeyDown}
      rows={rows}
      spellCheck={false}
      placeholder={language === 'python' ? '# Python — Cmd/Ctrl+Enter to run' : 'Write markdown…'}
      className="w-full resize-none rounded-md border border-line-200 bg-white px-3 py-2 font-mono text-[13px] leading-relaxed text-ink-800 focus:border-sage-600 focus:outline-none"
      style={{ fontFamily: 'JetBrains Mono, ui-monospace, monospace' }}
    />
  );
}

// Code cell editor: Monaco with the Atelier's framework-aware completions
// (Tab accepts a suggestion — `tabCompletion: 'on'`), Cmd/Ctrl+Enter runs, and
// the height auto-grows to fit the content. Mirrors the main Atelier editor.
function CodeEditor({
  value,
  onChange,
  onRun,
}: {
  value: string;
  onChange: (v: string) => void;
  onRun: () => void;
}) {
  const [height, setHeight] = useState(72);
  // Keep the Cmd/Ctrl+Enter command (bound once on mount) pointed at the latest
  // onRun, so it always runs the freshest cell source.
  const onRunRef = useRef(onRun);
  useEffect(() => {
    onRunRef.current = onRun;
  }, [onRun]);
  return (
    <div className="overflow-hidden rounded-md border border-line-200 bg-white">
      <Editor
        height={height}
        language="python"
        value={value}
        onChange={(v) => onChange(v ?? '')}
        theme="vs"
        onMount={(editor, monaco) => {
          registerGardenCompletions(editor, monaco);
          editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () =>
            onRunRef.current(),
          );
          const fit = () =>
            setHeight(Math.min(560, Math.max(72, editor.getContentHeight())));
          editor.onDidContentSizeChange(fit);
          fit();
        }}
        options={{
          minimap: { enabled: false },
          fontSize: 13,
          lineNumbers: 'off',
          folding: false,
          glyphMargin: false,
          lineDecorationsWidth: 6,
          lineNumbersMinChars: 0,
          scrollBeyondLastLine: false,
          tabCompletion: 'on',
          quickSuggestions: true,
          suggestOnTriggerCharacters: true,
          automaticLayout: true,
          padding: { top: 8, bottom: 8 },
          fontFamily: 'JetBrains Mono, ui-monospace, monospace',
          renderLineHighlight: 'none',
          overviewRulerLanes: 0,
          scrollbar: { alwaysConsumeMouseWheel: false },
        }}
      />
    </div>
  );
}

const NO_OUTPUT = 'Code executed successfully with no output.';

function CellOutput({ cell, onDiagnose }: { cell: Cell; onDiagnose?: () => void }) {
  const out = cell.outputs;
  if (!out) return null;
  const stdout = (out.stdout || '').trim();
  const showStdout = stdout && stdout !== NO_OUTPUT;
  const hasError = out.is_error;
  return (
    <div className="mt-2 space-y-2">
      {showStdout && (
        <pre
          className={`overflow-x-auto rounded-md px-3 py-2 text-xs leading-relaxed ${
            hasError
              ? 'border border-rust-700/30 bg-rust-700/5 text-rust-700'
              : 'bg-ink-900/95 text-cream-100'
          }`}
        >
          {stdout}
        </pre>
      )}
      {hasError && onDiagnose && (
        <button
          onClick={onDiagnose}
          title="Ask the notebook copilot to diagnose this error and rewrite the cell"
          className="inline-flex items-center gap-1.5 rounded-md border border-sage-300 bg-sage-100/70 px-2.5 py-1 text-xs font-medium text-sage-800 transition-colors hover:bg-sage-100"
        >
          <Sparkles size={13} /> Diagnose with copilot
        </button>
      )}
      {out.tables?.map((t, i) => (
        <TableCard key={t.id} tableRef={{ ...t, source: t.source ?? '' } as TableRef} idx={i} />
      ))}
      {out.plots?.map((p, i) => (
        <PlotCard key={p.id} plot={p} idx={i} />
      ))}
      {!showStdout && !out.tables?.length && !out.plots?.length && !hasError && (
        <p className="text-xs italic text-ink-300">No output.</p>
      )}
    </div>
  );
}

export function NotebookCell({
  cell,
  index,
  status,
  isFirst,
  isLast,
  onChange,
  onRun,
  onDelete,
  onMoveUp,
  onMoveDown,
  onAddBelow,
  onDiagnose,
}: {
  cell: Cell;
  index: number;
  status: CellStatus;
  isFirst: boolean;
  isLast: boolean;
  onChange: (source: string) => void;
  onRun: () => void;
  onDelete: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onAddBelow: (type: 'code' | 'markdown') => void;
  onDiagnose?: () => void;
}) {
  const isCode = cell.type === 'code';
  const [editing, setEditing] = useState(!cell.source.trim());
  const running = status === 'running';

  return (
    <div className="group rounded-xl border border-line-200 bg-cream-50/60 p-2.5">
      {/* Cell header: index/kind + actions */}
      <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-ink-400">
        <span className="font-mono tabular-nums">
          {isCode ? `[${index + 1}]` : 'md'}
        </span>
        <span className="uppercase tracking-wide">{isCode ? 'code' : 'markdown'}</span>
        {status === 'done' && <span className="text-sage-700">✓</span>}
        {status === 'error' && <span className="text-rust-700">error</span>}
        <div className="ml-auto flex items-center gap-0.5 opacity-60 transition-opacity group-hover:opacity-100">
          {!isCode && (
            <button
              title={editing ? 'Preview' : 'Edit'}
              onClick={() => setEditing((v) => !v)}
              className="rounded p-1 text-ink-400 hover:bg-line-200/60 hover:text-ink-700"
            >
              <Pencil size={13} />
            </button>
          )}
          <button
            title="Move up"
            disabled={isFirst}
            onClick={onMoveUp}
            className="rounded p-1 text-ink-400 hover:bg-line-200/60 hover:text-ink-700 disabled:opacity-30"
          >
            <ChevronUp size={13} />
          </button>
          <button
            title="Move down"
            disabled={isLast}
            onClick={onMoveDown}
            className="rounded p-1 text-ink-400 hover:bg-line-200/60 hover:text-ink-700 disabled:opacity-30"
          >
            <ChevronDown size={13} />
          </button>
          <button
            title="Delete cell"
            onClick={onDelete}
            className="rounded p-1 text-ink-400 hover:bg-rust-700/10 hover:text-rust-700"
          >
            <Trash2 size={13} />
          </button>
        </div>
      </div>

      {/* Body: code = editor + run; markdown = editor or preview */}
      {isCode ? (
        <div className="flex items-start gap-2">
          <button
            title="Run cell (Cmd/Ctrl+Enter)"
            onClick={onRun}
            disabled={running}
            className="mt-0.5 shrink-0 rounded-md border border-line-200 bg-white p-1.5 text-sage-700 hover:bg-sage-700/10 disabled:opacity-50"
          >
            {running ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
          </button>
          <div className="min-w-0 flex-1">
            <CodeEditor value={cell.source} onChange={onChange} onRun={onRun} />
            <CellOutput cell={cell} onDiagnose={onDiagnose} />
          </div>
        </div>
      ) : editing ? (
        <CellEditor value={cell.source} onChange={onChange} language="markdown" />
      ) : (
        <div
          className="cursor-text rounded-md px-1 text-sm text-ink-700"
          onDoubleClick={() => setEditing(true)}
        >
          {cell.source.trim() ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={CELL_MD}>
              {cell.source}
            </ReactMarkdown>
          ) : (
            <p className="italic text-ink-300">Empty markdown — double-click to edit.</p>
          )}
        </div>
      )}

      {/* Add-cell affordance below */}
      <div className="mt-1.5 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
        <button
          onClick={() => onAddBelow('code')}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-ink-400 hover:bg-line-200/60 hover:text-ink-700"
        >
          <Plus size={11} /> code
        </button>
        <button
          onClick={() => onAddBelow('markdown')}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-ink-400 hover:bg-line-200/60 hover:text-ink-700"
        >
          <Plus size={11} /> markdown
        </button>
      </div>
    </div>
  );
}
