import { useCallback, useEffect, useRef, useState } from "react";
import {
  FileSpreadsheet,
  Loader2,
  Play,
  Plus,
  Save,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { Button } from "../../components/ui";
import { NotebookCell, type CellStatus } from "./NotebookCell";
import {
  NotebookCopilotPanel,
  type DiagnoseRequest,
} from "./NotebookCopilotPanel";
import { useNotebookDoc, useSaveNotebook } from "../../api/hooks";
import {
  atelierNotebookService,
  hashSource,
  runCellToCompletion,
  type NotebookCell as Cell,
  type NotebookDataset,
} from "../../api/services/atelierNotebookService";

function uid(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return "c" + Math.random().toString(36).slice(2, 10);
  }
}

function newCell(type: "code" | "markdown"): Cell {
  return { id: uid(), type, source: "", outputs: null };
}

/**
 * A Jupyter-like demo/test space for the bespoke model in the editor. Cells run
 * against the LIVE editor buffer (the kernel re-imports only when the source
 * changes), an uploaded dataset auto-binds as `df`, and plot/table/stdout
 * outputs are tracked + persisted per model.
 */
export function AtelierNotebook({
  name,
  liveSource,
  version = null,
  onApplyToEditor,
  fill = false,
}: {
  name: string;
  liveSource: string;
  version?: number | null;
  /** Write a copilot model-source fix back to the editor buffer (the kernel
   * imports the model from there, so model-class rewrites can't live in a cell). */
  onApplyToEditor?: (code: string) => void;
  /** Fill the parent's height (instead of a fixed 58vh) so the inner cell list
   * scrolls — required inside the fixed-viewport fullscreen overlay, where
   * `xl:h-auto` would otherwise let the notebook grow past the screen unscrollably. */
  fill?: boolean;
}) {
  const nbName = name?.trim() || "untitled";
  const docQuery = useNotebookDoc(nbName, version);
  const save = useSaveNotebook();

  const [cells, setCells] = useState<Cell[]>([]);
  const [dataset, setDataset] = useState<NotebookDataset | null>(null);
  const [status, setStatus] = useState<Record<string, CellStatus>>({});
  const [runningAll, setRunningAll] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Notebook copilot (diagnose cell errors + rewrite cells). `diagnose` is a
  // nonce-keyed request raised from an errored cell's output.
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [diagnose, setDiagnose] = useState<DiagnoseRequest | null>(null);
  const diagnoseNonce = useRef(0);

  // Initialise from the loaded doc (or starter) exactly once per notebook key.
  const loadedKeyRef = useRef<string | null>(null);
  const readyRef = useRef(false);
  useEffect(() => {
    const key = `${nbName}__v${version ?? "draft"}`;
    if (loadedKeyRef.current === key) return;
    if (!docQuery.data) return;
    loadedKeyRef.current = key;
    readyRef.current = false;
    setCells(
      docQuery.data.cells?.length ? docQuery.data.cells : [newCell("code")],
    );
    setDataset(docQuery.data.dataset ?? null);
    setStatus({});
    // Allow autosave to fire only after this load settles.
    requestAnimationFrame(() => {
      readyRef.current = true;
    });
  }, [nbName, version, docQuery.data]);

  // The latest editor state + whether it has unsaved changes — kept in refs so the
  // unmount cleanup can flush WITHOUT a stale closure (fixes edits lost when the
  // user switches Atelier tabs before the 1s debounce fires).
  const latestRef = useRef({ name: nbName, version, cells, dataset });
  const dirtyRef = useRef(false);
  latestRef.current = { name: nbName, version, cells, dataset };

  const flushSave = useCallback(() => {
    if (!dirtyRef.current) return;
    dirtyRef.current = false;
    save.mutate(latestRef.current);
  }, [save]);

  // Debounced autosave whenever cells/dataset change post-load.
  useEffect(() => {
    if (!readyRef.current) return;
    dirtyRef.current = true;
    const t = setTimeout(flushSave, 1000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cells, dataset, nbName, version]);

  // Flush a pending save on UNMOUNT (e.g. switching center tabs). Empty deps so
  // the cleanup runs only when the notebook is torn down, never mid-edit.
  useEffect(() => {
    return () => {
      if (dirtyRef.current) save.mutate(latestRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const patchCell = useCallback((id: string, patch: Partial<Cell>) => {
    setCells((cs) => cs.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  }, []);

  const runCell = useCallback(
    async (cell: Cell) => {
      if (cell.type !== "code") return;
      setStatus((s) => ({ ...s, [cell.id]: "running" }));
      try {
        const out = await runCellToCompletion({
          name: nbName,
          version,
          source_code: liveSource,
          source_rev: hashSource(liveSource),
          code: cell.source,
          dataset_path: dataset?.path ?? null,
        });
        patchCell(cell.id, { outputs: out });
        setStatus((s) => ({
          ...s,
          [cell.id]: out.is_error ? "error" : "done",
        }));
        return out.is_error;
      } catch (e) {
        patchCell(cell.id, {
          outputs: {
            stdout: e instanceof Error ? e.message : String(e),
            plots: [],
            tables: [],
            is_error: true,
          },
        });
        setStatus((s) => ({ ...s, [cell.id]: "error" }));
        return true;
      }
    },
    [nbName, version, liveSource, dataset, patchCell],
  );

  const runAll = useCallback(async () => {
    setRunningAll(true);
    try {
      for (const cell of cells.filter((c) => c.type === "code")) {
        const failed = await runCell(cell);
        if (failed) break; // cells share kernel state — stop at the first error
      }
    } finally {
      setRunningAll(false);
    }
  }, [cells, runCell]);

  // Open the copilot and ask it to diagnose a failed cell (its code + traceback).
  const requestDiagnosis = useCallback((cell: Cell, index: number) => {
    setCopilotOpen(true);
    diagnoseNonce.current += 1;
    setDiagnose({
      nonce: diagnoseNonce.current,
      cellId: cell.id,
      cellIndex: index,
      code: cell.source,
      traceback: cell.outputs?.stdout ?? "",
    });
  }, []);

  // Apply a copilot code block: into the targeted cell, else as a new cell at end.
  // When patching a cell, also clear its (now-stale) error output + status so the
  // red traceform/badge/Diagnose button don't linger over freshly-fixed code.
  const applyCopilotCode = useCallback(
    (code: string, targetCellId: string | null) => {
      if (targetCellId && cells.some((c) => c.id === targetCellId)) {
        patchCell(targetCellId, { source: code, outputs: null });
        setStatus((s) => ({ ...s, [targetCellId]: "idle" }));
        return;
      }
      setCells((cs) => [...cs, { ...newCell("code"), source: code }]);
    },
    [cells, patchCell],
  );

  const addCell = (type: "code" | "markdown", afterIdx?: number) => {
    setCells((cs) => {
      const next = [...cs];
      const at = afterIdx == null ? cs.length : afterIdx + 1;
      next.splice(at, 0, newCell(type));
      return next;
    });
  };
  const deleteCell = (id: string) =>
    setCells((cs) => cs.filter((c) => c.id !== id));
  const moveCell = (id: string, dir: -1 | 1) =>
    setCells((cs) => {
      const i = cs.findIndex((c) => c.id === id);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= cs.length) return cs;
      const next = [...cs];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });

  const onUpload = async (file: File | undefined) => {
    if (!file) return;
    setUploading(true);
    try {
      const ds = await atelierNotebookService.uploadDataset(
        nbName,
        file,
        version,
      );
      setDataset({
        path: ds.path,
        filename: ds.filename,
        preview: ds.preview,
        kind: ds.kind,
      });
    } catch {
      /* surfaced by the next cell run if the path is unusable */
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div
      className={
        fill
          ? "flex h-full min-h-0 flex-col gap-2"
          : "flex h-[58vh] flex-col gap-2 xl:h-auto xl:min-h-[58vh]"
      }
    >
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-line-200 bg-white px-2.5 py-1.5">
        <Button
          size="sm"
          variant="primary"
          onClick={runAll}
          disabled={runningAll}
        >
          {runningAll ? (
            <Loader2 size={14} className="mr-1 animate-spin" />
          ) : (
            <Play size={14} className="mr-1" />
          )}
          Run all
        </Button>
        <Button size="sm" variant="secondary" onClick={() => addCell("code")}>
          <Plus size={14} className="mr-1" /> Code
        </Button>
        <Button size="sm" variant="ghost" onClick={() => addCell("markdown")}>
          <Plus size={14} className="mr-1" /> Markdown
        </Button>

        <span className="mx-1 h-4 w-px bg-line-200" />

        <input
          ref={fileRef}
          type="file"
          accept=".csv,.tsv,.txt,.parquet,.xlsx,.xls"
          className="hidden"
          onChange={(e) => onUpload(e.target.files?.[0])}
        />
        <Button
          size="sm"
          variant="secondary"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? (
            <Loader2 size={14} className="mr-1 animate-spin" />
          ) : (
            <UploadCloud size={14} className="mr-1" />
          )}
          Upload data
        </Button>
        {dataset ? (
          <span className="flex items-center gap-1 truncate text-xs text-ink-500">
            <FileSpreadsheet size={13} className="text-sage-700" />
            <span className="font-mono">{dataset.filename}</span>
            <span className="text-ink-300">→ bound as </span>
            <code className="rounded bg-cream-100 px-1 text-ink-700">df</code>
          </span>
        ) : (
          <span className="text-xs italic text-ink-300">
            No dataset — runs on a synthetic world. Upload an MFF CSV to test on
            your data.
          </span>
        )}

        <span className="ml-auto" />

        <Button
          size="sm"
          variant="secondary"
          onClick={() => {
            dirtyRef.current = false;
            save.mutate({ name: nbName, version, cells, dataset });
          }}
          disabled={save.isPending}
          title="Save the notebook now"
        >
          {save.isPending ? (
            <Loader2 size={14} className="mr-1 animate-spin" />
          ) : (
            <Save size={14} className="mr-1" />
          )}
          Save
        </Button>

        <Button
          size="sm"
          variant={copilotOpen ? "primary" : "ghost"}
          onClick={() => setCopilotOpen((v) => !v)}
          title="Notebook copilot — diagnose errors & rewrite cells"
        >
          <Sparkles size={14} className="mr-1" /> Copilot
        </Button>
        <span className="text-[11px] text-ink-300">
          source:{" "}
          <span className="text-ink-500">
            {version != null ? `v${version}` : "live editor"}
          </span>
        </span>
      </div>

      {/* Cells (+ optional copilot rail) */}
      <div className="flex min-h-0 flex-1 gap-2">
        <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
          {cells.map((cell, i) => (
            <NotebookCell
              key={cell.id}
              cell={cell}
              index={i}
              status={status[cell.id] ?? "idle"}
              isFirst={i === 0}
              isLast={i === cells.length - 1}
              onChange={(source) => patchCell(cell.id, { source })}
              onRun={() => runCell(cell)}
              onDelete={() => deleteCell(cell.id)}
              onMoveUp={() => moveCell(cell.id, -1)}
              onMoveDown={() => moveCell(cell.id, 1)}
              onAddBelow={(type) => addCell(type, i)}
              onDiagnose={() => requestDiagnosis(cell, i)}
            />
          ))}
          {cells.length === 0 && (
            <div className="rounded-xl border border-dashed border-line-300 p-6 text-center text-sm text-ink-300">
              Empty notebook — add a code or markdown cell to start.
            </div>
          )}
        </div>
        {/* Kept mounted (toggled via `hidden`) so the chat history + the
            consumed-diagnosis nonce survive closing/reopening the rail. */}
        <div
          className={
            copilotOpen
              ? "w-[21rem] shrink-0 overflow-hidden rounded-md border border-line-200 bg-white"
              : "hidden"
          }
        >
          <NotebookCopilotPanel
            sourceCode={liveSource}
            datasetPreview={dataset?.preview ?? null}
            cells={cells}
            diagnoseRequest={diagnose}
            onApplyCode={applyCopilotCode}
            onApplyToEditor={onApplyToEditor}
            onClose={() => setCopilotOpen(false)}
            className="h-full"
          />
        </div>
      </div>
    </div>
  );
}
