import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import Editor, { type OnMount } from '@monaco-editor/react';
import ReactMarkdown, { type Components } from 'react-markdown';
import { remarkPlugins, rehypePlugins, normalizeMath } from '../../lib/markdownMath';
import {
  AlignLeft,
  CircleCheck,
  FlaskConical,
  Loader2,
  Map as MapIcon,
  Maximize2,
  Minimize2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Sparkles,
  Sprout,
  Trash2,
  UploadCloud,
  WrapText,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import {
  Button,
  Card,
  DataTable,
  Drawer,
  EmptyState,
  SectionHeader,
  Tabs,
  type Column,
} from '../../components/ui';
import { GARDEN_STATUS, type GardenStatus } from '../../theme/colors';
import { useQueryClient } from '@tanstack/react-query';
import {
  copilotChatKeys,
  useDeleteGardenModel,
  useGardenModel,
  useGardenModels,
  useGardenSource,
  useGardenTest,
  useGardenVersions,
  usePromoteGardenModel,
  useRegisterGardenModel,
  useUpdateGardenDocs,
} from '../../api/hooks';
import type { CompatTier, GardenModel } from '../../api/services/modelGardenService';
import type { CopilotSurface } from '../../api/services/copilotService';
import { CopilotPanel } from '../../components/modelGarden/CopilotPanel';
import { AtelierNotebook } from '../../components/modelGarden/AtelierNotebook';
import {
  registerGardenCompletions,
  defineAtelierTheme,
  applyLintMarkers,
} from '../../components/modelGarden/gardenCompletions';
import { copilotService, type LintProblem } from '../../api/services/copilotService';

// Production-grade Monaco options shared by the Atelier code editor: framework
// IntelliSense triggers, bracket-pair colorization, hints, and live linting.
const CODE_EDITOR_OPTIONS = {
  scrollBeyondLastLine: false,
  fontFamily: 'JetBrains Mono, ui-monospace, monospace',
  fontLigatures: true,
  renderLineHighlight: 'line' as const,
  automaticLayout: true,
  tabCompletion: 'on' as const,
  quickSuggestions: { other: true, comments: false, strings: false },
  suggestOnTriggerCharacters: true,
  parameterHints: { enabled: true, cycle: true },
  hover: { enabled: true, above: false },
  bracketPairColorization: { enabled: true },
  guides: { bracketPairs: 'active' as const, indentation: true },
  stickyScroll: { enabled: true },
  smoothScrolling: true,
  cursorBlinking: 'smooth' as const,
  padding: { top: 10, bottom: 10 },
  suggest: { showWords: false },
};

const STARTER_TEMPLATE = `from mmm_framework.garden import CustomMMM


class MyCustomMMM(CustomMMM):
    """One line on what makes this model bespoke and when to use it.

    Subclass CustomMMM (a BayesianMMM subclass) and override the build / prior
    hooks to customize. Keep the (panel, model_config, trend_config) constructor
    so the agent can re-fit it on any project's data.
    """


GARDEN_MODEL = MyCustomMMM
`;

function GardenChip({ status }: { status: string }) {
  const s = GARDEN_STATUS[status as GardenStatus] ?? {
    fg: '#4a5a48',
    bg: '#f0ede0',
    label: status,
  };
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ color: s.fg, backgroundColor: s.bg }}
    >
      {s.label}
    </span>
  );
}

function errMessage(e: unknown): string {
  const any = e as { response?: { data?: { detail?: string } }; message?: string };
  return any?.response?.data?.detail ?? any?.message ?? 'Something went wrong.';
}

// Compact markdown renderer for the model's docs in the About panel — model docs
// are authored as markdown (headings, lists, code), so render them, not the raw text.
const DOCS_MD: Components = {
  h1: ({ children }) => (
    <h3 className="mb-1.5 mt-3 first:mt-0 font-display text-base font-semibold text-ink-900">{children}</h3>
  ),
  h2: ({ children }) => (
    <h4 className="mb-1 mt-3 font-display text-sm font-semibold text-ink-900">{children}</h4>
  ),
  h3: ({ children }) => <h5 className="mb-1 mt-2 text-sm font-semibold text-ink-800">{children}</h5>,
  p: ({ children }) => <p className="mb-2 leading-relaxed text-ink-700">{children}</p>,
  ul: ({ children }) => <ul className="mb-2 ml-4 list-disc space-y-1 text-ink-700">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 ml-4 list-decimal space-y-1 text-ink-700">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-sage-700 underline">
      {children}
    </a>
  ),
  strong: ({ children }) => <strong className="font-semibold text-ink-900">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-line-300 pl-3 italic text-ink-500">{children}</blockquote>
  ),
  pre: ({ children }) => <pre className="mb-2">{children}</pre>,
  code: ({ className, children }: { className?: string; children?: ReactNode }) => {
    const raw = String(children ?? '').replace(/\n$/, '');
    const isBlock = !!/language-(\w+)/.exec(className || '') || raw.includes('\n');
    return isBlock ? (
      <code className="my-2 block overflow-x-auto rounded-md bg-ink-900/95 p-3 font-mono text-xs leading-relaxed text-cream-100">
        {children}
      </code>
    ) : (
      <code className="rounded bg-cream-200 px-1 py-0.5 font-mono text-[0.85em] text-ink-800">{children}</code>
    );
  },
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="min-w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-line-200 px-2 py-1 text-left font-semibold text-ink-800">{children}</th>
  ),
  td: ({ children }) => <td className="border border-line-200 px-2 py-1 text-ink-700">{children}</td>,
};

function ToolButton({
  active,
  title,
  disabled,
  onClick,
  children,
}: {
  active?: boolean;
  title: string;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center rounded px-1.5 py-1 transition-colors disabled:opacity-40 ${
        active ? 'bg-sage-100 text-sage-800' : 'text-ink-500 hover:bg-cream-100 hover:text-ink-800'
      }`}
    >
      {children}
    </button>
  );
}

export function ModelGardenPage() {
  // Selection + editor state
  const [selName, setSelName] = useState<string | null>(null);
  const [selVersion, setSelVersion] = useState<number | null>(null);
  const [authoring, setAuthoring] = useState(false);
  const [code, setCode] = useState(STARTER_TEMPLATE);
  const [draftName, setDraftName] = useState('');
  const [draftDocs, setDraftDocs] = useState('');
  const [rightTab, setRightTab] = useState('compat');
  // Center editor tab: edit the model source ('code') or its docs ('docs').
  const [editorTab, setEditorTab] = useState<'code' | 'docs' | 'notebook'>('code');
  // In-place docs edit buffer for a registered version (null = mirror the
  // fetched docs); while authoring, docs live in `draftDocs` instead.
  const [docsDraft, setDocsDraft] = useState<string | null>(null);

  // Collapsible side panels (desktop) — give the editor/copilot room when needed.
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);

  // Expanded docs reading view (slide-over drawer).
  const [docsExpanded, setDocsExpanded] = useState(false);

  // IDE + copilot state
  const [fullscreen, setFullscreen] = useState(false);
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [wrap, setWrap] = useState(false);
  const [minimap, setMinimap] = useState(false);
  const [fontSize, setFontSize] = useState(13);
  const [problems, setProblems] = useState<LintProblem[]>([]);
  const [showProblems, setShowProblems] = useState(false);
  const [lintedOnce, setLintedOnce] = useState(false);
  const [ideBusy, setIdeBusy] = useState<null | 'lint' | 'format'>(null);

  // Monaco refs so live linting can paint inline markers + jump to a problem.
  const codeEditorRef = useRef<import('monaco-editor').editor.IStandaloneCodeEditor | null>(
    null,
  );
  const monacoRef = useRef<typeof import('monaco-editor') | null>(null);
  const lastLintSrcRef = useRef<string | null>(null);

  const qc = useQueryClient();

  // Data
  const { data: listData, isLoading: listLoading } = useGardenModels();
  const { data: versionsData } = useGardenVersions(selName);
  const { data: detail } = useGardenModel(selName, selVersion);
  const { data: sourceData } = useGardenSource(authoring ? null : selName, selVersion);

  // Mutations / jobs
  const register = useRegisterGardenModel();
  const promote = usePromoteGardenModel();
  const remove = useDeleteGardenModel();
  const updateDocs = useUpdateGardenDocs();
  const test = useGardenTest(selName, selVersion);

  const models = listData?.models ?? [];
  const versions = versionsData?.versions ?? [];

  // Editor shows the editable draft while authoring, else the fetched (read-only)
  // source — derived (no effect) to avoid a setState-in-effect cascade.
  const editorValue = authoring ? code : sourceData?.source_code ?? '';

  // The copilot chat is scoped to the model being worked on (per model/version
  // memory); a brand-new unsaved draft uses a stable '__draft__' key so the
  // conversation survives while authoring (the editor copilot uses this too).
  const DRAFT_KEY = '__draft__';
  const copilotName = selName ?? DRAFT_KEY;
  const copilotVersion = selVersion;

  // Reset the shared draft-chat bucket (both surfaces) so a fresh authoring
  // session doesn't inherit a previous abandoned draft's conversation.
  const clearDraftChat = () => {
    (['editor', 'notebook'] as CopilotSurface[]).forEach((surface) => {
      copilotService
        .saveChat({ name: DRAFT_KEY, version: null, surface, messages: [] })
        .catch(() => {});
      qc.setQueryData(copilotChatKeys.doc(DRAFT_KEY, null, surface), {
        messages: [],
        name: DRAFT_KEY,
        version: null,
        surface,
      });
    });
  };

  // On register, carry the authoring conversation (the '__draft__' chat) over to
  // the new (name, version) key so it continues seamlessly into v1, then clear
  // the draft bucket. Primes the query cache so the panel (which remounts on the
  // key change) reads the migrated chat immediately. Best-effort.
  const migrateDraftChat = async (newName: string, newVersion: number) => {
    for (const surface of ['editor', 'notebook'] as CopilotSurface[]) {
      try {
        const draft = await copilotService.getChat(DRAFT_KEY, null, surface);
        const messages = draft.messages ?? [];
        if (messages.length) {
          await copilotService.saveChat({ name: newName, version: newVersion, surface, messages });
          qc.setQueryData(copilotChatKeys.doc(newName, newVersion, surface), {
            messages,
            name: newName,
            version: newVersion,
            surface,
          });
        }
        await copilotService.saveChat({ name: DRAFT_KEY, version: null, surface, messages: [] });
        qc.setQueryData(copilotChatKeys.doc(DRAFT_KEY, null, surface), {
          messages: [],
          name: DRAFT_KEY,
          version: null,
          surface,
        });
      } catch {
        /* best-effort migration — never block registration */
      }
    }
  };

  // Docs editing mirrors the code editor: the editable draft while authoring,
  // else an in-place buffer over the fetched docs (null = mirror unchanged).
  const isPublished = detail?.status === 'published';
  const docsValue = authoring ? draftDocs : docsDraft ?? detail?.docs ?? '';
  const docsDirty =
    !authoring && docsDraft != null && docsDraft !== (detail?.docs ?? '');

  const saveDocs = () => {
    if (authoring || !selName || selVersion == null || docsDraft == null) return;
    updateDocs.mutate(
      { name: selName, version: selVersion, docs: docsDraft },
      { onSuccess: () => setDocsDraft(null) },
    );
  };

  const startAuthoring = () => {
    setAuthoring(true);
    setSelName(null);
    setSelVersion(null);
    setCode(STARTER_TEMPLATE);
    setDraftName('');
    setDraftDocs('');
    setDocsDraft(null);
    clearDraftChat();
    resetDiagnostics();
    test.reset();
  };

  const selectVersion = (m: GardenModel) => {
    setAuthoring(false);
    setSelName(m.name);
    setSelVersion(m.version);
    setDocsDraft(null);
    resetDiagnostics();
    updateDocs.reset();
    test.reset();
  };

  const onRegister = () => {
    register.mutate(
      { source_code: code, name: draftName.trim(), docs: draftDocs.trim() },
      {
        onSuccess: async (row) => {
          // Migrate the authoring chat to the new version BEFORE flipping the
          // selection, so the remounted panel reads the carried-over chat.
          await migrateDraftChat(row.name, row.version);
          setAuthoring(false);
          setSelName(row.name);
          setSelVersion(row.version);
          resetDiagnostics();
        },
      },
    );
  };

  const editAsNewVersion = () => {
    setCode(sourceData?.source_code ?? code);
    setAuthoring(true);
    setDraftName(selName ?? '');
    setDraftDocs(detail?.docs ?? '');
    setDocsDraft(null);
    resetDiagnostics();
    test.reset();
  };

  // Esc exits fullscreen.
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setFullscreen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [fullscreen]);

  // Live linting: debounced re-lint as the source changes (code tab only) →
  // inline squiggles + an up-to-date Problems count, without forcing the panel
  // open. Manual "Validate" still opens it. Identical source isn't re-fetched.
  useEffect(() => {
    if (editorTab !== 'code') return;
    const src = editorValue;
    if (!src.trim()) {
      lastLintSrcRef.current = src;
      setProblems([]);
      return;
    }
    if (src === lastLintSrcRef.current) return;
    let cancelled = false;
    const t = setTimeout(async () => {
      try {
        const r = await copilotService.lint(src);
        if (cancelled) return;
        lastLintSrcRef.current = src;
        setLintedOnce(true);
        setProblems(r.problems ?? []);
      } catch {
        /* auto-lint failures are non-fatal — manual Validate surfaces them */
      }
    }, 800);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [editorValue, editorTab]);

  // Paint inline markers from the latest problems (code tab + editor mounted).
  useEffect(() => {
    const ed = codeEditorRef.current;
    const monaco = monacoRef.current;
    if (ed && monaco && editorTab === 'code') applyLintMarkers(monaco, ed, problems);
  }, [problems, editorTab]);


  // The copilot is an inline panel that shares the right side with the
  // compatibility report; opening it collapses that panel so the editor keeps
  // usable width (and re-expands it on close).
  const toggleCopilot = (open: boolean) => {
    setCopilotOpen(open);
    setRightCollapsed(open);
  };

  // Apply a copilot-proposed code block: drop into authoring (a new draft seeded
  // from the current selection) if we were viewing a read-only version.
  const applyCode = (newCode: string) => {
    if (!authoring) {
      setAuthoring(true);
      setDraftName((p) => p || (selName ?? ''));
      setDraftDocs((p) => p || (detail?.docs ?? ''));
      test.reset();
    }
    setCode(newCode);
  };

  const runFormat = async () => {
    if (!authoring || ideBusy) return;
    setIdeBusy('format');
    try {
      const r = await copilotService.format(code);
      if (r.formatted != null) {
        setCode(r.formatted);
        setProblems([]);
        setShowProblems(false);
      } else {
        setProblems([{ severity: 'error', message: `Format failed: ${r.error ?? 'unknown error'}`, line: null }]);
        setShowProblems(true);
      }
    } catch (e) {
      setProblems([{ severity: 'error', message: errMessage(e), line: null }]);
      setShowProblems(true);
    } finally {
      setIdeBusy(null);
    }
  };

  const runLint = async () => {
    if (ideBusy) return;
    setIdeBusy('lint');
    try {
      const r = await copilotService.lint(editorValue);
      lastLintSrcRef.current = editorValue;
      setLintedOnce(true);
      setProblems(r.problems ?? []);
      setShowProblems(true);
    } catch (e) {
      setProblems([{ severity: 'error', message: errMessage(e), line: null }]);
      setShowProblems(true);
    } finally {
      setIdeBusy(null);
    }
  };

  // Store the Monaco editor + namespace on mount so live lint can paint inline
  // markers and the Problems panel can jump to a line.
  const handleCodeEditorMount: OnMount = (editor, monaco) => {
    registerGardenCompletions(editor, monaco);
    codeEditorRef.current = editor;
    monacoRef.current = monaco;
    applyLintMarkers(monaco, editor, problems);
    // Null the ref when the editor is disposed (switching to docs/notebook tab
    // unmounts it) so marker/jump code never touches a dead editor.
    editor.onDidDispose(() => {
      if (codeEditorRef.current === editor) codeEditorRef.current = null;
    });
  };

  const jumpToProblem = (p: LintProblem) => {
    const ed = codeEditorRef.current;
    // Bail if the code editor is gone (e.g. opened from another tab) or disposed.
    if (editorTab !== 'code' || !ed || ed.getModel() == null || p.line == null) return;
    ed.revealLineInCenter(p.line);
    ed.setPosition({ lineNumber: p.line, column: p.column ?? 1 });
    ed.focus();
  };

  // Clear diagnostics (squiggles + Problems list) — called at every model/
  // version/authoring switch, since the code editor is NOT remounted on a switch
  // and would otherwise keep the previous source's markers until the next lint.
  const resetDiagnostics = () => {
    setProblems([]);
    setLintedOnce(false);
    setShowProblems(false);
    lastLintSrcRef.current = null;
    const ed = codeEditorRef.current;
    const monaco = monacoRef.current;
    if (ed && monaco && ed.getModel() != null) applyLintMarkers(monaco, ed, []);
  };

  const errCount = problems.filter((p) => p.severity === 'error').length;
  const warnCount = problems.filter((p) => p.severity === 'warning').length;

  // The compat report shown on the right: a live test job (if running) else the
  // stored report on the selected version.
  const liveReport = test.job.data?.result;
  const storedReport = detail?.compat_report ?? null;
  const tiers: CompatTier[] =
    (liveReport?.tiers as CompatTier[] | undefined) ?? storedReport?.tiers ?? [];
  const score = liveReport?.score ?? storedReport?.score ?? null;
  const blockingPassed = liveReport?.blocking_passed ?? storedReport?.blocking_passed;

  const tierColumns: Column<CompatTier>[] = useMemo(
    () => [
      { key: 'name', header: 'Tier', render: (t) => <span className="font-medium">{t.name}</span> },
      {
        key: 'result',
        header: 'Result',
        render: (t) =>
          t.skipped ? (
            <span className="text-ink-300">skip</span>
          ) : t.passed ? (
            <span className="text-sage-700">pass</span>
          ) : (
            <span className="text-rust-700 font-semibold">FAIL</span>
          ),
      },
      {
        key: 'blocking',
        header: 'Gate',
        render: (t) => (t.blocking ? <span className="text-ink-500">blocking</span> : '—'),
      },
      { key: 'detail', header: 'Detail', render: (t) => <span className="text-ink-500">{t.detail}</span> },
    ],
    [],
  );

  const testing = test.start.isPending || ['pending', 'running'].includes(test.job.data?.status ?? '');

  return (
    <div className="flex flex-col h-full space-y-4">
      <SectionHeader
        level={1}
        title="Atelier"
        subtitle="Author, test, version & share bespoke models the agent can run."
        actions={
          <Button onClick={startAuthoring}>
            <Plus size={15} className="mr-1.5" /> New model
          </Button>
        }
      />

      <div
        className="grid grid-cols-1 gap-4 flex-1 min-h-0 xl:grid-cols-[var(--mg-cols)] xl:transition-[grid-template-columns] xl:duration-200 xl:ease-out"
        style={
          {
            '--mg-cols': `${leftCollapsed ? '3rem' : '19rem'} minmax(0, 1fr) ${
              rightCollapsed ? '3rem' : '23rem'
            }`,
          } as CSSProperties
        }
      >
        {/* ── Left: registry browser (collapsible) ── */}
        <div className="min-w-0">
          {leftCollapsed && (
            <button
              type="button"
              title="Expand garden"
              onClick={() => setLeftCollapsed(false)}
              className="hidden h-full max-h-[72vh] w-full flex-col items-center gap-3 rounded-lg border border-line-200 bg-cream-100 py-3 text-ink-400 transition hover:bg-cream-200 hover:text-ink-700 xl:flex"
            >
              <PanelLeftOpen size={16} />
              <Sprout size={15} />
              <span className="rotate-180 text-xs font-semibold uppercase tracking-wide [writing-mode:vertical-rl]">
                Garden
              </span>
            </button>
          )}
          <Card
            tone="cream"
            padding="sm"
            className={`h-full overflow-y-auto max-h-[72vh] ${leftCollapsed ? 'xl:hidden' : ''}`}
          >
            <div className="flex items-center justify-between px-1 pb-2">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-400">Garden</h2>
              <button
                type="button"
                title="Collapse garden"
                onClick={() => setLeftCollapsed(true)}
                className="hidden text-ink-300 transition hover:text-ink-700 xl:inline-flex"
              >
                <PanelLeftClose size={14} />
              </button>
            </div>
          {listLoading ? (
            <div className="flex justify-center py-8 text-ink-300">
              <Loader2 size={18} className="animate-spin" />
            </div>
          ) : models.length === 0 ? (
            <p className="px-1 py-4 text-sm text-ink-300 italic">
              No models yet. Author one with “New model”, or ask the agent to build one.
            </p>
          ) : (
            <ul className="space-y-1">
              {models.map((m) => {
                const active = !authoring && m.name === selName;
                return (
                  <li key={m.id}>
                    <button
                      onClick={() => selectVersion(m)}
                      className={`w-full rounded-md px-2.5 py-2 text-left transition ${
                        active ? 'bg-white shadow-sm' : 'hover:bg-white/60'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-sm font-medium text-ink-900">{m.name}</span>
                        <GardenChip status={m.status} />
                      </div>
                      <div className="mt-0.5 text-xs text-ink-400">
                        v{m.version}
                        {m.manifest?.class_name ? ` · ${m.manifest.class_name}` : ''}
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          {selName && versions.length > 1 && (
            <div className="mt-4 border-t border-line-200 pt-3">
              <h3 className="px-1 pb-1.5 text-xs font-semibold uppercase tracking-wide text-ink-400">
                Versions · {selName}
              </h3>
              <ul className="space-y-0.5">
                {versions.map((v) => (
                  <li key={v.id}>
                    <button
                      onClick={() => selectVersion(v)}
                      className={`flex w-full items-center justify-between rounded px-2 py-1 text-left text-sm transition ${
                        v.version === selVersion ? 'bg-white shadow-sm' : 'hover:bg-white/60'
                      }`}
                    >
                      <span className="text-ink-700">v{v.version}</span>
                      <GardenChip status={v.status} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          </Card>
        </div>

        {/* ── Center: editor + IDE tools ── */}
        <div
          className={
            fullscreen
              ? // z-[60] keeps the overlay above the app sidebar (z-50) so the
                // editor's left edge isn't covered by the navbar.
                'fixed inset-0 z-[60] flex flex-col gap-2 bg-cream-50 p-4'
              : 'flex flex-col min-h-0 min-w-0'
          }
        >
          {authoring && (
            <input
              value={draftName}
              onChange={(e) => setDraftName(e.target.value)}
              placeholder="model name (e.g. tight-adstock-prior)"
              className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:border-sage-600 focus:outline-none"
            />
          )}

          {/* Center editor: Code (source) | Docs (markdown). */}
          <Tabs
            tabs={[
              { id: 'code', label: 'Code' },
              { id: 'docs', label: 'Docs' },
              { id: 'notebook', label: 'Notebook' },
            ]}
            active={editorTab}
            onChange={(id) => setEditorTab(id as 'code' | 'docs' | 'notebook')}
          />

          {/* IDE toolbar */}
          <div className="flex items-center gap-0.5 rounded-md border border-line-200 bg-white px-1.5 py-1">
            <ToolButton
              title="Format (ruff)"
              disabled={!authoring || ideBusy !== null || editorTab !== 'code'}
              onClick={runFormat}
            >
              {ideBusy === 'format' ? <Loader2 size={15} className="animate-spin" /> : <AlignLeft size={15} />}
            </ToolButton>
            <ToolButton
              title="Validate (Problems)"
              disabled={ideBusy !== null || editorTab !== 'code'}
              onClick={runLint}
            >
              {ideBusy === 'lint' ? <Loader2 size={15} className="animate-spin" /> : <CircleCheck size={15} />}
            </ToolButton>
            <span className="mx-1 h-4 w-px bg-line-200" />
            <ToolButton title="Word wrap" active={wrap} onClick={() => setWrap((v) => !v)}>
              <WrapText size={15} />
            </ToolButton>
            <ToolButton title="Minimap" active={minimap} onClick={() => setMinimap((v) => !v)}>
              <MapIcon size={15} />
            </ToolButton>
            <ToolButton title="Smaller font" onClick={() => setFontSize((s) => Math.max(10, s - 1))}>
              <ZoomOut size={15} />
            </ToolButton>
            <ToolButton title="Larger font" onClick={() => setFontSize((s) => Math.min(22, s + 1))}>
              <ZoomIn size={15} />
            </ToolButton>
            <div className="ml-auto flex items-center gap-0.5">
              {editorTab === 'code' && (lintedOnce || problems.length > 0) && (
                <button
                  type="button"
                  onClick={() => setShowProblems((v) => !v)}
                  title="Toggle Problems"
                  className={`inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors hover:bg-cream-100 ${
                    errCount ? 'text-rust-700' : warnCount ? 'text-gold-700' : 'text-sage-700'
                  }`}
                >
                  {errCount || warnCount ? (
                    <span>
                      {errCount > 0 && `${errCount} error${errCount > 1 ? 's' : ''}`}
                      {errCount > 0 && warnCount > 0 && ' · '}
                      {warnCount > 0 && `${warnCount} warning${warnCount > 1 ? 's' : ''}`}
                    </span>
                  ) : (
                    <>
                      <CircleCheck size={13} /> No problems
                    </>
                  )}
                </button>
              )}
              {editorTab === 'code' && (
                <ToolButton title="Modeling copilot" active={copilotOpen} onClick={() => toggleCopilot(!copilotOpen)}>
                  <Sparkles size={15} className="mr-1" />
                  <span className="text-xs font-medium">Copilot</span>
                </ToolButton>
              )}
              <ToolButton
                title={fullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
                onClick={() => setFullscreen((v) => !v)}
              >
                {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
              </ToolButton>
            </div>
          </div>

          {/* Problems */}
          {editorTab === 'code' && showProblems && problems.length > 0 && (
            <div className="rounded-md border border-line-200 bg-white px-3 py-2 text-xs">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-semibold uppercase tracking-wide text-ink-400">Problems</span>
                <button onClick={() => setShowProblems(false)} className="text-ink-300 hover:text-ink-600">
                  <X size={13} />
                </button>
              </div>
              <ul className="max-h-32 space-y-0.5 overflow-y-auto">
                {problems.map((p, i) => (
                  <li key={i}>
                    <button
                      type="button"
                      onClick={() => jumpToProblem(p)}
                      disabled={p.line == null}
                      title={p.line != null ? 'Jump to line' : undefined}
                      className="flex w-full gap-2 rounded px-1 py-0.5 text-left transition-colors enabled:hover:bg-cream-100 disabled:cursor-default"
                    >
                      <span
                        className={
                          p.severity === 'error'
                            ? 'font-medium text-rust-700'
                            : p.severity === 'warning'
                              ? 'font-medium text-gold-700'
                              : 'font-medium text-sage-700'
                        }
                      >
                        {p.severity}
                      </span>
                      <span className="text-ink-600">
                        {p.line ? `L${p.line}: ` : ''}
                        {p.message}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Editor: code (source + copilot) or docs (markdown + live preview) */}
          {editorTab === 'code' ? (
            <div className={`flex gap-2 ${fullscreen ? 'flex-1 min-h-0' : 'h-[58vh]'}`}>
              <div className="min-h-0 min-w-0 flex-1 overflow-hidden rounded-md border border-line-200 bg-white">
                <Editor
                  height="100%"
                  defaultLanguage="python"
                  language="python"
                  value={editorValue}
                  onChange={(v) => authoring && setCode(v ?? '')}
                  beforeMount={defineAtelierTheme}
                  onMount={handleCodeEditorMount}
                  theme="atelier-light"
                  options={{
                    ...CODE_EDITOR_OPTIONS,
                    readOnly: !authoring,
                    minimap: { enabled: minimap },
                    fontSize,
                    wordWrap: wrap ? 'on' : 'off',
                  }}
                />
              </div>
              {/* Kept mounted (toggled via `hidden`) so an in-flight stream and
                  freshly-typed input survive closing/reopening the panel. */}
              <div
                className={
                  copilotOpen
                    ? 'w-[30rem] shrink-0 overflow-hidden rounded-md border border-line-200 bg-white'
                    : 'hidden'
                }
              >
                <CopilotPanel
                  key={`copilot-${copilotName}-${copilotVersion ?? 'draft'}`}
                  sourceCode={editorValue}
                  onApplyCode={applyCode}
                  name={copilotName}
                  version={copilotVersion}
                  active={copilotOpen}
                  onClose={() => toggleCopilot(false)}
                  className="h-full"
                />
              </div>
            </div>
          ) : editorTab === 'docs' ? (
            <div className={`flex flex-col min-h-0 ${fullscreen ? 'flex-1' : ''}`}>
              <div className={`flex gap-2 ${fullscreen ? 'flex-1 min-h-0' : 'h-[58vh]'}`}>
                <div className="min-h-0 min-w-0 flex-1 overflow-hidden rounded-md border border-line-200 bg-white">
                  <Editor
                    height="100%"
                    defaultLanguage="markdown"
                    language="markdown"
                    value={docsValue}
                    onChange={(v) => {
                      if (authoring) setDraftDocs(v ?? '');
                      else if (!isPublished) setDocsDraft(v ?? '');
                    }}
                    beforeMount={defineAtelierTheme}
                    theme="atelier-light"
                    options={{
                      readOnly: !authoring && isPublished,
                      minimap: { enabled: minimap },
                      fontSize,
                      wordWrap: wrap ? 'on' : 'off',
                      scrollBeyondLastLine: false,
                      fontFamily: 'JetBrains Mono, ui-monospace, monospace',
                      renderLineHighlight: 'line',
                      automaticLayout: true,
                    }}
                  />
                </div>
                <div className="min-h-0 w-[45%] shrink-0 overflow-y-auto rounded-md border border-line-200 bg-cream-50 p-4">
                  {docsValue.trim() ? (
                    <div className="text-sm text-ink-700">
                      <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={DOCS_MD}>
                        {normalizeMath(docsValue)}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm italic text-ink-300">
                      Live preview — write markdown on the left.
                    </p>
                  )}
                </div>
              </div>
              {/* Docs save bar */}
              <div className="mt-2 flex items-center gap-2 text-xs">
                {authoring ? (
                  <span className="italic text-ink-400">
                    Docs are saved with the draft when you Register.
                  </span>
                ) : isPublished ? (
                  <span className="italic text-ink-400">
                    Published versions are immutable — use “Edit as new version” to change docs.
                  </span>
                ) : (
                  <>
                    <Button
                      variant="secondary"
                      onClick={saveDocs}
                      disabled={!docsDirty || updateDocs.isPending}
                    >
                      {updateDocs.isPending ? (
                        <Loader2 size={14} className="mr-1.5 animate-spin" />
                      ) : null}
                      Save docs
                    </Button>
                    {docsDirty ? (
                      <span className="text-gold-700">Unsaved changes</span>
                    ) : updateDocs.isSuccess ? (
                      <span className="text-sage-700">Saved ✓</span>
                    ) : null}
                    {updateDocs.isError && (
                      <span className="text-rust-700">{errMessage(updateDocs.error)}</span>
                    )}
                  </>
                )}
              </div>
            </div>
          ) : (
            // Notebook: a Jupyter-like demo/test space for the model in the
            // editor — upload data, run cells against the LIVE buffer, track
            // plots/tables/markdown. Runs the source currently in the editor.
            <div className={`${fullscreen ? 'flex-1 min-h-0' : ''}`}>
              {editorValue.trim() ? (
                <AtelierNotebook
                  name={(authoring ? draftName : selName) || 'untitled'}
                  copilotName={copilotName}
                  copilotVersion={copilotVersion}
                  liveSource={editorValue}
                  onApplyToEditor={applyCode}
                  fill={fullscreen}
                />
              ) : (
                <div
                  className={`flex items-center justify-center rounded-md border border-dashed border-line-300 text-center text-sm text-ink-300 ${
                    fullscreen ? 'h-full' : 'h-[58vh]'
                  }`}
                >
                  Author a new model or select a version to open its demo notebook.
                </div>
              )}
            </div>
          )}

          {/* Action bar */}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {authoring ? (
              <>
                <Button
                  onClick={onRegister}
                  disabled={!draftName.trim() || !code.trim() || register.isPending}
                >
                  {register.isPending ? (
                    <Loader2 size={15} className="mr-1.5 animate-spin" />
                  ) : (
                    <UploadCloud size={15} className="mr-1.5" />
                  )}
                  Register draft
                </Button>
                <Button variant="ghost" onClick={() => setAuthoring(false)}>
                  Cancel
                </Button>
                {register.isError && (
                  <span className="text-xs text-rust-700">{errMessage(register.error)}</span>
                )}
              </>
            ) : detail ? (
              <>
                <Button
                  onClick={() => test.start.mutate()}
                  disabled={testing || detail.status === 'deprecated'}
                >
                  {testing ? (
                    <Loader2 size={15} className="mr-1.5 animate-spin" />
                  ) : (
                    <FlaskConical size={15} className="mr-1.5" />
                  )}
                  {testing ? 'Testing…' : 'Run compatibility test'}
                </Button>
                {detail.status === 'tested' && (
                  <Button
                    variant="primary"
                    onClick={() =>
                      promote.mutate({ name: detail.name, version: detail.version })
                    }
                    disabled={promote.isPending}
                  >
                    <Sprout size={15} className="mr-1.5" /> Publish
                  </Button>
                )}
                <Button variant="secondary" onClick={editAsNewVersion}>
                  Edit as new version
                </Button>
                {(detail.status === 'draft' || detail.status === 'deprecated') && (
                  <Button
                    variant="danger"
                    onClick={() => {
                      remove.mutate(
                        { name: detail.name, version: detail.version },
                        {
                          onSuccess: () => {
                            setSelName(null);
                            setSelVersion(null);
                          },
                        },
                      );
                    }}
                    disabled={remove.isPending}
                  >
                    <Trash2 size={15} className="mr-1.5" /> Delete
                  </Button>
                )}
                {promote.isError && (
                  <span className="text-xs text-rust-700">{errMessage(promote.error)}</span>
                )}
                {remove.isError && (
                  <span className="text-xs text-rust-700">{errMessage(remove.error)}</span>
                )}
              </>
            ) : null}
          </div>
        </div>

        {/* ── Right: compatibility + diagnostics (collapsible) ── */}
        <div className="min-w-0">
          {rightCollapsed && (
            <button
              type="button"
              title="Expand compatibility"
              onClick={() => setRightCollapsed(false)}
              className="hidden h-full max-h-[72vh] w-full flex-col items-center gap-3 rounded-lg border border-line-200 bg-white py-3 text-ink-400 shadow-sm transition hover:bg-cream-50 hover:text-ink-700 xl:flex"
            >
              <PanelRightOpen size={16} />
              <FlaskConical size={15} />
              <span className="text-xs font-semibold uppercase tracking-wide [writing-mode:vertical-rl]">
                Compatibility
              </span>
            </button>
          )}
          <Card
            tone="white"
            padding="sm"
            className={`h-full overflow-y-auto max-h-[72vh] ${rightCollapsed ? 'xl:hidden' : ''}`}
          >
            <div className="flex justify-end pb-1">
              <button
                type="button"
                title="Collapse compatibility"
                onClick={() => setRightCollapsed(true)}
                className="hidden text-ink-300 transition hover:text-ink-700 xl:inline-flex"
              >
                <PanelRightClose size={14} />
              </button>
            </div>
            {!detail && !authoring ? (
            <EmptyState
              icon={Sprout}
              title="Pick a model"
              description="Select a model on the left to see its compatibility report, or start a new one."
            />
          ) : authoring ? (
            <div className="space-y-2 text-sm text-ink-500">
              <h3 className="font-display text-base text-ink-900">Authoring a new model</h3>
              <p>
                Write a <code className="text-ink-700">BayesianMMM</code> subclass (subclass{' '}
                <code className="text-ink-700">CustomMMM</code>). Set a name + docs, then{' '}
                <strong>Register draft</strong>. Registration validates the source statically; the
                full compatibility suite runs (sandboxed) when you test it.
              </p>
            </div>
          ) : (
            <>
              <Tabs
                tabs={[
                  { id: 'compat', label: 'Compatibility' },
                  { id: 'about', label: 'About' },
                ]}
                active={rightTab}
                onChange={setRightTab}
              />
              {rightTab === 'compat' ? (
                <div className="mt-3 space-y-3">
                  {test.job.data?.status === 'error' && (
                    <p className="rounded border border-rust-100 bg-rust-100/40 px-3 py-1.5 text-xs text-rust-700">
                      Test failed: {test.job.data.error}
                    </p>
                  )}
                  {blockingPassed != null && (
                    <div className="flex items-center gap-3">
                      <GardenChip status={detail!.status} />
                      <span
                        className={`text-sm font-medium ${
                          blockingPassed ? 'text-sage-700' : 'text-rust-700'
                        }`}
                      >
                        {blockingPassed ? 'Compatible' : 'Not compatible'}
                      </span>
                      {score != null && (
                        <span className="text-xs text-ink-400">
                          fit-quality score {score}
                        </span>
                      )}
                    </div>
                  )}
                  {tiers.length > 0 ? (
                    <DataTable
                      columns={tierColumns}
                      rows={tiers}
                      rowKey={(t) => t.name}
                    />
                  ) : (
                    <p className="text-sm text-ink-300 italic">
                      No compatibility report yet — run the test.
                    </p>
                  )}
                  {(liveReport?.promoted ?? false) && (
                    <p className="text-xs text-sage-700">
                      ✓ Passed — promoted to <strong>tested</strong>.
                    </p>
                  )}
                </div>
              ) : (
                <dl className="mt-3 space-y-2 text-sm">
                  <div>
                    <dt className="text-ink-400">Class</dt>
                    <dd className="text-ink-900">{detail!.manifest?.class_name ?? '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-ink-400">Contract</dt>
                    <dd className="text-ink-900">
                      v{detail!.manifest?.contract_version ?? '—'}
                    </dd>
                  </div>
                  <div>
                    <div className="flex items-center justify-between">
                      <dt className="text-ink-400">Docs</dt>
                      {detail!.docs && (
                        <button
                          type="button"
                          title="Expand docs"
                          onClick={() => setDocsExpanded(true)}
                          className="inline-flex items-center gap-1 text-xs text-ink-400 transition hover:text-ink-700"
                        >
                          <Maximize2 size={12} /> Expand
                        </button>
                      )}
                    </div>
                    <dd className="mt-1 text-sm text-ink-700">
                      {detail!.docs ? (
                        <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={DOCS_MD}>
                          {normalizeMath(detail!.docs)}
                        </ReactMarkdown>
                      ) : (
                        '—'
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-ink-400">History</dt>
                    <dd className="text-ink-500 text-xs">
                      {detail!.status_history
                        .map((h) => h.status)
                        .join(' → ') || '—'}
                    </dd>
                  </div>
                </dl>
              )}
            </>
          )}
          </Card>
        </div>
      </div>

      <Drawer
        open={docsExpanded}
        onClose={() => setDocsExpanded(false)}
        title={detail ? `${detail.name} · docs` : 'Docs'}
        width="max-w-3xl"
      >
        {detail?.docs ? (
          <div className="text-sm text-ink-700">
            <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={DOCS_MD}>
              {normalizeMath(detail.docs)}
            </ReactMarkdown>
          </div>
        ) : (
          <p className="italic text-ink-400">No docs for this model.</p>
        )}
      </Drawer>
    </div>
  );
}

export default ModelGardenPage;
