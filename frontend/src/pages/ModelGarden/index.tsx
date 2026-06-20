import { useEffect, useMemo, useState, type ReactNode } from 'react';
import Editor, { type OnMount } from '@monaco-editor/react';
import {
  AlignLeft,
  CircleCheck,
  FlaskConical,
  Loader2,
  Map as MapIcon,
  Maximize2,
  Minimize2,
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
  EmptyState,
  SectionHeader,
  Tabs,
  type Column,
} from '../../components/ui';
import { GARDEN_STATUS, type GardenStatus } from '../../theme/colors';
import {
  useDeleteGardenModel,
  useGardenModel,
  useGardenModels,
  useGardenSource,
  useGardenTest,
  useGardenVersions,
  usePromoteGardenModel,
  useRegisterGardenModel,
} from '../../api/hooks';
import type { CompatTier, GardenModel } from '../../api/services/modelGardenService';
import { CopilotPanel } from '../../components/modelGarden/CopilotPanel';
import { copilotService, type LintProblem } from '../../api/services/copilotService';

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

// ── IDE: framework-aware autocomplete snippets ──────────────────────────────
const GARDEN_SNIPPETS: { label: string; detail: string; doc: string; insert: string }[] = [
  {
    label: 'custommmm',
    detail: 'CustomMMM subclass skeleton',
    doc: 'Minimal garden model: subclass CustomMMM and override _build_model.',
    insert: [
      'from mmm_framework.garden import CustomMMM',
      'import pymc as pm',
      'import pytensor.tensor as pt',
      '',
      '',
      'class ${1:MyMMM}(CustomMMM):',
      '    """${2:What makes this model bespoke and when to use it.}"""',
      '',
      '    def _build_model(self) -> pm.Model:',
      '        coords = self._build_coords()',
      '        x_media = self._prepare_raw_media_for_model()',
      '        with pm.Model(coords=coords) as model:',
      '            $0',
      '        return model',
      '',
      '',
      'GARDEN_MODEL = ${1:MyMMM}',
    ].join('\n'),
  },
  {
    label: 'build_model',
    detail: 'override _build_model (contract-complete)',
    doc: 'A _build_model registering every deterministic the read-ops consume.',
    insert: [
      'def _build_model(self) -> pm.Model:',
      '    coords = self._build_coords()',
      '    x_media_norm = self._prepare_raw_media_for_model()',
      '    n_obs = self.n_obs',
      '    with pm.Model(coords=coords) as model:',
      '        x_media = pm.Data("X_media_raw", x_media_norm, dims=("obs", "channel"))',
      '        intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)',
      '        pm.Deterministic("intercept_component", intercept + pt.zeros(n_obs), dims="obs")',
      '        contribs = []',
      '        for c, ch in enumerate(self.channel_names):',
      '            sat_kind, sat_params = self._build_channel_saturation(ch)',
      '            x_sat = _apply_saturation_pt(x_media[:, c], sat_kind, sat_params)',
      '            beta = pm.Gamma(f"beta_{ch}", mu=1.5, sigma=1.0)',
      '            contribs.append(beta * x_sat)',
      '        channels = pt.stack(contribs, axis=1)',
      '        pm.Deterministic("channel_contributions", channels, dims=("obs", "channel"))',
      '        media_total = channels.sum(axis=1)',
      '        pm.Deterministic("media_total", media_total)',
      '        sigma = pm.HalfNormal("sigma", sigma=0.5)',
      '        y_obs = pm.Normal("y_obs", mu=intercept + media_total, sigma=sigma, observed=self.y, dims="obs")',
      '        pm.Deterministic("y_obs_scaled", y_obs * self.y_std + self.y_mean, dims="obs")',
      '    return model',
    ].join('\n'),
  },
  {
    label: 'vectorized_adstock',
    detail: 'geometric carryover WITHOUT pytensor.scan',
    doc: 'Lower-triangular Toeplitz matmul: Sₜ = Σ ρ^(t-τ)·xₜ. Compiles instantly.',
    insert: [
      'import numpy as np',
      '',
      't = np.arange(n_obs)',
      'lag = t[:, None] - t[None, :]',
      'decay = pt.where(',
      '    pt.as_tensor_variable(lag >= 0),',
      '    ${1:rho} ** pt.as_tensor_variable(np.maximum(lag, 0)),',
      '    0.0,',
      ')',
      '${2:carryover} = decay @ ${3:media_inflow}  # (n_obs, n_channel), no scan',
    ].join('\n'),
  },
  {
    label: 'deterministic_channels',
    detail: 'register channel_contributions + media_total',
    doc: 'The two deterministics ROI/decomposition reporting needs.',
    insert: [
      'pm.Deterministic("channel_contributions", ${1:channels}, dims=("obs", "channel"))',
      'pm.Deterministic("media_total", ${1:channels}.sum(axis=1))',
    ].join('\n'),
  },
  {
    label: 'prior_normal',
    detail: 'pm.Normal prior',
    doc: 'Normal prior.',
    insert: 'pm.Normal("${1:name}", mu=${2:0.0}, sigma=${3:1.0})',
  },
  {
    label: 'prior_retention',
    detail: 'pm.Beta carryover/retention prior',
    doc: 'Beta(6, 2) ≈ mean 0.75 — a sticky carryover rate on (0, 1).',
    insert: 'pm.Beta("${1:adstock_alpha_channel}", alpha=${2:6.0}, beta=${3:2.0})',
  },
];

// Register framework completions once for the whole app lifetime.
let gardenCompletionsRegistered = false;
const registerGardenCompletions: OnMount = (_editor, monaco) => {
  if (gardenCompletionsRegistered) return;
  gardenCompletionsRegistered = true;
  monaco.languages.registerCompletionItemProvider('python', {
    provideCompletionItems(model, position) {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      return {
        suggestions: GARDEN_SNIPPETS.map((s) => ({
          label: s.label,
          kind: monaco.languages.CompletionItemKind.Snippet,
          insertText: s.insert,
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          detail: s.detail,
          documentation: s.doc,
          range,
        })),
      };
    },
  });
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

  // IDE + copilot state
  const [fullscreen, setFullscreen] = useState(false);
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [wrap, setWrap] = useState(false);
  const [minimap, setMinimap] = useState(false);
  const [fontSize, setFontSize] = useState(13);
  const [problems, setProblems] = useState<LintProblem[]>([]);
  const [showProblems, setShowProblems] = useState(false);
  const [ideBusy, setIdeBusy] = useState<null | 'lint' | 'format'>(null);

  // Data
  const { data: listData, isLoading: listLoading } = useGardenModels();
  const { data: versionsData } = useGardenVersions(selName);
  const { data: detail } = useGardenModel(selName, selVersion);
  const { data: sourceData } = useGardenSource(authoring ? null : selName, selVersion);

  // Mutations / jobs
  const register = useRegisterGardenModel();
  const promote = usePromoteGardenModel();
  const remove = useDeleteGardenModel();
  const test = useGardenTest(selName, selVersion);

  const models = listData?.models ?? [];
  const versions = versionsData?.versions ?? [];

  // Editor shows the editable draft while authoring, else the fetched (read-only)
  // source — derived (no effect) to avoid a setState-in-effect cascade.
  const editorValue = authoring ? code : sourceData?.source_code ?? '';

  const startAuthoring = () => {
    setAuthoring(true);
    setSelName(null);
    setSelVersion(null);
    setCode(STARTER_TEMPLATE);
    setDraftName('');
    setDraftDocs('');
    test.reset();
  };

  const selectVersion = (m: GardenModel) => {
    setAuthoring(false);
    setSelName(m.name);
    setSelVersion(m.version);
    test.reset();
  };

  const onRegister = () => {
    register.mutate(
      { source_code: code, name: draftName.trim(), docs: draftDocs.trim() },
      {
        onSuccess: (row) => {
          setAuthoring(false);
          setSelName(row.name);
          setSelVersion(row.version);
        },
      },
    );
  };

  const editAsNewVersion = () => {
    setCode(sourceData?.source_code ?? code);
    setAuthoring(true);
    setDraftName(selName ?? '');
    setDraftDocs(detail?.docs ?? '');
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
      setProblems(r.problems ?? []);
      setShowProblems(true);
    } catch (e) {
      setProblems([{ severity: 'error', message: errMessage(e), line: null }]);
      setShowProblems(true);
    } finally {
      setIdeBusy(null);
    }
  };

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

      <div className="grid grid-cols-1 lg:grid-cols-[19rem_1fr_23rem] gap-4 flex-1 min-h-0">
        {/* ── Left: registry browser ── */}
        <Card tone="cream" padding="sm" className="overflow-y-auto max-h-[72vh]">
          <h2 className="px-1 pb-2 text-xs font-semibold uppercase tracking-wide text-ink-400">
            Garden
          </h2>
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

        {/* ── Center: editor + IDE tools ── */}
        <div
          className={
            fullscreen
              ? 'fixed inset-0 z-40 flex flex-col gap-2 bg-cream-50 p-4'
              : 'flex flex-col min-h-0'
          }
        >
          {authoring && (
            <div className="grid grid-cols-1 sm:grid-cols-[1fr_2fr] gap-2">
              <input
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                placeholder="model name (e.g. tight-adstock-prior)"
                className="rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:border-sage-600 focus:outline-none"
              />
              <input
                value={draftDocs}
                onChange={(e) => setDraftDocs(e.target.value)}
                placeholder="docs — what it does & when to use it"
                className="rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:border-sage-600 focus:outline-none"
              />
            </div>
          )}

          {/* IDE toolbar */}
          <div className="flex items-center gap-0.5 rounded-md border border-line-200 bg-white px-1.5 py-1">
            <ToolButton title="Format (ruff)" disabled={!authoring || ideBusy !== null} onClick={runFormat}>
              {ideBusy === 'format' ? <Loader2 size={15} className="animate-spin" /> : <AlignLeft size={15} />}
            </ToolButton>
            <ToolButton title="Validate (Problems)" disabled={ideBusy !== null} onClick={runLint}>
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
              <ToolButton title="Modeling copilot" active={copilotOpen} onClick={() => setCopilotOpen((v) => !v)}>
                <Sparkles size={15} className="mr-1" />
                <span className="text-xs font-medium">Copilot</span>
              </ToolButton>
              <ToolButton
                title={fullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
                onClick={() => setFullscreen((v) => !v)}
              >
                {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
              </ToolButton>
            </div>
          </div>

          {/* Problems */}
          {showProblems && problems.length > 0 && (
            <div className="rounded-md border border-line-200 bg-white px-3 py-2 text-xs">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-semibold uppercase tracking-wide text-ink-400">Problems</span>
                <button onClick={() => setShowProblems(false)} className="text-ink-300 hover:text-ink-600">
                  <X size={13} />
                </button>
              </div>
              <ul className="max-h-28 space-y-1 overflow-y-auto">
                {problems.map((p, i) => (
                  <li key={i} className="flex gap-2">
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
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Editor + copilot */}
          <div className={`flex gap-2 ${fullscreen ? 'flex-1 min-h-0' : 'h-[58vh]'}`}>
            <div className="min-h-0 flex-1 overflow-hidden rounded-md border border-line-200 bg-white">
              <Editor
                height="100%"
                defaultLanguage="python"
                language="python"
                value={editorValue}
                onChange={(v) => authoring && setCode(v ?? '')}
                onMount={registerGardenCompletions}
                theme="vs"
                options={{
                  readOnly: !authoring,
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
            {copilotOpen && (
              <div className="w-[23rem] shrink-0 overflow-hidden rounded-md border border-line-200 bg-white">
                <CopilotPanel
                  sourceCode={editorValue}
                  onApplyCode={applyCode}
                  onClose={() => setCopilotOpen(false)}
                  className="h-full"
                />
              </div>
            )}
          </div>

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

        {/* ── Right: compatibility + diagnostics ── */}
        <Card tone="white" padding="sm" className="overflow-y-auto max-h-[72vh]">
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
                    <dt className="text-ink-400">Docs</dt>
                    <dd className="text-ink-700">{detail!.docs || '—'}</dd>
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
  );
}

export default ModelGardenPage;
