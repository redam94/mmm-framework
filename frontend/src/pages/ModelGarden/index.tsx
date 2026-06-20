import { useMemo, useState } from 'react';
import Editor from '@monaco-editor/react';
import { FlaskConical, Loader2, Plus, Sprout, Trash2, UploadCloud } from 'lucide-react';
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

export function ModelGardenPage() {
  // Selection + editor state
  const [selName, setSelName] = useState<string | null>(null);
  const [selVersion, setSelVersion] = useState<number | null>(null);
  const [authoring, setAuthoring] = useState(false);
  const [code, setCode] = useState(STARTER_TEMPLATE);
  const [draftName, setDraftName] = useState('');
  const [draftDocs, setDraftDocs] = useState('');
  const [rightTab, setRightTab] = useState('compat');

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

        {/* ── Center: editor ── */}
        <div className="flex flex-col min-h-0">
          {authoring && (
            <div className="mb-2 grid grid-cols-1 sm:grid-cols-[1fr_2fr] gap-2">
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

          <div className="overflow-hidden rounded-md border border-line-200 bg-white">
            <Editor
              height="58vh"
              defaultLanguage="python"
              language="python"
              value={editorValue}
              onChange={(v) => authoring && setCode(v ?? '')}
              theme="vs"
              options={{
                readOnly: !authoring,
                minimap: { enabled: false },
                fontSize: 13,
                scrollBeyondLastLine: false,
                fontFamily: 'JetBrains Mono, ui-monospace, monospace',
                renderLineHighlight: 'line',
                automaticLayout: true,
              }}
            />
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
