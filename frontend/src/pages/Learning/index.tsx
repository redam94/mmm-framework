import { useEffect, useMemo, useState } from 'react';
import { clsx } from 'clsx';
import {
  Compass,
  Download,
  FlaskConical,
  FolderOpen,
  Loader2,
  Plus,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import { Button, Card, DataTable, EmptyState, SectionHeader, type Column } from '../../components/ui';
import { useProjectStore } from '../../stores/projectStore';
import {
  useDeleteProgram,
  useLearningProgram,
  useLearningPrograms,
  useStartFit,
} from '../../api/hooks/useLearning';
import type {
  FundingVerdict,
  LearningProgram,
  LearningProgramStatus,
} from '../../api/services/learningService';
import { ProgramCreateWizard } from './ProgramCreateWizard';
import { DesignWaveStudio } from './DesignWaveStudio';
import { ImportExperimentsPanel } from './ImportExperimentsPanel';
import { EnbsCard } from './EnbsCard';
import { FundingLineChart } from './FundingLineChart';
import { SynergyHeatmap } from './SynergyHeatmap';
import { WaveTimeline } from './WaveTimeline';
import { effectiveChannels, errorDetail, fmtDollars, fmtNum, fmtSignedDollars } from './format';

const PROGRAM_STATUS: Record<LearningProgramStatus, { cls: string; label: string }> = {
  active: { cls: 'bg-sage-100 text-sage-800', label: 'Active' },
  stopped: { cls: 'bg-gold-100 text-gold-700', label: 'Stopped' },
  archived: { cls: 'bg-cream-200 text-ink-600', label: 'Archived' },
};

const VERDICT_CHIP: Record<FundingVerdict, string> = {
  FUND: 'bg-sage-100 text-sage-800',
  HOLD: 'bg-gold-100 text-gold-700',
  CUT: 'bg-rust-100 text-rust-700',
};

function ProgramStatusChip({ status }: { status: LearningProgramStatus }) {
  const s = PROGRAM_STATUS[status] ?? { cls: 'bg-cream-200 text-ink-600', label: status };
  return (
    <span className={clsx('inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium', s.cls)}>
      {s.label}
    </span>
  );
}

interface RecRow {
  channel: string;
  recommended: number;
  delta: number | null;
  sd: number | null;
  verdict: FundingVerdict | null;
}

function ProgramListCard({
  program,
  active,
  onClick,
}: {
  program: LearningProgram;
  active: boolean;
  onClick: () => void;
}) {
  const nWaves = program.summary?.evidence?.n_waves ?? null;
  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-full rounded-lg border p-3 text-left shadow-sm transition-colors',
        active ? 'border-sage-600 bg-sage-100/50' : 'border-line-200 bg-white hover:bg-cream-100',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="min-w-0 truncate text-sm font-medium text-ink-900">{program.name}</span>
        <ProgramStatusChip status={program.status} />
      </div>
      <p className="mt-1 text-xs text-ink-400">
        <span className="num">{(program.channels ?? []).length}</span> channels ·{' '}
        <span className="num">{fmtDollars(program.config?.budget)}</span>/geo-period
        {nWaves != null && (
          <>
            {' '}· <span className="num">{nWaves}</span> wave{nWaves === 1 ? '' : 's'}
          </>
        )}
      </p>
    </button>
  );
}

export function LearningPage() {
  const projectId = useProjectStore((s) => s.currentProjectId);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showWizard, setShowWizard] = useState(false);
  const [showDesigner, setShowDesigner] = useState(false);
  const [showImport, setShowImport] = useState(false);

  const programsQuery = useLearningPrograms(projectId);
  const programs = useMemo(() => programsQuery.data ?? [], [programsQuery.data]);
  const activeId =
    selectedId && programs.some((p) => p.id === selectedId)
      ? selectedId
      : programs[0]?.id ?? null;

  const detailQuery = useLearningProgram(projectId, activeId);
  const program = detailQuery.data?.program ?? programs.find((p) => p.id === activeId) ?? null;
  const waves = useMemo(() => detailQuery.data?.waves ?? [], [detailQuery.data]);

  const refit = useStartFit(projectId, activeId);
  const deleteProgram = useDeleteProgram(projectId);

  // a refit job belongs to one program — drop it when switching
  useEffect(() => {
    refit.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  const snapshot = useMemo(() => {
    if (program?.summary) return program.summary;
    const withSnap = [...waves].sort((a, b) => b.wave_index - a.wave_index).find((w) => w.snapshot);
    return withSnap?.snapshot ?? null;
  }, [program, waves]);

  const dims = useMemo(() => (program ? effectiveChannels(program) : []), [program]);

  const recRows: RecRow[] = useMemo(() => {
    if (!snapshot) return [];
    const centerMap = program?.config?.center ?? {};
    const verdictMap = new Map(snapshot.funding.map((f) => [f.channel, f.verdict]));
    return Object.keys(snapshot.recommendation).map((ch) => ({
      channel: ch,
      recommended: snapshot.recommendation[ch],
      delta: centerMap[ch] != null ? snapshot.recommendation[ch] - centerMap[ch] : null,
      sd: snapshot.allocation_sd?.[ch] ?? null,
      verdict: verdictMap.get(ch) ?? null,
    }));
  }, [snapshot, program]);

  const recColumns: Column<RecRow>[] = [
    {
      key: 'channel',
      header: 'Channel',
      render: (r) => <span className="font-medium text-ink-900">{r.channel}</span>,
    },
    {
      key: 'recommended',
      header: 'Recommended $/geo-period',
      numeric: true,
      render: (r) => fmtDollars(r.recommended),
    },
    {
      key: 'delta',
      header: 'Δ vs current',
      numeric: true,
      render: (r) =>
        r.delta == null ? (
          '—'
        ) : (
          <span className={r.delta >= 0 ? 'text-sage-800' : 'text-rust-700'}>
            {fmtSignedDollars(r.delta)}
          </span>
        ),
    },
    {
      key: 'sd',
      header: '± SD',
      numeric: true,
      render: (r) => (r.sd == null ? '—' : fmtDollars(r.sd)),
    },
    {
      key: 'verdict',
      header: 'Funding',
      render: (r) =>
        r.verdict ? (
          <span
            className={clsx(
              'rounded-full px-2 py-0.5 text-[11px] font-semibold',
              VERDICT_CHIP[r.verdict],
            )}
          >
            {r.verdict}
          </span>
        ) : (
          <span className="text-ink-300">—</span>
        ),
    },
  ];

  const warnings = useMemo(() => {
    if (!snapshot) return [];
    const out = [...(snapshot.warnings ?? []), ...(snapshot.diagnostics?.flags ?? [])];
    const unidentified = Object.entries(snapshot.evidence?.shape_identified ?? {})
      .filter(([, ok]) => !ok)
      .map(([ch]) => ch);
    if (unidentified.length > 0) {
      out.push(
        `Curve shape still prior-dominated for ${unidentified.join(', ')} — trust the funded set, not channel-by-channel magnitudes.`,
      );
    }
    return out;
  }, [snapshot]);

  const refitStatus = refit.job.data?.status ?? null;
  const refitting = refit.start.isPending || refitStatus === 'pending' || refitStatus === 'running';
  const refitError = refit.start.isError
    ? errorDetail(refit.start.error)
    : refitStatus === 'error'
      ? refit.job.data?.error ?? 'Refit failed'
      : null;

  const handleDelete = () => {
    if (!program) return;
    if (window.confirm(`Delete "${program.name}"? Its waves and state are removed.`)) {
      deleteProgram.mutate(program.id, { onSuccess: () => setSelectedId(null) });
    }
  };

  return (
    <div className="space-y-6">
      <SectionHeader
        level={1}
        title="Continuous learning"
        subtitle="Model-free response-surface programs: design a wave, read the funding line, stop when learning stops paying."
        actions={
          <Button onClick={() => setShowWizard(true)} disabled={!projectId}>
            <Plus className="h-4 w-4" /> Start a program
          </Button>
        }
      />

      {!projectId ? (
        <EmptyState
          icon={FolderOpen}
          title="No project selected"
          description="Pick a project from the switcher in the header to see its learning programs."
        />
      ) : programsQuery.isLoading ? (
        <div className="py-16 text-center text-sm text-ink-400">Loading learning programs…</div>
      ) : programs.length === 0 ? (
        <EmptyState
          icon={Compass}
          title="No learning programs yet"
          description="A learning program fits a spend→KPI response surface directly from designed geo experiments — no MMM required. Start one to design the first wave, or import past lift tests as evidence."
          action={<Button onClick={() => setShowWizard(true)}>Start a program</Button>}
        />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[280px,1fr] lg:items-start">
          {/* ── Program rail ── */}
          <aside className="space-y-2">
            {programs.map((p) => (
              <ProgramListCard
                key={p.id}
                program={p}
                active={p.id === activeId}
                onClick={() => setSelectedId(p.id)}
              />
            ))}
          </aside>

          {/* ── Program detail ── */}
          {program ? (
            <div className="min-w-0 space-y-6">
              <Card padding="md">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2.5">
                      <h2 className="font-display text-xl font-semibold text-ink-900">
                        {program.name}
                      </h2>
                      <ProgramStatusChip status={program.status} />
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      {dims.map((ch) => (
                        <span
                          key={ch}
                          className="rounded-full bg-cream-200 px-2 py-0.5 text-xs text-ink-600"
                        >
                          {ch}
                        </span>
                      ))}
                    </div>
                    <p className="mt-2 text-xs text-ink-400">
                      Budget <span className="num">{fmtDollars(program.config?.budget)}</span>
                      /period per geo · KPI unit worth{' '}
                      <span className="num">
                        ${fmtNum(program.config?.value_per_unit ?? null)}
                      </span>
                      {program.config?.kpi && <> · KPI {program.config.kpi}</>}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button variant="secondary" onClick={() => setShowImport(true)}>
                      <Download className="h-4 w-4" /> Import past experiments
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => refit.start.mutate({})}
                      disabled={refitting}
                      title="Refit the surface on all accumulated evidence"
                    >
                      {refitting ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="h-4 w-4" />
                      )}
                      Refit
                    </Button>
                    <Button onClick={() => setShowDesigner(true)}>
                      <FlaskConical className="h-4 w-4" /> Design next wave
                    </Button>
                    <button
                      onClick={handleDelete}
                      className="rounded-md p-2 text-ink-300 transition-colors hover:bg-rust-100 hover:text-rust-700"
                      title="Delete program"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {snapshot && (
                  <p className="mt-3 border-t border-line-200 pt-3 text-xs text-ink-400">
                    Evidence: <span className="num">{snapshot.evidence.n_waves}</span> waves ·{' '}
                    <span className="num">{snapshot.evidence.n_rows.toLocaleString()}</span>{' '}
                    geo-period rows ·{' '}
                    <span className="num">{snapshot.evidence.n_summaries}</span> imported readouts
                    · fitted{' '}
                    <span className="num">
                      {new Date(snapshot.fitted_at * 1000).toLocaleString()}
                    </span>
                    {snapshot.diagnostics.max_rhat != null && (
                      <>
                        {' '}· R̂ max{' '}
                        <span className="num">{fmtNum(snapshot.diagnostics.max_rhat, 3)}</span>
                      </>
                    )}
                    {snapshot.diagnostics.min_ess != null && (
                      <>
                        {' '}· ESS min{' '}
                        <span className="num">{Math.round(snapshot.diagnostics.min_ess)}</span>
                      </>
                    )}
                  </p>
                )}

                {refitting && (
                  <p className="mt-3 flex items-center gap-2 border-t border-line-200 pt-3 text-xs text-ink-400">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-sage-700" />
                    Refitting on all accumulated evidence — the readouts below refresh when it
                    lands.
                  </p>
                )}
                {refitError && <p className="mt-3 text-sm text-rust-600">{refitError}</p>}
              </Card>

              {warnings.length > 0 && (
                <div className="rounded-lg border border-gold-300 bg-gold-100/70 px-4 py-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-gold-700">
                    Watch-outs
                  </p>
                  <ul className="mt-1 list-inside list-disc space-y-0.5 text-xs text-gold-700">
                    {warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {snapshot ? (
                <>
                  <div className="grid gap-6 xl:grid-cols-[minmax(280px,360px),1fr] xl:items-start">
                    <EnbsCard regret={snapshot.regret} />
                    <Card padding="none">
                      <div className="px-5 pb-2 pt-4">
                        <h3 className="text-sm font-semibold text-ink-900">
                          Recommended allocation
                        </h3>
                        <p className="mt-0.5 text-xs text-ink-400">
                          Thompson-consensus allocation of{' '}
                          <span className="num">{fmtDollars(program.config?.budget)}</span> per
                          geo-period vs today's center — dollars are per geo, per period.
                        </p>
                      </div>
                      <DataTable
                        columns={recColumns}
                        rows={recRows}
                        rowKey={(r) => r.channel}
                        className="rounded-t-none border-x-0 border-b-0 shadow-none"
                      />
                    </Card>
                  </div>

                  <FundingLineChart snapshot={snapshot} />

                  <SynergyHeatmap snapshot={snapshot} channels={dims} />
                </>
              ) : (
                <EmptyState
                  icon={FlaskConical}
                  title="No fit yet"
                  description="Design the first wave (or import past experiments), record what happened, and the funding line, synergy map, and stopping rule will appear here."
                  action={
                    <Button onClick={() => setShowDesigner(true)}>
                      <FlaskConical className="h-4 w-4" /> Design the first wave
                    </Button>
                  }
                  secondary={
                    <button
                      className="text-sage-700 hover:underline"
                      onClick={() => setShowImport(true)}
                    >
                      Import past experiments instead
                    </button>
                  }
                />
              )}

              <div className="space-y-3">
                <SectionHeader
                  title="Waves"
                  subtitle="Each wave accumulates evidence — the fit always uses everything so far."
                />
                <WaveTimeline waves={waves} />
              </div>
            </div>
          ) : (
            <div className="py-16 text-center text-sm text-ink-400">Loading program…</div>
          )}
        </div>
      )}

      {showWizard && (
        <ProgramCreateWizard
          projectId={projectId}
          onClose={() => setShowWizard(false)}
          onCreated={(id) => setSelectedId(id)}
        />
      )}

      {program && (
        <>
          <DesignWaveStudio
            open={showDesigner}
            onClose={() => setShowDesigner(false)}
            projectId={projectId}
            program={program}
          />
          <ImportExperimentsPanel
            open={showImport}
            onClose={() => setShowImport(false)}
            projectId={projectId}
            programId={program.id}
          />
        </>
      )}
    </div>
  );
}
