import { useRef, useState } from 'react';
import { Gauge, Upload } from 'lucide-react';
import { Button, DataTable, EmptyState, SectionHeader, StatHero } from '../../components/ui';
import type { Column } from '../../components/ui/DataTable';
import { COLORS } from '../../theme/colors';
import { useProjectPacing, useUploadDelivery } from '../../api/hooks/usePacing';
import type { Pacing, PacingChannel, PacingStatus } from '../../api/services/pacingService';

// ── In-flight pacing — actual delivery vs the saved plan (issue #123) ──────────
// The planned series is auto-sourced server-side from the project's latest saved
// budget plan; the planner uploads actual delivery (CSV/JSON) and sees live
// pacing, off-pace flags, and an alert. Model-free — the expected-outcome delta
// is produced by the check_pacing agent tool, not this panel.

const STATUS_STYLE: Record<PacingStatus, { fg: string; bg: string; label: string }> = {
  'on-track': { fg: COLORS.sage800, bg: COLORS.sage100, label: 'On track' },
  'over-pacing': { fg: COLORS.rust700, bg: COLORS.rust100, label: 'Over-pacing' },
  'under-pacing': { fg: COLORS.steel700, bg: COLORS.steel100, label: 'Under-pacing' },
  'not-started': { fg: COLORS.ink400, bg: COLORS.cream200, label: 'Not started' },
};

function money(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1000) return `$${(v / 1000).toLocaleString(undefined, { maximumFractionDigits: 1 })}k`;
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function pct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(0)}%`;
}

function StatusChip({ status }: { status: PacingStatus }) {
  const s = STATUS_STYLE[status] ?? STATUS_STYLE['not-started'];
  return (
    <span
      className="inline-flex rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ backgroundColor: s.bg, color: s.fg }}
    >
      {s.label}
    </span>
  );
}

function UploadButton({
  onFile,
  pending,
  label = 'Upload delivery',
}: {
  onFile: (file: File) => void;
  pending: boolean;
  label?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.tsv,.txt,.json"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = '';
        }}
      />
      <Button
        variant="secondary"
        size="sm"
        onClick={() => inputRef.current?.click()}
        disabled={pending}
      >
        <Upload className="mr-1.5 h-3.5 w-3.5" />
        {pending ? 'Uploading…' : label}
      </Button>
    </>
  );
}

function AlertCallout({ pacing }: { pacing: Pacing }) {
  const alert = pacing.alert;
  if (!alert?.off_pace) {
    return (
      <div
        className="rounded-lg px-4 py-3 text-sm"
        style={{ backgroundColor: COLORS.sage100, color: COLORS.sage800 }}
      >
        All channels are pacing within {(alert?.threshold ?? 0.1) * 100}% of plan.
      </div>
    );
  }
  const worst = alert.worst;
  return (
    <div
      className="rounded-lg px-4 py-3 text-sm"
      style={{ backgroundColor: COLORS.rust100, color: COLORS.rust700 }}
    >
      <span className="font-semibold">
        Off-pace: {alert.n_flagged} channel{alert.n_flagged === 1 ? '' : 's'}
      </span>{' '}
      — <strong>{alert.flagged.join(', ')}</strong> diverged more than{' '}
      {(alert.threshold * 100).toFixed(0)}% from plan
      {worst ? ` (worst: ${worst.channel} ${pct(worst.divergence_pct)})` : ''}. Review before the
      divergence compounds.
    </div>
  );
}

function PacingView({ pacing, projectId }: { pacing: Pacing; projectId: string }) {
  const upload = useUploadDelivery(projectId);
  const [err, setErr] = useState<string | null>(null);
  const onFile = async (file: File) => {
    setErr(null);
    try {
      await upload.mutateAsync(file);
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Upload failed');
    }
  };

  const columns: Column<PacingChannel>[] = [
    { key: 'channel', header: 'Channel', render: (r) => <span className="font-medium text-ink-700">{r.channel}</span> },
    { key: 'planned', header: 'Planned (to date)', numeric: true, render: (r) => money(r.planned) },
    { key: 'actual', header: 'Actual', numeric: true, render: (r) => money(r.actual) },
    { key: 'divergence', header: 'Divergence', numeric: true, render: (r) => pct(r.divergence_pct) },
    { key: 'status', header: 'Status', render: (r) => <StatusChip status={r.status} /> },
  ];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-ink-500">
          Actual delivery vs the plan
          {pacing.plan_name ? (
            <>
              {' '}
              <span className="font-medium text-ink-700">{pacing.plan_name}</span>
            </>
          ) : null}
          {pacing.plan_basis ? (
            <span className="text-ink-400"> · {pacing.plan_basis} basis</span>
          ) : null}
          .
        </p>
        <UploadButton onFile={onFile} pending={upload.isPending} label="Update delivery" />
      </div>

      {err && <p className="text-sm text-rust-700">{err}</p>}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatHero
          label="Portfolio pacing vs plan"
          value={pct(pacing.divergence_pct)}
          delta={pacing.divergence_pct != null ? pacing.divergence_pct * 100 : null}
          deltaLabel={pct(pacing.divergence_pct)}
          increaseIsGood={false}
          hint={`${money(pacing.actual_total)} of ${money(pacing.planned_total)} planned`}
        />
        <StatHero
          label="Spent to date"
          value={money(pacing.actual_total)}
          hint={`Planned ${money(pacing.planned_total)}`}
        />
        <StatHero
          label="Off-pace channels"
          value={String(pacing.alert?.n_flagged ?? pacing.flagged.length)}
          hint={`Threshold ±${((pacing.threshold ?? 0.1) * 100).toFixed(0)}%`}
        />
      </div>

      <AlertCallout pacing={pacing} />

      <DataTable<PacingChannel>
        columns={columns}
        rows={pacing.channels}
        rowKey={(r) => r.channel}
      />
    </div>
  );
}

export function PacingPanel({ projectId }: { projectId: string }) {
  const { data: pacing, isLoading, isError } = useProjectPacing(projectId);
  const upload = useUploadDelivery(projectId);
  const [err, setErr] = useState<string | null>(null);
  const onFile = async (file: File) => {
    setErr(null);
    try {
      await upload.mutateAsync(file);
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Upload failed');
    }
  };

  return (
    <div className="space-y-6">
      <SectionHeader
        level={2}
        title="In-flight pacing"
        subtitle="Actual delivery vs the recommended plan — where money is drifting off-pace between fits."
      />

      {isLoading ? (
        <p className="text-sm text-ink-400">Loading pacing…</p>
      ) : isError || !pacing ? (
        <p className="text-sm text-rust-700">Failed to load pacing.</p>
      ) : pacing.available ? (
        <PacingView pacing={pacing} projectId={projectId} />
      ) : pacing.reason === 'no_plan' ? (
        <EmptyState
          icon={Gauge}
          title="No saved plan to pace against"
          description="Pacing compares actual delivery to your latest saved budget plan. Build and save a plan in the Planner, then upload delivery here."
        />
      ) : (
        <div className="space-y-3">
          {err && <p className="text-sm text-rust-700">{err}</p>}
          <EmptyState
            icon={Upload}
            title="Upload actual delivery to see pacing"
            description="A CSV (long: channel/period/spend, or wide: a period column + one column per channel) or JSON of spend to date. Re-uploading a period overwrites it."
            action={<UploadButton onFile={onFile} pending={upload.isPending} />}
          />
        </div>
      )}
    </div>
  );
}

export default PacingPanel;
