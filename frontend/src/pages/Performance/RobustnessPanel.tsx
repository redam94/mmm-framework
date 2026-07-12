import { useState } from 'react';
import { Layers } from 'lucide-react';
import { Button, DataTable, EmptyState, SectionHeader, StatHero } from '../../components/ui';
import type { Column } from '../../components/ui/DataTable';
import { COLORS } from '../../theme/colors';
import { useSpecCurveJob, useStartSpecCurve } from '../../api/hooks/useSpecCurve';
import type { SpecCurveResult } from '../../api/services/specCurveService';

// ── Robustness — spec-curve / model-averaging (issue #118) ─────────────────────
// Refit the model under a pre-registered grid of defensible modelling choices
// (adstock × saturation form), LOO-stack them, and show how much each channel's
// ROI depends on those choices. A robust channel keeps its sign with a contained
// spread; a spec-fragile one flips or swings — don't over-trust its point value.

type Verdict = 'robust' | 'spec-fragile';

function verdictFor(spread: number | null, signStable: boolean): Verdict {
  if (!signStable) return 'spec-fragile';
  return spread != null && spread > 50 ? 'spec-fragile' : 'robust';
}

const VERDICT_STYLE: Record<Verdict, { fg: string; bg: string; label: string }> = {
  robust: { fg: COLORS.sage800, bg: COLORS.sage100, label: 'Robust' },
  'spec-fragile': { fg: COLORS.rust700, bg: COLORS.rust100, label: 'Spec-fragile' },
};

function fmt(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? '—' : v.toFixed(2);
}

function VerdictChip({ verdict }: { verdict: Verdict }) {
  const s = VERDICT_STYLE[verdict];
  return (
    <span
      className="inline-flex rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ backgroundColor: s.bg, color: s.fg }}
    >
      {s.label}
    </span>
  );
}

interface Row {
  channel: string;
  bma: number | null;
  primary: number | null;
  min: number;
  max: number;
  spread: number | null;
  verdict: Verdict;
}

function ResultView({ result }: { result: SpecCurveResult }) {
  const rows: Row[] = result.channels.map((ch) => {
    const r = result.robustness[ch];
    const b = result.bma[ch];
    const verdict = verdictFor(r?.spread_pct ?? null, r?.sign_stable ?? false);
    return {
      channel: ch,
      bma: b?.mean ?? null,
      primary: r?.primary ?? null,
      min: r?.min ?? NaN,
      max: r?.max ?? NaN,
      spread: r?.spread_pct ?? null,
      verdict,
    };
  });
  const fragile = rows.filter((r) => r.verdict === 'spec-fragile').length;

  const columns: Column<Row>[] = [
    { key: 'channel', header: 'Channel', render: (r) => <span className="font-medium text-ink-700">{r.channel}</span> },
    { key: 'bma', header: 'BMA ROI', numeric: true, render: (r) => fmt(r.bma) },
    { key: 'primary', header: 'Primary', numeric: true, render: (r) => fmt(r.primary) },
    { key: 'range', header: 'Range', numeric: true, render: (r) => `${fmt(r.min)}–${fmt(r.max)}` },
    { key: 'spread', header: 'Spread', numeric: true, render: (r) => (r.spread == null ? '—' : `${r.spread.toFixed(0)}%`) },
    { key: 'verdict', header: 'Verdict', render: (r) => <VerdictChip verdict={r.verdict} /> },
  ];

  return (
    <div className="space-y-5">
      <p className="text-sm text-ink-500">
        {result.specs.length} defensible spec(s), model-averaged (BMA) via LOO stacking. Primary:{' '}
        <span className="font-medium text-ink-700">{result.primary ?? '—'}</span>.
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatHero label="Specifications fit" value={String(result.specs.length)} />
        <StatHero
          label="Robust channels"
          value={String(rows.length - fragile)}
          hint={`of ${rows.length}`}
        />
        <StatHero
          label="Spec-fragile"
          value={String(fragile)}
          increaseIsGood={false}
          delta={fragile > 0 ? fragile : null}
          deltaLabel={fragile > 0 ? 'review' : undefined}
          hint="sign flips or swings widely"
        />
      </div>
      {fragile > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-sm"
          style={{ backgroundColor: COLORS.rust100, color: COLORS.rust700 }}
        >
          <strong>{fragile} spec-fragile channel{fragile === 1 ? '' : 's'}</strong> — the ROI flips
          sign or swings widely across defensible specs. Anchor those with an experiment before
          committing budget to the point value.
        </div>
      )}
      <DataTable<Row> columns={columns} rows={rows} rowKey={(r) => r.channel} />
    </div>
  );
}

export function RobustnessPanel({ projectId }: { projectId: string }) {
  const start = useStartSpecCurve(projectId);
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const { data: job } = useSpecCurveJob(projectId, jobId);

  const run = async () => {
    setErr(null);
    try {
      const { job_id } = await start.mutateAsync({ rationale: 'adstock × saturation forms' });
      setJobId(job_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Could not start the spec-curve.');
    }
  };

  const running = start.isPending || job?.status === 'pending' || job?.status === 'running';

  return (
    <div className="space-y-6">
      <SectionHeader
        level={2}
        title="Robustness (spec-curve)"
        subtitle="How much each channel's ROI depends on the modelling choices — refit across a pre-registered spec set and model-averaged."
        actions={
          <Button onClick={run} disabled={running} size="sm">
            {running ? 'Running…' : job?.status === 'done' ? 'Re-run sweep' : 'Run spec-curve'}
          </Button>
        }
      />

      {err && <p className="text-sm text-rust-700">{err}</p>}

      {running ? (
        <div className="rounded-lg border border-line-200 bg-white p-6 text-sm text-ink-500 shadow-sm">
          Fitting {job?.n_specs ?? 'several'} specifications with NUTS — this takes a few minutes.
          The panel updates automatically when the sweep finishes.
        </div>
      ) : job?.status === 'error' ? (
        <p className="text-sm text-rust-700">
          {job.error || 'The spec-curve failed.'}
        </p>
      ) : job?.status === 'done' && job.result ? (
        <ResultView result={job.result} />
      ) : (
        <EmptyState
          icon={Layers}
          title="Run a robustness check"
          description="Refit the model across a pre-registered grid of defensible specifications (adstock × saturation form) and model-average them, to see which channel ROIs hold up and which are spec-fragile. Requires a fitted baseline; the sweep runs in the background."
          action={
            <Button onClick={run} disabled={running}>
              Run spec-curve
            </Button>
          }
        />
      )}
    </div>
  );
}

export default RobustnessPanel;
