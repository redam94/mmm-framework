import { useState } from 'react';
import { Scale } from 'lucide-react';
import { Button, DataTable, EmptyState, SectionHeader, StatHero } from '../../components/ui';
import type { Column } from '../../components/ui/DataTable';
import { COLORS } from '../../theme/colors';
import { apiErrorMessage } from '../../api/client';
import { useVarianceBridge } from '../../api/hooks/useVariance';
import type { BridgeProvenance, BridgeRow, VarianceBridge } from '../../api/services/varianceService';

// ── Variance to plan (#227) — committed forecast vs realized KPI ─────────────
// Two buckets on the COMMITTED posterior: per-channel delivery variance (a
// paired counterfactual) plus a LABELLED unexplained remainder; the rows sum
// to actual − committed exactly. The interval verdict LEADS: a realized total
// inside the committed band is within the committed uncertainty, not a gap
// owed a story. Needs a committed plan of record + uploaded delivery +
// uploaded realized-KPI actuals; a bridge that cannot be built refuses at
// POST time with the stated reason (409).

const PROV_STYLE: Record<string, { fg: string; bg: string; label: string }> = {
  modelled: { fg: COLORS.sage800, bg: COLORS.sage100, label: 'Modelled' },
  supplied: { fg: COLORS.steel700, bg: COLORS.steel100, label: 'Supplied' },
  residual: { fg: COLORS.ink400, bg: COLORS.cream200, label: 'Residual' },
};

function num(v: number | null | undefined, signed = false): string {
  if (v == null || !Number.isFinite(v)) return '—';
  const s = v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return signed && v >= 0 ? `+${s}` : s;
}

function ProvChip({ prov }: { prov: BridgeProvenance }) {
  const s = PROV_STYLE[prov] ?? PROV_STYLE.residual;
  return (
    <span
      className="inline-flex rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ backgroundColor: s.bg, color: s.fg }}
    >
      {s.label}
    </span>
  );
}

function VerdictCallout({ bridge }: { bridge: VarianceBridge }) {
  const within = bridge.within_committed_interval;
  const pct =
    bridge.interval_mass != null ? `${Math.round(bridge.interval_mass * 100)}%` : 'committed';
  if (within === true) {
    return (
      <div
        className="rounded-lg px-4 py-3 text-sm"
        style={{ backgroundColor: COLORS.sage100, color: COLORS.sage800 }}
      >
        <span className="font-semibold">Within the committed interval.</span> The realized total
        sits inside the {pct} band that was committed to — this bridge explains composition, not a
        surprise.
      </div>
    );
  }
  if (within === false) {
    return (
      <div
        className="rounded-lg px-4 py-3 text-sm"
        style={{ backgroundColor: COLORS.rust100, color: COLORS.rust700 }}
      >
        <span className="font-semibold">Outside the committed interval.</span> The miss exceeds the{' '}
        {pct} uncertainty that was committed to.
      </div>
    );
  }
  return (
    <div
      className="rounded-lg px-4 py-3 text-sm"
      style={{ backgroundColor: COLORS.cream200, color: COLORS.ink400 }}
    >
      The within-interval verdict is unavailable for this commitment — unavailable, not passed.
    </div>
  );
}

function BridgeView({ bridge }: { bridge: VarianceBridge }) {
  const showDollars = bridge.value_per_kpi != null && !bridge.dollar_headline_suppressed;
  const columns: Column<BridgeRow>[] = [
    {
      key: 'name',
      header: 'Line',
      render: (r) => <span className="font-medium text-ink-700">{r.name}</span>,
    },
    { key: 'kpi', header: 'KPI units', numeric: true, render: (r) => num(r.value, true) },
    ...(showDollars
      ? [
          {
            key: 'dollars',
            header: 'Dollars',
            numeric: true,
            render: (r: BridgeRow) => num(r.value * (bridge.value_per_kpi as number), true),
          } as Column<BridgeRow>,
        ]
      : []),
    { key: 'prov', header: 'Provenance', render: (r) => <ProvChip prov={r.provenance} /> },
    {
      key: 'note',
      header: 'Note',
      render: (r) => (
        <span className="text-xs text-ink-400">{r.source_note || r.note || r.basis || ''}</span>
      ),
    },
  ];

  const band =
    bridge.committed_lower != null
      ? `${Math.round((bridge.interval_mass ?? 0.9) * 100)}% band ${num(bridge.committed_lower)} – ${num(bridge.committed_upper)}`
      : undefined;

  return (
    <div className="space-y-5">
      <VerdictCallout bridge={bridge} />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatHero
          label={`Realized KPI (${bridge.period_set.length} periods)`}
          value={num(bridge.actual_kpi)}
        />
        <StatHero label="Committed forecast" value={num(bridge.committed_kpi)} hint={band} />
        <StatHero
          label="Gap the bridge closes exactly"
          value={num(bridge.gap, true)}
          hint={`Delivery ${num(bridge.delivery_total, true)} · Unexplained ${num(bridge.unexplained, true)}`}
        />
      </div>

      <DataTable<BridgeRow> columns={columns} rows={bridge.rows} rowKey={(r) => r.name} />

      {bridge.refusals.length > 0 && (
        <div
          className="rounded-lg px-4 py-3 text-sm"
          style={{ backgroundColor: COLORS.rust100, color: COLORS.rust700 }}
        >
          {bridge.refusals.map((r) => (
            <p key={r}>{r}</p>
          ))}
        </div>
      )}
      {bridge.caveats.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-xs text-ink-400">
          {bridge.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function VarianceBridgePanel({ projectId }: { projectId: string }) {
  const { start, job, reset } = useVarianceBridge(projectId);
  const [refusal, setRefusal] = useState<string | null>(null);

  const run = async () => {
    setRefusal(null);
    reset();
    try {
      await start.mutateAsync([]);
    } catch (e) {
      // A 409 is a stated refusal (no committed plan, missing delivery or
      // actuals, changed dataset) — show it verbatim, it names the fix.
      setRefusal(apiErrorMessage(e, 'The bridge could not be started.'));
    }
  };

  const bridge = job.data?.status === 'done' ? job.data.result : null;
  const pending =
    start.isPending || (!!job.data && ['pending', 'running'].includes(job.data.status));

  return (
    <div className="space-y-6">
      <SectionHeader
        level={2}
        title="Variance to plan"
        subtitle="The committed forecast vs realized KPI — delivery-driven vs unexplained, and the bridge closes exactly."
        actions={
          <Button variant="secondary" size="sm" onClick={run} disabled={pending}>
            <Scale className="mr-1.5 h-3.5 w-3.5" />
            {pending ? 'Building…' : bridge ? 'Rebuild bridge' : 'Build bridge'}
          </Button>
        }
      />

      {refusal && (
        <div
          className="rounded-lg px-4 py-3 text-sm"
          style={{ backgroundColor: COLORS.rust100, color: COLORS.rust700 }}
        >
          {refusal}
        </div>
      )}
      {job.data?.status === 'error' && (
        <div
          className="rounded-lg px-4 py-3 text-sm"
          style={{ backgroundColor: COLORS.rust100, color: COLORS.rust700 }}
        >
          {job.data.error}
        </div>
      )}

      {bridge ? (
        <BridgeView bridge={bridge} />
      ) : (
        !refusal &&
        job.data?.status !== 'error' && (
          <EmptyState
            icon={Scale}
            title={pending ? 'Building the bridge…' : 'No bridge built yet'}
            description={
              pending
                ? 'Re-running the committed forecast under actual spend on the committed posterior.'
                : 'Needs a committed plan of record, uploaded delivery covering its window, and uploaded realized-KPI actuals. A bridge that cannot be built refuses with the reason.'
            }
          />
        )
      )}
    </div>
  );
}
