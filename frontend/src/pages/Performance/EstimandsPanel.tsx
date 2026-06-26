import { useMemo, useState } from 'react';
import { LineChart } from 'lucide-react';
import { Card, DataTable, EmptyState, SectionHeader } from '../../components/ui';
import type { Column } from '../../components/ui/DataTable';
import { COLORS } from '../../theme/colors';
import { useProjectEstimands } from '../../api/hooks/useEstimands';
import type {
  Evidence,
  EstimandCell,
  EstimandGroup,
  EstimandModel,
  EstimandRunSummary,
  ProjectEstimands,
} from '../../api/services/estimandsService';

// ── Estimands — the declarative causal lens, grouped for comparison ────────────
// Estimands (contribution ROI, marginal ROAS, incremental contribution, …) are
// clustered by (metric × KPI): models measuring the SAME metric on the SAME KPI
// sit side by side; a model on a different KPI is shown separately because the
// numbers are not directly comparable. Two ROI flavors (contribution vs
// counterfactual) stay distinct — same kind, different number.

const EVIDENCE_STYLE: Record<Evidence, { fg: string; bg: string; label: string }> = {
  strong: { fg: COLORS.sage800, bg: COLORS.sage100, label: 'Strong' },
  below: { fg: COLORS.rust700, bg: COLORS.rust100, label: 'Below ref' },
  uncertain: { fg: COLORS.steel700, bg: COLORS.steel100, label: 'Uncertain' },
  na: { fg: COLORS.ink400, bg: COLORS.cream200, label: 'N/A' },
};

function prettyKind(kind: string): string {
  if (!kind || kind === 'mmm') return 'MMM';
  return kind.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function shortDate(ts: number | null): string {
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: '2-digit',
  });
}

function modelHeader(m: EstimandModel): string {
  const d = shortDate(m.created_at);
  return d ? `${prettyKind(m.model_kind)} · ${d}` : prettyKind(m.model_kind);
}

function fmtVal(group: EstimandGroup, v: number | null): string {
  if (v == null || !Number.isFinite(v)) return '—';
  if (group.is_ratio) return v.toFixed(2);
  const a = Math.abs(v);
  if (a >= 100) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(2);
}

function modelHasData(m: EstimandModel): boolean {
  return m.rows.some((r) => r.status === 'ok' && r.mean != null);
}

function EvidenceDot({ evidence }: { evidence: Evidence }) {
  const s = EVIDENCE_STYLE[evidence];
  return (
    <span
      title={s.label}
      aria-label={s.label}
      className="inline-block h-2 w-2 shrink-0 rounded-full"
      style={{ backgroundColor: s.fg }}
    />
  );
}

function ValueCell({ cell, group }: { cell: EstimandCell | undefined; group: EstimandGroup }) {
  if (!cell || cell.status !== 'ok' || cell.mean == null) {
    return <span className="text-ink-300">—</span>;
  }
  const ci =
    cell.lower != null && cell.upper != null
      ? `[${fmtVal(group, cell.lower)}, ${fmtVal(group, cell.upper)}]`
      : '';
  const tips: string[] = [];
  if (cell.prob_positive != null) tips.push(`P(>0) ${(cell.prob_positive * 100).toFixed(0)}%`);
  if (cell.prob_profitable != null)
    tips.push(`P(profitable) ${(cell.prob_profitable * 100).toFixed(0)}%`);
  return (
    <div className="flex flex-col items-end gap-0.5" title={tips.join(' · ') || undefined}>
      <div className="flex items-center gap-1.5">
        <EvidenceDot evidence={cell.evidence} />
        <span className="num font-medium text-ink-900">{fmtVal(group, cell.mean)}</span>
      </div>
      {ci && <span className="num text-xs text-ink-400">{ci}</span>}
    </div>
  );
}

/** One comparability cluster rendered as a channels × models pivot. */
function GroupCard({ group, models }: { group: EstimandGroup; models: EstimandModel[] }) {
  const shown = models.filter(modelHasData);
  if (shown.length === 0) return null;

  // channels present in at least one shown model, in the group's union order
  const channels = group.channels.filter((ch) =>
    shown.some((m) => m.rows.some((r) => r.channel === ch)),
  );

  type Row = { channel: string; cells: Record<string, EstimandCell | undefined> };
  const rows: Row[] = channels.map((ch) => ({
    channel: ch,
    cells: Object.fromEntries(
      shown.map((m) => [m.run_id, m.rows.find((r) => r.channel === ch)]),
    ),
  }));

  const columns: Column<Row>[] = [
    {
      key: 'channel',
      header: 'Channel',
      render: (r) =>
        r.channel === '—' ? (
          <span className="italic text-ink-400">overall</span>
        ) : (
          <span className="font-medium text-ink-700">{r.channel}</span>
        ),
    },
    ...shown.map(
      (m): Column<Row> => ({
        key: m.run_id,
        header: modelHeader(m),
        numeric: true,
        render: (r) => <ValueCell cell={r.cells[m.run_id]} group={group} />,
      }),
    ),
  ];

  const comparable = shown.length > 1;
  const refHint = group.is_ratio ? 'vs 1.0 (break-even)' : 'vs 0 (no effect)';

  return (
    <Card padding="md" className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <h3 className="font-display text-lg font-semibold tracking-tight text-ink-900">
            {group.label}
          </h3>
          {group.units && <span className="text-xs text-ink-400">{group.units}</span>}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-ink-400">evidence {refHint}</span>
          <span
            className="rounded-full px-2 py-0.5 text-xs font-medium"
            style={
              comparable
                ? { backgroundColor: COLORS.sage100, color: COLORS.sage800 }
                : { backgroundColor: COLORS.cream200, color: COLORS.ink600 }
            }
          >
            {comparable ? `Comparable · ${shown.length} models` : 'Single model'}
          </span>
        </div>
      </div>
      <DataTable<Row> columns={columns} rows={rows} rowKey={(r) => r.channel} />
    </Card>
  );
}

function ModelChip({
  run,
  selected,
  onToggle,
}: {
  run: EstimandRunSummary;
  selected: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={selected}
      className="rounded-full border px-3 py-1 text-xs transition-colors"
      style={
        selected
          ? { backgroundColor: COLORS.sage100, color: COLORS.sage800, borderColor: COLORS.sage300 }
          : { backgroundColor: '#ffffff', color: COLORS.ink600, borderColor: COLORS.line300 }
      }
    >
      <span className="font-medium">{prettyKind(run.model_kind)}</span>
      <span className="text-ink-400"> · {run.kpi || 'KPI?'}</span>
      {run.created_at && <span className="text-ink-400"> · {shortDate(run.created_at)}</span>}
    </button>
  );
}

function kpiLabel(kpi: string): string {
  return kpi || 'Unspecified KPI';
}

export function EstimandsPanel({ projectId }: { projectId: string }) {
  const { data, isLoading, isError } = useProjectEstimands(projectId);

  // Default selection: the latest run per distinct model. Once the user toggles,
  // `selected` holds their explicit choice.
  const [selected, setSelected] = useState<Set<string> | null>(null);
  const defaultSelected = useMemo(
    () => new Set((data?.runs ?? []).filter((r) => r.is_latest_for_model).map((r) => r.run_id)),
    [data],
  );
  const active = selected ?? defaultSelected;

  const toggle = (runId: string) => {
    const next = new Set(active);
    if (next.has(runId)) next.delete(runId);
    else next.add(runId);
    setSelected(next);
  };

  const groupsByKpi = useMemo(() => {
    const out = new Map<string, EstimandGroup[]>();
    for (const g of data?.groups ?? []) {
      const arr = out.get(g.kpi) ?? [];
      arr.push(g);
      out.set(g.kpi, arr);
    }
    return out;
  }, [data]);

  if (isLoading) return <p className="text-sm text-ink-400">Loading estimands…</p>;
  if (isError) return <p className="text-sm text-rust-700">Failed to load estimands.</p>;

  const payload = data as ProjectEstimands | undefined;
  if (!payload || payload.runs.length === 0) {
    return (
      <EmptyState
        icon={LineChart}
        title="No estimands yet"
        description="Estimands are captured when a model is fitted. Fit a model in the Workspace, or backfill existing models with: python -m mmm_framework.api.backfill --what estimands"
      />
    );
  }

  const kpis = payload.kpis.length ? payload.kpis : Array.from(groupsByKpi.keys());
  const anyVisible = (payload.groups ?? []).some((g) =>
    g.models.some((m) => active.has(m.run_id) && modelHasData(m)),
  );

  return (
    <div className="space-y-6">
      <Card tone="cream" padding="md" className="space-y-3">
        <div>
          <p className="text-sm text-ink-700">
            Estimands are grouped by <span className="font-medium">metric × KPI</span>. Models
            measuring the same metric on the same KPI are shown together; metrics on different
            KPIs aren&apos;t directly comparable and stay in separate sections.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-ink-400">
            Models
          </span>
          {payload.runs.map((run) => (
            <ModelChip
              key={run.run_id}
              run={run}
              selected={active.has(run.run_id)}
              onToggle={() => toggle(run.run_id)}
            />
          ))}
          <button
            type="button"
            onClick={() => setSelected(new Set(payload.runs.map((r) => r.run_id)))}
            className="ml-1 text-xs text-sage-700 underline-offset-2 hover:underline"
          >
            Select all
          </button>
          <button
            type="button"
            onClick={() => setSelected(new Set(defaultSelected))}
            className="text-xs text-ink-400 underline-offset-2 hover:underline"
          >
            Latest only
          </button>
        </div>
      </Card>

      {!anyVisible ? (
        <EmptyState
          icon={LineChart}
          title="No models selected"
          description="Pick at least one model above to see its estimands."
        />
      ) : (
        kpis.map((kpi) => {
          const groups = groupsByKpi.get(kpi) ?? [];
          const visible = groups.filter((g) =>
            g.models.some((m) => active.has(m.run_id) && modelHasData(m)),
          );
          if (visible.length === 0) return null;
          return (
            <section key={kpi || '__none__'} className="space-y-3">
              <SectionHeader
                level={2}
                title={kpiLabel(kpi)}
                subtitle={`${visible.length} metric${visible.length === 1 ? '' : 's'} on this KPI`}
              />
              {visible.map((g) => (
                <GroupCard
                  key={g.key}
                  group={g}
                  models={g.models.filter((m) => active.has(m.run_id))}
                />
              ))}
            </section>
          );
        })
      )}
    </div>
  );
}

export default EstimandsPanel;
