import { ClipboardCheck } from 'lucide-react';
import { DataTable, EmptyState, SectionHeader, StatHero } from '../../components/ui';
import type { Column } from '../../components/ui/DataTable';
import { COLORS } from '../../theme/colors';
import { useProjectScorecard } from '../../api/hooks/useScorecard';
import type { ScorecardRow } from '../../api/services/scorecardService';

// ── Recommendation scorecard — predicted vs realized (issue #109) ──────────────
// Each channel's realized experiment readout next to the ROI the model predicted
// for it, whether the realized value landed inside the predicted interval, and
// the model's calibration over time. Nothing builds trust faster than watching a
// model's past calls come true — or honestly own the misses.

function fmt(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? '—' : v.toFixed(2);
}

function HitChip({ hit }: { hit: boolean | null }) {
  if (hit == null) {
    return (
      <span
        className="inline-flex rounded-full px-2 py-0.5 text-xs font-medium"
        style={{ backgroundColor: COLORS.cream200, color: COLORS.ink400 }}
      >
        No prediction
      </span>
    );
  }
  const s = hit
    ? { fg: COLORS.sage800, bg: COLORS.sage100, label: 'In interval' }
    : { fg: COLORS.rust700, bg: COLORS.rust100, label: 'Missed' };
  return (
    <span
      className="inline-flex rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ backgroundColor: s.bg, color: s.fg }}
    >
      {s.label}
    </span>
  );
}

export function ScorecardPanel({ projectId }: { projectId: string }) {
  const { data, isLoading, isError } = useProjectScorecard(projectId);

  if (isLoading) return <p className="text-sm text-ink-400">Loading scorecard…</p>;
  if (isError || !data) return <p className="text-sm text-rust-700">Failed to load the scorecard.</p>;

  if (data.n_recommendations === 0) {
    return (
      <div className="space-y-6">
        <SectionHeader
          level={2}
          title="Recommendation scorecard"
          subtitle="Predicted vs realized — how the model's past calls held up against experiments."
        />
        <EmptyState
          icon={ClipboardCheck}
          title="No realized outcomes yet"
          description="Once an experiment reads out (calibrated or completed), its measured return is scored against what the model predicted for that channel. Run and record an experiment to start the accountability log."
        />
      </div>
    );
  }

  const cov = data.calibration.coverage;
  const columns: Column<ScorecardRow>[] = [
    { key: 'channel', header: 'Channel', render: (r) => <span className="font-medium text-ink-700">{r.channel}</span> },
    {
      key: 'predicted',
      header: 'Predicted ROI',
      numeric: true,
      render: (r) =>
        r.predicted == null ? (
          '—'
        ) : (
          <span>
            {fmt(r.predicted)}
            {r.predicted_lower != null && r.predicted_upper != null && (
              <span className="ml-1 text-xs text-ink-400">
                [{fmt(r.predicted_lower)}, {fmt(r.predicted_upper)}]
              </span>
            )}
          </span>
        ),
    },
    { key: 'realized', header: 'Realized', numeric: true, render: (r) => fmt(r.realized) },
    {
      key: 'error',
      header: 'Error',
      numeric: true,
      render: (r) => (r.error == null ? '—' : `${r.error >= 0 ? '+' : ''}${fmt(r.error)}`),
    },
    { key: 'hit', header: 'Calibration', render: (r) => <HitChip hit={r.in_interval} /> },
    { key: 'date', header: 'Read out', render: (r) => <span className="text-xs text-ink-400">{r.end_date ?? '—'}</span> },
  ];

  return (
    <div className="space-y-6">
      <SectionHeader
        level={2}
        title="Recommendation scorecard"
        subtitle="Predicted vs realized — how the model's past calls held up against experiments."
      />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatHero label="Recommendations scored" value={String(data.n_recommendations)} />
        <StatHero
          label="Interval calibration"
          value={cov == null ? '—' : `${(cov * 100).toFixed(0)}%`}
          hint={`${data.calibration.hits}/${data.calibration.n_with_interval} realized in interval`}
          increaseIsGood
        />
        <StatHero
          label="Missed intervals"
          value={String(data.calibration.n_with_interval - data.calibration.hits)}
          increaseIsGood={false}
          hint="realized outside the predicted range"
        />
      </div>
      {cov != null && data.calibration.n_with_interval >= 3 && (
        <div
          className="rounded-lg px-4 py-3 text-sm"
          style={
            cov >= 0.7
              ? { backgroundColor: COLORS.sage100, color: COLORS.sage800 }
              : { backgroundColor: COLORS.rust100, color: COLORS.rust700 }
          }
        >
          {cov >= 0.7
            ? `Well-calibrated: ${(cov * 100).toFixed(0)}% of realized returns landed inside the model's predicted interval.`
            : `Under-calibrated: only ${(cov * 100).toFixed(0)}% of realized returns fell inside the predicted interval — the model's intervals may be too tight. Widen priors or fold in more experiments.`}
        </div>
      )}
      <DataTable<ScorecardRow>
        columns={columns}
        rows={data.rows}
        rowKey={(r) => r.experiment_id ?? `${r.channel}-${r.end_date}`}
      />
    </div>
  );
}

export default ScorecardPanel;
