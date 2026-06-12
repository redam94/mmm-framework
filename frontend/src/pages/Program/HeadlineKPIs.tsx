import { StatHero } from '../../components/ui';
import { useCalibrationCoverage, useProjectHistory } from '../../api/hooks/useMeasurement';

function pctDelta(first: number | null | undefined, last: number | null | undefined): number | null {
  if (first == null || last == null || !Number.isFinite(first) || Math.abs(first) < 1e-12) return null;
  return (100 * (last - first)) / Math.abs(first);
}

/** The four numbers the measurement program is trying to move. */
export function HeadlineKPIs({ projectId }: { projectId: string | null }) {
  const { data: history } = useProjectHistory(projectId);
  const { data: coverage } = useCalibrationCoverage(projectId);

  const portfolio = history?.portfolio ?? [];
  const first = portfolio[0];
  const last = portfolio[portfolio.length - 1];
  if (!last) return null;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <StatHero
        label="Portfolio marginal ROI"
        value={last.marginal_roi != null ? last.marginal_roi.toFixed(2) : '—'}
        delta={pctDelta(first?.marginal_roi, last.marginal_roi)}
        increaseIsGood
        hint="return on the next dollar"
      />
      <StatHero
        label="Misallocation proxy"
        value={
          last.expected_uplift != null
            ? Math.round(last.expected_uplift).toLocaleString()
            : '—'
        }
        delta={pctDelta(first?.expected_uplift, last.expected_uplift)}
        increaseIsGood={false}
        hint="KPI left vs optimal allocation"
      />
      <StatHero
        label="Spend experiment-backed"
        value={coverage ? coverage.spend_weighted_coverage_pct.toFixed(0) : '—'}
        unit="%"
        hint={coverage ? `${coverage.channels.filter((c) => c.tier !== 'model_only').length}/${coverage.channels.length} channels` : undefined}
      />
      <StatHero
        label="Mean ROI uncertainty"
        value={last.mean_ci_width != null ? last.mean_ci_width.toFixed(2) : '—'}
        delta={pctDelta(first?.mean_ci_width, last.mean_ci_width)}
        increaseIsGood={false}
        hint="avg 90% CI width — contracts as you calibrate"
      />
    </div>
  );
}
