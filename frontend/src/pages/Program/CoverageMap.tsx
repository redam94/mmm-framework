import { useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { TierBadge } from '../../components/ui';
import { useCalibrationCoverage } from '../../api/hooks/useMeasurement';

function evidenceAge(days: number | null): string {
  if (days == null) return 'never tested';
  if (days < 14) return `${Math.round(days)}d old`;
  if (days < 120) return `${Math.round(days / 7)}wk old`;
  return `${(days / 30.4).toFixed(0)}mo old`;
}

/** Channels × evidence tier: where causal anchors exist, and how fresh. */
export function CoverageMap({ projectId }: { projectId: string | null }) {
  const navigate = useNavigate();
  const { data, isLoading } = useCalibrationCoverage(projectId);
  if (isLoading || !data || data.channels.length === 0) return null;

  return (
    <section className="rounded-lg border border-line-200 bg-white p-5 shadow-sm">
      <div className="flex items-baseline justify-between">
        <h2 className="font-display text-lg font-semibold text-ink-900">Calibration coverage</h2>
        <span className="text-xs text-ink-400">
          <span className="num">{data.spend_weighted_coverage_pct.toFixed(0)}%</span> of spend
          experiment-backed
        </span>
      </div>
      <ul className="mt-4 divide-y divide-line-200">
        {data.channels.map((c) => (
          <li key={c.channel} className="flex items-center gap-3 py-2.5">
            <span className="w-36 truncate text-sm font-medium text-ink-900">{c.channel}</span>
            <TierBadge tier={c.tier} />
            <span className="flex-1 text-xs text-ink-400">
              {c.tier === 'running'
                ? c.in_flight_status === 'completed'
                  ? 'readout in — awaiting calibration'
                  : `test in market${c.in_flight_started ? ` since ${c.in_flight_started}` : ''}`
                : c.tier === 'model_only'
                  ? c.n_experiments > 0
                    ? `${c.n_experiments} experiment(s), none calibrated`
                    : 'no experiments yet'
                  : `evidence ${evidenceAge(c.evidence_age_days)}${c.retest_due ? ' — re-test due' : ''}`}
            </span>
            {c.spend_share != null && (
              <span className="text-xs text-ink-400 num">{(c.spend_share * 100).toFixed(0)}% spend</span>
            )}
          </li>
        ))}
      </ul>
      <button
        onClick={() => navigate('/experiments')}
        className="mt-3 flex items-center gap-1 text-xs font-medium text-sage-800 hover:underline"
      >
        Plan the next experiment <ArrowRight className="h-3 w-3" />
      </button>
    </section>
  );
}
