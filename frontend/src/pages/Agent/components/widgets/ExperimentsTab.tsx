import { useNavigate } from 'react-router-dom';
import { ExternalLink, TestTubes } from 'lucide-react';
import { useExperimentRegistry } from '../../../../api/hooks/useMeasurement';
import { StatusChip } from '../../../../components/ui';
import { EmptyTabState } from '../common/EmptyTabState';

/**
 * Read-mostly registry view inside the workspace: the session's experiment
 * portfolio at a glance, with quick-actions that hand the actual work to the
 * agent (plan / readout / calibrate are chat-driven in here — the full
 * lifecycle UI lives at /experiments).
 */
export function ExperimentsTab({
  projectId,
  onQuickAction,
  disabled,
}: {
  projectId: string | null;
  onQuickAction: (msg: string) => void;
  disabled: boolean;
}) {
  const navigate = useNavigate();
  const { data: experiments = [], isLoading } = useExperimentRegistry(projectId);

  const active = experiments.filter((e) => !['abandoned', 'cancelled'].includes(e.status));
  const completed = active.filter((e) => e.status === 'completed');

  const quick = (msg: string) => !disabled && onQuickAction(msg);

  if (!isLoading && active.length === 0) {
    return (
      <div className="space-y-4">
        <EmptyTabState
          icon={<TestTubes size={28} />}
          title="No experiments in this project yet"
          hint="Ask the agent to compute experiment priorities after a fit — it will rank channels by what learning is worth (EIG × EVOI) and draft the designs."
        />
        <div className="flex flex-wrap justify-center gap-2">
          <button
            onClick={() => quick('Compute the EIG/EVOI experiment priorities for the fitted model and recommend which experiments to run next.')}
            disabled={disabled}
            className="rounded-lg border border-line-300 bg-white px-3 py-1.5 text-sm font-medium text-sage-800 shadow-sm transition-colors hover:bg-sage-100 disabled:opacity-50"
          >
            Compute experiment priorities
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-line-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-line-200 px-5 py-4">
        <div className="flex items-center gap-2">
          <TestTubes size={16} className="text-sage-700" />
          <h3 className="text-sm font-semibold text-ink-900">Experiment registry</h3>
        </div>
        <button
          onClick={() => navigate('/experiments')}
          className="flex items-center gap-1 text-xs font-medium text-sage-800 hover:underline"
        >
          Open full view <ExternalLink size={12} />
        </button>
      </div>

      <ul className="divide-y divide-line-200">
        {active.map((e) => (
          <li key={e.id} className="flex items-center gap-3 px-5 py-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-ink-900">{e.channel}</span>
                <StatusChip status={e.status} />
              </div>
              <p className="mt-0.5 truncate text-xs text-ink-400">
                {e.design_type ?? 'design tbd'}
                {e.start_date && (
                  <span className="num"> · {e.start_date} → {e.end_date ?? '…'}</span>
                )}
                {e.value != null && e.se != null && (
                  <span className="num"> · {e.value.toFixed(2)} ± {e.se.toFixed(2)} {e.estimand ?? ''}</span>
                )}
              </p>
            </div>
            {e.status === 'running' && (
              <button
                onClick={() => quick(`The ${e.channel} experiment (id ${e.id}) has finished — help me record its readout with record_experiment_readout.`)}
                disabled={disabled}
                className="shrink-0 text-xs font-medium text-sage-800 hover:underline disabled:opacity-50"
              >
                Record readout
              </button>
            )}
          </li>
        ))}
      </ul>

      {completed.length > 0 && (
        <div className="flex items-center justify-between gap-3 border-t border-gold-300 bg-gold-100/50 px-5 py-3">
          <p className="text-xs text-gold-700">
            {completed.length} readout(s) measured but not yet in the model.
          </p>
          <button
            onClick={() => quick('Apply the completed experiment readouts as calibration likelihoods (apply_experiment_calibration) and refit the model.')}
            disabled={disabled}
            className="shrink-0 rounded-md bg-sage-700 px-2.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-sage-800 disabled:opacity-50"
          >
            Calibrate &amp; refit
          </button>
        </div>
      )}
    </div>
  );
}
