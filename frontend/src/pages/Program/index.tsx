import { useNavigate } from 'react-router-dom';
import { Compass } from 'lucide-react';
import { Button, EmptyState, SectionHeader } from '../../components/ui';
import { useProjectStore } from '../../stores/projectStore';
import { usePortfolio } from '../../api/hooks/usePortfolio';
import {
  useExperimentPriorities,
  useExperimentRegistry,
} from '../../api/hooks/useMeasurement';
import { CycleStageRing, type CycleStage } from './CycleStageRing';
import { HeadlineKPIs } from './HeadlineKPIs';
import { NextBestActions } from './NextBestActions';
import { CoverageMap } from './CoverageMap';
import { RecentActivity } from './RecentActivity';

/**
 * Infer where the program sits in the T₀–T₅ loop from observable state.
 * Priority order mirrors the loop's own urgency: readouts waiting to be
 * calibrated beat everything; live experiments beat planning; decayed
 * evidence sends the program back to re-evaluation.
 */
function inferStage(
  hasRuns: boolean,
  experiments: { status: string }[],
  retestDue: boolean,
  prioritiesStale: boolean,
): { stage: CycleStage; reason: string } {
  if (!hasRuns) return { stage: 0, reason: 'No baseline fit yet' };
  const by = (s: string) => experiments.filter((e) => e.status === s).length;
  if (by('completed') > 0)
    return { stage: 3, reason: `${by('completed')} readout(s) waiting for calibration` };
  if (by('running') > 0) return { stage: 2, reason: `${by('running')} experiment(s) in flight` };
  if (by('planned') > 0 || by('draft') > 0)
    return { stage: 2, reason: 'Experiments planned — launch when ready' };
  if (retestDue || prioritiesStale)
    return { stage: 5, reason: retestDue ? 'Evidence decayed — re-evaluate priorities' : 'Priorities stale — refit to refresh' };
  if (by('calibrated') > 0) return { stage: 4, reason: 'Calibrated posterior ready for allocation' };
  return { stage: 1, reason: 'Baseline fitted — prioritize the first experiments' };
}

export function ProgramPage() {
  const navigate = useNavigate();
  const { currentProjectId } = useProjectStore();
  const { data: portfolio } = usePortfolio(currentProjectId);
  const { data: priorities } = useExperimentPriorities(currentProjectId);
  const { data: experiments = [] } = useExperimentRegistry(currentProjectId);

  if (!currentProjectId) {
    return (
      <div className="mx-auto max-w-3xl py-16">
        <EmptyState
          icon={Compass}
          title="Pick a project to see its measurement program"
          description="Each project tracks its own cycle: baseline fits, experiment portfolio, calibrations, and the performance trajectory they produce. Use the project switcher in the header."
        />
      </div>
    );
  }

  const hasRuns = (portfolio?.model_runs?.length ?? 0) > 0;
  const retestDue = (priorities?.channels ?? []).some((c) => c.retest_due);
  const { stage, reason } = inferStage(hasRuns, experiments, retestDue, !!priorities?.stale);

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <SectionHeader
        level={1}
        title="Measurement program"
        subtitle="Fit → prioritize → experiment → calibrate → allocate → re-evaluate. Each cycle contracts the uncertainty that matters for the budget."
      />

      <CycleStageRing current={stage} reason={reason} />

      {!hasRuns ? (
        <EmptyState
          icon={Compass}
          title="Start at T₀ — fit a baseline model"
          description="Upload data and fit the first MMM in the Workspace. Run metrics, experiment priorities, and the performance trajectory all flow from that first fit."
          action={<Button onClick={() => navigate('/workspace')}>Open the workspace</Button>}
        />
      ) : (
        <>
          <HeadlineKPIs projectId={currentProjectId} />
          <NextBestActions projectId={currentProjectId} />
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <CoverageMap projectId={currentProjectId} />
            <RecentActivity projectId={currentProjectId} />
          </div>
        </>
      )}
    </div>
  );
}

export default ProgramPage;
