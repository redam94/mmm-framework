import { useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Compass, FlaskConical, FolderOpen, Plus } from 'lucide-react';
import { Button, EmptyState, SectionHeader, Tabs } from '../../components/ui';
import { useProjectStore } from '../../stores/projectStore';
import { useExperimentRegistry, useExperimentPriorities } from '../../api/hooks/useMeasurement';
import { PriorityMatrix } from './PriorityMatrix';
import { LifecycleBoard } from './LifecycleBoard';
import { RetestSchedule } from './RetestSchedule';
import { ExperimentDrawer } from './ExperimentDrawer';
import { LogExperimentModal } from './LogExperimentModal';
import { DesignStudio } from './DesignStudio';

type View = 'matrix' | 'board' | 'schedule';

const VIEW_TABS = [
  { id: 'matrix', label: 'Priority matrix' },
  { id: 'board', label: 'Lifecycle board' },
  { id: 'schedule', label: 'Re-test schedule' },
];

export function ExperimentsPage() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const navigate = useNavigate();
  const projectId = useProjectStore((s) => s.currentProjectId);

  const [view, setView] = useState<View>('matrix');
  const [showLog, setShowLog] = useState(false);
  const [showDesigner, setShowDesigner] = useState(false);

  const registry = useExperimentRegistry(projectId);
  const prioritiesQuery = useExperimentPriorities(projectId);

  const experiments = registry.data ?? [];
  const priorities = prioritiesQuery.data ?? null;
  const loading = registry.isLoading || prioritiesQuery.isLoading;

  // Channels the designer can target + the best default (top test_now row)
  const designChannels = useMemo(() => {
    const fromPriorities = (priorities?.channels ?? []).map((c) => c.channel);
    if (fromPriorities.length > 0) return fromPriorities;
    return Array.from(new Set(experiments.map((e) => e.channel)));
  }, [priorities, experiments]);
  const topChannel = useMemo(() => {
    const rows = priorities?.channels ?? [];
    return (
      rows.find((c) => c.quadrant === 'test_now')?.channel ?? rows[0]?.channel ?? null
    );
  }, [priorities]);

  const noProject = !projectId;
  const noBaseline = !!projectId && !loading && priorities === null && experiments.length === 0;

  const prioritiesMissing = (
    <EmptyState
      icon={FlaskConical}
      title="No experiment priorities yet"
      description="Priorities come from a fitted model's ROI uncertainty. Fit a baseline model (T₀) in the Workspace to compute them — the lifecycle board still works from logged experiments alone."
      action={
        <Button variant="secondary" onClick={() => navigate('/workspace')}>
          Open Workspace
        </Button>
      }
    />
  );

  return (
    <div className="space-y-6">
      <SectionHeader
        level={1}
        title="Experiments"
        subtitle="Plan, pre-register, run, and calibrate — prioritized by what learning is worth."
        actions={
          <>
            <Button variant="secondary" onClick={() => setShowLog(true)} disabled={!projectId}>
              <Plus className="h-4 w-4" /> Log experiment
            </Button>
            <Button
              onClick={() => setShowDesigner(true)}
              disabled={!projectId || designChannels.length === 0}
              title="Randomized geo lift, matched-market DiD, or flighting — with power analysis"
            >
              <Compass className="h-4 w-4" /> Design experiment
            </Button>
          </>
        }
      />

      {noProject ? (
        <EmptyState
          icon={FolderOpen}
          title="No project selected"
          description="Pick a project from the switcher in the header to see its experiments and learning priorities."
        />
      ) : loading ? (
        <div className="py-16 text-center text-sm text-ink-400">Loading experiments…</div>
      ) : noBaseline ? (
        <EmptyState
          icon={FlaskConical}
          title="Nothing to prioritize yet"
          description="Fit a baseline model (T₀) in the Workspace to compute experiment priorities, or log an experiment you're already running."
          action={
            <Button onClick={() => navigate('/workspace')}>Go to Workspace</Button>
          }
          secondary={
            <button
              className="text-sage-700 hover:underline"
              onClick={() => setShowLog(true)}
            >
              Log an experiment manually
            </button>
          }
        />
      ) : (
        <>
          <Tabs
            tabs={VIEW_TABS.map((t) =>
              t.id === 'board' ? { ...t, badge: experiments.length } : t,
            )}
            active={view}
            onChange={(id) => setView(id as View)}
          />

          {view === 'matrix' &&
            (priorities ? <PriorityMatrix priorities={priorities} /> : prioritiesMissing)}
          {view === 'board' && (
            <LifecycleBoard
              experiments={experiments}
              onOpen={(id) => navigate(`/experiments/${id}`)}
            />
          )}
          {view === 'schedule' &&
            (priorities ? <RetestSchedule priorities={priorities} /> : prioritiesMissing)}
        </>
      )}

      <ExperimentDrawer
        experimentId={experimentId ?? null}
        onClose={() => navigate('/experiments')}
      />

      {showLog && <LogExperimentModal onClose={() => setShowLog(false)} />}

      <DesignStudio
        open={showDesigner}
        onClose={() => setShowDesigner(false)}
        projectId={projectId}
        channels={designChannels}
        defaultChannel={topChannel}
        priorities={priorities}
      />
    </div>
  );
}
