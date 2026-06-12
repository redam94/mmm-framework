import { useLocation, useNavigate } from 'react-router-dom';
import { FolderOpen, LineChart } from 'lucide-react';
import { Button, EmptyState, SectionHeader, Tabs } from '../../components/ui';
import { useProjectStore } from '../../stores/projectStore';
import { useProjectHistory } from '../../api/hooks/useMeasurement';
import { TrajectoryPanels } from './TrajectoryPanels';
import { AgreementLog } from './AgreementLog';
import { RunsTimeline } from './RunsTimeline';

type TabId = 'trajectories' | 'agreement' | 'runs';

const TABS = [
  { id: 'trajectories', label: 'Trajectories' },
  { id: 'agreement', label: 'Agreement' },
  { id: 'runs', label: 'Runs' },
];

function tabFromPath(pathname: string): TabId {
  if (pathname.endsWith('/agreement')) return 'agreement';
  if (pathname.endsWith('/runs')) return 'runs';
  return 'trajectories';
}

export function PerformancePage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { currentProjectId } = useProjectStore();
  const active = tabFromPath(location.pathname);

  const { data: history, isLoading } = useProjectHistory(currentProjectId);

  const onTabChange = (id: string) => {
    navigate(id === 'trajectories' ? '/performance' : `/performance/${id}`);
  };

  const noProject = !currentProjectId;
  const noHistory = !noProject && !isLoading && (history?.runs.length ?? 0) === 0;

  return (
    <div className="space-y-6">
      <SectionHeader
        level={1}
        title="Performance"
        subtitle="How measurement quality and media decisions improved cycle over cycle."
      />

      <Tabs tabs={TABS} active={active} onChange={onTabChange} />

      {noProject ? (
        <EmptyState
          icon={FolderOpen}
          title="Pick a project"
          description="Performance trajectories are scoped to a project — choose one from the header to see how measurement quality evolved."
        />
      ) : active === 'runs' ? (
        // The runs timeline handles its own loading and empty states.
        <RunsTimeline />
      ) : isLoading ? (
        <p className="text-sm text-ink-400">Loading history…</p>
      ) : noHistory || !history ? (
        <EmptyState
          icon={LineChart}
          title="No fitted runs yet — fit a baseline in the Workspace"
          description="Once a model is fitted, each cycle's ROI estimates, budget shares, and portfolio metrics are tracked here."
          action={<Button onClick={() => navigate('/workspace')}>Go to Workspace</Button>}
        />
      ) : active === 'trajectories' ? (
        <TrajectoryPanels history={history} />
      ) : (
        <AgreementLog projectId={currentProjectId} />
      )}
    </div>
  );
}

export default PerformancePage;
