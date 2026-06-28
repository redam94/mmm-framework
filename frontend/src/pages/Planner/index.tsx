import { useMemo, useState } from 'react';
import { Telescope } from 'lucide-react';
import { Card, EmptyState, SectionHeader, Tabs } from '../../components/ui';
import { ReportViewer } from '../../components/common';
import { useProjectStore } from '../../stores/projectStore';
import { useExperimentPriorities } from '../../api/hooks/useMeasurement';
import { useBudgetPlans } from '../../api/hooks/useBudgetPlans';
import { PlannerStudio } from './PlannerStudio';
import { ScenarioStudio } from './ScenarioStudio';
import { SavedPlans } from './SavedPlans';

/**
 * Almanac — the planner desk. Build a budget plan (optimal allocation, optional
 * per-geo + forward flighting calendar), run a what-if, save/compare/reload
 * plans, export an executable CSV flight plan, and view the per-session report.
 * (B1, B4, B5, B6, U4.)
 */
export function PlannerPage() {
  const currentProjectId = useProjectStore((s) => s.currentProjectId);
  const { data: priorities } = useExperimentPriorities(currentProjectId);
  const plansQuery = useBudgetPlans(
    currentProjectId ? { project_id: currentProjectId } : undefined,
  );
  const [tab, setTab] = useState('optimize');
  const [activeThread] = useState<string | null>(
    () => localStorage.getItem('mmm.activeThreadId'),
  );

  const channels = useMemo(
    () => (priorities?.channels ?? []).map((c) => c.channel),
    [priorities],
  );
  const modelId = priorities?.run_id ?? null;
  const plans = plansQuery.data?.plans ?? [];

  if (!currentProjectId) {
    return (
      <div className="mx-auto max-w-3xl py-16">
        <EmptyState
          icon={Telescope}
          title="Pick a project to start planning"
          description="Choose a project from the switcher in the header to build and save budget plans."
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <SectionHeader
        level={1}
        title="Almanac"
        subtitle="Allocate budget · plan flights · run what-ifs"
      />

      {!priorities && (
        <p className="rounded-lg border border-gold-300 bg-gold-50/60 px-3 py-2 text-sm text-gold-800">
          Fit a baseline model (T₀) in the Workspace to plan against it — the
          optimizer and what-if read the project's latest fit.
        </p>
      )}

      <Card padding="md">
        <Tabs
          tabs={[
            { id: 'optimize', label: 'Build plan' },
            { id: 'scenario', label: 'What-if' },
          ]}
          active={tab}
          onChange={setTab}
        />
        <div className="mt-4">
          {tab === 'optimize' ? (
            <PlannerStudio
              projectId={currentProjectId}
              channels={channels}
              modelId={modelId}
              onSaved={() => plansQuery.refetch()}
            />
          ) : (
            <ScenarioStudio
              projectId={currentProjectId}
              channels={channels}
              modelId={modelId}
              onSaved={() => plansQuery.refetch()}
            />
          )}
        </div>
      </Card>

      <Card padding="md">
        <h2 className="mb-3 font-display text-lg font-semibold text-ink-900">Saved plans</h2>
        <SavedPlans plans={plans} />
      </Card>

      <Card padding="md">
        <h2 className="mb-1 font-display text-lg font-semibold text-ink-900">
          Latest session report
        </h2>
        <p className="mb-3 text-xs text-ink-400">
          The model report for your active Workspace session — view or download it
          here without leaving the planner.
        </p>
        <ReportViewer threadId={activeThread} kind="mmm" height={480} />
      </Card>
    </div>
  );
}
