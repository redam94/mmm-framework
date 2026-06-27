import { useEffect, useState } from 'react';
import {
  Activity, BarChart2, BookOpen, Database, Download,
  ExternalLink, Layers, Maximize2, Minimize2, Network,
  ShieldCheck, SlidersHorizontal, TestTubes,
} from 'lucide-react';
import {
  WorkflowChecklist, AssumptionsLog, DataFilesWidget, useCausalPanels,
  type DagPayload,
} from '../../../../components/causal/CausalWidgets';
import { CausalPlanner } from '../../../../components/causal/CausalPlanner';
import { API_BASE, selectTables } from '../../constants';
import { BrandingModal } from '../branding/BrandingModal';
import { DashWidget } from '../common/DashWidget';
import { EmptyTabState } from '../common/EmptyTabState';
import { PlotCard } from '../plots/PlotCard';
import { PythonOutputWidget } from '../python/PythonOutputWidget';
import { TableCard } from '../tables/TableCard';
import { ArtifactsPanel } from '../widgets/ArtifactsPanel';
import { DatasetPanel } from '../widgets/DatasetPanel';
import { DecompositionWidget } from '../widgets/DecompositionWidget';
import { KnowledgeTab } from '../widgets/KnowledgeTab';
import { ModelSpecWidget } from '../widgets/ModelSpecWidget';
import { PriorConfigWidget } from '../widgets/PriorConfigWidget';
import { SeasonalityTrendWidget } from '../widgets/SeasonalityTrendWidget';
import { WorkspaceFilesWidget } from '../widgets/WorkspaceFilesWidget';
import { ExperimentsTab } from '../widgets/ExperimentsTab';
import { ModeSwitcher } from '../widgets/ModeSwitcher';
import { GardenModelConfigWidget } from '../widgets/GardenModelConfigWidget';
import { DatasetRolesWidget } from '../widgets/DatasetRolesWidget';
import { useExperimentRegistry } from '../../../../api/hooks/useMeasurement';
import { useGardenModel } from '../../../../api/hooks/useModelGarden';
import { EdaTab } from './EdaTab';
import { ValidationTab } from './ValidationTab';
import { DataStudioModal } from '../dataStudio/DataStudioModal';
import type { CommitPayload } from '../dataStudio/useDataStudio';
import type { ModelingMode } from '../../../../api/services/sessionService';
import type { Artifact, DashboardData, ModelSpec, OutlierAction, PythonOutput } from '../../types';

// Native dashboard ROI table row (server-driven; all metrics optional). Distinct
// from the experiment-priority PriorityChannel shape — this carries prob_profitable.
interface RoiMetricRow {
  channel: string;
  roi_mean?: number;
  roi_hdi_low?: number;
  roi_hdi_high?: number;
  prob_profitable?: number;
}

// A captured plot ref handed straight to PlotCard (which reads the loose figure
// blob internally); we only touch `id` here for the React key.
interface PlotRef { id?: string; [key: string]: unknown }

export function WorkspaceTabs({
  rightExpanded, onToggleExpand, activeTab, onTabChange,
  causal, dashboardData, artifacts, pythonOutputs,
  threadId, apiKey, modelName, projectId, workspaceRefreshKey,
  chatLoading,
  onApplySpec, onUnlockField, onQuickAction,
  onRerunArtifact, onDeleteArtifact, onLoadRun, onClearPython,
  onResolveOutlierAction, onDatasetCommitted,
}: {
  rightExpanded: boolean;
  onToggleExpand: () => void;
  activeTab: string;
  onTabChange: (tab: string) => void;
  causal: ReturnType<typeof useCausalPanels>;
  dashboardData: DashboardData;
  artifacts: Artifact[];
  pythonOutputs: PythonOutput[];
  threadId: string | null;
  apiKey: string | null;
  modelName: string | null;
  projectId: string | null;
  workspaceRefreshKey: number;
  chatLoading: boolean;
  onApplySpec: (newSpec: ModelSpec) => void;
  onUnlockField: (path: string | string[]) => void;
  onQuickAction: (msg: string) => void;
  onRerunArtifact: (a: Artifact) => void;
  onDeleteArtifact: (id: string) => void;
  onLoadRun: (runName: string) => void;
  onClearPython: () => void;
  onResolveOutlierAction: (action: OutlierAction) => Promise<string | null>;
  onDatasetCommitted: (payload: CommitPayload) => void;
}) {
  const [brandingOpen, setBrandingOpen] = useState(false);
  const [studioOpen, setStudioOpen] = useState(false);
  // Modeling mode (hydrated from the session by ModeSwitcher). MMM keeps the full
  // ROI/experiment surface; non-MMM modes hide the Experiments tab.
  const [modelingMode, setModelingMode] = useState<ModelingMode>('mmm');
  const isMmm = modelingMode === 'mmm';
  // If the Experiments tab is hidden by a mode switch while it's active, fall back.
  useEffect(() => {
    if (!isMmm && activeTab === 'experiments') onTabChange('results');
  }, [isMmm, activeTab, onTabChange]);
  // Experiment registry (read-only here; lifecycle work is chat- or /experiments-driven)
  const { data: registryExperiments = [] } = useExperimentRegistry(projectId);
  const activeExperiments = registryExperiments.filter(
    (e) => !['abandoned', 'cancelled'].includes(e.status),
  );
  const modelCompleted = dashboardData.model_status === 'completed';
  const modelSpec = dashboardData.model_spec as ModelSpec | undefined;
  const hasSpec = !!modelSpec;
  const hasDecomp = dashboardData.decomposition?.length > 0;

  // A loaded garden model carries a garden_ref; its manifest (fetched here) tells
  // us the family kind + its config_schema / required roles. model_kind drives
  // whether the MMM config widgets apply (an awareness model is still mmm-kind +
  // has bespoke params; a CFA/LCA is non-MMM, so the channel/adstock widgets hide).
  const gardenRef = modelSpec?.garden_ref;
  const { data: gardenModel } = useGardenModel(gardenRef?.name ?? null, gardenRef?.version ?? null);
  const modelKind = (gardenModel?.manifest?.model_kind as string) ?? 'mmm';
  const showMmmConfig = modelKind === 'mmm';

  // Structured tables, bucketed by group (unknown groups fall into "repl").
  const edaTables = selectTables(dashboardData.tables, 'eda');
  const resultsTables = selectTables(dashboardData.tables, 'results');
  const replTables = selectTables(dashboardData.tables, 'repl');

  // EDA tab badge = error/warning issue count; dot = any proposed outlier action.
  const edaIssueCount = (dashboardData.eda?.issues || [])
    .filter(i => i.severity === 'error' || i.severity === 'warning').length;
  const edaHasProposed = (dashboardData.eda?.outlier_actions || [])
    .some(a => a.status === 'proposed');

  return (
    <div className={`${rightExpanded ? 'w-full' : 'w-2/3'} bg-cream-50 overflow-hidden flex flex-col`}>
      {brandingOpen && (
        <BrandingModal
          projectId={projectId}
          apiKey={apiKey}
          modelName={modelName}
          onClose={() => setBrandingOpen(false)}
        />
      )}
      {/* One slim row: tab bar left, settings + expand right (the global app
          Header already names the page — no duplicate title here). */}
      <div className="px-5 pt-3 pb-0 bg-cream-50 sticky top-0 z-10 border-b border-line-200 flex items-end justify-between gap-3">
        {(() => {
          // Six groups, in workflow order: think → look at the data → specify
          // → read results → decide experiments → reference material.
          const tabs = [
            { id: 'plan',      label: 'Plan',      icon: <Network size={14} />,
              badge: `${causal.workflow.filter(s => s.status === 'done').length}/9`,
              dot: !!causal.dag || causal.assumptions.length > 0 },
            { id: 'data',      label: 'Data',      icon: <Database size={14} />,
              badge: edaIssueCount > 0 ? String(edaIssueCount)
                : causal.files.length > 0 ? String(causal.files.length) : null,
              dot: !!dashboardData.dataset || edaHasProposed },
            { id: 'model',     label: 'Model',     icon: <Layers size={14} />,
              dot: hasSpec },
            { id: 'results',   label: 'Results',   icon: <BarChart2 size={14} />,
              badge: (dashboardData.plots?.length || 0) > 0 ? String(dashboardData.plots.length) : null,
              dot: modelCompleted || hasDecomp || !!dashboardData.roi_metrics },
            { id: 'validation', label: 'Validation', icon: <ShieldCheck size={14} />,
              dot: modelCompleted },
            // Experiments are an MMM-only surface (lift tests on media channels);
            // hidden in the general / causal / descriptive modes.
            ...(isMmm ? [{ id: 'experiments', label: 'Experiments', icon: <TestTubes size={14} />,
              badge: activeExperiments.length > 0 ? String(activeExperiments.length) : null,
              dot: activeExperiments.some(e => e.status === 'completed') }] : []),
            { id: 'library',   label: 'Library',   icon: <BookOpen size={14} />,
              badge: (artifacts.length + pythonOutputs.length) > 0
                ? String(artifacts.length + pythonOutputs.length) : null },
          ];
          return (
            <div className="flex items-center gap-1 overflow-x-auto -mb-px">
              {tabs.map(t => {
                const active = activeTab === t.id;
                return (
                  <button
                    key={t.id}
                    onClick={() => onTabChange(t.id)}
                    className={`flex items-center gap-2 px-3.5 py-2.5 text-sm font-medium rounded-t-lg border-b-2 transition-colors shrink-0 ${
                      active
                        ? 'border-sage-700 text-sage-800 bg-white'
                        : 'border-transparent text-ink-400 hover:text-ink-900 hover:bg-cream-100'
                    }`}
                  >
                    <span className={active ? 'text-sage-700' : 'text-ink-300'}>{t.icon}</span>
                    <span>{t.label}</span>
                    {t.badge && (
                      <span className={`text-[10px] font-semibold rounded-full px-1.5 py-0.5 ${
                        active ? 'bg-sage-100 text-sage-800' : 'bg-cream-200 text-ink-600'
                      }`}>
                        {t.badge}
                      </span>
                    )}
                    {!t.badge && t.dot && (
                      <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-sage-600' : 'bg-line-400'}`} />
                    )}
                  </button>
                );
              })}
            </div>
          );
        })()}
        <div className="flex items-center gap-1.5 pb-1.5 shrink-0">
          <ModeSwitcher threadId={threadId} value={modelingMode} onChange={setModelingMode} />
          <button
            onClick={() => setBrandingOpen(true)}
            title="Branding & preferences"
            className="p-2 rounded-lg text-ink-300 hover:text-ink-900 hover:bg-cream-100 transition-colors"
          >
            <SlidersHorizontal size={15} />
          </button>
          <button
            onClick={onToggleExpand}
            title={rightExpanded ? 'Restore chat' : 'Full screen'}
            className="p-2 rounded-lg text-ink-300 hover:text-ink-900 hover:bg-cream-100 transition-colors"
          >
            {rightExpanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
        </div>
      </div>

      {/* Tab panels — only one is rendered at a time */}
      <div className="flex-1 overflow-y-auto p-5">
        <div className="grid grid-cols-1 gap-4">

          {activeTab === 'plan' && (
            causal.workflow.length > 0
              ? <WorkflowChecklist steps={causal.workflow} onOverride={causal.overrideWorkflow} />
              : <EmptyTabState
                  icon={<BookOpen size={28} />}
                  title="Start a conversation to begin"
                  hint="Type a message to the copilot and the scientific workflow checklist will appear here as you progress."
                />
          )}

          {activeTab === 'plan' && (
            <>
              <CausalPlanner
                // Prefer the mid-stream agent-pushed DAG; fall back to the
                // last fetched one from the causal panels hook.
                dag={(dashboardData.dag as DagPayload | undefined) ?? causal.dag}
                threadId={threadId}
                chatStreaming={chatLoading}
                onSaved={causal.refresh}
              />
              <AssumptionsLog
                threadId={threadId}
                assumptions={causal.assumptions}
                onRefresh={causal.refresh}
              />
            </>
          )}

          {activeTab === 'data' && (
            <>
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <button
                  onClick={() => setStudioOpen(true)}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-sage-700 hover:bg-sage-800 text-white text-sm font-semibold shadow-sm transition-colors"
                >
                  <Database size={16} /> Upload &amp; clean data
                </button>
                <p className="text-xs text-ink-300">Stage a CSV/Excel file, run interactive EDA, then convert it into the working dataset.</p>
              </div>
              {studioOpen && (
                <DataStudioModal
                  threadId={threadId}
                  apiKey={apiKey}
                  modelName={modelName}
                  chatLoading={chatLoading}
                  onClose={() => setStudioOpen(false)}
                  onCommitted={(payload) => { onDatasetCommitted(payload); setStudioOpen(false); onTabChange('data'); }}
                />
              )}
              <DataFilesWidget files={causal.files} onDelete={causal.deleteFile} />
              <WorkspaceFilesWidget
                threadId={threadId}
                apiKey={apiKey}
                modelName={modelName}
                refreshKey={workspaceRefreshKey}
              />
              <DatasetRolesWidget spec={modelSpec} gardenModel={gardenModel} />
              {dashboardData.dataset ? (
                <DatasetPanel dataset={dashboardData.dataset} threadId={threadId} />
              ) : (
                <EmptyTabState
                  icon={<Database size={28} />}
                  title="No dataset loaded yet"
                  hint="Ask the agent to generate synthetic data or upload a CSV — it'll call `inspect_dataset` and details will appear here."
                />
              )}
            </>
          )}

          {activeTab === 'data' && (
            <EdaTab
              eda={dashboardData.eda}
              tables={edaTables}
              disabled={chatLoading}
              onResolveAction={onResolveOutlierAction}
              onNavigate={onTabChange}
            />
          )}

          {activeTab === 'experiments' && (
            <ExperimentsTab
              projectId={projectId}
              onQuickAction={onQuickAction}
              disabled={chatLoading}
            />
          )}

          {activeTab === 'library' && (
            <KnowledgeTab
              projectId={projectId}
              apiKey={apiKey}
              modelName={modelName}
            />
          )}

          {activeTab === 'model' && (
            <>
              {hasSpec ? (
                <>
                  {/* A bespoke garden model: show its identity + model_params +
                      likelihood. For non-MMM kinds this REPLACES the MMM widgets;
                      for an mmm-kind garden model (e.g. awareness) it sits above
                      them and the inference section stays with the MMM widget. */}
                  {gardenRef && (
                    <GardenModelConfigWidget
                      spec={modelSpec as ModelSpec}
                      gardenModel={gardenModel}
                      modelKind={modelKind}
                      // A bespoke model can be reconfigured + re-fit even after a
                      // fit (unlike the MMM widgets, which lock post-fit); only a
                      // streaming turn disables editing.
                      editable={!chatLoading}
                      fitted={modelCompleted}
                      onApplySpec={onApplySpec}
                      showInference={!showMmmConfig}
                    />
                  )}
                  {showMmmConfig && (
                    <>
                      <ModelSpecWidget
                        spec={dashboardData.model_spec}
                        editable={!modelCompleted}
                        onApplySpec={onApplySpec}
                        lockedFields={dashboardData.locked_fields || []}
                        onUnlock={modelCompleted ? undefined : onUnlockField}
                      />
                      <SeasonalityTrendWidget
                        spec={dashboardData.model_spec}
                        onQuickAction={onQuickAction}
                        modelCompleted={modelCompleted}
                      />
                      <PriorConfigWidget
                        spec={dashboardData.model_spec}
                        editable={!modelCompleted}
                        onApplySpec={onApplySpec}
                      />
                    </>
                  )}
                  {modelCompleted && (
                    <DashWidget title="Model Successfully Fit" dotColor="bg-green-500 animate-pulse" color="green">
                      <p className="text-sm text-ink-700">{dashboardData.summary}</p>
                    </DashWidget>
                  )}
                </>
              ) : (
                <EmptyTabState
                  icon={<Layers size={28} />}
                  title="No model configured yet"
                  hint="Ask the agent to configure a model (Step 3 of the workflow) — it'll call `configure_model` and a spec will appear here."
                />
              )}
            </>
          )}

          {activeTab === 'results' && (
            <>
              {!modelCompleted && !hasDecomp && !dashboardData.roi_metrics && !dashboardData.report_path && resultsTables.length === 0 && (
                <EmptyTabState
                  icon={<BarChart2 size={28} />}
                  title="No results yet"
                  hint="Fit a model (Step 5), then ask for the decomposition or ROI."
                />
              )}

              {hasDecomp && <DecompositionWidget decomposition={dashboardData.decomposition} />}

              {dashboardData.roi_metrics && (() => {
                const table = (
                  <div className="overflow-x-auto rounded-xl border border-line-200">
                    <table className="w-full text-left text-sm">
                      <thead className="bg-cream-50 text-ink-400 uppercase text-xs">
                        <tr>
                          {['Channel', 'Mean ROI', '94% HDI', 'Prob. Profitable'].map(h => (
                            <th key={h} className="px-4 py-3 font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-line-200">
                        {(dashboardData.roi_metrics as RoiMetricRow[]).map((row) => (
                          <tr key={row.channel} className="bg-white hover:bg-cream-100 transition-colors">
                            <td className="px-4 py-3 font-medium text-ink-900">{row.channel}</td>
                            <td className="px-4 py-3 text-emerald-600 font-semibold">{row.roi_mean?.toFixed(2)}x</td>
                            <td className="px-4 py-3 text-ink-400">[{row.roi_hdi_low?.toFixed(2)}, {row.roi_hdi_high?.toFixed(2)}]</td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <div className="w-14 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                  <div className="h-full bg-emerald-500" style={{ width: `${(row.prob_profitable || 0) * 100}%` }} />
                                </div>
                                <span className="text-ink-700 text-xs font-medium">{((row.prob_profitable || 0) * 100).toFixed(1)}%</span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                );
                return (
                  <DashWidget title="ROI Performance" dotColor="bg-emerald-500" color="emerald" expandContent={table}>
                    {table}
                  </DashWidget>
                );
              })()}

              {resultsTables.length > 0 && (
                <DashWidget title={`Tables (${resultsTables.length})`} dotColor="bg-indigo-500" color="indigo">
                  <div className="space-y-4">
                    {resultsTables.map((t, idx) => (
                      <TableCard key={t.id} tableRef={t} idx={idx} />
                    ))}
                  </div>
                </DashWidget>
              )}

              {dashboardData.report_path && (
                <DashWidget title="Full MMM Report" dotColor="bg-violet-500" color="violet"
                  expandTitle="MMM Report"
                  expandContent={
                    <div className="h-[80vh]">
                      <iframe src={`${API_BASE}/report`} className="w-full h-full rounded-xl border border-line-200" title="MMM Report" sandbox="allow-scripts allow-same-origin" />
                    </div>
                  }
                >
                  <div className="flex flex-col gap-3">
                    <p className="text-sm text-ink-400">Full analysis report with diagnostics, ROI, and channel decomposition.</p>
                    <div className="flex gap-3">
                      <a href={`${API_BASE}/report/download`} download="mmm_report.html"
                        className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm rounded-xl transition-colors font-medium">
                        <Download size={15} /> Download
                      </a>
                      <a href={`${API_BASE}/report`} target="_blank" rel="noreferrer"
                        className="flex items-center gap-2 px-4 py-2 bg-cream-100 hover:bg-gray-200 text-ink-700 text-sm rounded-xl transition-colors font-medium border border-line-200">
                        <ExternalLink size={15} /> Open Tab
                      </a>
                    </div>
                    <div className="rounded-xl overflow-hidden border border-line-200" style={{ height: '340px' }}>
                      <iframe src={`${API_BASE}/report`} className="w-full h-full" title="Preview" sandbox="allow-scripts allow-same-origin" />
                    </div>
                  </div>
                </DashWidget>
              )}
            </>
          )}

          {activeTab === 'results' && (
            <>
              {dashboardData.plots?.length > 0 ? (
                <DashWidget title={`Visualizations (${dashboardData.plots.length})`} dotColor="bg-fuchsia-500" color="fuchsia">
                  <div className="space-y-4">
                    {(dashboardData.plots as PlotRef[]).map((plot, idx: number) => (
                      <PlotCard key={plot?.id ?? idx} plot={plot} idx={idx} />
                    ))}
                  </div>
                </DashWidget>
              ) : replTables.length === 0 ? (
                <EmptyTabState
                  icon={<Activity size={28} />}
                  title="No plots yet"
                  hint="Ask the agent to run execute_python with fig.show() — charts appear here automatically."
                />
              ) : null}

              {replTables.length > 0 && (
                <DashWidget title={`Tables (${replTables.length})`} dotColor="bg-indigo-500" color="indigo">
                  <div className="space-y-4">
                    {replTables.map((t, idx) => (
                      <TableCard key={t.id} tableRef={t} idx={idx} />
                    ))}
                  </div>
                </DashWidget>
              )}
            </>
          )}

          {activeTab === 'validation' && <ValidationTab projectId={projectId} />}

          {activeTab === 'library' && (
            <>
              <ArtifactsPanel
                artifacts={artifacts}
                onRerun={onRerunArtifact}
                onDelete={onDeleteArtifact}
                onLoadRun={onLoadRun}
              />
              <PythonOutputWidget
                outputs={pythonOutputs}
                onClear={onClearPython}
                onExport={threadId ? () => window.open(`${API_BASE}/sessions/${threadId}/export`, '_blank') : undefined}
              />
            </>
          )}

        </div>
      </div>
    </div>
  );
}
