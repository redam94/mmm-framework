import { useState } from 'react';
import {
  Activity, BarChart2, BookOpen, BrainCircuit, Database, Download,
  ExternalLink, FileCode, FlaskConical, Layers, Maximize2, Minimize2, Network,
  SlidersHorizontal,
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
import { EdaTab } from './EdaTab';
import type { Artifact, DashboardData, OutlierAction, PythonOutput } from '../../types';

export function WorkspaceTabs({
  rightExpanded, onToggleExpand, activeTab, onTabChange,
  causal, dashboardData, artifacts, pythonOutputs,
  threadId, apiKey, modelName, projectId, workspaceRefreshKey,
  chatLoading,
  onApplySpec, onUnlockField, onQuickAction,
  onRerunArtifact, onDeleteArtifact, onLoadRun, onClearPython,
  onResolveOutlierAction,
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
  onApplySpec: (newSpec: any) => void;
  onUnlockField: (path: string) => void;
  onQuickAction: (msg: string) => void;
  onRerunArtifact: (a: Artifact) => void;
  onDeleteArtifact: (id: string) => void;
  onLoadRun: (runName: string) => void;
  onClearPython: () => void;
  onResolveOutlierAction: (action: OutlierAction) => Promise<string | null>;
}) {
  const [brandingOpen, setBrandingOpen] = useState(false);
  const modelCompleted = dashboardData.model_status === 'completed';
  const hasSpec = !!dashboardData.model_spec;
  const hasDecomp = dashboardData.decomposition?.length > 0;

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
    <div className={`${rightExpanded ? 'w-full' : 'w-2/3'} bg-gray-50 overflow-hidden flex flex-col`}>
      {brandingOpen && (
        <BrandingModal
          projectId={projectId}
          apiKey={apiKey}
          modelName={modelName}
          onClose={() => setBrandingOpen(false)}
        />
      )}
      {/* Header — title + tab bar + settings + fullscreen toggle */}
      <div className="px-5 pt-5 pb-0 bg-gray-50 sticky top-0 z-10 border-b border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-bold text-gray-900">Project Workspace</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setBrandingOpen(true)}
              title="Branding & preferences"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-gray-200 text-gray-500 hover:text-gray-800 shadow-sm transition-all text-sm font-medium"
            >
              <SlidersHorizontal size={14} /> Settings
            </button>
            <button
              onClick={onToggleExpand}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-gray-200 text-gray-500 hover:text-gray-800 shadow-sm transition-all text-sm font-medium"
            >
              {rightExpanded ? <><Minimize2 size={14} /> Restore Chat</> : <><Maximize2 size={14} /> Full Screen</>}
            </button>
          </div>
        </div>

        {/* Tab bar */}
        {(() => {
          const tabs = [
            { id: 'workflow',  label: 'Workflow',  icon: <BookOpen size={14} />,
              badge: `${causal.workflow.filter(s => s.status === 'done').length}/9` },
            { id: 'causal',    label: 'Causal',    icon: <Network size={14} />,
              badge: causal.assumptions.length > 0 ? String(causal.assumptions.length) : null,
              dot: !!causal.dag },
            { id: 'data',      label: 'Data',      icon: <Database size={14} />,
              badge: causal.files.length > 0 ? String(causal.files.length) : null,
              dot: !!dashboardData.dataset },
            { id: 'eda',       label: 'EDA',       icon: <FlaskConical size={14} />,
              badge: edaIssueCount > 0 ? String(edaIssueCount) : null,
              dot: edaHasProposed },
            { id: 'knowledge', label: 'Knowledge', icon: <BrainCircuit size={14} />,
              badge: null, dot: false },
            { id: 'model',     label: 'Model',     icon: <Layers size={14} />,
              dot: hasSpec },
            { id: 'results',   label: 'Results',   icon: <BarChart2 size={14} />,
              dot: modelCompleted || hasDecomp || !!dashboardData.roi_metrics },
            { id: 'plots',     label: 'Plots',     icon: <Activity size={14} />,
              badge: (dashboardData.plots?.length || 0) > 0 ? String(dashboardData.plots.length) : null,
              dot: (dashboardData.plots?.length || 0) > 0 },
            { id: 'artifacts', label: 'Artifacts', icon: <FileCode size={14} />,
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
                        ? 'border-indigo-500 text-indigo-700 bg-white'
                        : 'border-transparent text-gray-500 hover:text-gray-800 hover:bg-gray-100'
                    }`}
                  >
                    <span className={active ? 'text-indigo-600' : 'text-gray-400'}>{t.icon}</span>
                    <span>{t.label}</span>
                    {t.badge && (
                      <span className={`text-[10px] font-semibold rounded-full px-1.5 py-0.5 ${
                        active ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-100 text-gray-600'
                      }`}>
                        {t.badge}
                      </span>
                    )}
                    {!t.badge && t.dot && (
                      <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-indigo-500' : 'bg-gray-300'}`} />
                    )}
                  </button>
                );
              })}
            </div>
          );
        })()}
      </div>

      {/* Tab panels — only one is rendered at a time */}
      <div className="flex-1 overflow-y-auto p-5">
        <div className="grid grid-cols-1 gap-4">

          {activeTab === 'workflow' && (
            causal.workflow.length > 0
              ? <WorkflowChecklist steps={causal.workflow} onOverride={causal.overrideWorkflow} />
              : <EmptyTabState
                  icon={<BookOpen size={28} />}
                  title="Start a conversation to begin"
                  hint="Type a message to the copilot and the scientific workflow checklist will appear here as you progress."
                />
          )}

          {activeTab === 'causal' && (
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
              <DataFilesWidget files={causal.files} onDelete={causal.deleteFile} />
              <WorkspaceFilesWidget
                threadId={threadId}
                apiKey={apiKey}
                modelName={modelName}
                refreshKey={workspaceRefreshKey}
              />
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

          {activeTab === 'eda' && (
            <EdaTab
              eda={dashboardData.eda}
              tables={edaTables}
              disabled={chatLoading}
              onResolveAction={onResolveOutlierAction}
              onNavigate={onTabChange}
            />
          )}

          {activeTab === 'knowledge' && (
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
                  {modelCompleted && (
                    <DashWidget title="Model Successfully Fit" dotColor="bg-green-500 animate-pulse" color="green">
                      <p className="text-sm text-gray-700">{dashboardData.summary}</p>
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
                  <div className="overflow-x-auto rounded-xl border border-gray-200">
                    <table className="w-full text-left text-sm">
                      <thead className="bg-gray-50 text-gray-500 uppercase text-xs">
                        <tr>
                          {['Channel', 'Mean ROI', '94% HDI', 'Prob. Profitable'].map(h => (
                            <th key={h} className="px-4 py-3 font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {dashboardData.roi_metrics.map((row: any) => (
                          <tr key={row.channel} className="bg-white hover:bg-gray-50 transition-colors">
                            <td className="px-4 py-3 font-medium text-gray-900">{row.channel}</td>
                            <td className="px-4 py-3 text-emerald-600 font-semibold">{row.roi_mean?.toFixed(2)}x</td>
                            <td className="px-4 py-3 text-gray-500">[{row.roi_hdi_low?.toFixed(2)}, {row.roi_hdi_high?.toFixed(2)}]</td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <div className="w-14 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                  <div className="h-full bg-emerald-500" style={{ width: `${(row.prob_profitable || 0) * 100}%` }} />
                                </div>
                                <span className="text-gray-700 text-xs font-medium">{((row.prob_profitable || 0) * 100).toFixed(1)}%</span>
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
                      <iframe src={`${API_BASE}/report`} className="w-full h-full rounded-xl border border-gray-200" title="MMM Report" sandbox="allow-scripts allow-same-origin" />
                    </div>
                  }
                >
                  <div className="flex flex-col gap-3">
                    <p className="text-sm text-gray-500">Full analysis report with diagnostics, ROI, and channel decomposition.</p>
                    <div className="flex gap-3">
                      <a href={`${API_BASE}/report/download`} download="mmm_report.html"
                        className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm rounded-xl transition-colors font-medium">
                        <Download size={15} /> Download
                      </a>
                      <a href={`${API_BASE}/report`} target="_blank" rel="noreferrer"
                        className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-xl transition-colors font-medium border border-gray-200">
                        <ExternalLink size={15} /> Open Tab
                      </a>
                    </div>
                    <div className="rounded-xl overflow-hidden border border-gray-200" style={{ height: '340px' }}>
                      <iframe src={`${API_BASE}/report`} className="w-full h-full" title="Preview" sandbox="allow-scripts allow-same-origin" />
                    </div>
                  </div>
                </DashWidget>
              )}
            </>
          )}

          {activeTab === 'plots' && (
            <>
              {dashboardData.plots?.length > 0 ? (
                <DashWidget title={`Visualizations (${dashboardData.plots.length})`} dotColor="bg-fuchsia-500" color="fuchsia">
                  <div className="space-y-4">
                    {dashboardData.plots.map((plot: any, idx: number) => (
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

          {activeTab === 'artifacts' && (
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
