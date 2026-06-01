import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FolderPlusIcon,
  ChatBubbleLeftRightIcon,
  CubeIcon,
  CurrencyDollarIcon,
  ArrowRightIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  FolderIcon,
  ChevronDownIcon,
  ClipboardDocumentListIcon,
} from '@heroicons/react/24/outline';
import {
  useProjects,
  useCreateProject,
  useSessions,
  useModels,
  useAnalysisPlans,
} from '../../api/hooks';
import type { AnalysisPlanInfo } from '../../api/hooks';
import { useBudgetPlans } from '../../api/hooks/useBudgetPlans';
import type { BudgetPlanInfo } from '../../api/hooks/useBudgetPlans';
import { useProjectStore } from '../../stores/projectStore';
import type { ProjectResponse } from '../../api/services/projectService';
import type { SessionInfo } from '../../api/services/sessionService';

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatTs(ts: number | string) {
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  const now = Date.now();
  const diffMs = now - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH < 24) return `${diffH}h ago`;
  return d.toLocaleDateString();
}

function statusColor(status: string) {
  if (status === 'completed') return 'text-green-600 bg-green-50';
  if (status === 'running' || status === 'queued') return 'text-blue-600 bg-blue-50';
  if (status === 'failed') return 'text-red-600 bg-red-50';
  return 'text-gray-500 bg-gray-50';
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ProjectSwitcher({
  projects,
  currentId,
  onSelect,
  onCreateNew,
}: {
  projects: ProjectResponse[];
  currentId: string | null;
  onSelect: (id: string | null) => void;
  onCreateNew: () => void;
}) {
  const [open, setOpen] = useState(false);
  const current = projects.find((p) => p.project_id === currentId);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-lg shadow-sm hover:bg-gray-50 transition-colors text-sm font-medium text-gray-700"
      >
        <FolderIcon className="h-4 w-4 text-gray-400" />
        <span>{current?.name ?? 'All Projects'}</span>
        <ChevronDownIcon className="h-3.5 w-3.5 text-gray-400" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute left-0 mt-1 z-20 w-64 bg-white rounded-xl border border-gray-200 shadow-lg overflow-hidden">
            <button
              onClick={() => { onSelect(null); setOpen(false); }}
              className={`w-full flex items-center gap-2 px-4 py-2.5 text-sm hover:bg-gray-50 transition-colors ${!currentId ? 'font-semibold text-indigo-600' : 'text-gray-700'}`}
            >
              All Projects
            </button>
            {projects.map((p) => (
              <button
                key={p.project_id}
                onClick={() => { onSelect(p.project_id); setOpen(false); }}
                className={`w-full flex items-center gap-2 px-4 py-2.5 text-sm hover:bg-gray-50 transition-colors ${p.project_id === currentId ? 'font-semibold text-indigo-600' : 'text-gray-700'}`}
              >
                <FolderIcon className="h-4 w-4 text-gray-400" />
                <span className="flex-1 text-left truncate">{p.name}</span>
                <span className="text-xs text-gray-400">
                  {p.model_count}m · {p.session_count}s
                </span>
              </button>
            ))}
            <div className="border-t border-gray-100">
              <button
                onClick={() => { onCreateNew(); setOpen(false); }}
                className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-indigo-600 hover:bg-indigo-50 transition-colors"
              >
                <FolderPlusIcon className="h-4 w-4" />
                New Project
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function NewProjectModal({ onClose, onCreated }: { onClose: () => void; onCreated: (id: string) => void }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const create = useCreateProject();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    const project = await create.mutateAsync({ name: name.trim(), description: description.trim() || undefined });
    onCreated(project.project_id);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/40" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md">
        <div className="px-6 py-4 border-b border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900">New Project</h2>
        </div>
        <form onSubmit={handleSubmit} className="px-6 py-4 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Project name</label>
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Q4 2024 MMM Analysis"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description (optional)</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <div className="flex gap-3 justify-end pt-2">
            <button type="button" onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 transition-colors">Cancel</button>
            <button
              type="submit"
              disabled={!name.trim() || create.isPending}
              className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
            >
              {create.isPending ? 'Creating…' : 'Create Project'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function SessionsPanel({ projectId }: { projectId: string | null }) {
  const navigate = useNavigate();
  const { data, isLoading } = useSessions(
    projectId ? { project_id: projectId, limit: 8 } : { limit: 8 }
  );
  const sessions = data?.sessions ?? [];

  return (
    <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <ChatBubbleLeftRightIcon className="h-5 w-5 text-indigo-500" />
          <h2 className="font-semibold text-gray-900">Agent Sessions</h2>
        </div>
        <button
          onClick={() => navigate('/chat')}
          className="flex items-center gap-1.5 text-sm text-indigo-600 hover:text-indigo-700 font-medium transition-colors"
        >
          New session <ArrowRightIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      {isLoading ? (
        <div className="px-5 py-8 text-center text-sm text-gray-400">Loading…</div>
      ) : sessions.length === 0 ? (
        <div className="px-5 py-8 text-center">
          <ChatBubbleLeftRightIcon className="h-8 w-8 text-gray-200 mx-auto mb-2" />
          <p className="text-sm text-gray-400">No sessions yet.</p>
          <button
            onClick={() => navigate('/chat')}
            className="mt-3 text-sm text-indigo-600 hover:underline"
          >
            Start your first session →
          </button>
        </div>
      ) : (
        <ul className="divide-y divide-gray-50">
          {sessions.map((s: SessionInfo) => (
            <li key={s.thread_id} className="flex items-center gap-3 px-5 py-3 hover:bg-gray-50 transition-colors group">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-800 truncate">{s.name}</p>
                <p className="text-xs text-gray-400 flex items-center gap-1 mt-0.5">
                  <ClockIcon className="h-3 w-3" />
                  {formatTs(s.updated_at)}
                  {s.artifact_count > 0 && (
                    <span className="ml-1.5 text-gray-300">· {s.artifact_count} artifacts</span>
                  )}
                </p>
              </div>
              <button
                onClick={() => navigate(`/chat?session=${s.thread_id}`)}
                className="flex items-center gap-1.5 text-xs text-indigo-600 font-medium opacity-0 group-hover:opacity-100 transition-opacity hover:underline"
              >
                Resume <ArrowRightIcon className="h-3 w-3" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function ModelsPanel({ projectId }: { projectId: string | null }) {
  const navigate = useNavigate();
  const { data, isLoading } = useModels(
    projectId ? { project_id: projectId, limit: 8 } as any : { limit: 8 } as any
  );
  const models = data?.models ?? [];

  return (
    <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <CubeIcon className="h-5 w-5 text-teal-500" />
          <h2 className="font-semibold text-gray-900">Fitted Models</h2>
        </div>
        <button
          onClick={() => navigate('/chat')}
          className="flex items-center gap-1.5 text-sm text-teal-600 hover:text-teal-700 font-medium transition-colors"
        >
          New fit <ArrowRightIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      {isLoading ? (
        <div className="px-5 py-8 text-center text-sm text-gray-400">Loading…</div>
      ) : models.length === 0 ? (
        <div className="px-5 py-8 text-center">
          <CubeIcon className="h-8 w-8 text-gray-200 mx-auto mb-2" />
          <p className="text-sm text-gray-400">No models fitted yet.</p>
          <button
            onClick={() => navigate('/chat')}
            className="mt-3 text-sm text-teal-600 hover:underline"
          >
            Fit your first model →
          </button>
        </div>
      ) : (
        <ul className="divide-y divide-gray-50">
          {models.map((m: any) => (
            <li key={m.model_id} className="flex items-center gap-3 px-5 py-3 hover:bg-gray-50 transition-colors group">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-800 truncate">{m.name || m.model_id}</p>
                <p className="text-xs text-gray-400 flex items-center gap-1.5 mt-0.5">
                  <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium ${statusColor(m.status)}`}>
                    {m.status === 'completed' && <CheckCircleIcon className="h-3 w-3" />}
                    {m.status === 'failed' && <ExclamationTriangleIcon className="h-3 w-3" />}
                    {m.status}
                  </span>
                  <span className="text-gray-300">·</span>
                  <ClockIcon className="h-3 w-3" />
                  {formatTs(m.created_at)}
                </p>
              </div>
              {m.status === 'completed' && (
                <button
                  onClick={() => navigate(`/chat?session=${m.model_id}`)}
                  className="flex items-center gap-1.5 text-xs text-teal-600 font-medium opacity-0 group-hover:opacity-100 transition-opacity hover:underline"
                >
                  View <ArrowRightIcon className="h-3 w-3" />
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function AnalysisPlansPanel() {
  const navigate = useNavigate();
  const { data, isLoading } = useAnalysisPlans();
  const plans = data?.plans ?? [];

  return (
    <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <ClipboardDocumentListIcon className="h-5 w-5 text-violet-500" />
          <h2 className="font-semibold text-gray-900">Analysis Plans</h2>
        </div>
        <button
          onClick={() => navigate('/analysis-plan')}
          className="flex items-center gap-1.5 text-sm text-violet-600 hover:text-violet-700 font-medium transition-colors"
        >
          Open DAG editor <ArrowRightIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      {isLoading ? (
        <div className="px-5 py-8 text-center text-sm text-gray-400">Loading…</div>
      ) : plans.length === 0 ? (
        <div className="px-5 py-8 text-center">
          <ClipboardDocumentListIcon className="h-8 w-8 text-gray-200 mx-auto mb-2" />
          <p className="text-sm text-gray-400">No locked plans yet.</p>
          <p className="text-xs text-gray-400 mt-1 max-w-48 mx-auto">
            Build a DAG in Analysis Plan and click "Lock Analysis Plan" to pre-register your analysis.
          </p>
        </div>
      ) : (
        <ul className="divide-y divide-gray-50">
          {plans.slice(0, 8).map((p: AnalysisPlanInfo) => (
            <li key={p.id} className="flex items-start gap-3 px-5 py-3 hover:bg-gray-50 transition-colors group">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-800 truncate">{p.name}</p>
                <p className="text-xs text-gray-400 flex items-center gap-1 mt-0.5">
                  <ClockIcon className="h-3 w-3" />
                  {formatTs(p.locked_at)}
                  {Array.isArray(p.payload?.assumptions) && (
                    <span className="ml-1.5 text-gray-300">
                      · {(p.payload.assumptions as unknown[]).length} assumptions
                    </span>
                  )}
                </p>
              </div>
              <button
                onClick={() => navigate(`/chat?session=${p.thread_id}`)}
                className="flex items-center gap-1.5 text-xs text-violet-600 font-medium opacity-0 group-hover:opacity-100 transition-opacity hover:underline shrink-0"
              >
                Session <ArrowRightIcon className="h-3 w-3" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function BudgetPlansPanel({ projectId }: { projectId: string | null }) {
  const navigate = useNavigate();
  const { data, isLoading } = useBudgetPlans(
    projectId ? { project_id: projectId } : undefined
  );
  const plans = (data?.plans ?? []).slice(0, 8);

  return (
    <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <CurrencyDollarIcon className="h-5 w-5 text-amber-500" />
          <h2 className="font-semibold text-gray-900">Budget Plans</h2>
        </div>
        <button
          onClick={() => navigate('/chat')}
          className="flex items-center gap-1.5 text-sm text-amber-600 hover:text-amber-700 font-medium transition-colors"
        >
          Open planner <ArrowRightIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      {isLoading ? (
        <div className="px-5 py-8 text-center text-sm text-gray-400">Loading…</div>
      ) : plans.length === 0 ? (
        <div className="px-5 py-8 text-center">
          <CurrencyDollarIcon className="h-8 w-8 text-gray-200 mx-auto mb-2" />
          <p className="text-sm text-gray-400">No saved plans yet.</p>
          <button
            onClick={() => navigate('/chat')}
            className="mt-3 text-sm text-amber-600 hover:underline"
          >
            Create your first plan →
          </button>
        </div>
      ) : (
        <ul className="divide-y divide-gray-50">
          {plans.map((p: BudgetPlanInfo) => {
            const positive = p.outcome_change >= 0;
            return (
              <li
                key={p.plan_id}
                className="flex items-center gap-3 px-5 py-3 hover:bg-gray-50 transition-colors group"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-800 truncate">{p.name}</p>
                  <p className="text-xs text-gray-400 flex items-center gap-1.5 mt-0.5">
                    <ClockIcon className="h-3 w-3" />
                    {formatTs(p.created_at)}
                    <span
                      className={`ml-1 px-1.5 py-0.5 rounded text-[11px] font-medium ${
                        positive ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'
                      }`}
                    >
                      {positive ? '+' : ''}
                      {p.outcome_change_pct.toFixed(1)}%
                    </span>
                  </p>
                </div>
                <button
                  onClick={() => navigate('/chat')}
                  className="flex items-center gap-1.5 text-xs text-amber-600 font-medium opacity-0 group-hover:opacity-100 transition-opacity hover:underline"
                >
                  View <ArrowRightIcon className="h-3 w-3" />
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

// ── Dashboard Page ─────────────────────────────────────────────────────────────

export function DashboardPage() {
  const { data: projectsData } = useProjects();
  const { currentProjectId, setProject } = useProjectStore();
  const [showNewProject, setShowNewProject] = useState(false);

  const projects = projectsData?.projects ?? [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-sm text-gray-500 mt-0.5">Your MMM sessions, models, and plans</p>
        </div>
        <ProjectSwitcher
          projects={projects}
          currentId={currentProjectId}
          onSelect={setProject}
          onCreateNew={() => setShowNewProject(true)}
        />
      </div>

      {/* Four panel grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SessionsPanel projectId={currentProjectId} />
        <ModelsPanel projectId={currentProjectId} />
        <AnalysisPlansPanel />
        <BudgetPlansPanel projectId={currentProjectId} />
      </div>

      {showNewProject && (
        <NewProjectModal
          onClose={() => setShowNewProject(false)}
          onCreated={(id) => setProject(id)}
        />
      )}
    </div>
  );
}

export default DashboardPage;
