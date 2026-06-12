import { useState } from 'react';
import { ChevronDown, Folder, FolderPlus, Pencil } from 'lucide-react';
import { clsx } from 'clsx';
import { useProjects } from '../../api/hooks';
import type { ProjectResponse } from '../../api/services/projectService';
import { useProjectStore } from '../../stores/projectStore';
import { ProjectOnboardingWizard } from '../onboarding/ProjectOnboardingWizard';

/**
 * Global project switcher (lives in the Header — every measurement-program
 * page is project-scoped). Reads/writes projectStore.currentProjectId.
 */
export function ProjectSwitcher() {
  const { currentProjectId, setProject } = useProjectStore();
  const { data } = useProjects();
  const projects: ProjectResponse[] = data?.projects ?? [];
  const [open, setOpen] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [editing, setEditing] = useState<ProjectResponse | null>(null);
  const current = projects.find((p) => p.project_id === currentProjectId);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-md border border-line-300 bg-white px-3 py-1.5 text-sm font-medium text-ink-700 shadow-sm transition-colors hover:bg-cream-100"
      >
        <Folder className="h-4 w-4 text-ink-400" />
        <span className="max-w-44 truncate">{current?.name ?? 'All projects'}</span>
        <ChevronDown className="h-3.5 w-3.5 text-ink-400" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-20 mt-1 w-72 overflow-hidden rounded-lg border border-line-200 bg-white shadow-lg">
            <button
              onClick={() => {
                setProject(null);
                setOpen(false);
              }}
              className={clsx(
                'flex w-full items-center gap-2 px-4 py-2.5 text-sm transition-colors hover:bg-cream-100',
                !currentProjectId ? 'font-semibold text-sage-800' : 'text-ink-700',
              )}
            >
              All projects
            </button>
            {projects.map((p) => (
              <div
                key={p.project_id}
                className="group flex items-center transition-colors hover:bg-cream-100"
              >
                <button
                  onClick={() => {
                    setProject(p.project_id);
                    setOpen(false);
                  }}
                  className={clsx(
                    'flex min-w-0 flex-1 items-center gap-2 px-4 py-2.5 text-sm',
                    p.project_id === currentProjectId
                      ? 'font-semibold text-sage-800'
                      : 'text-ink-700',
                  )}
                >
                  <Folder className="h-4 w-4 shrink-0 text-ink-300" />
                  <span className="flex-1 truncate text-left">{p.name}</span>
                  {(p.model_count !== undefined || p.session_count !== undefined) && (
                    <span className="text-xs text-ink-300 num">
                      {p.model_count ?? 0}m · {p.session_count ?? 0}s
                    </span>
                  )}
                </button>
                <button
                  onClick={() => {
                    setEditing(p);
                    setOpen(false);
                  }}
                  title="Edit project info"
                  className="mr-2 rounded p-1.5 text-ink-300 opacity-0 transition-opacity hover:bg-cream-200 hover:text-ink-600 focus:opacity-100 group-hover:opacity-100"
                >
                  <Pencil className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            <div className="border-t border-line-200">
              <button
                onClick={() => {
                  setShowNew(true);
                  setOpen(false);
                }}
                className="flex w-full items-center gap-2 px-4 py-2.5 text-sm text-sage-800 transition-colors hover:bg-sage-100"
              >
                <FolderPlus className="h-4 w-4" />
                New project
              </button>
            </div>
          </div>
        </>
      )}

      {showNew && (
        <ProjectOnboardingWizard
          onClose={() => setShowNew(false)}
          onCreated={(id) => setProject(id)}
        />
      )}

      {editing && (
        <ProjectOnboardingWizard
          project={editing}
          onClose={() => setEditing(null)}
          onCreated={(id) => setProject(id)}
        />
      )}
    </div>
  );
}
