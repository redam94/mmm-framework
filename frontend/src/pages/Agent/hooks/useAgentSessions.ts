import { useCallback, useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useProjectStore } from '../../../stores/projectStore';
import { API_BASE, authHeaders } from '../constants';
import type { Project, Session } from '../types';

export function useAgentSessions({ apiKey, modelName }: {
  apiKey: string | null;
  modelName: string | null;
}) {
  const [searchParams] = useSearchParams();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [threadId, setThreadId] = useState<string | null>(() => localStorage.getItem('mmm.activeThreadId'));
  const [projects, setProjects] = useState<Project[]>([]);
  // ONE source of project truth: the shared zustand store the Header's
  // ProjectSwitcher drives (it persists itself). Switching projects anywhere
  // in the app switches the workspace too. null = All projects (no filter).
  const projectId = useProjectStore(s => s.currentProjectId);
  const setProjectId = useProjectStore(s => s.setProject);
  // Gate session loading until /projects has resolved (success or failure) so
  // we know whether the stored project still exists.
  const [projectsReady, setProjectsReady] = useState(false);

  const loadProjects = useCallback(async (): Promise<Project[]> => {
    const data = await fetch(`${API_BASE}/projects`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json());
    return Array.isArray(data?.projects) ? data.projects : [];
  }, [apiKey, modelName]);

  // Effect P (mount): validate the stored selection against the live project
  // list (a deleted project must not filter sessions forever), then unlock
  // session loading. On error, degrade to no filter.
  useEffect(() => {
    (async () => {
      try {
        const list = await loadProjects();
        setProjects(list);
        const current = useProjectStore.getState().currentProjectId;
        if (current && !list.some(p => p.project_id === current)) {
          setProjectId(null);
        }
      } catch (e) {
        console.error('Failed to load projects (degrading to single implicit project)', e);
        setProjects([]);
      } finally {
        setProjectsReady(true);
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Follow ?session=<thread_id>: set on launch AND whenever the app Sidebar's
  // session list navigates while the workspace is already mounted.
  useEffect(() => {
    const sessionParam = searchParams.get('session');
    if (sessionParam) setThreadId(sessionParam);
  }, [searchParams]);

  // Persist active session
  useEffect(() => {
    if (threadId) localStorage.setItem('mmm.activeThreadId', threadId);
  }, [threadId]);

  // Load session list (filtered by project); auto-create one if none, auto-select
  // first if no active. Gated on projectsReady so the filter is known up front.
  useEffect(() => {
    if (!projectsReady) return;
    (async () => {
      try {
        const url = projectId
          ? `${API_BASE}/sessions?project_id=${encodeURIComponent(projectId)}`
          : `${API_BASE}/sessions`;
        const raw = await fetch(url, { headers: authHeaders(apiKey, modelName) }).then(r => r.json());
        let list: Session[] = Array.isArray(raw) ? raw : (raw?.sessions ?? []);
        if (list.length === 0) {
          const created = await fetch(`${API_BASE}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
            body: JSON.stringify(projectId ? { project_id: projectId } : {}),
          }).then(r => r.json());
          list = [created];
        }
        setSessions(list);
        if (!threadId || !list.some(s => s.thread_id === threadId)) {
          setThreadId(list[0].thread_id);
        }
      } catch (e) { console.error('Failed to load sessions', e); }
    })();
  }, [projectId, projectsReady]); // eslint-disable-line react-hooks/exhaustive-deps

  const refreshSessions = useCallback(async () => {
    try {
      const url = projectId
        ? `${API_BASE}/sessions?project_id=${encodeURIComponent(projectId)}`
        : `${API_BASE}/sessions`;
      const raw = await fetch(url, { headers: authHeaders(apiKey, modelName) }).then(r => r.json());
      setSessions(Array.isArray(raw) ? raw : (raw?.sessions ?? []));
    } catch (e) { console.error(e); }
  }, [projectId, apiKey, modelName]);

  // Project actions
  const handleProjectSelect = useCallback((id: string) => {
    setProjectId(id);
  }, [setProjectId]);

  const handleProjectCreate = useCallback(async (name: string, description?: string) => {
    try {
      const created: Project = await fetch(`${API_BASE}/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ name, description }),
      }).then(r => r.json());
      const list = await loadProjects().catch(() => null);
      if (list) setProjects(list);
      else setProjects(prev => [...prev, created]);
      if (created?.project_id) setProjectId(created.project_id);
    } catch (e) { console.error('Failed to create project', e); }
  }, [apiKey, modelName, loadProjects, setProjectId]);

  // ── Session actions ────────────────────────────────────────────────────────
  const handleCreateSession = async () => {
    const created: Session = await fetch(`${API_BASE}/sessions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
      body: JSON.stringify(projectId ? { project_id: projectId } : {}),
    }).then(r => r.json());
    await refreshSessions();
    setThreadId(created.thread_id);
  };

  const handleRenameSession = async (id: string, name: string) => {
    await fetch(`${API_BASE}/sessions/${id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) }, body: JSON.stringify({ name }),
    });
    refreshSessions();
  };

  const handleDeleteSession = async (id: string) => {
    await fetch(`${API_BASE}/sessions/${id}`, { method: 'DELETE', headers: authHeaders(apiKey, modelName) });
    const remaining = sessions.filter(s => s.thread_id !== id);
    setSessions(remaining);
    if (id === threadId) {
      if (remaining.length > 0) setThreadId(remaining[0].thread_id);
      else handleCreateSession();
    }
  };

  return {
    sessions,
    threadId,
    setThreadId,
    projects,
    projectId,
    handleProjectSelect,
    handleProjectCreate,
    handleCreateSession,
    handleRenameSession,
    handleDeleteSession,
  };
}
