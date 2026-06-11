import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';
import { ModelSwitcher } from '../../components/common';
import { useCausalPanels } from '../../components/causal/CausalWidgets';
import { API_BASE, authHeaders } from './constants';
import { specLeafDiff, specWithDefaults } from './utils/spec';
import { useAgentSessions } from './hooks/useAgentSessions';
import { useChatStream } from './hooks/useChatStream';
import { ChatPanel } from './components/chat/ChatPanel';
import { SessionSidebar } from './components/sidebar/SessionSidebar';
import { WorkspaceTabs } from './components/tabs/WorkspaceTabs';
import { PendingSpecChangesModal } from './components/widgets/PendingSpecChangesModal';
import type { Artifact, OutlierAction } from './types';

// ─── Main AgentPage ───────────────────────────────────────────────────────────

export function AgentPage() {
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [rightExpanded, setRightExpanded] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [workspaceRefreshKey, setWorkspaceRefreshKey] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { apiKey, modelName } = useAuthStore();
  const {
    sessions, threadId, setThreadId, projects, projectId,
    handleProjectSelect, handleProjectCreate,
    handleCreateSession, handleRenameSession, handleDeleteSession,
  } = useAgentSessions({ apiKey, modelName });
  const causal = useCausalPanels(threadId);
  const [activeTab, setActiveTab] = useState<string>(() => localStorage.getItem('mmm.activeTab') || 'workflow');

  // Artifact state lives here (it is shared with the Artifacts tab); the chat
  // stream hook hands artifacts back via this stable callback.
  const handleArtifactsLoaded = useCallback((arts: Artifact[]) => setArtifacts(arts), []);

  const {
    messages, dashboardData, pythonOutputs, loading,
    send, stop: handleStop, clear: handleClearChat, loadThreadState,
    setDashboardData, setPythonOutputs, setLoading,
  } = useChatStream({
    threadId,
    apiKey,
    modelName,
    // Refresh artifacts + causal panels after the turn so newly-saved
    // snippets, assumptions, files, DAG, and workflow status all show up.
    onTurnSettled: async () => {
      try {
        const arts = await fetch(`${API_BASE}/artifacts/${threadId}`).then(r => r.json());
        if (Array.isArray(arts)) setArtifacts(arts);
      } catch { /* ignore */ }
      causal.refresh();
      // Refresh workspace output files (newly generated reports/CSVs/PNGs).
      setWorkspaceRefreshKey(k => k + 1);
    },
    onArtifactsLoaded: handleArtifactsLoaded,
  });

  useEffect(() => { localStorage.setItem('mmm.activeTab', activeTab); }, [activeTab]);

  // Re-load whenever active session changes
  useEffect(() => {
    if (threadId) loadThreadState(threadId);
  }, [threadId, loadThreadState]);

  const handleBack = async () => {
    if (!threadId || loading) return;
    // Fetch timeline, find a checkpoint before the latest human message
    try {
      const timeline: any[] = await fetch(`${API_BASE}/history/${threadId}`).then(r => r.json());
      if (!Array.isArray(timeline) || timeline.length < 2) return;
      // timeline is newest-first. Find checkpoints in chronological order
      // and rewind to the one before the latest user-visible state change.
      // Strategy: target = the checkpoint where message_count drops by ≥1
      // compared to current head — equivalent to "previous turn boundary".
      const head = timeline[0];
      let target: any = null;
      for (let i = 1; i < timeline.length; i++) {
        if (timeline[i].message_count < head.message_count) {
          target = timeline[i];
          break;
        }
      }
      if (!target) target = timeline[timeline.length - 1];
      await fetch(`${API_BASE}/rewind/${threadId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_id: target.checkpoint_id }),
      });
      await loadThreadState(threadId);
    } catch (e) { console.error('Back failed', e); }
  };

  const handleRetry = async () => {
    if (!threadId || loading) return;
    // Find the last human message in current state
    const lastHuman = [...messages].reverse().find(m => m.type === 'human');
    if (!lastHuman) return;
    try {
      const timeline: any[] = await fetch(`${API_BASE}/history/${threadId}`).then(r => r.json());
      if (!Array.isArray(timeline)) return;
      // Rewind to the checkpoint with the smallest message_count that is
      // still ≥ (messages_before_last_human). That's the state right before
      // the most recent user turn. Walk oldest→newest.
      const ordered = [...timeline].reverse(); // oldest first
      const targetCount = messages.findIndex(m => m.id === lastHuman.id);
      // Find first checkpoint whose message_count ≥ targetCount but whose
      // last_human_index < targetCount (i.e. the human is not yet in it).
      let chosen: any = null;
      for (const cp of ordered) {
        if ((cp.last_human_index ?? -1) < targetCount) chosen = cp;
        else break;
      }
      if (!chosen) chosen = ordered[0];
      await fetch(`${API_BASE}/rewind/${threadId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_id: chosen.checkpoint_id }),
      });
      await loadThreadState(threadId);
      handleSend(lastHuman.content);
    } catch (e) { console.error('Retry failed', e); }
  };

  const handleSend = async (messageOverride?: string) => {
    if (!threadId) return;
    const textToSend = messageOverride || input;
    if (!textToSend.trim()) return;
    if (!messageOverride) setInput('');
    await send(textToSend);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    setLoading(true);
    try {
      const url = threadId ? `${API_BASE}/upload?thread_id=${encodeURIComponent(threadId ?? "")}` : `${API_BASE}/upload`;
      const res = await fetch(url, {
        method: 'POST',
        headers: authHeaders(apiKey, modelName),
        body: fd,
      });
      const data = await res.json();
      if (data.path) {
        causal.refresh();
        handleSend(`I have uploaded a dataset at \`${data.path}\`. Please load it using the execute_python tool and run some basic EDA on it. Don't build a model yet.`);
      }
    } catch (e) { console.error('File upload failed', e); }
    finally { setLoading(false); if (fileInputRef.current) fileInputRef.current.value = ''; }
  };

  // Manual edits are server-authoritative: PATCH the spec straight into agent
  // state. The server diffs what changed and locks those leaf fields so the LLM
  // can't silently overwrite them. (No LLM round-trip — manual takes priority.)
  const handleApplySpec = useCallback(async (newSpec: any) => {
    // Lock ONLY the leaves the user actually changed in the editor. The widget's
    // draft starts from specWithDefaults(current), so diffing against that same
    // defaulted baseline cancels default-vs-default and yields just the edits.
    const baseline = specWithDefaults(dashboardData.model_spec);
    const lock_paths = specLeafDiff(baseline, newSpec);
    setDashboardData((prev: any) => ({ ...prev, model_spec: newSpec }));  // optimistic
    try {
      const res = await fetch(`${API_BASE}/spec/${encodeURIComponent(threadId ?? "")}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ model_spec: newSpec, lock_paths }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || 'spec update failed');
      setDashboardData((prev: any) => ({
        ...prev,
        model_spec: data.model_spec,
        locked_fields: data.locked_fields,
        pending_spec_changes: data.pending_spec_changes,
      }));
    } catch (e) {
      console.error('Apply spec failed', e);
    }
  }, [threadId, apiKey, modelName, dashboardData.model_spec, setDashboardData]);

  // Confirm / decline an LLM-proposed change to a user-locked field.
  const handleResolveSpecChange = useCallback(async (path: string, action: 'approve' | 'reject') => {
    try {
      const res = await fetch(`${API_BASE}/spec/${encodeURIComponent(threadId ?? "")}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ path, action }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || 'resolve failed');
      setDashboardData((prev: any) => ({
        ...prev,
        model_spec: data.model_spec,
        locked_fields: data.locked_fields,
        pending_spec_changes: data.pending_spec_changes,
      }));
    } catch (e) {
      console.error('Resolve spec change failed', e);
    }
  }, [threadId, apiKey, modelName, setDashboardData]);

  // Hand a locked field back to the LLM.
  const handleUnlockField = useCallback(async (path: string) => {
    try {
      const res = await fetch(`${API_BASE}/spec/${encodeURIComponent(threadId ?? "")}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ model_spec: dashboardData.model_spec || {}, unlock_paths: [path] }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || 'unlock failed');
      setDashboardData((prev: any) => ({
        ...prev,
        locked_fields: data.locked_fields,
        pending_spec_changes: data.pending_spec_changes,
      }));
    } catch (e) {
      console.error('Unlock failed', e);
    }
  }, [threadId, apiKey, modelName, dashboardData.model_spec, setDashboardData]);

  // Confirm a proposed outlier treatment from the EDA tab. Returns an error
  // string (shown inline under the action row) or null on success. The fresh
  // eda envelope from the response is merged immediately; dataset-side effects
  // (dataset_path switch, plots) arrive on the next turn/state reload.
  const handleResolveOutlierAction = useCallback(async (action: OutlierAction): Promise<string | null> => {
    if (!threadId) return 'No active session';
    try {
      const res = await fetch(`${API_BASE}/outliers/${encodeURIComponent(threadId)}/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ action_ids: [action.action_id], reason: 'Confirmed in EDA tab' }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) return data?.error || `Apply failed (HTTP ${res.status})`;
      setDashboardData(prev => ({ ...prev, eda: data.eda ?? prev.eda }));
      // Same refresh as onTurnSettled: artifacts + causal panels + workspace files.
      try {
        const arts = await fetch(`${API_BASE}/artifacts/${threadId}`).then(r => r.json());
        if (Array.isArray(arts)) setArtifacts(arts);
      } catch { /* ignore */ }
      causal.refresh();
      setWorkspaceRefreshKey(k => k + 1);
      return null;
    } catch (e) {
      return (e instanceof Error && e.message) || 'Network error applying outlier action';
    }
  }, [threadId, apiKey, modelName, setDashboardData, causal]);

  const handleRerunArtifact = (a: Artifact) => {
    if (a.kind !== 'code_snippet') return;
    const code = String(a.payload?.code ?? '');
    if (!code.trim()) return;
    handleSend(
      `Please re-run this saved code snippet using \`execute_python\`:\n\n\`\`\`python\n${code}\n\`\`\``
    );
  };

  const handleDeleteArtifact = async (id: string) => {
    await fetch(`${API_BASE}/artifacts/${id}`, { method: 'DELETE' });
    setArtifacts(prev => prev.filter(a => a.id !== id));
  };

  const handleLoadRun = async (runName: string) => {
    // Direct load — no LLM round-trip. The endpoint loads the model into the
    // session and writes model_status=completed into agent state.
    if (!runName) return;
    try {
      const resp = await fetch(`${API_BASE}/sessions/${threadId}/load-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: runName }),
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(body?.detail || 'Load failed');
      setDashboardData((prev: any) => ({
        ...prev,
        model_status: 'completed',
        summary: body?.message ?? `Model ${runName} loaded.`,
        error: undefined,
      }));
    } catch (e: any) {
      setDashboardData((prev: any) => ({ ...prev, error: String(e?.message ?? e) }));
    }
  };

  const lastAssistantHasContent = messages.length > 0 && messages[messages.length - 1].type === 'ai';
  const canRetry = !loading && messages.some(m => m.type === 'human');
  const canBack = !loading && messages.length >= 2;

  const activeSession = sessions.find(s => s.thread_id === threadId);

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-900 overflow-hidden font-sans">
      <PendingSpecChangesModal
        // Hold the modal until the turn settles: resolving mid-stream would race
        // the agent's own aupdate_state writes on this thread.
        changes={loading ? [] : (dashboardData.pending_spec_changes || [])}
        onResolve={handleResolveSpecChange}
      />
      {/* ── Shared top bar (matches main app Header) ── */}
      <header className="flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm sm:px-6 z-20">
        <button
          onClick={() => navigate('/dashboard')}
          className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-indigo-600 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          <span className="hidden sm:inline">Dashboard</span>
        </button>
        <div className="h-6 w-px bg-gray-200" />
        <h1 className="text-lg font-semibold text-gray-900 flex-1">MMM Chat</h1>
        <span className="text-sm text-gray-400 truncate max-w-48 hidden md:block">
          {activeSession?.name ?? ''}
        </span>
        <div className="h-6 w-px bg-gray-200 hidden md:block" />
        <ModelSwitcher theme="light" />
      </header>

      {/* ── Panel row ── */}
      <div className="flex flex-1 overflow-hidden">
      <SessionSidebar
        sessions={sessions}
        activeId={threadId}
        onSelect={setThreadId}
        onCreate={handleCreateSession}
        onRename={handleRenameSession}
        onDelete={handleDeleteSession}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(v => !v)}
        projects={projects}
        projectId={projectId}
        onProjectSelect={handleProjectSelect}
        onProjectCreate={handleProjectCreate}
      />

      {/* ── Left: Chat (1/3 width) ── */}
      {!rightExpanded && (
        <ChatPanel
          messages={messages}
          loading={loading}
          input={input}
          onInputChange={setInput}
          canBack={canBack}
          canRetry={canRetry}
          lastAssistantHasContent={lastAssistantHasContent}
          onBack={handleBack}
          onRetry={handleRetry}
          onClear={handleClearChat}
          onSend={handleSend}
          onStop={handleStop}
          onNavigate={setActiveTab}
          fileInputRef={fileInputRef}
          onFileUpload={handleFileUpload}
        />
      )}

      {/* ── Right: Workspace Dashboard (2/3 or full) ── */}
      <WorkspaceTabs
        rightExpanded={rightExpanded}
        onToggleExpand={() => setRightExpanded(v => !v)}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        causal={causal}
        dashboardData={dashboardData}
        artifacts={artifacts}
        pythonOutputs={pythonOutputs}
        threadId={threadId}
        apiKey={apiKey}
        modelName={modelName}
        projectId={projectId}
        workspaceRefreshKey={workspaceRefreshKey}
        chatLoading={loading}
        onApplySpec={handleApplySpec}
        onUnlockField={handleUnlockField}
        onQuickAction={handleSend}
        onRerunArtifact={handleRerunArtifact}
        onDeleteArtifact={handleDeleteArtifact}
        onLoadRun={handleLoadRun}
        onClearPython={() => setPythonOutputs([])}
        onResolveOutlierAction={handleResolveOutlierAction}
      />
      </div>
    </div>
  );
}
