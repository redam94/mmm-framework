import { useState } from 'react';
import { Check, FolderOpen, Loader2, Plus } from 'lucide-react';
import type { Project } from '../../types';

export function ProjectPicker({ projects, projectId, onSelect, onCreate }: {
  projects: Project[];
  projectId: string | null;
  onSelect: (id: string) => void;
  onCreate: (name: string, description?: string) => Promise<void> | void;
}) {
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);

  // Degraded mode: /projects unavailable. Hide the picker entirely.
  if (projects.length === 0 && projectId == null) return null;

  const submit = async () => {
    const n = name.trim();
    if (!n) return;
    setBusy(true);
    try { await onCreate(n); setName(''); setCreating(false); }
    finally { setBusy(false); }
  };

  return (
    <div className="px-3 py-2.5 border-b border-gray-200 bg-gray-50/60">
      <div className="flex items-center gap-1 mb-1.5">
        <FolderOpen size={12} className="text-indigo-500 shrink-0" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider flex-1">Project</span>
        <button
          onClick={() => setCreating(v => !v)}
          className="p-1 rounded hover:bg-indigo-50 text-indigo-600"
          title="New project"
        ><Plus size={12} /></button>
      </div>
      {creating ? (
        <div className="flex items-center gap-1">
          <input
            autoFocus value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') submit(); if (e.key === 'Escape') { setCreating(false); setName(''); } }}
            placeholder="Project name…"
            disabled={busy}
            className="flex-1 text-xs bg-white border border-indigo-300 rounded px-1.5 py-1 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          />
          <button onClick={submit} disabled={busy || !name.trim()}
            className="p-1 rounded bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40" title="Create">
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
          </button>
        </div>
      ) : (
        <select
          value={projectId ?? ''}
          onChange={e => onSelect(e.target.value)}
          className="w-full text-xs bg-white border border-gray-200 rounded-lg px-2 py-1.5 text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {projects.map(p => (
            <option key={p.project_id} value={p.project_id}>
              {p.name}{typeof p.session_count === 'number' ? ` (${p.session_count})` : ''}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
