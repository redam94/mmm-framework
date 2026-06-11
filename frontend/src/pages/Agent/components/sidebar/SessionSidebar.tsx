import { useState } from 'react';
import { ChevronRight, MessagesSquare, Pencil, Plus, Trash2 } from 'lucide-react';
import { ProjectPicker } from './ProjectPicker';
import type { Project, Session } from '../../types';

export function SessionSidebar({
  sessions, activeId, onSelect, onCreate, onRename, onDelete, collapsed, onToggle,
  projects, projectId, onProjectSelect, onProjectCreate,
}: {
  sessions: Session[]; activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
  collapsed: boolean;
  onToggle: () => void;
  projects: Project[];
  projectId: string | null;
  onProjectSelect: (id: string) => void;
  onProjectCreate: (name: string, description?: string) => Promise<void> | void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');

  if (collapsed) {
    return (
      <div className="w-12 border-r border-gray-200 bg-white flex flex-col items-center py-3 gap-2 shrink-0">
        <button onClick={onToggle} className="p-2 rounded-lg hover:bg-gray-100 text-gray-500" title="Show sessions">
          <MessagesSquare size={16} />
        </button>
        <button onClick={onCreate} className="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white" title="New session">
          <Plus size={16} />
        </button>
      </div>
    );
  }

  return (
    <div className="w-56 border-r border-gray-200 bg-white flex flex-col shrink-0">
      <ProjectPicker
        projects={projects}
        projectId={projectId}
        onSelect={onProjectSelect}
        onCreate={onProjectCreate}
      />
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Sessions</span>
        <div className="flex items-center gap-1">
          <button onClick={onCreate} className="p-1.5 rounded-md hover:bg-indigo-50 text-indigo-600" title="New session">
            <Plus size={14} />
          </button>
          <button onClick={onToggle} className="p-1.5 rounded-md hover:bg-gray-100 text-gray-400" title="Collapse">
            <ChevronRight size={14} className="rotate-180" />
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto py-2">
        {sessions.length === 0 && (
          <div className="px-3 py-6 text-xs text-gray-400 text-center">No sessions yet.</div>
        )}
        {sessions.map(s => {
          const active = s.thread_id === activeId;
          const isEditing = editingId === s.thread_id;
          return (
            <div key={s.thread_id} className={`group mx-2 mb-1 rounded-lg ${active ? 'bg-indigo-50 border border-indigo-200' : 'hover:bg-gray-50 border border-transparent'}`}>
              <div className="flex items-center gap-1 px-2 py-1.5">
                {isEditing ? (
                  <input
                    autoFocus value={editName}
                    onChange={e => setEditName(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') { onRename(s.thread_id, editName.trim() || s.name); setEditingId(null); }
                      if (e.key === 'Escape') setEditingId(null);
                    }}
                    onBlur={() => { onRename(s.thread_id, editName.trim() || s.name); setEditingId(null); }}
                    className="flex-1 text-xs bg-white border border-indigo-300 rounded px-1.5 py-1 focus:outline-none"
                  />
                ) : (
                  <button onClick={() => onSelect(s.thread_id)} className="flex-1 text-left text-xs text-gray-700 truncate">
                    {s.name}
                  </button>
                )}
                {!isEditing && (
                  <>
                    <button
                      onClick={() => { setEditName(s.name); setEditingId(s.thread_id); }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded text-gray-400 hover:text-indigo-600"
                      title="Rename"
                    ><Pencil size={11} /></button>
                    <button
                      onClick={() => { if (confirm(`Delete "${s.name}"?`)) onDelete(s.thread_id); }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded text-gray-400 hover:text-red-600"
                      title="Delete"
                    ><Trash2 size={11} /></button>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
