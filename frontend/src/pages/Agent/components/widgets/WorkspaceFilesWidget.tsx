import { useEffect, useState } from 'react';
import { Download, File as FileIcon, FolderOpen } from 'lucide-react';
import { PanelShellLite } from '../common/PanelShell';
import { API_BASE, authHeaders, fmtBytes } from '../../constants';
import type { WorkspaceFile } from '../../types';

export function WorkspaceFilesWidget({ threadId, apiKey, modelName, refreshKey }: {
  threadId: string | null;
  apiKey: string | null;
  modelName: string | null;
  refreshKey: number;
}) {
  const [files, setFiles] = useState<WorkspaceFile[]>([]);

  useEffect(() => {
    if (!threadId) { setFiles([]); return; }
    let cancelled = false;
    fetch(`${API_BASE}/workspace/${encodeURIComponent(threadId ?? "")}/files`, { headers: authHeaders(apiKey, modelName) })
      .then(r => r.json())
      .then(data => { if (!cancelled) setFiles(Array.isArray(data?.files) ? data.files : []); })
      .catch(() => { if (!cancelled) setFiles([]); });
    return () => { cancelled = true; };
  }, [threadId, apiKey, modelName, refreshKey]);

  if (files.length === 0) {
    return (
      <PanelShellLite title="Workspace Outputs" icon={<FolderOpen size={16} className="text-teal-600" />} color="teal">
        <p className="text-sm text-ink-300 italic">No generated files yet. When the agent writes reports, CSVs, or PNGs via <code className="text-xs bg-cream-100 px-1 rounded">execute_python</code>, they appear here for download.</p>
      </PanelShellLite>
    );
  }

  return (
    <PanelShellLite title={`Workspace Outputs (${files.length})`} icon={<FolderOpen size={16} className="text-teal-600" />} color="teal">
      <div className="space-y-2">
        {files.map(f => (
          <div key={f.id} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-line-200 bg-white">
            <FileIcon size={14} className="text-teal-600 shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className="text-sm font-semibold text-ink-900 truncate">{f.name}</span>
                <span className="text-[10px] uppercase tracking-wider text-teal-700 bg-teal-50 rounded px-1.5 py-0.5 border border-teal-200">{f.kind}</span>
                <span className="text-[10px] text-ink-300">{fmtBytes(f.size_bytes)}</span>
              </div>
              <p className="text-[11px] text-ink-300 font-mono mt-0.5 truncate">{f.path}</p>
            </div>
            <a
              href={`${API_BASE}/files/${f.id}/download`}
              download
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-cream-100 hover:bg-gray-200 text-ink-700 text-xs font-semibold transition-colors border border-line-200 shrink-0"
            >
              <Download size={12} /> Save
            </a>
          </div>
        ))}
      </div>
    </PanelShellLite>
  );
}
