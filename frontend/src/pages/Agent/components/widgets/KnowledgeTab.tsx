import React, { useCallback, useEffect, useRef, useState } from 'react';
import { BrainCircuit, FileText, Loader2, Search, Trash2, UploadCloud } from 'lucide-react';
import { EmptyTabState } from '../common/EmptyTabState';
import { PanelShellLite } from '../common/PanelShell';
import { API_BASE, authHeaders, fmtBytes } from '../../constants';
import type { KbDocument, KbSearchResult } from '../../types';

const KB_STATUS_STYLE: Record<string, string> = {
  ready:   'bg-emerald-50 text-emerald-700 border-emerald-200',
  pending: 'bg-amber-50 text-amber-700 border-amber-200',
  error:   'bg-red-50 text-red-700 border-red-200',
};

export function KnowledgeTab({ projectId, apiKey, modelName }: {
  projectId: string | null;
  apiKey: string | null;
  modelName: string | null;
}) {
  const [docs, setDocs] = useState<KbDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<KbSearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    if (!projectId) { setDocs([]); return; }
    try {
      const data = await fetch(`${API_BASE}/projects/${projectId}/kb`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json());
      setDocs(Array.isArray(data?.documents) ? data.documents : []);
    } catch { /* leave as-is */ }
  }, [projectId, apiKey, modelName]);

  useEffect(() => {
    if (!projectId) { setDocs([]); return; }
    setLoading(true);
    fetchDocs().finally(() => setLoading(false));
  }, [projectId, fetchDocs]);

  // Poll while any document is still ingesting (pending → ready/error).
  useEffect(() => {
    if (!projectId) return;
    if (!docs.some(d => d.status === 'pending')) return;
    const t = setInterval(fetchDocs, 3000);
    return () => clearInterval(t);
  }, [projectId, docs, fetchDocs]);

  const uploadFile = useCallback(async (file: File) => {
    if (!projectId) return;
    setError(null);
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      // NOTE: do not set Content-Type — the browser sets the multipart boundary.
      const res = await fetch(`${API_BASE}/projects/${projectId}/kb`, {
        method: 'POST', headers: authHeaders(apiKey, modelName), body: fd,
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        setError(e?.detail ?? e?.error ?? `Upload failed (${res.status})`);
      }
      await fetchDocs();
    } catch {
      setError('Upload failed — is the API running?');
    } finally {
      setUploading(false);
    }
  }, [projectId, apiKey, modelName, fetchDocs]);

  const onPickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) uploadFile(f);
    if (fileRef.current) fileRef.current.value = '';
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) uploadFile(f);
  };

  const deleteDoc = async (id: string) => {
    if (!confirm('Remove this document from the knowledge base?')) return;
    await fetch(`${API_BASE}/kb/${id}`, { method: 'DELETE', headers: authHeaders(apiKey, modelName) });
    fetchDocs();
  };

  const runSearch = async () => {
    const q = query.trim();
    if (!q || !projectId) return;
    setSearching(true);
    setError(null);
    try {
      const params = new URLSearchParams({ q, k: '6' });
      const data = await fetch(`${API_BASE}/projects/${projectId}/kb/search?${params}`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json());
      setResults(Array.isArray(data?.results) ? data.results : []);
    } catch {
      setError('Search failed.');
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  if (!projectId) {
    return (
      <EmptyTabState
        icon={<BrainCircuit size={28} />}
        title="No project selected"
        hint="Select or create a project in the sidebar to manage its knowledge base."
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* Upload zone */}
      <PanelShellLite title="Knowledge Base" icon={<BrainCircuit size={16} className="text-indigo-600" />} color="indigo">
        <input ref={fileRef} type="file" className="hidden" onChange={onPickFile}
          accept=".txt,.md,.markdown,.csv,.pdf,.docx,.xlsx" />
        <div
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`flex flex-col items-center justify-center gap-2 py-8 px-4 rounded-xl border-2 border-dashed cursor-pointer transition-colors ${
            dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-line-200 hover:border-indigo-300 hover:bg-cream-100'
          }`}
        >
          {uploading ? (
            <><Loader2 size={22} className="text-indigo-500 animate-spin" />
              <p className="text-sm text-ink-400">Uploading & ingesting…</p></>
          ) : (
            <><UploadCloud size={22} className="text-indigo-400" />
              <p className="text-sm text-ink-600 font-medium">Drop a file or click to upload</p>
              <p className="text-xs text-ink-300">txt · md · csv · pdf · docx · xlsx</p></>
          )}
        </div>
        {error && <p className="mt-2 text-xs text-red-600 bg-red-50 border border-red-200 rounded px-3 py-1.5">{error}</p>}

        {/* Document list */}
        <div className="mt-4 space-y-2">
          {loading && docs.length === 0 ? (
            <div className="flex items-center justify-center py-6 text-ink-300"><Loader2 size={18} className="animate-spin" /></div>
          ) : docs.length === 0 ? (
            <p className="text-sm text-ink-300 italic text-center py-2">No documents yet. Upload context files the agent can look up.</p>
          ) : docs.map(d => (
            <div key={d.id} className="flex items-start gap-2 px-3 py-2 rounded-lg border border-line-200 bg-white">
              <FileText size={14} className="text-indigo-600 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="text-sm font-semibold text-ink-900 truncate">{d.name}</span>
                  <span className="text-[10px] uppercase tracking-wider text-indigo-700 bg-indigo-50 rounded px-1.5 py-0.5 border border-indigo-200">{d.kind}</span>
                  <span className="text-[10px] text-ink-300">{fmtBytes(d.size_bytes)}</span>
                  {d.n_chunks > 0 && <span className="text-[10px] text-ink-300">{d.n_chunks} chunk{d.n_chunks !== 1 ? 's' : ''}</span>}
                  <span className={`text-[10px] uppercase tracking-wider font-semibold rounded border px-1.5 py-0.5 ${KB_STATUS_STYLE[d.status] ?? KB_STATUS_STYLE.error}`}>
                    {d.status === 'pending' && <Loader2 size={9} className="inline animate-spin mr-1" />}
                    {d.status}
                  </span>
                </div>
              </div>
              <button onClick={() => deleteDoc(d.id)} className="p-1 rounded text-ink-300 hover:text-red-500 shrink-0" title="Delete">
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      </PanelShellLite>

      {/* Search */}
      <PanelShellLite title="Search Knowledge Base" icon={<Search size={16} className="text-teal-600" />} color="teal">
        <div className="flex items-center gap-2">
          <input
            type="text" value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && runSearch()}
            placeholder="Search the knowledge base…"
            className="flex-1 text-sm border border-line-200 rounded-lg px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-teal-400"
          />
          <button onClick={runSearch} disabled={searching || !query.trim()}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-teal-600 text-white text-sm font-medium hover:bg-teal-500 disabled:opacity-40">
            {searching ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />} Search
          </button>
        </div>
        {results != null && (
          <div className="mt-3 space-y-2">
            {results.length === 0 ? (
              <p className="text-sm text-ink-300 italic">No matches found.</p>
            ) : results.map((r, i) => (
              <div key={i} className="rounded-lg border border-line-200 bg-cream-50 px-3 py-2">
                <div className="flex items-baseline gap-2 mb-1">
                  <FileText size={12} className="text-teal-600 shrink-0" />
                  <span className="text-xs font-semibold text-ink-700 truncate flex-1">{r.document}</span>
                  <span className="text-[10px] text-ink-300">#{r.chunk_index}</span>
                  <span className="text-[10px] font-mono text-teal-600 bg-teal-50 rounded px-1.5 py-0.5 border border-teal-200">{r.score.toFixed(3)}</span>
                </div>
                <p className="text-xs text-ink-600 leading-relaxed whitespace-pre-wrap line-clamp-6">{r.text}</p>
              </div>
            ))}
          </div>
        )}
      </PanelShellLite>
    </div>
  );
}
