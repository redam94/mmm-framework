import React, { useState } from 'react';
import {
  BarChart2, BookOpen, ChevronDown, ChevronRight, Copy, Download,
  ExternalLink, FileCode, Layers, Play, Trash2,
} from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { DashWidget } from '../common/DashWidget';
import { API_BASE } from '../../constants';
import { truncate } from '../../utils/text';
import type { Artifact } from '../../types';

// ─── ProjectDocsWidget ───────────────────────────────────────────────────────

function ProjectDocsWidget({ artifacts, onDelete }: {
  artifacts: Artifact[];
  onDelete: (id: string) => void;
}) {
  const reports = artifacts.filter(a => a.kind === 'project_report');
  const slides  = artifacts.filter(a => a.kind === 'project_slides');
  if (reports.length === 0 && slides.length === 0) return null;

  const latest = (arr: Artifact[]) => arr.sort((a, b) => b.created_at - a.created_at)[0];
  const reportArt = latest(reports);
  const slidesArt = latest(slides);

  const DocCard = ({ art, icon, label, viewUrl, downloadUrl }: {
    art: Artifact; icon: React.ReactNode; label: string;
    viewUrl: string; downloadUrl: string;
  }) => (
    <div className="flex items-center gap-3 p-4 bg-white rounded-xl border border-gray-200 shadow-sm">
      <div className="w-10 h-10 rounded-xl bg-indigo-50 flex items-center justify-center text-indigo-600 shrink-0">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-gray-900">{label}</p>
        <p className="text-xs text-gray-400 truncate">{new Date(art.created_at * 1000).toLocaleString()}</p>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <a
          href={viewUrl}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-colors"
        >
          <ExternalLink size={12} /> Open
        </a>
        <a
          href={downloadUrl}
          download
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-semibold transition-colors border border-gray-200"
        >
          <Download size={12} /> Save
        </a>
        <button
          onClick={() => onDelete(art.id)}
          className="p-1.5 rounded-lg hover:bg-red-50 text-red-400 hover:text-red-600 transition-colors"
          title="Remove"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );

  return (
    <DashWidget
      title="Project Documents"
      dotColor="bg-indigo-500"
      color="indigo"
      icon={<BookOpen size={15} className="text-indigo-600 shrink-0" />}
    >
      <div className="space-y-2.5">
        {reportArt && (
          <DocCard
            art={reportArt}
            icon={<BookOpen size={18} />}
            label="Project Report (HTML)"
            viewUrl={`${API_BASE}/project-report`}
            downloadUrl={`${API_BASE}/project-report/download`}
          />
        )}
        {slidesArt && (
          <DocCard
            art={slidesArt}
            icon={<Layers size={18} />}
            label="Presentation Slides (Reveal.js)"
            viewUrl={`${API_BASE}/project-slides`}
            downloadUrl={`${API_BASE}/project-slides/download`}
          />
        )}
        {(reports.length > 1 || slides.length > 1) && (
          <p className="text-xs text-gray-400 text-center pt-1">Showing latest version of each. Older versions available in history.</p>
        )}
      </div>
    </DashWidget>
  );
}

// ─── ModelRunsWidget ─────────────────────────────────────────────────────────

function ModelRunsWidget({
  runs,
  onLoad,
  onDelete,
}: {
  runs: Artifact[];
  onLoad: (runName: string) => void;
  onDelete: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (runs.length === 0) return null;

  return (
    <DashWidget
      title={`Model Run History (${runs.length})`}
      dotColor="bg-emerald-500"
      color="emerald"
      icon={<BarChart2 size={15} className="text-emerald-600 shrink-0" />}
    >
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full text-xs">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-3 py-2 text-left font-semibold text-emerald-700 border-b border-gray-200">Run</th>
              <th className="px-3 py-2 text-left font-semibold text-emerald-700 border-b border-gray-200">Timestamp</th>
              <th className="px-3 py-2 text-left font-semibold text-emerald-700 border-b border-gray-200">KPI</th>
              <th className="px-3 py-2 text-center font-semibold text-emerald-700 border-b border-gray-200">Channels</th>
              <th className="px-3 py-2 text-center font-semibold text-emerald-700 border-b border-gray-200">Draws</th>
              <th className="px-3 py-2 text-center font-semibold text-emerald-700 border-b border-gray-200">Actions</th>
            </tr>
          </thead>
          <tbody>
            {runs.map(a => {
              const r = a.payload ?? {};
              const isExp = expanded === a.id;
              const ts = r.timestamp_iso
                ? new Date(r.timestamp_iso).toLocaleString()
                : new Date(a.created_at * 1000).toLocaleString();
              const channels: string[] = r.channels ?? [];
              const draws = r.inference?.draws ?? r.draws ?? '—';
              return (
                <React.Fragment key={a.id}>
                  <tr
                    className="even:bg-gray-50 hover:bg-emerald-50 transition-colors cursor-pointer"
                    onClick={() => setExpanded(isExp ? null : a.id)}
                  >
                    <td className="px-3 py-2 text-gray-700 border-b border-gray-100 font-mono font-semibold">
                      <span className="flex items-center gap-1.5">
                        {isExp ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                        {r.run_name ?? a.id.slice(0, 8)}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-500 border-b border-gray-100">{ts}</td>
                    <td className="px-3 py-2 text-gray-700 border-b border-gray-100 font-medium">{r.kpi ?? '—'}</td>
                    <td className="px-3 py-2 text-center text-gray-600 border-b border-gray-100">{channels.length}</td>
                    <td className="px-3 py-2 text-center text-gray-600 border-b border-gray-100">{draws}</td>
                    <td className="px-3 py-2 text-center border-b border-gray-100">
                      <div className="flex items-center justify-center gap-1.5" onClick={e => e.stopPropagation()}>
                        <button
                          onClick={() => onLoad(r.run_name)}
                          className="px-2 py-1 rounded bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 font-semibold text-[11px] flex items-center gap-1"
                          title="Load this run into the agent"
                        >
                          <Play size={10} /> Load
                        </button>
                        <button
                          onClick={() => onDelete(a.id)}
                          className="p-1 rounded hover:bg-red-50 text-red-400 hover:text-red-600"
                          title="Remove from history"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </td>
                  </tr>
                  {isExp && (
                    <tr>
                      <td colSpan={6} className="px-4 py-3 bg-emerald-50/60 border-b border-gray-200">
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Channels</p>
                            <div className="flex flex-wrap gap-1">
                              {channels.length > 0
                                ? channels.map(ch => <span key={ch} className="px-2 py-0.5 rounded-full bg-white border border-emerald-200 text-emerald-700 text-[11px]">{ch}</span>)
                                : <span className="text-gray-400">none</span>}
                            </div>
                          </div>
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Controls</p>
                            <div className="flex flex-wrap gap-1">
                              {(r.controls ?? []).length > 0
                                ? (r.controls as string[]).map(c => <span key={c} className="px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-600 text-[11px]">{c}</span>)
                                : <span className="text-gray-400">none</span>}
                            </div>
                          </div>
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Inference</p>
                            <p className="text-gray-700 font-mono">
                              {r.inference?.chains ?? '?'} chains × {r.inference?.draws ?? '?'} draws
                              {r.inference?.tune ? ` (${r.inference.tune} tune)` : ''}
                            </p>
                          </div>
                          <div>
                            <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Trend / Seasonality</p>
                            <p className="text-gray-700">{r.trend ?? '—'}</p>
                          </div>
                          {r.model_path && (
                            <div className="col-span-2">
                              <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Saved path</p>
                              <p className="text-gray-500 font-mono text-[11px] truncate">{r.model_path}</p>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </DashWidget>
  );
}

// ─── ArtifactsPanel ──────────────────────────────────────────────────────────

export function ArtifactsPanel({ artifacts, onRerun, onDelete, onLoadRun }: {
  artifacts: Artifact[];
  onRerun: (a: Artifact) => void;
  onDelete: (id: string) => void;
  onLoadRun: (runName: string) => void;
}) {
  if (artifacts.length === 0) return null;
  const codeArtifacts = artifacts.filter(a => a.kind === 'code_snippet');
  const reportArtifacts = artifacts.filter(a => a.kind === 'report');
  const modelRunArtifacts = artifacts.filter(a => a.kind === 'model_run');
  const projectDocArtifacts = artifacts.filter(a => a.kind === 'project_report' || a.kind === 'project_slides');

  return (
    <div className="space-y-4">
      <ProjectDocsWidget artifacts={projectDocArtifacts} onDelete={onDelete} />
      <ModelRunsWidget runs={modelRunArtifacts} onLoad={onLoadRun} onDelete={onDelete} />
      {(codeArtifacts.length > 0 || reportArtifacts.length > 0) && (
        <DashWidget title={`Code & Reports (${codeArtifacts.length + reportArtifacts.length})`} dotColor="bg-amber-500" color="amber">
          <div className="space-y-3">
            {codeArtifacts.length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5">Code Snippets</p>
                <div className="space-y-2">
                  {codeArtifacts.map(a => {
                    const code = String(a.payload?.code ?? '');
                    return (
                      <div key={a.id} className="rounded-lg border border-gray-200 bg-gray-50 overflow-hidden">
                        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-200 bg-white">
                          <FileCode size={12} className="text-amber-600 shrink-0" />
                          <span className="text-[11px] text-gray-500 flex-1 truncate">
                            {new Date(a.created_at * 1000).toLocaleString()}
                          </span>
                          <button
                            onClick={() => navigator.clipboard.writeText(code)}
                            className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-800"
                            title="Copy"
                          ><Copy size={11} /></button>
                          <a
                            href={`${API_BASE}/artifacts/${a.id}/download`}
                            download
                            className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-800"
                            title="Download"
                          ><Download size={11} /></a>
                          <button
                            onClick={() => onRerun(a)}
                            className="p-1 rounded hover:bg-indigo-50 text-indigo-600"
                            title="Rerun"
                          ><Play size={11} /></button>
                          <button
                            onClick={() => onDelete(a.id)}
                            className="p-1 rounded hover:bg-red-50 text-red-500"
                            title="Delete"
                          ><Trash2 size={11} /></button>
                        </div>
                        <div className="overflow-x-auto max-h-40 bg-[#fafafa]">
                          <SyntaxHighlighter
                            language="python"
                            style={oneLight}
                            PreTag="div"
                            customStyle={{ margin: 0, padding: '0.5rem 0.75rem', fontSize: '0.6875rem', background: '#fafafa' }}
                            codeTagProps={{ style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' } }}
                          >
                            {truncate(code, 600)}
                          </SyntaxHighlighter>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {reportArtifacts.length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5">Reports</p>
                <div className="space-y-1.5">
                  {reportArtifacts.map(a => (
                    <div key={a.id} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
                      <ExternalLink size={12} className="text-violet-600 shrink-0" />
                      <span className="text-xs text-gray-700 flex-1 truncate font-mono">{a.payload?.path ?? a.id}</span>
                      <a
                        href={`${API_BASE}/artifacts/${a.id}/download`}
                        download
                        className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-800"
                        title="Download"
                      ><Download size={11} /></a>
                      <button onClick={() => onDelete(a.id)} className="p-1 rounded hover:bg-red-50 text-red-500" title="Delete">
                        <Trash2 size={11} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </DashWidget>
      )}
    </div>
  );
}
