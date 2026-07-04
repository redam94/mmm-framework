import { Download, ExternalLink, FileText } from 'lucide-react';
import { API_BASE_URL } from '../../api/client';

export type ReportKind =
  | 'mmm'
  | 'project'
  | 'client'
  | 'project-slides'
  | 'prefit'
  | 'defense'
  | 'client-slides';

const REPORTS: Record<ReportKind, { path: string; label: string; file: string }> = {
  mmm: { path: '/report', label: 'MMM report', file: 'mmm_report.html' },
  project: { path: '/project-report', label: 'Project report', file: 'project_report.html' },
  client: { path: '/client-report', label: 'Client report', file: 'client_report.html' },
  'project-slides': { path: '/project-slides', label: 'Slides', file: 'project_slides.html' },
  prefit: { path: '/prefit-report', label: 'Model design readout', file: 'model_design_readout.html' },
  defense: { path: '/model-defense', label: 'Model defense', file: 'model_defense.html' },
  'client-slides': { path: '/client-slides', label: 'Client slides', file: 'client_slides.html' },
};

/**
 * In-app, PER-SESSION report viewer + download (U4). The agent writes report
 * HTML per session; these endpoints take an optional `?thread_id=`, so a report
 * is scoped to its session rather than the global dev artifact. Renders a
 * sandboxed iframe preview plus download / open-in-new-tab links.
 */
export function ReportViewer({
  threadId,
  kind = 'mmm',
  height = 420,
  title,
  className,
}: {
  threadId: string | null;
  kind?: ReportKind;
  height?: number;
  title?: string;
  className?: string;
}) {
  const cfg = REPORTS[kind];
  if (!threadId) {
    return (
      <div className={className}>
        <p className="rounded-lg border border-line-200 bg-cream-50 px-3 py-4 text-sm text-ink-400">
          <FileText className="mb-1 inline h-4 w-4" /> No active session — open a
          Workspace session and generate a report to view it here.
        </p>
      </div>
    );
  }
  const q = `?thread_id=${encodeURIComponent(threadId)}`;
  const viewUrl = `${API_BASE_URL}${cfg.path}${q}`;
  const dlUrl = `${API_BASE_URL}${cfg.path}/download${q}`;

  return (
    <div className={className}>
      <div className="mb-2 flex items-center gap-3">
        <span className="text-sm font-medium text-ink-700">{title ?? cfg.label}</span>
        <a
          href={dlUrl}
          download={cfg.file}
          className="inline-flex items-center gap-1 rounded-md border border-line-300 bg-white px-2.5 py-1 text-xs text-ink-700 hover:bg-cream-100"
        >
          <Download className="h-3.5 w-3.5" /> Download
        </a>
        <a
          href={viewUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 rounded-md border border-line-300 bg-white px-2.5 py-1 text-xs text-ink-700 hover:bg-cream-100"
        >
          <ExternalLink className="h-3.5 w-3.5" /> Open
        </a>
      </div>
      <iframe
        src={viewUrl}
        title={title ?? cfg.label}
        style={{ height }}
        className="w-full rounded-lg border border-line-200 bg-white"
        sandbox="allow-scripts allow-same-origin allow-popups"
      />
    </div>
  );
}
