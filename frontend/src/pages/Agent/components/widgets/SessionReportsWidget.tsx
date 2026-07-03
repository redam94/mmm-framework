import type { ReactNode } from 'react';
import {
  BookOpen, Download, ExternalLink, FileText, Layers,
  Presentation, ScrollText, ShieldCheck,
} from 'lucide-react';
import { DashWidget } from '../common/DashWidget';
import { API_BASE } from '../../constants';
import type { Artifact, DashboardData } from '../../types';

// Every deliverable the agent can generate for a session, with its serve
// endpoints. All report endpoints accept ?thread_id= so each row is scoped to
// THIS oracle session (reports are stored per-session in the workspace).
interface DocKind {
  kind: string;      // artifact kind persisted by the chat stream
  dashKey: string;   // dashboard_data key set by the generating tool
  label: string;
  desc: string;
  view?: string;     // inline-viewable endpoint (HTML); absent for binary docs
  download: string;
  file: string;      // suggested filename for the download attribute
  icon: ReactNode;
}

const DOC_KINDS: DocKind[] = [
  {
    kind: 'report', dashKey: 'report_path',
    label: 'MMM technical report',
    desc: 'Full diagnostics, ROI, decomposition and posterior checks.',
    view: '/report', download: '/report/download', file: 'mmm_report.html',
    icon: <FileText size={18} />,
  },
  {
    kind: 'client_report', dashKey: 'client_report_path',
    label: 'Client readout',
    desc: 'The editorial Media Performance Readout for clients and planners.',
    view: '/client-report', download: '/client-report/download', file: 'mmm_client_report.html',
    icon: <BookOpen size={18} />,
  },
  {
    kind: 'prefit_report', dashKey: 'prefit_report_path',
    label: 'Model design readout (pre-fit)',
    desc: 'Pre-registered assumptions, priors, prior-predictive checks and SBC.',
    view: '/prefit-report', download: '/prefit-report/download', file: 'model_design_readout.html',
    icon: <ScrollText size={18} />,
  },
  {
    kind: 'model_defense', dashKey: 'model_defense_path',
    label: 'Model defense',
    desc: 'The causal-rigor evidence behind the fitted model.',
    view: '/model-defense', download: '/model-defense/download', file: 'mmm_model_defense.html',
    icon: <ShieldCheck size={18} />,
  },
  {
    kind: 'project_report', dashKey: 'project_report_path',
    label: 'Project report',
    desc: 'Project findings across the session.',
    view: '/project-report', download: '/project-report/download', file: 'mmm_project_report.html',
    icon: <BookOpen size={18} />,
  },
  {
    kind: 'project_slides', dashKey: 'project_slides_path',
    label: 'Project slides',
    desc: 'Reveal.js slideshow of the project findings.',
    view: '/project-slides', download: '/project-slides/download', file: 'mmm_project_slides.html',
    icon: <Layers size={18} />,
  },
  {
    kind: 'client_slides', dashKey: 'client_slides_path',
    label: 'Client slides',
    desc: 'Client-ready slideshow (no MCMC internals).',
    view: '/client-slides', download: '/client-slides/download', file: 'mmm_client_slides.html',
    icon: <Layers size={18} />,
  },
  {
    kind: 'slide_deck', dashKey: 'slide_deck_path',
    label: 'Slide deck (PowerPoint)',
    desc: 'The .pptx deck rendered from the fitted model.',
    download: '/slide-deck/download', file: 'mmm_slide_deck.pptx',
    icon: <Presentation size={18} />,
  },
];

/** Per-session inventory of every report/deck the agent has generated, each
 * viewable and downloadable from the oracle. Availability is the union of the
 * session's persisted artifacts (survives reload) and the live dashboard keys
 * (present the moment a tool finishes, before the artifact lands). */
export function SessionReportsWidget({ dashboardData, artifacts, threadId }: {
  dashboardData: DashboardData;
  artifacts: Artifact[];
  threadId: string | null;
}) {
  const rq = threadId ? `?thread_id=${encodeURIComponent(threadId)}` : '';
  const latestByKind = new Map<string, Artifact>();
  for (const a of artifacts) {
    const prev = latestByKind.get(a.kind);
    if (!prev || a.created_at > prev.created_at) latestByKind.set(a.kind, a);
  }

  const rows = DOC_KINDS
    .map((d) => {
      const art = latestByKind.get(d.kind);
      const dashPath = (dashboardData as Record<string, unknown>)[d.dashKey];
      if (!art && !dashPath) return null;
      return { d, art };
    })
    .filter((r): r is { d: DocKind; art: Artifact | undefined } => r !== null);

  if (rows.length === 0) return null;

  return (
    <DashWidget
      title={`Reports & documents (${rows.length})`}
      dotColor="bg-violet-500"
      color="violet"
      icon={<FileText size={15} className="text-violet-600 shrink-0" />}
    >
      <div className="space-y-2.5">
        {rows.map(({ d, art }) => (
          <div key={d.kind} className="flex items-center gap-3 p-4 bg-white rounded-xl border border-line-200 shadow-sm">
            <div className="w-10 h-10 rounded-xl bg-violet-50 flex items-center justify-center text-violet-600 shrink-0">
              {d.icon}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-ink-900">{d.label}</p>
              <p className="text-xs text-ink-400 truncate">
                {d.desc}
                {art && (
                  <span className="text-ink-300"> · {new Date(art.created_at * 1000).toLocaleString()}</span>
                )}
              </p>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              {d.view && (
                <a
                  href={`${API_BASE}${d.view}${rq}`}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-semibold transition-colors"
                >
                  <ExternalLink size={12} /> Open
                </a>
              )}
              <a
                href={`${API_BASE}${d.download}${rq}`}
                download={d.file}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-cream-100 hover:bg-gray-200 text-ink-700 text-xs font-semibold transition-colors border border-line-200"
              >
                <Download size={12} /> Download
              </a>
            </div>
          </div>
        ))}
        <p className="text-xs text-ink-300 text-center pt-1">
          All documents are stored in this session&apos;s workspace and stay downloadable after reload.
        </p>
      </div>
    </DashWidget>
  );
}
