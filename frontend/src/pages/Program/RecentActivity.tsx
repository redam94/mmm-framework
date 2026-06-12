import { useNavigate } from 'react-router-dom';
import { ArrowRight, Clock, MessageSquareText, Package } from 'lucide-react';
import { useSessions } from '../../api/hooks/useSessions';
import { usePortfolio } from '../../api/hooks/usePortfolio';

function formatTs(ts: number | null | undefined): string {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const diffMin = Math.floor((Date.now() - d.getTime()) / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH < 24) return `${diffH}h ago`;
  return d.toLocaleDateString();
}

/** Condensed sessions + model runs feed (the Workspace's recent footprint). */
export function RecentActivity({ projectId }: { projectId: string | null }) {
  const navigate = useNavigate();
  const { data: sessionsData } = useSessions(
    projectId ? { project_id: projectId, limit: 5 } : { limit: 5 },
  );
  const { data: portfolio } = usePortfolio(projectId);
  const sessions = sessionsData?.sessions ?? [];
  const runs = (portfolio?.model_runs ?? []).slice(0, 5);

  return (
    <section className="rounded-lg border border-line-200 bg-white p-5 shadow-sm">
      <div className="flex items-baseline justify-between">
        <h2 className="font-display text-lg font-semibold text-ink-900">Recent activity</h2>
        <button
          onClick={() => navigate('/workspace')}
          className="flex items-center gap-1 text-xs font-medium text-sage-800 hover:underline"
        >
          New session <ArrowRight className="h-3 w-3" />
        </button>
      </div>

      <div className="mt-3 space-y-4">
        <div>
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-ink-300">
            Model runs
          </h3>
          {runs.length === 0 ? (
            <p className="mt-1.5 text-sm text-ink-300">No fits yet.</p>
          ) : (
            <ul className="mt-1 divide-y divide-line-200">
              {runs.map((r) => (
                <li key={r.model_id} className="flex items-center gap-2.5 py-2">
                  <Package className="h-4 w-4 shrink-0 text-steel-600" strokeWidth={1.75} />
                  <span className="flex-1 truncate text-sm text-ink-700 num">{r.run_name}</span>
                  <span className="text-xs text-ink-300">{formatTs(r.created_at)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-ink-300">
            Sessions
          </h3>
          {sessions.length === 0 ? (
            <p className="mt-1.5 text-sm text-ink-300">No sessions yet.</p>
          ) : (
            <ul className="mt-1 divide-y divide-line-200">
              {sessions.map((s: any) => (
                <li key={s.thread_id} className="group flex items-center gap-2.5 py-2">
                  <MessageSquareText
                    className="h-4 w-4 shrink-0 text-sage-700"
                    strokeWidth={1.75}
                  />
                  <span className="min-w-0 flex-1 truncate text-sm text-ink-700">{s.name}</span>
                  <span className="flex items-center gap-1 text-xs text-ink-300">
                    <Clock className="h-3 w-3" />
                    {formatTs(s.updated_at)}
                  </span>
                  <button
                    onClick={() => navigate(`/workspace?session=${s.thread_id}`)}
                    className="text-xs font-medium text-sage-800 opacity-0 transition-opacity hover:underline group-hover:opacity-100"
                  >
                    Resume
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
