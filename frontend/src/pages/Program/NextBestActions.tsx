import { useNavigate } from 'react-router-dom';
import {
  ArrowRight,
  CheckCircle2,
  FlaskConical,
  History,
  Package,
  RefreshCw,
  Scale,
} from 'lucide-react';
import { usePortfolio } from '../../api/hooks/usePortfolio';
import type { NextAction } from '../../api/services/portfolioService';

type ActionType = NextAction['type'] | 'retest';

const ACTION_STYLE: Record<ActionType, { icon: typeof FlaskConical; ring: string; iconColor: string }> = {
  calibrate: { icon: Scale, ring: 'border-rust-600/30 bg-rust-100/40', iconColor: 'text-rust-600' },
  refresh: { icon: RefreshCw, ring: 'border-gold-300 bg-gold-100/40', iconColor: 'text-gold-600' },
  fit: { icon: Package, ring: 'border-steel-300 bg-steel-100/40', iconColor: 'text-steel-600' },
  experiment: { icon: FlaskConical, ring: 'border-sage-300 bg-sage-100/40', iconColor: 'text-sage-700' },
  retest: { icon: History, ring: 'border-gold-300 bg-gold-100/40', iconColor: 'text-gold-600' },
};

function UrgencyChip({ urgency }: { urgency: NextAction['urgency'] }) {
  const cls =
    urgency === 'high'
      ? 'bg-rust-100 text-rust-700'
      : urgency === 'medium'
        ? 'bg-gold-100 text-gold-700'
        : 'bg-cream-200 text-ink-600';
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${cls}`}>
      {urgency}
    </span>
  );
}

/** The program's queue: what the loop says to do next (incl. decay re-tests). */
export function NextBestActions({ projectId }: { projectId: string | null }) {
  const navigate = useNavigate();
  const { data, isLoading } = usePortfolio(projectId);
  const actions = (data?.next_actions ?? []) as (NextAction & { type: ActionType })[];
  const lastFitAt = data?.last_fit_at;

  if (isLoading) return null;

  const target = (type: ActionType) =>
    type === 'experiment' || type === 'retest' ? '/experiments' : '/workspace';
  const targetLabel = (type: ActionType) =>
    type === 'experiment' || type === 'retest' ? 'Open experiments' : 'Open workspace';

  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xs font-bold uppercase tracking-wider text-ink-400">Next best actions</h2>
        {lastFitAt != null && (
          <span className="text-xs text-ink-300">
            Last fit: <span className="num">{new Date(lastFitAt * 1000).toLocaleDateString()}</span>
          </span>
        )}
      </div>

      {actions.length === 0 ? (
        <div className="flex items-center gap-2.5 rounded-lg border border-sage-300 bg-white px-5 py-4">
          <CheckCircle2 className="h-5 w-5 text-sage-700" />
          <p className="text-sm text-ink-600">
            All caught up — the model is fresh, no readouts are waiting, and no evidence has decayed
            past the re-test threshold.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {actions.map((a, i) => {
            const style = ACTION_STYLE[a.type] ?? ACTION_STYLE.fit;
            const Icon = style.icon;
            return (
              <div
                key={i}
                className={`flex flex-col gap-2 rounded-lg border bg-white p-4 shadow-sm ${style.ring}`}
              >
                <div className="flex items-center gap-2">
                  <Icon className={`h-5 w-5 shrink-0 ${style.iconColor}`} strokeWidth={1.75} />
                  <p className="flex-1 text-sm font-semibold leading-snug text-ink-900">{a.title}</p>
                  <UrgencyChip urgency={a.urgency} />
                </div>
                <p className="text-xs leading-relaxed text-ink-400">{a.detail}</p>
                {a.design && (
                  <div className="space-y-0.5 rounded-md border border-line-200 bg-cream-100/60 px-2.5 py-1.5 text-[11px] text-ink-600">
                    {a.design.design_type && (
                      <p>
                        <span className="font-medium text-ink-700">Design:</span> {a.design.design_type}
                      </p>
                    )}
                    {a.design.min_duration_periods != null && (
                      <p>
                        <span className="font-medium text-ink-700">Duration:</span>{' '}
                        <span className="num">≥ {a.design.min_duration_periods}</span> periods
                      </p>
                    )}
                    {a.design.target_se != null && (
                      <p>
                        <span className="font-medium text-ink-700">Target SE:</span>{' '}
                        <span className="num">≤ {a.design.target_se.toFixed(2)}</span>
                      </p>
                    )}
                  </div>
                )}
                <button
                  onClick={() => navigate(target(a.type))}
                  className="mt-auto flex items-center gap-1 self-start text-xs font-medium text-sage-800 hover:underline"
                >
                  {targetLabel(a.type)} <ArrowRight className="h-3 w-3" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
