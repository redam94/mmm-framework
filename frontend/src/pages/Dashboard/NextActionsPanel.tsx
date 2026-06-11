import { useNavigate } from 'react-router-dom';
import {
  ArrowPathIcon,
  ArrowRightIcon,
  BeakerIcon,
  CheckCircleIcon,
  CubeIcon,
  ScaleIcon,
} from '@heroicons/react/24/outline';
import { usePortfolio } from '../../api/hooks/usePortfolio';
import type { NextAction } from '../../api/services/portfolioService';

const ACTION_STYLE: Record<NextAction['type'], { icon: typeof BeakerIcon; ring: string; iconColor: string }> = {
  calibrate: { icon: ScaleIcon, ring: 'border-red-200 bg-red-50/40', iconColor: 'text-red-500' },
  refresh: { icon: ArrowPathIcon, ring: 'border-amber-200 bg-amber-50/40', iconColor: 'text-amber-500' },
  fit: { icon: CubeIcon, ring: 'border-indigo-200 bg-indigo-50/40', iconColor: 'text-indigo-500' },
  experiment: { icon: BeakerIcon, ring: 'border-violet-200 bg-violet-50/40', iconColor: 'text-violet-500' },
};

function UrgencyChip({ urgency }: { urgency: NextAction['urgency'] }) {
  const cls =
    urgency === 'high'
      ? 'bg-red-100 text-red-700'
      : urgency === 'medium'
        ? 'bg-amber-100 text-amber-700'
        : 'bg-gray-100 text-gray-600';
  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${cls}`}>
      {urgency}
    </span>
  );
}

export function NextActionsPanel({ projectId }: { projectId: string | null }) {
  const navigate = useNavigate();
  const { data, isLoading } = usePortfolio(projectId);
  const actions = data?.next_actions ?? [];
  const lastFitAt = data?.last_fit_at;

  if (isLoading) return null;

  return (
    <section>
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-bold text-gray-600 uppercase tracking-wider">Next Best Actions</h2>
        {lastFitAt != null && (
          <span className="text-xs text-gray-400">
            Last fit: {new Date(lastFitAt * 1000).toLocaleDateString()}
          </span>
        )}
      </div>

      {actions.length === 0 ? (
        <div className="flex items-center gap-2.5 bg-white rounded-2xl border border-emerald-200 px-5 py-4">
          <CheckCircleIcon className="h-5 w-5 text-emerald-500" />
          <p className="text-sm text-gray-600">
            All caught up — the model is fresh and no experiment results are waiting to be calibrated.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {actions.map((a, i) => {
            const style = ACTION_STYLE[a.type] ?? ACTION_STYLE.fit;
            const Icon = style.icon;
            return (
              <div key={i} className={`bg-white rounded-2xl border shadow-sm p-4 flex flex-col gap-2 ${style.ring}`}>
                <div className="flex items-center gap-2">
                  <Icon className={`h-5 w-5 shrink-0 ${style.iconColor}`} />
                  <p className="flex-1 text-sm font-semibold text-gray-900 leading-snug">{a.title}</p>
                  <UrgencyChip urgency={a.urgency} />
                </div>
                <p className="text-xs text-gray-500 leading-relaxed">{a.detail}</p>
                {a.design && (
                  <div className="text-[11px] text-gray-500 bg-white/70 rounded-lg border border-gray-100 px-2.5 py-1.5 space-y-0.5">
                    {a.design.design_type && <p><span className="font-medium text-gray-600">Design:</span> {a.design.design_type}</p>}
                    {a.design.min_duration_periods != null && (
                      <p><span className="font-medium text-gray-600">Duration:</span> ≥ {a.design.min_duration_periods} periods</p>
                    )}
                    {a.design.target_se != null && (
                      <p><span className="font-medium text-gray-600">Target SE:</span> ≤ {a.design.target_se.toFixed(2)}</p>
                    )}
                  </div>
                )}
                <button
                  onClick={() => navigate('/chat')}
                  className="mt-auto self-start flex items-center gap-1 text-xs font-medium text-indigo-600 hover:underline"
                >
                  Open agent session <ArrowRightIcon className="h-3 w-3" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
