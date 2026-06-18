import { useNavigate } from 'react-router-dom';
import { ArrowRight, CheckCircle2, Circle } from 'lucide-react';
import { useOnboardingStatus } from '../../api/hooks/useProjects';

/**
 * Self-serve onboarding: the path-to-first-model checklist a new user follows.
 * Driven by GET /projects/{id}/onboarding-status; it disappears once the project
 * is fully onboarded (an analyst never sees it).
 */
export function OnboardingChecklist({ projectId }: { projectId: string | null }) {
  const navigate = useNavigate();
  const { data } = useOnboardingStatus(projectId ?? undefined);

  // Hide while loading, on error, and once the project is fully set up.
  if (!data || data.complete) return null;

  const next = data.steps.find((s) => !s.done);

  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xs font-bold uppercase tracking-wider text-ink-400">Get started</h2>
        <span className="text-xs text-ink-300">
          <span className="num">{data.completed}</span>/<span className="num">{data.total}</span> steps ·{' '}
          <span className="num">{data.percent}%</span>
        </span>
      </div>

      <div className="rounded-lg border border-line-200 bg-white p-5 shadow-sm">
        <div className="mb-4 h-1.5 w-full overflow-hidden rounded-full bg-cream-200">
          <div
            className="h-full rounded-full bg-sage-700 transition-all"
            style={{ width: `${data.percent}%` }}
          />
        </div>

        <ul className="space-y-2.5">
          {data.steps.map((s) => {
            const isNext = next != null && s.key === next.key;
            return (
              <li key={s.key} className="flex items-start gap-2.5">
                {s.done ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-sage-700" strokeWidth={2} />
                ) : (
                  <Circle
                    className={`mt-0.5 h-4 w-4 shrink-0 ${isNext ? 'text-rust-600' : 'text-ink-300'}`}
                    strokeWidth={2}
                  />
                )}
                <div className="flex-1">
                  <p
                    className={`text-sm leading-snug ${
                      s.done
                        ? 'text-ink-400 line-through'
                        : isNext
                          ? 'font-semibold text-ink-900'
                          : 'text-ink-700'
                    }`}
                  >
                    {s.title}
                  </p>
                  {isNext && <p className="mt-0.5 text-xs leading-relaxed text-ink-400">{s.hint}</p>}
                </div>
              </li>
            );
          })}
        </ul>

        {next && (
          <button
            onClick={() => navigate('/workspace')}
            className="mt-4 flex items-center gap-1.5 self-start text-xs font-medium text-sage-800 hover:underline"
          >
            Continue in the workspace <ArrowRight className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </section>
  );
}
