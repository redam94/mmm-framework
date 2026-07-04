import ReactMarkdown from 'react-markdown';
import { ShieldCheck, Loader2, AlertTriangle, History, MessageSquare } from 'lucide-react';
import { mdComponents } from '../common/markdown';
import { remarkPlugins, rehypePlugins, normalizeMath } from '../../../../lib/markdownMath';
import { PlotCard } from '../plots/PlotCard';
import { TableCard } from '../tables/TableCard';
import { useValidation } from '../../../../api/hooks/useValidation';
import type { ValidationCheck } from '../../../../api/services/validationService';
import type { TableRef } from '../../types';

const MD = mdComponents();

// One-click model validation — runs the same checks as the agent's validate_model
// / run_* tools, but deterministically (no LLM) against the project's latest fit.
const CHECKS: { id: ValidationCheck; label: string; hint: string; slow?: boolean }[] = [
  { id: 'validate', label: 'Validate model', hint: 'Full battery — start here' },
  { id: 'ppc', label: 'Posterior predictive', hint: 'Goodness of fit' },
  { id: 'residuals', label: 'Residuals', hint: 'Autocorrelation / normality' },
  { id: 'channels', label: 'Channels', hint: 'VIF / collinearity' },
  { id: 'refutation', label: 'Refutation', hint: 'Confounding robustness' },
  { id: 'cross_validation', label: 'Cross-validation', hint: 'Out-of-time (slow, refits)', slow: true },
];

// Labels for history rows — includes chat-only checks (SBC, prior predictive)
// that the button strip doesn't offer.
const CHECK_LABELS: Record<string, string> = {
  ...Object.fromEntries(CHECKS.map((c) => [c.id, c.label])),
  sbc: 'Calibration (SBC)',
  prior_predictive: 'Prior predictive',
};

export function ValidationTab({ projectId }: { projectId: string | null }) {
  const { start, job, history, load, jobId, check } = useValidation(projectId);
  const data = job.data;
  const running =
    start.isPending || data?.status === 'pending' || data?.status === 'running';

  if (!projectId) {
    return (
      <div className="p-4 text-sm text-ink-400">
        Pick a project to validate its fitted model.
      </div>
    );
  }

  return (
    <div className="space-y-4 p-1">
      <div>
        <h2 className="flex items-center gap-2 font-display text-lg font-semibold text-ink-900">
          <ShieldCheck size={18} className="text-sage-700" />
          Model validation
        </h2>
        <p className="mt-1 text-sm text-ink-400">
          Run validation checks against the project&apos;s latest fitted model —
          deterministic, no chat needed. Results render below.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {CHECKS.map((c) => {
          const active = check === c.id && running;
          return (
            <button
              key={c.id}
              type="button"
              disabled={running}
              onClick={() => start.mutate(c.id)}
              title={c.hint}
              className="rounded-lg border border-line-300 bg-white px-3 py-2 text-left text-sm transition-colors hover:bg-cream-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <span className="flex items-center gap-1.5 font-medium text-ink-800">
                {active && <Loader2 size={13} className="animate-spin" />}
                {c.label}
                {c.slow && (
                  <span className="rounded bg-gold-100 px-1 text-[10px] text-gold-700">
                    slow
                  </span>
                )}
              </span>
              <span className="block text-xs text-ink-400">{c.hint}</span>
            </button>
          );
        })}
      </div>

      {running && (
        <div className="flex items-center gap-2 text-sm text-ink-500" role="status">
          <Loader2 size={14} className="animate-spin" />
          Running {CHECKS.find((c) => c.id === check)?.label ?? 'validation'}…
        </div>
      )}

      {data?.status === 'error' && (
        <div className="flex items-start gap-2 rounded-lg border border-rust-600 bg-rust-100 p-3 text-sm text-rust-700">
          <AlertTriangle size={15} className="mt-0.5 shrink-0" />
          <span>{data.error || 'Validation failed.'}</span>
        </div>
      )}

      {data?.status === 'done' && data.result && (
        <div className="space-y-4">
          {data.result.content && (
            <div className="prose prose-sm max-w-none text-ink-700">
              <ReactMarkdown
                remarkPlugins={remarkPlugins}
                rehypePlugins={rehypePlugins}
                components={MD}
              >
                {normalizeMath(data.result.content)}
              </ReactMarkdown>
            </div>
          )}
          {data.result.plots?.map((p, i) => (
            <PlotCard key={p.id || i} plot={p} idx={i} />
          ))}
          {data.result.tables?.map((t, i) => (
            <TableCard key={t.id || i} tableRef={t as unknown as TableRef} idx={i} />
          ))}
        </div>
      )}

      {!data && !running && (
        <div className="rounded-lg border border-dashed border-line-300 bg-cream-50 p-6 text-center text-sm text-ink-400">
          Pick a check above. <strong>Validate model</strong> runs the full
          battery (convergence, posterior-predictive, residuals, channel
          identifiability, confounding robustness).
        </div>
      )}

      {(history.data?.length ?? 0) > 0 && (
        <div className="pt-2">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-ink-700">
            <History size={14} className="text-ink-400" />
            Validation history
          </h3>
          <div className="overflow-hidden rounded-lg border border-line-200 bg-white">
            {history.data!.map((h) => {
              const selected = h.job_id === jobId;
              return (
                <button
                  key={h.job_id}
                  type="button"
                  onClick={() => load(h)}
                  className={`flex w-full items-center gap-3 border-b border-line-200 px-3 py-2 text-left text-sm last:border-b-0 transition-colors hover:bg-cream-100 ${
                    selected ? 'bg-cream-100' : 'bg-white'
                  }`}
                  title="Open this run's result"
                >
                  <span
                    className={`h-2 w-2 shrink-0 rounded-full ${
                      h.status === 'done'
                        ? 'bg-emerald-500'
                        : h.status === 'error'
                          ? 'bg-rust-600'
                          : 'bg-gold-600 animate-pulse'
                    }`}
                  />
                  <span className="min-w-0 flex-1 truncate font-medium text-ink-800">
                    {CHECK_LABELS[h.check] ?? h.check}
                  </span>
                  {h.source === 'chat' && (
                    <span
                      className="flex items-center gap-1 rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-600"
                      title="Run by the chat agent"
                    >
                      <MessageSquare size={10} /> chat
                    </span>
                  )}
                  <span className="shrink-0 text-xs text-ink-300">
                    {h.created_at ? new Date(h.created_at * 1000).toLocaleString() : ''}
                  </span>
                </button>
              );
            })}
          </div>
          <p className="mt-1.5 text-xs text-ink-300">
            Every run is kept — UI checks and the ones the chat agent ran. Click a
            row to re-open its result.
          </p>
        </div>
      )}
    </div>
  );
}

export default ValidationTab;
