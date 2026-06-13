import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ShieldCheckIcon, ShieldExclamationIcon } from '@heroicons/react/24/outline';
import { Card } from '../../components/ui';
import { runsService } from '../../api/services/runsService';
import type {
  ConvergenceBlock,
  LearningBlock,
  LearningParameter,
  LearningVerdict,
  RunInfo,
} from '../../api/services/runsService';
import { useProjectStore } from '../../stores/projectStore';

// ── Model health — should you believe this fit? ───────────────────────────────
// Surfaces the fit-time diagnostics the backend already computes: sampler
// convergence (did the computation work?) and prior→posterior learning (did
// the DATA inform the parameters, or are posteriors re-stating the priors?).

const VERDICT_STYLE: Record<LearningVerdict, { cls: string; hint: string }> = {
  strong: {
    cls: 'bg-sage-100 text-sage-800',
    hint: 'the data clearly informed this parameter',
  },
  moderate: { cls: 'bg-steel-100 text-steel-700', hint: 'the data narrowed the prior somewhat' },
  weak: { cls: 'bg-gold-100 text-gold-700', hint: 'little narrowing beyond the prior' },
  relocated: {
    cls: 'bg-steel-100 text-steel-700',
    hint: 'the posterior moved ≥1 prior-SD without narrowing — the prior was tight in the wrong place',
  },
  'prior-dominated': {
    cls: 'bg-rust-100 text-rust-700',
    hint: 'the posterior is essentially the prior — the data did not speak',
  },
  undetermined: { cls: 'bg-cream-200 text-ink-600', hint: 'degenerate prior; not diagnosable' },
};

function VerdictBadge({ verdict }: { verdict: LearningVerdict }) {
  const s = VERDICT_STYLE[verdict] ?? VERDICT_STYLE.undetermined;
  return (
    <span
      title={s.hint}
      className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-medium ${s.cls}`}
    >
      {verdict}
    </span>
  );
}

function fmt(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits);
}

function ConvergenceTile({
  label,
  value,
  failed,
  hint,
}: {
  label: string;
  value: string;
  failed: boolean;
  hint: string;
}) {
  return (
    <div
      className={`rounded-lg border p-3 ${
        failed ? 'border-rust-600/40 bg-rust-100/50' : 'border-line-200 bg-white'
      }`}
      title={hint}
    >
      <p className="text-xs font-medium uppercase tracking-wider text-ink-400">{label}</p>
      <p className={`num mt-1 text-xl font-semibold ${failed ? 'text-rust-700' : 'text-ink-900'}`}>
        {value}
      </p>
      <p className={`mt-0.5 text-[11px] ${failed ? 'text-rust-600' : 'text-ink-400'}`}>{hint}</p>
    </div>
  );
}

function ConvergenceSection({ conv }: { conv: ConvergenceBlock }) {
  return (
    <Card padding="md">
      <div className="flex items-center gap-2">
        {conv.ok ? (
          <ShieldCheckIcon className="h-5 w-5 text-sage-700" />
        ) : (
          <ShieldExclamationIcon className="h-5 w-5 text-rust-600" />
        )}
        <h3 className="text-sm font-semibold text-ink-900">Sampler convergence</h3>
        <span
          className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
            conv.ok ? 'bg-sage-100 text-sage-800' : 'bg-rust-100 text-rust-700'
          }`}
        >
          {conv.ok ? 'clean' : `${conv.flags.length} flag${conv.flags.length > 1 ? 's' : ''}`}
        </span>
      </div>
      <p className="mt-0.5 text-xs text-ink-400">
        Whether the computation worked. A clean pass validates the <em>sampling</em>, not the
        causal reading — a converged model can still be confounded or misspecified.
      </p>
      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <ConvergenceTile
          label="Divergences"
          value={conv.divergences == null ? '—' : conv.divergences.toLocaleString()}
          failed={conv.flags.includes('divergences')}
          hint="any divergent transition means the posterior geometry was not explored faithfully"
        />
        <ConvergenceTile
          label="Max R-hat"
          value={fmt(conv.rhat_max, 3)}
          failed={conv.flags.includes('rhat')}
          hint={`chains agree when R-hat ≤ ${conv.rhat_threshold}`}
        />
        <ConvergenceTile
          label="Min bulk ESS"
          value={conv.ess_bulk_min == null ? '—' : Math.round(conv.ess_bulk_min).toLocaleString()}
          failed={conv.flags.includes('ess')}
          hint={`effective draws behind the worst-estimated parameter; ≥ ${Math.round(conv.ess_threshold)} is comfortable`}
        />
      </div>
    </Card>
  );
}

function LearningSection({ learning }: { learning: LearningBlock }) {
  const [showAll, setShowAll] = useState(false);
  const flagged = learning.parameters.filter(
    (p) => p.verdict === 'prior-dominated' || p.verdict === 'weak' || p.verdict === 'relocated',
  );
  const rows = showAll ? learning.parameters : flagged.length > 0 ? flagged : learning.parameters;
  const shown = rows.slice(0, showAll ? 40 : 10);
  const nPriorDominated = learning.verdict_counts['prior-dominated'] ?? 0;

  return (
    <Card padding="md">
      <h3 className="text-sm font-semibold text-ink-900">Prior → posterior learning</h3>
      <p className="mt-0.5 text-xs text-ink-400">
        A credible interval doesn't say how much of the belief came from the data versus the
        prior. <span className="font-medium text-ink-600">Prior-dominated</span> parameters are
        re-stating an assumption — treat their ROI reads as un-evidenced until an experiment or
        more variation pins them.
      </p>

      <div className="mt-3 flex flex-wrap gap-2">
        {(Object.entries(learning.verdict_counts) as [LearningVerdict, number][]).map(
          ([verdict, n]) => (
            <span key={verdict} className="flex items-center gap-1.5 text-xs text-ink-600">
              <VerdictBadge verdict={verdict} />
              <span className="num">{n}</span>
            </span>
          ),
        )}
        <span className="text-xs text-ink-400">
          of <span className="num">{learning.n_parameters}</span> parameters
        </span>
      </div>

      {nPriorDominated > 0 && (
        <p className="mt-2 rounded-lg bg-rust-100/60 px-3 py-2 text-xs text-rust-700">
          {nPriorDominated} parameter{nPriorDominated > 1 ? 's' : ''} where the posterior ≈ the
          prior. The model's answer there is your assumption, fed back to you — a calibration
          experiment is the honest way to learn it.
        </p>
      )}

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-line-200 text-[10px] uppercase tracking-wider text-ink-400">
              <th className="py-1.5 pr-3 font-semibold">Parameter</th>
              <th className="py-1.5 pr-3 font-semibold">Verdict</th>
              <th className="py-1.5 pr-3 font-semibold" title="1 − Var_post/Var_prior — how much the data narrowed the prior">
                Contraction
              </th>
              <th className="py-1.5 pr-3 font-semibold" title="prior–posterior overlap; ~1 means nothing learned">
                Overlap
              </th>
              <th className="py-1.5 pr-3 font-semibold" title="posterior-mean shift in prior SDs">
                Shift (z)
              </th>
              <th className="py-1.5 font-semibold">Posterior</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((p: LearningParameter) => (
              <tr key={p.parameter} className="border-b border-line-200 last:border-0">
                <td className="py-1.5 pr-3">
                  <code className="num text-ink-700">{p.parameter}</code>
                </td>
                <td className="py-1.5 pr-3">
                  <VerdictBadge verdict={p.verdict} />
                </td>
                <td className="num py-1.5 pr-3 text-ink-700">{fmt(p.contraction)}</td>
                <td className="num py-1.5 pr-3 text-ink-700">{fmt(p.overlap)}</td>
                <td className="num py-1.5 pr-3 text-ink-700">{fmt(p.shift_z)}</td>
                <td className="num py-1.5 text-ink-700">
                  {fmt(p.post_mean)} ± {fmt(p.post_sd)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(flagged.length > 10 || learning.parameters.length > shown.length || showAll) && (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="mt-2 text-xs font-medium text-sage-700 hover:underline"
        >
          {showAll
            ? 'Show flagged parameters only'
            : `Show all ${learning.parameters.length} diagnosed parameters`}
        </button>
      )}
      {learning.truncated && (
        <p className="mt-1 text-[11px] text-ink-300">
          Snapshot keeps the {learning.parameters.length} least-learned parameters; the rest
          diagnosed as informed.
        </p>
      )}
    </Card>
  );
}

export function ModelHealthPanel() {
  const { currentProjectId } = useProjectStore();
  const { data: runs = [], isLoading } = useQuery({
    queryKey: ['runs', currentProjectId],
    queryFn: () => runsService.listRuns(currentProjectId),
    staleTime: 30000,
  });

  // Newest run that carries a health snapshot (older fits predate it).
  const run: RunInfo | undefined = useMemo(
    () => runs.find((r) => r.diagnostics != null),
    [runs],
  );
  const diag = run?.diagnostics;

  if (isLoading) return <p className="text-sm text-ink-400">Loading model health…</p>;

  if (!run || !diag) {
    return (
      <Card padding="md">
        <h3 className="text-sm font-semibold text-ink-900">Model health</h3>
        <p className="mt-1 text-sm text-ink-400">
          No health snapshot recorded yet — fits made from now on stamp sampler convergence and
          prior→posterior learning diagnostics here. Refit in the Workspace to populate this view.
        </p>
      </Card>
    );
  }

  return (
    <div className="max-w-4xl space-y-6">
      <p className="text-sm text-ink-400">
        Diagnostics for{' '}
        <code className="rounded bg-cream-200 px-1 text-xs">{run.run_name ?? run.run_id}</code>
        {runs[0] !== run && ' (newer fits exist without a snapshot)'} — first whether the sampler
        converged, then whether the data (not the priors) drove the estimates.
      </p>

      {diag.convergence && <ConvergenceSection conv={diag.convergence} />}
      {diag.convergence_error && (
        <p className="text-xs text-rust-600">Convergence snapshot failed: {diag.convergence_error}</p>
      )}

      {diag.learning && <LearningSection learning={diag.learning} />}
      {diag.learning_error && (
        <p className="text-xs text-rust-600">Learning snapshot failed: {diag.learning_error}</p>
      )}

      <p className="text-xs text-ink-400">
        Green here means the <em>computation</em> is trustworthy. It does not certify the causal
        reading — confounding, functional-form misspecification, and structural breaks all pass
        these checks silently. Identification risk is tracked on the Program page's contract card
        and retired only by experiments.
      </p>
    </div>
  );
}

export default ModelHealthPanel;
