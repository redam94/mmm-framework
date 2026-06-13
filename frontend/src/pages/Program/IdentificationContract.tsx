import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronDownIcon, ChevronRightIcon, ScaleIcon } from '@heroicons/react/24/outline';
import { Card } from '../../components/ui';
import { useCalibrationCoverage } from '../../api/hooks/useMeasurement';
import { runsService } from '../../api/services/runsService';

// ── Identification contract ───────────────────────────────────────────────────
// The seven causal assumptions every MMM read rests on (the docs'
// identification-assumptions page), each with its testability and whatever
// live evidence the program actually has. The point is honesty at the point
// of decision: a coverage map says what's been tested; this card says what's
// being ASSUMED.

type Testability = 'untestable' | 'partially testable' | 'testable';

const TESTABILITY_STYLE: Record<Testability, string> = {
  untestable: 'bg-rust-100 text-rust-700',
  'partially testable': 'bg-gold-100 text-gold-700',
  testable: 'bg-sage-100 text-sage-800',
};

interface AssumptionRow {
  name: string;
  testability: Testability;
  statement: string;
  /** what the platform does about it + any live evidence */
  evidence: React.ReactNode;
}

function Row({ a }: { a: AssumptionRow }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="border-b border-line-200 py-2 last:border-0">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 text-left"
      >
        {open ? (
          <ChevronDownIcon className="h-3.5 w-3.5 shrink-0 text-ink-300" />
        ) : (
          <ChevronRightIcon className="h-3.5 w-3.5 shrink-0 text-ink-300" />
        )}
        <span className="flex-1 text-sm font-medium text-ink-900">{a.name}</span>
        <span
          className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium ${TESTABILITY_STYLE[a.testability]}`}
        >
          {a.testability}
        </span>
      </button>
      {open && (
        <div className="mt-1.5 space-y-1.5 pl-5">
          <p className="text-xs text-ink-600">{a.statement}</p>
          <div className="text-xs text-ink-400">{a.evidence}</div>
        </div>
      )}
    </li>
  );
}

export function IdentificationContract({ projectId }: { projectId: string }) {
  const { data: coverage } = useCalibrationCoverage(projectId);
  const { data: runs = [] } = useQuery({
    queryKey: ['runs', projectId],
    queryFn: () => runsService.listRuns(projectId),
    staleTime: 30000,
  });
  const latest = runs[0];

  const backedPct = coverage?.spend_weighted_coverage_pct ?? null;
  const nControls = latest?.controls?.length ?? 0;

  const rows: AssumptionRow[] = useMemo(
    () => [
      {
        name: 'No unobserved confounding',
        testability: 'untestable',
        statement:
          'After controlling for the declared confounders, spend is as-good-as-random with respect to the outcome. Nothing in the data can verify this — a demand shock that drives both spend and sales biases every untested ROI.',
        evidence: (
          <>
            {backedPct != null ? (
              <>
                <span className="num font-medium text-ink-600">{backedPct.toFixed(0)}%</span> of
                spend is experiment-backed —{' '}
              </>
            ) : null}
            randomized experiments are the only mechanism that retires this risk (a coin flip
            can't be confounded). Channels still model-only inherit it in full.
          </>
        ),
      },
      {
        name: 'Positivity / overlap',
        testability: 'partially testable',
        statement:
          'Each channel must vary enough — including dark or near-dark periods — for the data to identify its zero-out counterfactual. A channel that never goes dark has its ROI extrapolated from the prior, not measured.',
        evidence: (
          <>
            Screened pre-fit by the Workspace EDA's identification checks (near-constant spend,
            zero-inflation, observations-per-parameter). When variation is missing, design an
            experiment instead of tightening a prior.
          </>
        ),
      },
      {
        name: 'Correct functional form',
        testability: 'testable',
        statement:
          'The true response must lie inside the chosen adstock kernel and saturation family. Pressure tests put worst-channel error at 80–96% under shape misspecification — with green diagnostics throughout.',
        evidence: (
          <>
            {latest
              ? `Latest fit: ${latest.channels?.length ?? 0} channels, trend "${latest.trend ?? '—'}". `
              : null}
            Refit under alternative shapes and compare — findings that survive across forms are
            robust; ones that don't are artifacts of the choice.
          </>
        ),
      },
      {
        name: 'No interference (SUTVA)',
        testability: 'untestable',
        statement:
          "A treated geo's outcome depends only on its own treatment — no spillover from neighbors, national halo, or shared retargeting pools.",
        evidence: (
          <>
            The design studio screens matched pairs on residual correlation and flags spillover
            risk; channels that can't be geo-isolated should use budget-neutral flighting instead.
          </>
        ),
      },
      {
        name: 'Sequential ignorability (mediation)',
        testability: 'untestable',
        statement:
          'Splitting effects into direct and mediated paths (nested models) needs unconfoundedness on both the media→mediator and mediator→outcome links. The split is prior-sensitive.',
        evidence: (
          <>
            Only binds when a mediator is modeled. Experiments calibrate the <em>total</em>{' '}
            effect; treat the direct/indirect split as a modeling assumption, not a measurement.
          </>
        ),
      },
      {
        name: 'Exogenous spend (no reverse causality)',
        testability: 'untestable',
        statement:
          'Budgets were not set by looking at concurrent or forecast outcomes. Spend that chases demand makes media look causal when planners were just good forecasters.',
        evidence: (
          <>
            {nControls > 0 ? (
              <>
                <span className="num font-medium text-ink-600">{nControls}</span> control
                {nControls === 1 ? '' : 's'} declared in the latest fit
                {latest?.controls?.length ? ` (${latest.controls.join(', ')})` : ''} —{' '}
              </>
            ) : (
              <>No controls declared in the latest fit — </>
            )}
            declaring the planning signals (demand forecasts, promo calendar) as confounders is
            what converts anticipatory budgeting back into a conditionally random design.
          </>
        ),
      },
      {
        name: 'Stable structure over the window',
        testability: 'partially testable',
        statement:
          'One set of parameters generated the whole window — no unmodeled trend breaks or evolving seasonality. Stress tests measured −41% silent error from a single unmodeled break.',
        evidence: (
          <>
            {latest?.trend ? (
              <>
                Latest fit uses a <span className="font-medium text-ink-600">{latest.trend}</span>{' '}
                trend.{' '}
              </>
            ) : null}
            Known breaks (launches, distribution changes) belong in the spec as changepoints;
            out-of-time backtests are the drift detector.
          </>
        ),
      },
    ],
    [backedPct, latest, nControls],
  );

  return (
    <Card padding="md">
      <div className="flex items-center gap-2">
        <ScaleIcon className="h-5 w-5 text-ink-400" />
        <h3 className="text-sm font-semibold text-ink-900">Identification contract</h3>
      </div>
      <p className="mt-0.5 text-xs text-ink-400">
        Every causal number on this page rests on these seven assumptions. Three are untestable
        from observational data — they are retired by experiments or carried as risk, never
        proven by fit quality.
      </p>
      <ul className="mt-2">
        {rows.map((a) => (
          <Row key={a.name} a={a} />
        ))}
      </ul>
      <p className="mt-3 border-t border-line-200 pt-2.5 text-[11px] leading-relaxed text-ink-400">
        Green diagnostics validate <em>computation</em>, not causality — the pressure-test suite
        holds eight worlds where every check passes and the answer is silently wrong. Model
        health (sampling, prior-vs-data learning) lives on the Performance page; this card is the
        part no diagnostic can certify.
      </p>
    </Card>
  );
}

export default IdentificationContract;
