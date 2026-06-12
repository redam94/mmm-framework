import { CalendarClock } from 'lucide-react';
import { Card, EmptyState, TierBadge } from '../../components/ui';
import { COLORS } from '../../theme/colors';
import type { PrioritiesPayload, PriorityChannel } from '../../api/services/measurementService';

const HALF_LIFE_WEEKS = 39; // illustrative — per-channel server-side
const EIG_THRESHOLD = 0.15; // nats
const T_MAX = 104;

/** eig(t) as posterior certainty decays: sigma_eff grows, a fresh test regains more info. */
function eigAt(t: number, roiSd: number, sigmaExp: number): number {
  const sigmaEff = roiSd * Math.exp((0.5 * Math.LN2 * t) / HALF_LIFE_WEEKS);
  return 0.5 * Math.log(1 + (sigmaEff / sigmaExp) ** 2);
}

function DecaySparkline({ ch }: { ch: PriorityChannel }) {
  if (ch.roi_sd == null || ch.sigma_exp == null || ch.sigma_exp <= 0) {
    return <span className="text-xs text-ink-300">—</span>;
  }
  const W = 160;
  const H = 44;
  const PAD = 4;

  const ts: number[] = [];
  for (let t = 0; t <= T_MAX; t += 2) ts.push(t);
  const ys = ts.map((t) => eigAt(t, ch.roi_sd as number, ch.sigma_exp as number));
  const yMax = Math.max(...ys, EIG_THRESHOLD * 1.3);

  const px = (t: number) => PAD + (t / T_MAX) * (W - 2 * PAD);
  const py = (y: number) => H - PAD - (y / yMax) * (H - 2 * PAD);

  const path = ts.map((t, i) => `${i === 0 ? 'M' : 'L'}${px(t).toFixed(1)},${py(ys[i]).toFixed(1)}`).join(' ');

  const now = Math.min(ch.weeks_since_evidence ?? 0, T_MAX);
  const nowY = eigAt(now, ch.roi_sd, ch.sigma_exp);

  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`EIG decay trajectory for ${ch.channel}`}
    >
      <line
        x1={PAD}
        x2={W - PAD}
        y1={py(EIG_THRESHOLD)}
        y2={py(EIG_THRESHOLD)}
        stroke={COLORS.line400}
        strokeWidth={1}
        strokeDasharray="3 3"
      />
      <path d={path} fill="none" stroke={COLORS.steel600} strokeWidth={1.5} />
      <circle
        cx={px(now)}
        cy={py(nowY)}
        r={3.5}
        fill={ch.retest_due ? COLORS.rust600 : COLORS.sage600}
        stroke="#ffffff"
        strokeWidth={1}
      />
    </svg>
  );
}

export function RetestSchedule({ priorities }: { priorities: PrioritiesPayload }) {
  const withEvidence = priorities.channels
    .filter((c) => c.weeks_since_evidence != null)
    .sort((a, b) => (b.weeks_since_evidence ?? 0) - (a.weeks_since_evidence ?? 0));
  const neverTested = priorities.channels.filter((c) => c.weeks_since_evidence == null);

  if (withEvidence.length === 0) {
    return (
      <EmptyState
        icon={CalendarClock}
        title="No calibrated experiments yet"
        description="Re-test scheduling starts after your first calibration — once a channel has experimental evidence, its information value decays and re-tests get queued here."
      />
    );
  }

  return (
    <div className="space-y-4">
      <Card padding="none">
        <ul className="divide-y divide-line-200">
          {withEvidence.map((ch) => (
            <li key={ch.channel} className="flex flex-wrap items-center gap-x-5 gap-y-2 px-5 py-3">
              <span className="w-32 shrink-0 font-medium text-ink-900">{ch.channel}</span>
              <TierBadge tier={ch.retest_due ? 'stale' : 'calibrated'} />
              <span className="text-sm text-ink-600">
                evidence <span className="num">{(ch.weeks_since_evidence as number).toFixed(0)}</span> wk old
              </span>
              <span className="text-sm text-ink-600">
                EIG <span className="num">{ch.eig != null ? ch.eig.toFixed(2) : '—'}</span>
                <span className="text-ink-300"> → </span>
                <span className="num">{ch.eig_decayed != null ? ch.eig_decayed.toFixed(2) : '—'}</span>
                <span className="text-ink-400"> nats</span>
              </span>
              <div className="ml-auto">
                <DecaySparkline ch={ch} />
              </div>
            </li>
          ))}
        </ul>
        <p className="border-t border-line-200 px-5 py-2.5 text-xs text-ink-400">
          Trajectory: effective ROI uncertainty re-inflates as evidence ages (half-life{' '}
          <span className="num">{HALF_LIFE_WEEKS}</span> wk shown here — it's per-channel
          server-side), so the value of a fresh test climbs back. Dashed line marks the{' '}
          <span className="num">{EIG_THRESHOLD.toFixed(2)}</span>-nat re-test threshold.
        </p>
      </Card>

      {neverTested.length > 0 && (
        <Card tone="cream" padding="sm">
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-ink-400">
            Never tested
          </p>
          <ul className="space-y-1">
            {neverTested.map((ch) => (
              <li key={ch.channel} className="flex items-center gap-3 text-sm text-ink-400">
                <span className="w-32 shrink-0">{ch.channel}</span>
                <TierBadge tier="model_only" />
                <span className="text-xs">no experimental evidence on record</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
