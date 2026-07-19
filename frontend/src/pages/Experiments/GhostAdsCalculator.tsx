import { useState } from 'react';
import { Ghost, X } from 'lucide-react';
import { Button } from '../../components/ui';
import { useGhostAdsPower } from '../../api/hooks/useMeasurement';
import { useProjectStore } from '../../stores/projectStore';
import type { GhostAdsPowerPayload } from '../../api/services/measurementService';

function errorDetail(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data
    ?.detail;
  return detail ?? String(e);
}

const inputCls =
  'w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm num text-ink-900 ' +
  'focus:outline-none focus:ring-2 focus:ring-sage-600';

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs font-medium text-ink-600">{label}</span>
      {children}
      {hint && <span className="mt-0.5 block text-[11px] text-ink-400">{hint}</span>}
    </label>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`rounded-lg border px-3 py-2.5 ${accent ? 'border-sage-300 bg-sage-100/50' : 'border-line-200 bg-white'}`}>
      <div className="text-[11px] font-medium uppercase tracking-wider text-ink-400">{label}</div>
      <div className={`num text-lg ${accent ? 'font-semibold text-sage-800' : 'text-ink-900'}`}>{value}</div>
    </div>
  );
}

/**
 * Ghost-ads (user-level RCT) power calculator — a standalone, pre-fit tool: no
 * dataset or model needed. Treated users see the real ad; ghost/PSA users are
 * would-have-been-served controls. Two-proportion MDE + users-required, with
 * ITT vs treatment-on-treated dilution and a rare-event flag.
 */
export function GhostAdsCalculator({ onClose }: { onClose: () => void }) {
  const projectId = useProjectStore((s) => s.currentProjectId);
  const power = useGhostAdsPower(projectId);

  const [users, setUsers] = useState(500_000);
  const [baselinePct, setBaselinePct] = useState(2.0);
  const [treatedPct, setTreatedPct] = useState(50);
  const [exposurePct, setExposurePct] = useState(100);
  const [targetLiftPct, setTargetLiftPct] = useState<number | ''>('');
  const [result, setResult] = useState<GhostAdsPowerPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setError(null);
    try {
      const res = await power.mutateAsync({
        users_reached: users,
        baseline_rate: baselinePct / 100,
        treated_fraction: treatedPct / 100,
        exposure_rate: exposurePct / 100,
        ...(targetLiftPct !== '' ? { target_lift_abs: (Number(targetLiftPct) / 100) * (baselinePct / 100) } : {}),
        simulate: true,
      });
      setResult(res);
    } catch (e) {
      setError(errorDetail(e));
      setResult(null);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/40 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl bg-cream-50 p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-ink-900">
            <Ghost className="h-5 w-5 text-sage-700" /> Ghost-ads power calculator
          </h3>
          <button onClick={onClose} className="rounded p-1 text-ink-400 hover:bg-cream-100">
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="mb-4 text-xs leading-relaxed text-ink-500">
          User-level incrementality: treated users see the real ad, ghost/PSA users a placebo the ad
          server would have served. Standalone — needs no dataset or fitted model.
        </p>

        <div className="grid grid-cols-2 gap-3">
          <Field label="Users reached (total)">
            <input type="number" min={1000} step={10000} value={users} className={inputCls}
              onChange={(e) => setUsers(Number(e.target.value))} />
          </Field>
          <Field label="Baseline conversion %" hint="control conversion rate">
            <input type="number" min={0.01} max={99} step={0.1} value={baselinePct} className={inputCls}
              onChange={(e) => setBaselinePct(Number(e.target.value))} />
          </Field>
          <Field label="Treated share %" hint="rest are ghost controls">
            <input type="number" min={1} max={99} step={5} value={treatedPct} className={inputCls}
              onChange={(e) => setTreatedPct(Number(e.target.value))} />
          </Field>
          <Field label="Exposure rate %" hint="share actually reached (ITT dilution)">
            <input type="number" min={1} max={100} step={5} value={exposurePct} className={inputCls}
              onChange={(e) => setExposurePct(Number(e.target.value))} />
          </Field>
          <Field label="Target lift % (relative)" hint="optional — users needed to detect it">
            <input type="number" min={0} step={1} value={targetLiftPct} placeholder="e.g. 10" className={inputCls}
              onChange={(e) => setTargetLiftPct(e.target.value === '' ? '' : Number(e.target.value))} />
          </Field>
          <div className="flex items-end">
            <Button onClick={run} disabled={power.isPending || users <= 0}>
              {power.isPending ? 'Computing…' : 'Compute power'}
            </Button>
          </div>
        </div>

        {error && <p className="mt-3 text-sm text-rust-700">{error}</p>}

        {result && (
          <div className="mt-4 space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <Stat accent label="MDE (relative lift)" value={`${(result.mde_rel * 100).toFixed(1)}%`} />
              <Stat label="MDE (absolute)" value={`${(result.mde_abs * 100).toFixed(3)} pp`} />
              <Stat label="Incremental conversions at MDE" value={Math.round(result.incremental_at_mde).toLocaleString()} />
              <Stat label="Split" value={`${Math.round(result.n_treated).toLocaleString()} / ${Math.round(result.n_ghost).toLocaleString()}`} />
            </div>
            {result.exposure_rate < 1 && (
              <p className="rounded-md bg-cream-100 px-3 py-2 text-xs text-ink-600">
                Dilution: ITT MDE {(result.itt_mde * 100).toFixed(3)} pp → treatment-on-treated MDE{' '}
                {(result.tot_mde * 100).toFixed(3)} pp (only {(result.exposure_rate * 100).toFixed(0)}% reached).
              </p>
            )}
            {result.users_required_for_target != null && (
              <p className="rounded-md border border-sage-300 bg-sage-100/50 px-3 py-2 text-xs text-sage-800">
                To detect the target lift: <span className="num font-semibold">{result.users_required_for_target.toLocaleString()}</span>{' '}
                users needed — the current audience gives {(100 * (result.power_at_target ?? 0)).toFixed(0)}% power.
              </p>
            )}
            {result.rare_event_regime && (
              <p className="rounded-md border border-rust-300 bg-rust-100/40 px-3 py-2 text-xs text-rust-800">
                ⚠️ Rare-event regime — few expected conversions per arm; the normal approximation is
                optimistic{result.simulation ? ` (simulated power ${(result.simulation.empirical_power * 100).toFixed(0)}% vs analytic ${(result.simulation.analytic_power * 100).toFixed(0)}%)` : ''}.
              </p>
            )}
            {result.simulation && !result.rare_event_regime && (
              <p className="text-[11px] text-ink-400">
                Simulation check: empirical power {(result.simulation.empirical_power * 100).toFixed(0)}%,
                false-positive rate {(result.simulation.empirical_fpr * 100).toFixed(1)}% (nominal 5%),{' '}
                {result.simulation.n_sims.toLocaleString()} draws.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
