import { useEffect, useMemo, useState } from 'react';
import { Loader2, Save, Sparkles } from 'lucide-react';
import { Button } from '../../components/ui';
import { usePlannerOptimization } from '../../api/hooks/usePlanner';
import { useSaveBudgetPlan } from '../../api/hooks/useBudgetPlans';
import type {
  FlightingRequest,
  PlannerOptimizeRequest,
} from '../../api/services/plannerService';
import { AllocationResult } from './AllocationResult';

const inputCls =
  'w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600';
const numInputCls =
  'w-full rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm num text-right focus:outline-none focus:ring-2 focus:ring-sage-600';
const labelCls = 'mb-1 block text-xs font-medium text-ink-700';

const PATTERNS = ['even', 'front_loaded', 'back_loaded', 'pulsed'] as const;

function errorDetail(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
  return detail ?? String(e);
}

/**
 * The optimization studio: compute a budget plan (national or per-geo) with an
 * optional forward flighting calendar, then persist it as a named plan — no chat
 * round-trip. (B1 + B4 + B6.)
 */
export function PlannerStudio({
  projectId,
  channels,
  modelId,
  onSaved,
}: {
  projectId: string;
  channels: string[];
  modelId?: string | null;
  onSaved?: () => void;
}) {
  const [budgetMode, setBudgetMode] = useState<'reallocate' | 'change'>('reallocate');
  const [budgetChangePct, setBudgetChangePct] = useState(0);
  const [byGeo, setByGeo] = useState(false);
  const [withFlighting, setWithFlighting] = useState(true);
  const [pattern, setPattern] = useState<string>('even');
  const [nPeriods, setNPeriods] = useState(13);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [minMult, setMinMult] = useState(0);
  const [maxMult, setMaxMult] = useState(2);
  // Per-channel overrides: only channels the user edits are sent; the rest fall
  // back to the default min/max on the backend.
  const [perChannel, setPerChannel] = useState<Record<string, { lo?: number; hi?: number }>>({});
  const [planName, setPlanName] = useState('');
  const [saved, setSaved] = useState<string | null>(null);

  const optimization = usePlannerOptimization(projectId);
  const save = useSaveBudgetPlan();

  // Any input change invalidates a computed plan (mirrors DesignStudio).
  useEffect(() => {
    optimization.reset();
    setSaved(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    budgetMode,
    budgetChangePct,
    byGeo,
    withFlighting,
    pattern,
    nPeriods,
    minMult,
    maxMult,
    perChannel,
  ]);

  const result =
    optimization.job.data?.status === 'done' ? optimization.job.data.result : null;
  const jobStatus = optimization.job.data?.status ?? null;
  const running =
    optimization.start.isPending || jobStatus === 'pending' || jobStatus === 'running';
  const jobError =
    jobStatus === 'error'
      ? optimization.job.data?.error
      : optimization.start.isError
        ? errorDetail(optimization.start.error)
        : null;

  const flighting: FlightingRequest | null = useMemo(
    () => (withFlighting ? { pattern, n_periods: nPeriods } : null),
    [withFlighting, pattern, nPeriods],
  );

  // Per-channel [lo, hi] spend-multiplier bounds — only for channels the user
  // explicitly set; an unset bound inherits the default min/max.
  const channelBounds = useMemo(() => {
    const out: Record<string, [number, number]> = {};
    for (const ch of channels) {
      const pc = perChannel[ch];
      if (pc && (pc.lo !== undefined || pc.hi !== undefined)) {
        out[ch] = [pc.lo ?? minMult, pc.hi ?? maxMult];
      }
    }
    return Object.keys(out).length ? out : null;
  }, [perChannel, channels, minMult, maxMult]);

  const run = () => {
    const body: PlannerOptimizeRequest = {
      by_geo: byGeo,
      flighting,
      min_multiplier: minMult,
      max_multiplier: maxMult,
      channel_bounds: channelBounds,
      ...(budgetMode === 'change' ? { budget_change_pct: budgetChangePct } : {}),
    };
    optimization.start.mutate(body);
  };

  const onSave = async () => {
    if (!result) return;
    try {
      await save.mutateAsync({
        name: planName.trim() || `Plan ${new Date().toLocaleDateString()}`,
        project_id: projectId,
        model_id: modelId ?? null,
        kind: 'optimization',
        plan_payload: result as unknown as Record<string, unknown>,
      });
      setSaved(planName.trim() || 'plan');
      setPlanName('');
      onSaved?.();
    } catch {
      /* surfaced via save.isError below */
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className={labelCls}>Budget</span>
          <select
            value={budgetMode}
            onChange={(e) => setBudgetMode(e.target.value as 'reallocate' | 'change')}
            className={inputCls}
          >
            <option value="reallocate">Reallocate current spend</option>
            <option value="change">Change total by %</option>
          </select>
        </label>
        {budgetMode === 'change' && (
          <label className="block">
            <span className={labelCls}>Total budget change %</span>
            <input
              type="number"
              step={5}
              value={budgetChangePct}
              onChange={(e) => setBudgetChangePct(Number(e.target.value))}
              className={inputCls}
            />
          </label>
        )}
      </div>

      <label className="flex items-center gap-2 text-sm text-ink-700">
        <input
          type="checkbox"
          checked={byGeo}
          onChange={(e) => setByGeo(e.target.checked)}
          className="h-3.5 w-3.5 rounded border-line-300 text-sage-600 focus:ring-sage-600"
        />
        Allocate per geography / DMA (geo panels only)
      </label>

      <div className="rounded-lg border border-line-200 bg-cream-50 px-3 py-3">
        <label className="flex items-center gap-2 text-sm font-medium text-ink-700">
          <input
            type="checkbox"
            checked={withFlighting}
            onChange={(e) => setWithFlighting(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-line-300 text-sage-600 focus:ring-sage-600"
          />
          Add a forward flighting calendar
        </label>
        {withFlighting && (
          <div className="mt-3 grid grid-cols-2 gap-3">
            <label className="block">
              <span className={labelCls}>Pattern</span>
              <select
                value={pattern}
                onChange={(e) => setPattern(e.target.value)}
                className={inputCls}
              >
                {PATTERNS.map((p) => (
                  <option key={p} value={p}>
                    {p.replace('_', ' ')}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className={labelCls}>Periods (weeks)</span>
              <input
                type="number"
                min={1}
                max={104}
                value={nPeriods}
                onChange={(e) => setNPeriods(Number(e.target.value))}
                className={inputCls}
              />
            </label>
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={() => setShowAdvanced((s) => !s)}
        className="text-xs font-medium text-steel-600 hover:text-steel-700"
      >
        {showAdvanced ? '− Hide' : '+ Show'} spend constraints
      </button>
      {showAdvanced && (
        <div className="space-y-3 rounded-lg border border-line-200 bg-white px-3 py-3">
          <p className="text-[11px] text-ink-400">
            Bounds are multiples of each channel's <em>current</em> spend (1.0 = hold,
            0.8–1.2 = ±20%). They cap how far the optimizer can shift budget and keep
            recommendations inside the range the model has evidence for.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className={labelCls}>Default min ×</span>
              <input
                type="number"
                min={0}
                step={0.25}
                value={minMult}
                onChange={(e) => setMinMult(Number(e.target.value))}
                className={inputCls}
              />
            </label>
            <label className="block">
              <span className={labelCls}>Default max ×</span>
              <input
                type="number"
                min={0}
                step={0.25}
                value={maxMult}
                onChange={(e) => setMaxMult(Number(e.target.value))}
                className={inputCls}
              />
            </label>
          </div>

          {channels.length > 0 ? (
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <span className="text-xs font-medium text-ink-700">Per-channel bounds</span>
                {Object.keys(perChannel).length > 0 && (
                  <button
                    type="button"
                    onClick={() => setPerChannel({})}
                    className="text-[11px] text-steel-600 hover:text-steel-700"
                  >
                    Reset to default
                  </button>
                )}
              </div>
              <div className="mb-1 grid grid-cols-[1fr_5rem_5rem] gap-2 text-[10px] font-semibold uppercase tracking-wider text-ink-400">
                <span>Channel</span>
                <span className="text-right">Min ×</span>
                <span className="text-right">Max ×</span>
              </div>
              <div className="space-y-1">
                {channels.map((ch) => (
                  <div key={ch} className="grid grid-cols-[1fr_5rem_5rem] items-center gap-2">
                    <span className="truncate text-sm text-ink-700" title={ch}>
                      {ch}
                    </span>
                    <input
                      type="number"
                      min={0}
                      step={0.25}
                      value={perChannel[ch]?.lo ?? minMult}
                      onChange={(e) =>
                        setPerChannel((p) => ({
                          ...p,
                          [ch]: { ...p[ch], lo: Number(e.target.value) },
                        }))
                      }
                      className={numInputCls}
                      aria-label={`${ch} minimum spend multiplier`}
                    />
                    <input
                      type="number"
                      min={0}
                      step={0.25}
                      value={perChannel[ch]?.hi ?? maxMult}
                      onChange={(e) =>
                        setPerChannel((p) => ({
                          ...p,
                          [ch]: { ...p[ch], hi: Number(e.target.value) },
                        }))
                      }
                      className={numInputCls}
                      aria-label={`${ch} maximum spend multiplier`}
                    />
                  </div>
                ))}
              </div>
              <p className="mt-1.5 text-[11px] text-ink-400">
                Freeze a line at 1.0–1.0, cap a partner channel, or floor a committed
                buy — channels left at the default inherit the min/max above.
              </p>
            </div>
          ) : (
            <p className="text-[11px] text-ink-400">
              Per-channel rows appear once a baseline model is fit (channels come from
              the latest fit).
            </p>
          )}
        </div>
      )}

      <div className="flex items-center gap-3">
        <Button onClick={run} disabled={running}>
          {running ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Sparkles className="h-4 w-4" />
          )}
          {running ? 'Optimizing…' : 'Build plan'}
        </Button>
        {running && (
          <span className="text-xs text-ink-400">
            Loading the latest fit and re-optimizing per posterior draw — a moment.
          </span>
        )}
      </div>

      {jobError && <p className="text-sm text-rust-600">{jobError}</p>}

      {result && (
        <div className="space-y-4 border-t border-line-200 pt-4">
          <AllocationResult plan={result} />

          <div className="flex flex-wrap items-end gap-2 border-t border-line-200 pt-4">
            <label className="block grow">
              <span className={labelCls}>Save this plan as</span>
              <input
                type="text"
                value={planName}
                placeholder="e.g. Q3 reallocation"
                onChange={(e) => setPlanName(e.target.value)}
                className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
            </label>
            <Button variant="secondary" onClick={onSave} disabled={save.isPending}>
              <Save className="h-4 w-4" />
              {save.isPending ? 'Saving…' : 'Save plan'}
            </Button>
          </div>
          {saved && (
            <p className="text-sm text-sage-800">Saved — it's in your plans below.</p>
          )}
          {save.isError && (
            <p className="text-sm text-rust-600">{errorDetail(save.error)}</p>
          )}
        </div>
      )}
    </div>
  );
}
