import { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { Dices, Lock } from 'lucide-react';
import { Button, Drawer, TierBadge } from '../../components/ui';
import { COLORS } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import {
  useComputeDesign,
  useDesignOptions,
  useTransitionExperiment,
} from '../../api/hooks/useMeasurement';
import { useUpsertExperiment } from '../../api/hooks/usePortfolio';
import type {
  DesignKey,
  ExperimentDesignPayload,
  PrioritiesPayload,
} from '../../api/services/measurementService';

function errorDetail(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data
    ?.detail;
  return detail ?? String(e);
}

const DESIGN_LABEL: Record<DesignKey, { name: string; hint: string }> = {
  geo_lift: {
    name: 'Randomized geo lift',
    hint: 'Matched pairs, treatment randomized within pair — the gold standard when you have ≥4 geos.',
  },
  matched_market_did: {
    name: 'Matched-market DiD',
    hint: 'Pseudo-experimental: business picks the treated markets; matching + placebo bar do the defending.',
  },
  national_flighting: {
    name: 'Randomized flighting',
    hint: 'Budget-neutral on/off spend pulses that manufacture exogenous variance in a national series.',
  },
};

/**
 * The design studio: turn a priority into a runnable, pre-registerable
 * experiment. Geo data → randomized matched-pair geo lift (or matched-market
 * DiD) with a DiD power analysis; national data → block-randomized
 * budget-neutral flighting. The "Pre-register" button locks the design into
 * the registry (status: planned).
 */
export function DesignStudio({
  open,
  onClose,
  projectId,
  channels,
  defaultChannel,
  priorities,
}: {
  open: boolean;
  onClose: () => void;
  projectId: string | null;
  channels: string[];
  defaultChannel: string | null;
  priorities: PrioritiesPayload | null;
}) {
  const [channel, setChannel] = useState<string>(defaultChannel ?? channels[0] ?? '');
  const [designKey, setDesignKey] = useState<DesignKey | null>(null);
  const [duration, setDuration] = useState(8);
  const [geoDesign, setGeoDesign] = useState<'holdout' | 'scaling'>('scaling');
  const [intensity, setIntensity] = useState(50);
  const [amplitude, setAmplitude] = useState(50);
  const [blockWeeks, setBlockWeeks] = useState(2);
  const [seed, setSeed] = useState(42);
  const [design, setDesign] = useState<ExperimentDesignPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [registered, setRegistered] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setChannel(defaultChannel ?? channels[0] ?? '');
      setDesign(null);
      setError(null);
      setRegistered(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, defaultChannel]);

  const options = useDesignOptions(open ? projectId : null, channel || null);
  const effectiveKey: DesignKey | null =
    designKey && options.data?.designs.includes(designKey)
      ? designKey
      : options.data?.recommended ?? null;

  const compute = useComputeDesign(projectId);
  const upsert = useUpsertExperiment();
  const transition = useTransitionExperiment();

  const priorityRow = useMemo(
    () => priorities?.channels.find((c) => c.channel === channel) ?? null,
    [priorities, channel],
  );

  const runCompute = async (seedOverride?: number) => {
    if (!effectiveKey || !channel) return;
    setError(null);
    const s = seedOverride ?? seed;
    try {
      const result = await compute.mutateAsync({
        channel,
        design_key: effectiveKey,
        duration,
        design: geoDesign,
        intensity_pct: intensity,
        amplitude_pct: amplitude,
        block_weeks: blockWeeks,
        seed: s,
      });
      setSeed(s);
      setDesign(result);
      setRegistered(null);
    } catch (e) {
      setError(errorDetail(e));
      setDesign(null);
    }
  };

  const preregister = async () => {
    if (!design || !projectId) return;
    setError(null);
    try {
      const exp = await upsert.mutateAsync({
        channel: design.channel,
        project_id: projectId,
        status: 'draft',
        design_type: design.design_key,
        recommending_run_id: priorities?.run_id,
        design: design as unknown as Record<string, unknown>,
        priority: priorityRow as unknown as Record<string, unknown>,
      });
      await transition.mutateAsync({
        id: exp.id,
        body: { status: 'planned', note: 'pre-registered from the design studio' },
      });
      setRegistered(exp.id);
    } catch (e) {
      setError(errorDetail(e));
    }
  };

  const isGeo = effectiveKey === 'geo_lift' || effectiveKey === 'matched_market_did';

  return (
    <Drawer open={open} onClose={onClose} title="Design the next experiment" width="max-w-2xl">
      <div className="space-y-5">
        {/* ── Inputs ── */}
        <div className="grid grid-cols-2 gap-3">
          <label className="block text-sm">
            <span className="mb-1 block font-medium text-ink-700">Channel</span>
            <select
              value={channel}
              onChange={(e) => { setChannel(e.target.value); setDesign(null); }}
              className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
            >
              {channels.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="mb-1 block font-medium text-ink-700">Test duration (weeks)</span>
            <input
              type="number" min={4} max={26} value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
            />
          </label>
        </div>

        {priorityRow && (
          <p className="rounded-md bg-cream-100 px-3 py-2 text-xs text-ink-600">
            Priority context: <span className="font-medium">{priorityRow.quadrant}</span> — EIG{' '}
            <span className="num">{(priorityRow.eig ?? 0).toFixed(2)}</span> nats, EVOI{' '}
            <span className="num">{Math.round(priorityRow.evoi ?? 0).toLocaleString()}</span> KPI units.
          </p>
        )}

        {/* ── Design family ── */}
        <div>
          <span className="mb-1.5 block text-sm font-medium text-ink-700">Design</span>
          <div className="space-y-2">
            {(options.data?.designs ?? []).map((key) => (
              <button
                key={key}
                onClick={() => { setDesignKey(key); setDesign(null); }}
                className={`block w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                  effectiveKey === key
                    ? 'border-sage-600 bg-sage-100/60'
                    : 'border-line-200 bg-white hover:bg-cream-100'
                }`}
              >
                <span className="flex items-center gap-2 text-sm font-medium text-ink-900">
                  {DESIGN_LABEL[key].name}
                  {options.data?.recommended === key && (
                    <span className="rounded-full bg-sage-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sage-800">
                      recommended
                    </span>
                  )}
                </span>
                <span className="mt-0.5 block text-xs text-ink-400">{DESIGN_LABEL[key].hint}</span>
              </button>
            ))}
            {options.data && options.data.n_geos < 4 && (
              <p className="text-xs text-ink-400">
                Geo designs need ≥ 4 geographies — this dataset has {options.data.n_geos || 'no'} geo
                breakdown, so flighting is the available randomization.
              </p>
            )}
          </div>
        </div>

        {/* ── Family-specific parameters ── */}
        {isGeo ? (
          <div className="grid grid-cols-2 gap-3">
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-ink-700">Treatment</span>
              <select
                value={geoDesign}
                onChange={(e) => { setGeoDesign(e.target.value as 'holdout' | 'scaling'); setDesign(null); }}
                className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
              >
                <option value="scaling">Spend lift (scaling)</option>
                <option value="holdout">Go dark (holdout)</option>
              </select>
            </label>
            {geoDesign === 'scaling' && (
              <label className="block text-sm">
                <span className="mb-1 block font-medium text-ink-700">Spend lift %</span>
                <input
                  type="number" min={10} max={200} step={10} value={intensity}
                  onChange={(e) => setIntensity(Number(e.target.value))}
                  className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
                />
              </label>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-ink-700">Amplitude ±%</span>
              <input
                type="number" min={10} max={100} step={5} value={amplitude}
                onChange={(e) => setAmplitude(Number(e.target.value))}
                className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
            </label>
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-ink-700">Block length (weeks)</span>
              <input
                type="number" min={1} max={8} value={blockWeeks}
                onChange={(e) => setBlockWeeks(Number(e.target.value))}
                className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
            </label>
          </div>
        )}

        <div className="flex items-center gap-2">
          <Button onClick={() => runCompute()} disabled={!channel || !effectiveKey || compute.isPending}>
            {compute.isPending ? 'Computing…' : design ? 'Recompute' : 'Compute design'}
          </Button>
          <Button
            variant="secondary"
            onClick={() => runCompute(seed + 1)}
            disabled={!design || compute.isPending}
            title="New random assignment / block order (seed +1)"
          >
            <Dices className="h-4 w-4" /> Re-randomize
          </Button>
        </div>

        {error && <p className="text-sm text-rust-600">{error}</p>}

        {/* ── Results ── */}
        {design && (
          <div className="space-y-4 border-t border-line-200 pt-4">
            <div className="grid grid-cols-3 gap-3">
              <Metric label="SE (ROAS)" value={design.se_roas.toFixed(2)} />
              <Metric label="MDE @ 80% power" value={design.mde_roas.toFixed(2)} />
              <Metric
                label="Weekly spend Δ"
                value={Math.round(design.weekly_spend_delta).toLocaleString()}
              />
            </div>

            {isGeo && design.assignment && (
              <>
                <div>
                  <h4 className="mb-1.5 text-sm font-semibold text-ink-900">
                    Assignment{' '}
                    <span className="font-normal text-ink-400">
                      ({design.randomized ? 'randomized within matched pairs' : 'observational — largest geo treated'})
                    </span>
                  </h4>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-line-200 text-left text-xs uppercase tracking-wider text-ink-400">
                        <th className="py-1.5">Treatment</th>
                        <th>Control</th>
                        <th className="text-right" title="Raw pre-period KPI correlation — inflated by shared trend/seasonality">
                          Raw r
                        </th>
                        <th className="text-right" title="Co-movement after removing trend, seasonality, and spend response — what the DiD's precision actually depends on">
                          Residual r
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line-200">
                      {design.assignment.map((p) => (
                        <tr key={p.treatment}>
                          <td className="py-1.5 font-medium text-ink-900">{p.treatment}</td>
                          <td className="text-ink-700">{p.control}</td>
                          <td className="text-right num text-ink-400">{p.correlation.toFixed(2)}</td>
                          <td
                            className={`text-right num ${
                              (p.residual_correlation ?? 1) < 0.2 ? 'text-rust-600' : 'text-ink-700'
                            }`}
                          >
                            {p.residual_correlation != null ? p.residual_correlation.toFixed(2) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="mt-1.5 text-xs text-ink-400">
                    Matched on model-structured residuals (trend, seasonality, and {design.channel}{' '}
                    spend removed) + covariate distance; pairing solved globally.
                    {design.diagnostics?.max_balance_abs_std_diff != null && (
                      <>
                        {' '}Worst covariate imbalance{' '}
                        <span className={`num ${design.diagnostics.max_balance_abs_std_diff > 0.25 ? 'text-gold-700' : ''}`}>
                          {design.diagnostics.max_balance_abs_std_diff.toFixed(2)}
                        </span>{' '}
                        std (&lt; 0.25 is balanced).
                      </>
                    )}
                    {design.se_source === 'placebo_calibrated' && (
                      <> Power is simulation-calibrated against {design.placebo?.n_windows ?? 0} historical placebo windows.</>
                    )}
                  </p>
                  {design.diagnostics?.parallel_trends_warning && (
                    <p className="mt-1.5 text-xs text-rust-600">
                      ⚠ Weak residual co-movement in at least one pair — after removing shared
                      structure the pairs barely track each other; parallel trends is shaky.
                      Consider fewer, better-matched pairs.
                    </p>
                  )}
                </div>

                {design.power_curve && (
                  <div>
                    <h4 className="mb-1 text-sm font-semibold text-ink-900">Power curve</h4>
                    <Plot
                      data={[
                        {
                          x: design.power_curve.map((p) => p.duration),
                          y: design.power_curve.map((p) => p.mde_roas),
                          type: 'scatter', mode: 'lines+markers',
                          line: { color: COLORS.sage600 },
                          name: 'MDE (ROAS)',
                        } as any,
                      ]}
                      layout={mmmPlotlyLayout({
                        height: 220, margin: { t: 10, l: 50, r: 20, b: 40 },
                        xaxis: { title: { text: 'test weeks' } },
                        yaxis: { title: { text: 'MDE (ROAS)' } },
                        showlegend: false,
                      })}
                      config={PLOTLY_CONFIG as any}
                      style={{ width: '100%' }}
                      useResizeHandler
                    />
                    <p className="text-xs text-ink-400">
                      Smallest true ROAS effect the test detects with 80% power — longer tests buy precision.
                      {design.placebo?.p95_abs != null && (
                        <> Placebo bar: the pre-period produces chance "lifts" up to ±
                          <span className="num">{Math.round(design.placebo.p95_abs).toLocaleString()}</span> KPI
                          units (95%).</>
                      )}
                    </p>
                  </div>
                )}
              </>
            )}

            {!isGeo && design.schedule && (
              <div>
                <h4 className="mb-1 text-sm font-semibold text-ink-900">
                  Flighting schedule <span className="font-normal text-ink-400">(budget-neutral)</span>
                </h4>
                <Plot
                  data={[
                    {
                      x: design.schedule.map((s) => s.week_offset + 1),
                      y: design.schedule.map((s) => s.multiplier),
                      type: 'bar',
                      marker: {
                        color: design.schedule.map((s) =>
                          s.multiplier > 1 ? COLORS.sage600 : COLORS.steel300,
                        ),
                      },
                    } as any,
                  ]}
                  layout={mmmPlotlyLayout({
                    height: 200, margin: { t: 10, l: 50, r: 20, b: 40 },
                    xaxis: { title: { text: 'test week' } },
                    yaxis: { title: { text: 'spend ×' } },
                    showlegend: false,
                  })}
                  config={PLOTLY_CONFIG as any}
                  style={{ width: '100%' }}
                  useResizeHandler
                />
                {design.identification && (
                  <p className="text-xs text-ink-400">
                    <span className="num">{Math.round(design.identification.exogenous_share * 100)}%</span> of
                    test-window spend variance is randomized (clean) — historical variance co-moves with
                    demand and identifies nothing.
                  </p>
                )}
              </div>
            )}

            <div className="rounded-md bg-cream-100 px-3 py-2.5">
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-wider text-ink-400">
                Analysis plan (locked at pre-registration)
              </h4>
              <p className="text-xs leading-relaxed text-ink-600">{design.analysis_plan}</p>
            </div>

            {registered ? (
              <div className="flex items-center gap-2 rounded-md border border-sage-300 bg-sage-100/60 px-3 py-2.5 text-sm text-sage-800">
                <Lock className="h-4 w-4" /> Pre-registered (status: planned) — track it on the lifecycle board.
              </div>
            ) : (
              <Button onClick={preregister} disabled={upsert.isPending || transition.isPending}>
                <Lock className="h-4 w-4" />
                {upsert.isPending || transition.isPending ? 'Registering…' : 'Pre-register this design'}
              </Button>
            )}
          </div>
        )}

        {!design && options.data && (
          <p className="text-xs text-ink-400">
            Designs are computed from the latest run's dataset ({options.data.n_weeks} weeks
            {options.data.n_geos >= 4 ? `, ${options.data.n_geos} geos` : ', national'}).
            <TierBadge tier="model_only" compact /> Randomization is seeded and reproducible.
          </p>
        )}
      </div>
    </Drawer>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line-200 bg-white px-3 py-2.5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">{label}</div>
      <div className="mt-0.5 font-display text-xl font-semibold text-ink-900 num">{value}</div>
    </div>
  );
}
