import { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import type { Config, Data, PlotMouseEvent } from 'plotly.js';
import { Dices, Lock, FlaskConical, Loader2, Sparkles } from 'lucide-react';
import { Button, Drawer, TierBadge } from '../../components/ui';
import { COLORS } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import {
  useComputeDesign,
  useDesignOptions,
  useExperimentOptimization,
  useExperimentSimulation,
  useStructuralIdentification,
  useTransitionExperiment,
} from '../../api/hooks/useMeasurement';
import { useUpsertExperiment } from '../../api/hooks/usePortfolio';
import type {
  CandidateEval,
  DesignKey,
  DesignRequest,
  ExperimentAnchor,
  ExperimentEconomicsPayload,
  ExperimentNetValuePayload,
  ExperimentOptimizationPayload,
  ExperimentSimulation,
  ExperimentVerdict,
  OpportunityCost,
  OptimizeRequest,
  PrioritiesPayload,
  ExperimentDesignPayload,
  SimulateRequest,
  StructuralIdentificationPayload,
  StructuralParamIdent,
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
  // named analysis methodology (synthetic_control / tbr / gbr / did_mmt);
  // null = the design family's default estimator (pooled/per-pair DiD)
  const [method, setMethod] = useState<string | null>(null);
  const [duration, setDuration] = useState(8);
  const [geoDesign, setGeoDesign] = useState<'holdout' | 'scaling'>('scaling');
  const [intensity, setIntensity] = useState(50);
  const [amplitude, setAmplitude] = useState(50);
  const [blockWeeks, setBlockWeeks] = useState(2);
  const [seed, setSeed] = useState(42);
  const [design, setDesign] = useState<ExperimentDesignPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [registered, setRegistered] = useState<string | null>(null);
  // the Pareto-front candidate the active design was loaded from (if any) — kept
  // whole so the simulation can reproduce its exact basis (levels / n_pairs).
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateEval | null>(null);

  useEffect(() => {
    if (open) {
      setChannel(defaultChannel ?? channels[0] ?? '');
      setDesign(null);
      setMethod(null);
      setError(null);
      setRegistered(null);
      setSelectedCandidate(null);
      simulation.reset();
      optimization.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, defaultChannel]);

  // The simulation is anchored to a computed design + the current inputs; clear
  // it whenever any input that would change the economics changes (mirrors how
  // setDesign(null) invalidates the analysis plan on edit). The optimizer only
  // depends on the channel (it explores the design grid itself).
  useEffect(() => {
    simulation.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [channel, designKey, duration, geoDesign, intensity, amplitude, blockWeeks]);

  useEffect(() => {
    optimization.reset();
    identification.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [channel]);

  const options = useDesignOptions(open ? projectId : null, channel || null);
  const effectiveKey: DesignKey | null =
    designKey && options.data?.designs.includes(designKey)
      ? designKey
      : options.data?.recommended ?? null;

  const compute = useComputeDesign(projectId);
  const upsert = useUpsertExperiment();
  const transition = useTransitionExperiment();
  const simulation = useExperimentSimulation(projectId);
  const optimization = useExperimentOptimization(projectId);
  const identification = useStructuralIdentification(projectId);

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
        ...(method ? { method } : {}),
      });
      setSeed(s);
      setDesign(result);
      setRegistered(null);
      setSelectedCandidate(null);
      simulation.reset();
    } catch (e) {
      setError(errorDetail(e));
      setDesign(null);
    }
  };

  // Load any Pareto-front candidate as the active design: sync the visible
  // inputs (so runSimulation / the input fields reflect it) and recompute the
  // full design payload from that candidate's parameters. The user can then run
  // the simulation or pre-register it like a manually-computed design.
  const applyCandidate = async (c: CandidateEval) => {
    const key = c.design_key as DesignKey;
    const isFlighting = c.mode === 'flighting' || key === 'national_flighting';
    setDesignKey(key);
    setDuration(c.duration);
    let body: DesignRequest;
    if (isFlighting) {
      const levels = Array.from(
        new Set((c.schedule ?? []).map((s) => Number(s.multiplier.toFixed(4)))),
      ).sort((a, b) => a - b);
      // amplitude is a budget-neutral ± peak, valid in (0, 100]; clamp the
      // fallback (only used for a degenerate schedule) so it can't 400.
      const amp = Math.min(100, Math.max(1, Math.round(Math.abs(c.intensity_pct))));
      setAmplitude(amp);
      if (c.block_weeks) setBlockWeeks(c.block_weeks);
      body = {
        channel,
        design_key: key,
        duration: c.duration,
        block_weeks: c.block_weeks ?? blockWeeks,
        seed,
        ...(levels.length >= 2 ? { levels } : { amplitude_pct: amp }),
      };
    } else {
      const mode: 'holdout' | 'scaling' = c.mode === 'holdout' ? 'holdout' : 'scaling';
      // intensity_pct is SIGNED for scaling (a negative is a spend reduction) —
      // preserve the sign; only holdout (server forces −100%) uses the magnitude.
      const intens =
        mode === 'holdout'
          ? Math.round(Math.abs(c.intensity_pct))
          : Math.round(c.intensity_pct);
      setGeoDesign(mode);
      setIntensity(intens);
      body = {
        channel,
        design_key: key,
        duration: c.duration,
        design: mode,
        intensity_pct: intens,
        seed,
        ...(c.n_pairs ? { n_pairs: c.n_pairs } : {}),
      };
    }
    setError(null);
    setRegistered(null);
    setSelectedCandidate(c);
    simulation.reset();
    try {
      const result = await compute.mutateAsync(body);
      setDesign(result);
    } catch (e) {
      setError(errorDetail(e));
      setDesign(null);
      setSelectedCandidate(null);
    }
  };

  const runSimulation = () => {
    if (!effectiveKey || !channel) return;
    const body: SimulateRequest = {
      channel,
      design_key: effectiveKey,
      design: geoDesign,
      intensity_pct: intensity,
      amplitude_pct: amplitude,
      block_weeks: blockWeeks,
      duration,
      seed,
      kpi_kind: 'revenue',
    };
    // When the design was loaded from a candidate, simulate on its EXACT basis
    // (multi-level flighting levels / geo footprint) — not an on/off, full-pairs
    // approximation that would misreport power & opportunity cost.
    if (selectedCandidate) {
      const isFlighting =
        selectedCandidate.mode === 'flighting' || effectiveKey === 'national_flighting';
      if (isFlighting) {
        const levels = Array.from(
          new Set((selectedCandidate.schedule ?? []).map((s) => Number(s.multiplier.toFixed(4)))),
        ).sort((a, b) => a - b);
        if (levels.length >= 2) body.levels = levels;
      } else if (selectedCandidate.n_pairs) {
        body.n_pairs = selectedCandidate.n_pairs;
      }
    }
    simulation.start.mutate(body);
  };

  const runOptimization = (opts?: Partial<OptimizeRequest>) => {
    if (!channel) return;
    // a new front re-indexes the candidates — drop any stale selection
    setSelectedCandidate(null);
    optimization.start.mutate({ channel, kpi_kind: 'revenue', ...opts });
  };

  const preregister = async () => {
    if (!design || !projectId) return;
    setError(null);
    try {
      // Fold the model-backed anchor into the saved design so the lifecycle
      // board / drawer can show power vs the expected effect. The simulation is
      // reset on every input change (and on recompute), so a `done` result here
      // is guaranteed to match the design being registered. Shape mirrors the
      // agent path's `model_anchor` (nested `verdict`) for one canonical reader.
      const simResult =
        simulation.job.data?.status === 'done' ? simulation.job.data.result : null;
      const anchor = simResult?.anchor ?? null;
      const designToSave: Record<string, unknown> = {
        ...(design as unknown as Record<string, unknown>),
      };
      if (anchor && (anchor.prob_detectable != null || anchor.assurance != null)) {
        designToSave.model_anchor = {
          expected_effect: {
            roas_at_current_median: anchor.roas_at_current_median,
            incremental_roas_median: anchor.incremental_roas_median,
            incremental_roas_hdi: anchor.incremental_roas_hdi,
          },
          verdict: {
            verdict: anchor.verdict,
            assurance: anchor.assurance,
            prob_detectable: anchor.prob_detectable,
            incremental_roas_median: anchor.incremental_roas_median,
            recommended_duration: anchor.recommended_duration,
          },
        };
      }
      const exp = await upsert.mutateAsync({
        channel: design.channel,
        project_id: projectId,
        status: 'draft',
        design_type: design.design_key,
        recommending_run_id: priorities?.run_id,
        design: designToSave,
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
              onChange={(e) => { setChannel(e.target.value); setDesign(null); setSelectedCandidate(null); }}
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
              onChange={(e) => { setDuration(Number(e.target.value)); setSelectedCandidate(null); }}
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
                onClick={() => { setDesignKey(key); setMethod(null); setDesign(null); setSelectedCandidate(null); }}
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

        {/* ── Analysis method (named estimator from the planning.methods registry) ── */}
        {(() => {
          const family = isGeo ? 'geo' : 'switchback';
          const rows = (options.data?.methods ?? []).filter((m) => m.family === family);
          if (rows.length === 0) return null;
          return (
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-ink-700">Analysis method</span>
              <select
                value={method ?? ''}
                onChange={(e) => {
                  setMethod(e.target.value || null);
                  setDesign(null);
                  setSelectedCandidate(null);
                }}
                className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
              >
                <option value="">
                  {isGeo ? 'Default (matched-pair DiD)' : 'Default (on/off contrast)'}
                </option>
                {rows.map((m) => (
                  <option key={m.key} value={m.key} disabled={!m.supported}>
                    {m.name}
                    {!m.supported && m.reason ? ` — ${m.reason}` : ''}
                  </option>
                ))}
              </select>
              {method && (
                <span className="mt-1 block text-xs text-ink-400">
                  {rows.find((m) => m.key === method)?.description}
                </span>
              )}
            </label>
          );
        })()}

        {/* ── Family-specific parameters ── */}
        {isGeo ? (
          <div className="grid grid-cols-2 gap-3">
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-ink-700">Treatment</span>
              <select
                value={geoDesign}
                onChange={(e) => { setGeoDesign(e.target.value as 'holdout' | 'scaling'); setDesign(null); setSelectedCandidate(null); }}
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
                  onChange={(e) => { setIntensity(Number(e.target.value)); setSelectedCandidate(null); }}
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
                onChange={(e) => { setAmplitude(Number(e.target.value)); setSelectedCandidate(null); }}
                className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
            </label>
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-ink-700">Block length (weeks)</span>
              <input
                type="number" min={1} max={8} value={blockWeeks}
                onChange={(e) => { setBlockWeeks(Number(e.target.value)); setSelectedCandidate(null); }}
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

        {/* ── Pareto-front optimizer ── */}
        <OptimizerPanel
          channel={channel}
          jobStatus={optimization.job.data?.status ?? null}
          jobError={optimization.job.data?.error ?? null}
          result={
            optimization.job.data?.status === 'done'
              ? optimization.job.data.result
              : null
          }
          isStarting={optimization.start.isPending}
          startError={
            optimization.start.isError ? errorDetail(optimization.start.error) : null
          }
          onRun={runOptimization}
          onSelect={applyCandidate}
          selectedIndex={selectedCandidate?.index ?? null}
          applying={compute.isPending}
        />

        <IdentificationPanel
          channel={channel}
          jobStatus={identification.job.data?.status ?? null}
          jobError={identification.job.data?.error ?? null}
          result={
            identification.job.data?.status === 'done'
              ? identification.job.data.result
              : null
          }
          isStarting={identification.start.isPending}
          startError={
            identification.start.isError ? errorDetail(identification.start.error) : null
          }
          onRun={(opts) => identification.start.mutate({ channel, ...opts })}
        />

        {/* ── Results ── */}
        {design && (
          <div className="space-y-4 border-t border-line-200 pt-4">
            {selectedCandidate != null && (
              <div className="flex items-center gap-2 rounded-md border border-steel-300 bg-steel-100/60 px-3 py-2 text-xs text-steel-700">
                <Sparkles className="h-3.5 w-3.5 shrink-0" />
                Loaded from the Pareto front — review it below, then run the
                simulation or pre-register it.
              </div>
            )}
            <div className="grid grid-cols-3 gap-3">
              <Metric label="SE (ROAS)" value={design.se_roas.toFixed(2)} />
              <Metric label="MDE @ 80% power" value={design.mde_roas.toFixed(2)} />
              <Metric
                label={isGeo ? 'Weekly spend Δ' : 'Hi–lo spend swing'}
                value={Math.round(design.weekly_spend_delta).toLocaleString()}
              />
            </div>
            {!isGeo && design.budget_neutral && (
              <p className="-mt-1 text-xs text-ink-400">
                The swing is the per-week spend difference between high and low
                blocks (what powers the ROAS estimate); the design is
                budget-neutral, so the <span className="font-medium">net</span>{' '}
                weekly spend is unchanged.
              </p>
            )}

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
                        } as Data,
                      ]}
                      layout={mmmPlotlyLayout({
                        height: 220, margin: { t: 10, l: 50, r: 20, b: 40 },
                        xaxis: { title: { text: 'test weeks' } },
                        yaxis: { title: { text: 'MDE (ROAS)' } },
                        showlegend: false,
                      })}
                      config={PLOTLY_CONFIG as Partial<Config>}
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
                    } as Data,
                  ]}
                  layout={mmmPlotlyLayout({
                    height: 200, margin: { t: 10, l: 50, r: 20, b: 40 },
                    xaxis: { title: { text: 'test week' } },
                    yaxis: { title: { text: 'spend ×' } },
                    showlegend: false,
                  })}
                  config={PLOTLY_CONFIG as Partial<Config>}
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
              {design.method_name && (
                <p className="mb-1 text-xs font-medium text-ink-700">
                  Method: {design.method_name}
                </p>
              )}
              <p className="text-xs leading-relaxed text-ink-600">{design.analysis_plan}</p>
            </div>

            {/* ── Model-backed risk & methodology simulation ── */}
            <SimulationPanels
              jobStatus={simulation.job.data?.status ?? null}
              jobError={simulation.job.data?.error ?? null}
              result={
                simulation.job.data?.status === 'done'
                  ? simulation.job.data.result
                  : null
              }
              isStarting={simulation.start.isPending}
              startError={
                simulation.start.isError ? errorDetail(simulation.start.error) : null
              }
              onRun={runSimulation}
            />

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

/** A single estimand's power %, coloured by whether it clears the 80% target. */
function PowerStat({
  label,
  value,
  lower,
}: {
  label: string;
  value: number | null;
  /** conservative power at the model's lower 95% effect bound */
  lower?: number | null;
}) {
  const pct = value != null ? `${Math.round(value * 100)}%` : '—';
  const ok = value != null && value >= 0.8;
  return (
    <div>
      <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div className={`num text-base font-semibold ${ok ? 'text-sage-800' : 'text-ink-900'}`}>
        {pct}
      </div>
      {lower != null && (
        <div className="num text-[10px] text-ink-400" title="power if the channel's true effect is at the model's lower 95% bound">
          lo 95%: {Math.round(lower * 100)}%
        </div>
      )}
    </div>
  );
}

// ── Model-backed economics panels ─────────────────────────────────────────────

const fmtInt = (x: number | null | undefined): string =>
  x == null || !Number.isFinite(x) ? '—' : Math.round(x).toLocaleString();

const fmtSignedInt = (x: number | null | undefined): string =>
  x == null || !Number.isFinite(x)
    ? '—'
    : `${x >= 0 ? '+' : ''}${Math.round(x).toLocaleString()}`;

const fmtPct = (x: number | null | undefined, digits = 1): string =>
  x == null || !Number.isFinite(x) ? '—' : `${(x * 100).toFixed(digits)}%`;

const fmtNum = (x: number | null | undefined, digits = 2): string =>
  x == null || !Number.isFinite(x) ? '—' : x.toFixed(digits);

const VERDICT_STYLE: Record<ExperimentVerdict, string> = {
  powered: 'bg-sage-100 text-sage-800',
  overpowered: 'bg-gold-100 text-gold-700',
  underpowered: 'bg-rust-100 text-rust-600',
  inconclusive: 'bg-rust-100 text-rust-600',
};

const LTC_BASIS_LABEL: Record<string, string> = {
  kpi_per_week: 'EVOI per week ÷ KPI cost per week',
  channel_already_precise: 'channel already precise — little to learn',
  net_neutral_design: 'budget-neutral design — no KPI cost basis',
  unavailable: 'EVOI unavailable (no model loopback)',
};

function SimulationPanels({
  jobStatus,
  jobError,
  result,
  isStarting,
  startError,
  onRun,
}: {
  jobStatus: 'pending' | 'running' | 'done' | 'error' | null;
  jobError: string | null;
  result: ExperimentEconomicsPayload | null;
  isStarting: boolean;
  startError: string | null;
  onRun: () => void;
}) {
  const running = isStarting || jobStatus === 'pending' || jobStatus === 'running';

  return (
    <div className="space-y-4 border-t border-line-200 pt-4">
      <div className="flex items-center gap-3">
        <Button variant="secondary" onClick={onRun} disabled={running}>
          {running ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <FlaskConical className="h-4 w-4" />
          )}
          {running ? 'Simulating…' : 'Run model-backed risk & simulation'}
        </Button>
        {running && (
          <span className="text-xs text-ink-400">
            Loading the latest fit and running A/A · A/B on history — this can take a moment.
          </span>
        )}
      </div>

      {startError && <p className="text-sm text-rust-600">{startError}</p>}
      {jobStatus === 'error' && jobError && (
        <p className="text-sm text-rust-600">{jobError}</p>
      )}

      {result && (
        <>
          {result.net_value && result.net_value.net_value != null && (
            <NetValueCard nv={result.net_value} />
          )}

          {result.opportunity_cost ? (
            <OpportunityCostPanel
              oc={result.opportunity_cost}
              modelAnchored={result.model_anchored}
            />
          ) : result.model_anchored === false ? (
            <p className="rounded-md bg-cream-100 px-3 py-2.5 text-xs text-ink-600">
              No fitted model — fit a model for opportunity cost. The methodology
              comparison below still runs on historical data.
            </p>
          ) : result.opportunity_cost_error ? (
            <p className="text-xs text-rust-600">{result.opportunity_cost_error}</p>
          ) : null}

          {result.anchor && <AnchorPanel anchor={result.anchor} />}

          {result.simulation && <MethodologyPanel sim={result.simulation} />}
        </>
      )}
    </div>
  );
}

/** Phase-3 headline: reallocation gain (decayed, EVPI-capped) − test loss. */
function NetValueCard({ nv }: { nv: ExperimentNetValuePayload }) {
  const dollar = nv.unit === '$';
  const fmt = (v: number | null | undefined) =>
    v == null ? '—' : `${dollar ? '$' : ''}${Math.round(v).toLocaleString()}${dollar ? '' : ' KPI'}`;
  const positive = (nv.net_value ?? 0) > 0;
  const gain = nv.reallocation_gain ?? 0;
  const loss = Math.max(nv.test_loss ?? 0, 0);
  const total = Math.max(gain + loss, 1e-9);
  return (
    <div
      className={`rounded-lg border px-4 py-3 ${
        positive ? 'border-sage-300 bg-sage-100/50' : 'border-rust-300 bg-rust-100/30'
      }`}
    >
      <div className="flex items-baseline justify-between">
        <h4 className="text-sm font-semibold text-ink-900">Is this test worth running?</h4>
        <span className={`num text-lg font-semibold ${positive ? 'text-sage-800' : 'text-rust-800'}`}>
          {(nv.net_value ?? 0) >= 0 ? '+' : '−'}
          {fmt(Math.abs(nv.net_value ?? 0))}
        </span>
      </div>
      {/* gain-vs-loss bar */}
      <div className="mt-2 flex h-2.5 w-full overflow-hidden rounded-full bg-line-200">
        <div className="h-full bg-sage-600" style={{ width: `${(100 * gain) / total}%` }} />
        <div className="h-full bg-rust-500" style={{ width: `${(100 * loss) / total}%` }} />
      </div>
      <p className="mt-2 text-xs text-ink-600">
        Expected reallocation gain <span className="num text-sage-800">{fmt(gain)}</span> (EVOI,
        decay-adjusted{nv.decay_factor != null ? ` ×${nv.decay_factor.toFixed(2)}` : ''}
        {nv.evpi_cap != null ? ', EVPI-capped' : ''}) vs expected test loss{' '}
        <span className="num text-rust-800">{fmt(loss)}</span>
        {nv.prob_net_positive != null && (
          <>
            {' '}— <span className="num font-medium">{Math.round(nv.prob_net_positive * 100)}%</span>{' '}
            chance net-positive
          </>
        )}
        .
      </p>
      <p className="mt-1 text-xs text-ink-400">
        {nv.breakeven_horizon_weeks == null
          ? nv.basis !== 'insufficient'
            ? 'The decayed learning value never repays the test loss within 10× the horizon.'
            : ''
          : nv.breakeven_horizon_weeks <= 0
            ? 'Break-even immediately — the test itself is not expected to lose money.'
            : `Break-even in ≈ ${Math.round(nv.breakeven_horizon_weeks)} weeks of reallocation value.`}{' '}
        Basis: {nv.basis === 'model_anchored' ? 'model-anchored realized precision' : nv.basis}.
        {nv.net_value_p5 != null && nv.net_value_p95 != null && (
          <>
            {' '}
            90% interval [{fmt(nv.net_value_p5)}, {fmt(nv.net_value_p95)}].
          </>
        )}
      </p>
    </div>
  );
}

function OpportunityCostPanel({
  oc,
  modelAnchored,
}: {
  oc: OpportunityCost;
  modelAnchored: boolean;
}) {
  const hasMargin = oc.margin_source !== 'none';
  return (
    <div>
      <h4 className="mb-1.5 text-sm font-semibold text-ink-900">
        Opportunity cost &amp; short-term risk{' '}
        <span className="font-normal text-ink-400">
          (window-only deviation from business-as-usual)
        </span>
      </h4>

      <div className="grid grid-cols-3 gap-3">
        <Metric label="Forgone KPI (median)" value={fmtInt(oc.forgone_kpi_median)} />
        <Metric label="% of treated-window KPI" value={fmtPct(oc.pct_of_window_kpi)} />
        <Metric label="Prob of KPI loss" value={fmtPct(oc.prob_kpi_loss, 0)} />
      </div>

      <p className="mt-2 text-xs text-ink-600">
        KPI delta (90% interval): [
        <span className="num">{fmtInt(oc.kpi_delta_p5)}</span>,{' '}
        <span className="num">{fmtInt(oc.kpi_delta_p95)}</span>] — worst case forgo up to{' '}
        <span className="num">{fmtInt(oc.forgone_kpi_p95)}</span> KPI (95%). Spend Δ{' '}
        <span className="num">{fmtSignedInt(oc.spend_delta)}</span>, spend at risk{' '}
        <span className="num">{fmtInt(oc.spend_at_risk)}</span>.
      </p>

      {hasMargin ? (
        <div className="mt-3 grid grid-cols-2 gap-3">
          <Metric label="Net $ impact (median)" value={fmtSignedInt(oc.net_profit_impact_median)} />
          <Metric label="Opportunity cost $ (median)" value={fmtInt(oc.opportunity_cost_dollar_median)} />
        </div>
      ) : (
        <p className="mt-2 text-xs text-ink-400">
          Supply a gross margin to see net-$ impact.
        </p>
      )}

      {hasMargin && (
        <p className="mt-2 text-xs text-ink-600">
          Net $ 90% interval: [
          <span className="num">{fmtSignedInt(oc.net_profit_impact_p5)}</span>,{' '}
          <span className="num">{fmtSignedInt(oc.net_profit_impact_p95)}</span>]; downside (p95)
          opportunity cost <span className="num">{fmtInt(oc.opportunity_cost_dollar_p95)}</span>.
          Margin source: {oc.margin_source}.
        </p>
      )}

      <p className="mt-2 text-xs text-ink-400">
        Learning-vs-cost ratio{' '}
        <span className="num text-ink-700">{fmtNum(oc.learning_to_cost_ratio)}</span>{' '}
        <span className="text-ink-400">
          ({LTC_BASIS_LABEL[oc.learning_to_cost_basis] ?? oc.learning_to_cost_basis})
        </span>
        .{' '}
        {oc.duration_effective} effective week{oc.duration_effective === 1 ? '' : 's'},{' '}
        {oc.n_test_rows} treated row{oc.n_test_rows === 1 ? '' : 's'}.
      </p>

      {!modelAnchored && (
        <p className="mt-1.5 text-xs text-rust-600">
          Fit a model for opportunity cost — only design-level figures are available.
        </p>
      )}
      {oc.low_information && (
        <p className="mt-1.5 text-xs text-gold-700">
          ⚠ Near-zero posterior spread on the KPI delta — the intervals are deceptively
          tight; treat the point estimate cautiously.
        </p>
      )}
      {oc.extrapolation_warning && (
        <p className="mt-1.5 text-xs text-gold-700">
          ⚠ Scaled spend exceeds the channel's observed range — that response is
          extrapolation, not evidence.
        </p>
      )}
    </div>
  );
}

function AnchorPanel({ anchor }: { anchor: ExperimentAnchor }) {
  return (
    <div className="rounded-md border border-line-200 bg-white px-3 py-2.5">
      <h4 className="mb-1.5 flex items-center gap-2 text-sm font-semibold text-ink-900">
        Model anchor
        <span
          className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
            VERDICT_STYLE[anchor.verdict] ?? 'bg-cream-200 text-ink-600'
          }`}
        >
          {anchor.verdict}
        </span>
      </h4>
      <p className="text-xs text-ink-600">
        Expected incremental ROAS{' '}
        <span className="num font-medium text-ink-900">
          {fmtNum(anchor.incremental_roas_median)}
        </span>{' '}
        (90% HDI [
        <span className="num">{fmtNum(anchor.incremental_roas_hdi?.[0])}</span>,{' '}
        <span className="num">{fmtNum(anchor.incremental_roas_hdi?.[1])}</span>]). Assurance{' '}
        <span className="num">{fmtPct(anchor.assurance, 0)}</span> of detecting the expected effect.
        {anchor.recommended_duration != null && (
          <>
            {' '}Reach power at ≈{' '}
            <span className="num">{anchor.recommended_duration}</span> weeks.
          </>
        )}
      </p>
      {anchor.extrapolation_warning && (
        <p className="mt-1.5 text-xs text-gold-700">
          ⚠ The anchored effect relies on spend beyond the observed range.
        </p>
      )}
    </div>
  );
}

function MethodologyPanel({ sim }: { sim: ExperimentSimulation }) {
  const chosen = sim.methodologies.find((m) => m.key === sim.chosen_key) ?? null;
  const curve = chosen?.power_curve ?? [];
  return (
    <div>
      <h4 className="mb-1.5 text-sm font-semibold text-ink-900">
        Methodology comparison{' '}
        <span className="font-normal text-ink-400">(A/A · A/B)</span>
      </h4>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line-200 text-left text-xs uppercase tracking-wider text-ink-400">
            <th className="py-1.5">Method</th>
            <th
              className="text-right"
              title="Empirical false-positive rate of the analytic decision rule on un-treated history (should be near α=0.05)"
            >
              A/A FPR
            </th>
            <th className="text-right" title="Smallest detectable effect in ROAS units at 80% power">
              MDE (ROAS)
            </th>
            <th className="text-right" title="Power to detect the model's expected effect, at calibrated size">
              Power @ exp.
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line-200">
          {sim.methodologies.map((m) => (
            <tr key={m.key}>
              <td className="py-1.5 font-medium text-ink-900">
                {m.label}
                {m.key === sim.chosen_key && (
                  <span className="ml-1.5 rounded-full bg-sage-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sage-800">
                    recommended
                  </span>
                )}
              </td>
              <td className={`text-right num ${m.fpr_inflated ? 'text-rust-600' : 'text-ink-700'}`}>
                {fmtNum(m.fpr)}
                <span className="ml-1 text-[10px] text-ink-400">
                  [{fmtNum(m.fpr_ci?.[0])}–{fmtNum(m.fpr_ci?.[1])}]
                </span>
              </td>
              <td className="text-right num text-ink-700">{fmtNum(m.empirical_mde_roas)}</td>
              <td className="text-right num text-ink-700">{fmtPct(m.power_at_expected_effect, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {curve.length > 1 && (
        <div className="mt-2">
          <Plot
            data={[
              {
                x: curve.map((p) => p.effect),
                y: curve.map((p) => p.power),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: COLORS.sage600 },
                name: 'power',
              } as Data,
            ]}
            layout={mmmPlotlyLayout({
              height: 200,
              margin: { t: 10, l: 50, r: 20, b: 40 },
              xaxis: { title: { text: 'injected effect (KPI units)' } },
              yaxis: { title: { text: 'power' }, range: [0, 1] },
              showlegend: false,
            })}
            config={PLOTLY_CONFIG as Partial<Config>}
            style={{ width: '100%' }}
            useResizeHandler
          />
          <p className="text-xs text-ink-400">
            Empirical power vs injected effect for{' '}
            <span className="font-medium">{chosen?.label}</span> (the recommended estimator).
          </p>
        </div>
      )}

      <p className="mt-1.5 text-xs text-ink-400">
        Injection basis: {sim.injection_basis}. {sim.caveats.join(' ')}
      </p>
    </div>
  );
}

// ── Pareto-front experiment optimizer ─────────────────────────────────────────

const MODE_LABEL: Record<CandidateEval['mode'], string> = {
  holdout: 'go-dark holdout',
  scaling: 'spend lift',
  flighting: 'flighting',
};

const fmtSignedPct = (x: number | null | undefined): string =>
  x == null || !Number.isFinite(x) ? '—' : `${x >= 0 ? '+' : ''}${Math.round(x)}%`;

function OptimizerPanel({
  channel,
  jobStatus,
  jobError,
  result,
  isStarting,
  startError,
  onRun,
  onSelect,
  selectedIndex,
  applying,
}: {
  channel: string;
  jobStatus: 'pending' | 'running' | 'done' | 'error' | null;
  jobError: string | null;
  result: ExperimentOptimizationPayload | null;
  isStarting: boolean;
  startError: string | null;
  onRun: (opts: Partial<OptimizeRequest>) => void;
  onSelect: (c: CandidateEval) => void;
  selectedIndex: number | null;
  applying: boolean;
}) {
  const running = isStarting || jobStatus === 'pending' || jobStatus === 'running';
  // design-space ranges the optimizer searches within
  const [durMin, setDurMin] = useState(4);
  const [durMax, setDurMax] = useState(12);
  const [intMin, setIntMin] = useState(50);
  const [intMax, setIntMax] = useState(100);
  const [includeHoldout, setIncludeHoldout] = useState(true);

  const run = () =>
    onRun({
      duration_min: durMin,
      duration_max: durMax,
      intensity_min: intMin,
      intensity_max: intMax,
      include_holdout: includeHoldout,
    });

  return (
    <div className="space-y-4 border-t border-line-200 pt-4">
      <div>
        {/* ── design-space ranges ── */}
        <div className="mb-3 grid grid-cols-2 gap-3">
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-ink-700">
              Duration range (weeks)
            </span>
            <div className="flex items-center gap-1.5">
              <input
                type="number" min={1} max={52} value={durMin}
                onChange={(e) => setDurMin(Number(e.target.value))}
                className="w-full rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
              <span className="text-ink-400">–</span>
              <input
                type="number" min={1} max={52} value={durMax}
                onChange={(e) => setDurMax(Number(e.target.value))}
                className="w-full rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
            </div>
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-ink-700">
              Spend variation range (%)
            </span>
            <div className="flex items-center gap-1.5">
              <input
                type="number" min={-100} max={500} step={10} value={intMin}
                onChange={(e) => setIntMin(Number(e.target.value))}
                className="w-full rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
              <span className="text-ink-400">–</span>
              <input
                type="number" min={-100} max={500} step={10} value={intMax}
                onChange={(e) => setIntMax(Number(e.target.value))}
                className="w-full rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
            </div>
          </label>
        </div>
        <label className="mb-3 flex items-center gap-2 text-xs text-ink-600">
          <input
            type="checkbox"
            checked={includeHoldout}
            onChange={(e) => setIncludeHoldout(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-line-300 text-sage-600 focus:ring-sage-600"
          />
          Also test a go-dark holdout (−100%)
        </label>

        <div className="flex items-center gap-3">
          <Button variant="secondary" onClick={run} disabled={!channel || running}>
            {running ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {running ? 'Optimizing…' : 'Find best experiments (Pareto)'}
          </Button>
          {running && (
            <span className="text-xs text-ink-400">
              Exploring the design grid (footprint × intensity × duration) for the
              non-dominated front — this can take a moment.
            </span>
          )}
        </div>
        {!result && !running && (
          <p className="mt-1.5 text-xs text-ink-400">
            Optimize the experiment setup itself: the Pareto front trades off MDE
            (precision) × statistical power (target 80%) × short-term cost × duration,
            with a recommended runnable design — searched within the ranges above
            (−100% = go dark, positive = scale up).
          </p>
        )}
      </div>

      {startError && <p className="text-sm text-rust-600">{startError}</p>}
      {jobStatus === 'error' && jobError && (
        <p className="text-sm text-rust-600">{jobError}</p>
      )}

      {result && (
        <>
          {result.n_candidates > 0 ? (
            <ParetoScatter result={result} onSelect={onSelect} selectedIndex={selectedIndex} />
          ) : (
            <p className="rounded-md bg-cream-100 px-3 py-2.5 text-xs text-ink-600">
              No feasible design found for this channel and dataset.
            </p>
          )}
          {result.recommended && (
            <RecommendedCard
              result={result}
              onSelect={onSelect}
              selectedIndex={selectedIndex}
              applying={applying}
            />
          )}
          {result.pareto.length > 0 && (
            <ParetoTable
              pareto={result.pareto}
              onSelect={onSelect}
              selectedIndex={selectedIndex}
              applying={applying}
            />
          )}
        </>
      )}
    </div>
  );
}

const IDENT_HINT: Record<string, string> = {
  beta: 'widen the spend contrast',
  alpha: 'sharpen / lengthen the spend pulses',
  lam: 'add ≥3 in-support spend levels',
};

function IdentRow({
  name,
  paramKey,
  d,
}: {
  name: string;
  paramKey: 'beta' | 'alpha' | 'lam';
  d: StructuralParamIdent;
}) {
  return (
    <tr className="border-t border-line-200">
      <td className="py-1.5 pr-3 text-xs font-medium text-ink-700">{name}</td>
      {d.claimed ? (
        <>
          <td className="num py-1.5 pr-3 text-xs text-ink-900">{fmtPct(d.contraction, 0)}</td>
          <td className="num py-1.5 pr-3 text-xs text-ink-700">
            {d.mde != null ? d.mde.toPrecision(3) : '—'}
            {d.mde_relative != null && (
              <span className="text-ink-400"> ({fmtPct(d.mde_relative, 0)} of est.)</span>
            )}
          </td>
          <td className="num py-1.5 text-xs text-ink-700">{fmtPct(d.power, 0)}</td>
        </>
      ) : (
        <td colSpan={3} className="py-1.5 text-xs text-gold-700">
          not identified by this design — {IDENT_HINT[paramKey]}
        </td>
      )}
    </tr>
  );
}

/**
 * Structural identification: design a multi-level flighting schedule and show
 * how well its refit would pin the channel's saturation / adstock / beta — an
 * optimistic Laplace UPPER BOUND on the next refit, never a guarantee.
 */
function IdentificationPanel({
  channel,
  jobStatus,
  jobError,
  result,
  isStarting,
  startError,
  onRun,
}: {
  channel: string;
  jobStatus: 'pending' | 'running' | 'done' | 'error' | null;
  jobError: string | null;
  result: StructuralIdentificationPayload | null;
  isStarting: boolean;
  startError: string | null;
  onRun: (opts: { levels?: number[]; block_weeks?: number; duration?: number }) => void;
}) {
  const running = isStarting || jobStatus === 'pending' || jobStatus === 'running';
  const [levelsText, setLevelsText] = useState('0.5, 1, 1.5');
  const [blockWeeks, setBlockWeeks] = useState(''); // '' = auto (adstock cool-down)
  const [duration, setDuration] = useState(12);

  const run = () => {
    const levels = levelsText
      .split(/[,\s]+/)
      .map(Number)
      .filter((m) => Number.isFinite(m) && m >= 0)
      .slice(0, 8);
    onRun({
      ...(levels.length >= 2 ? { levels } : {}),
      ...(blockWeeks !== '' ? { block_weeks: Math.max(1, Math.round(Number(blockWeeks))) } : {}),
      duration,
    });
  };

  const struct = result?.structural ?? null;

  return (
    <div className="space-y-4 border-t border-line-200 pt-4">
      <div>
        <div className="mb-3 grid grid-cols-3 gap-3">
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-ink-700">
              Spend levels (× current)
            </span>
            <input
              value={levelsText}
              onChange={(e) => setLevelsText(e.target.value)}
              placeholder="0.5, 1, 1.5"
              className="num w-full rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-ink-700">
              Block weeks (blank = washout)
            </span>
            <input
              type="number" min={1} max={13} value={blockWeeks}
              onChange={(e) => setBlockWeeks(e.target.value)}
              placeholder="auto"
              className="num w-full rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-ink-700">
              Duration (weeks)
            </span>
            <input
              type="number" min={4} max={52} value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="num w-full rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
            />
          </label>
        </div>

        <div className="flex items-center gap-3">
          <Button variant="secondary" onClick={run} disabled={!channel || running}>
            {running ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <FlaskConical className="h-4 w-4" />
            )}
            {running ? 'Analyzing…' : 'Check structural identification'}
          </Button>
          {running && (
            <span className="text-xs text-ink-400">
              Designing the multi-level flighting schedule and scoring the Laplace
              information it would add…
            </span>
          )}
        </div>
        {!result && !running && (
          <p className="mt-1.5 text-xs text-ink-400">
            Would the next refit actually pin this channel's saturation curve, adstock
            carryover, and coefficient? Designs a multi-level flighting schedule and
            reports each parameter's expected posterior contraction — an optimistic
            upper bound, not a guarantee.
          </p>
        )}
      </div>

      {(startError || jobError) && (
        <p className="text-sm text-rust-600">{startError ?? jobError}</p>
      )}

      {result && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full bg-steel-100 px-2.5 py-1 text-steel-700">
              <span className="num">{result.n_levels}</span> spend levels
            </span>
            <span className="rounded-full bg-steel-100 px-2.5 py-1 text-steel-700">
              <span className="num">{result.block_weeks}</span>w blocks ·{' '}
              <span className="num">{result.duration}</span>w total
            </span>
            <span
              className={
                result.block_ge_cooldown
                  ? 'rounded-full bg-sage-100 px-2.5 py-1 text-sage-800'
                  : 'rounded-full bg-gold-100 px-2.5 py-1 text-gold-700'
              }
            >
              adstock washout {result.cooldown_weeks ?? '?'}w{' '}
              {result.block_ge_cooldown ? '≤ block ✓' : '> block — carryover smears the contrast'}
            </span>
            {result.extrapolation_warning && (
              <span className="rounded-full bg-gold-100 px-2.5 py-1 text-gold-700">
                top level clamped to historical spend support
              </span>
            )}
          </div>

          {!result.structural_gated ? (
            <p className="rounded-md bg-cream-100 px-3 py-2 text-xs text-ink-600">
              Structural readout unavailable
              {result.structural_gate_reason ? ` — ${result.structural_gate_reason}` : ''}.
              The reduced-form curve / marginal power still applies.
            </p>
          ) : struct ? (
            <>
              <table className="w-full">
                <thead>
                  <tr className="text-left text-[11px] uppercase tracking-wider text-ink-400">
                    <th className="pb-1 pr-3 font-medium">Parameter</th>
                    <th className="pb-1 pr-3 font-medium">Contraction</th>
                    <th className="pb-1 pr-3 font-medium">MDE</th>
                    <th className="pb-1 font-medium">Power vs 0</th>
                  </tr>
                </thead>
                <tbody>
                  <IdentRow name="Coefficient β" paramKey="beta" d={struct.params.beta} />
                  <IdentRow name="Adstock α (carryover)" paramKey="alpha" d={struct.params.alpha} />
                  <IdentRow name="Saturation ψ" paramKey="lam" d={struct.params.lam} />
                </tbody>
              </table>
              {struct.identifies_anything ? (
                <p className="text-xs text-ink-600">
                  Binding identification:{' '}
                  <span className="num font-medium">{fmtPct(struct.binding_contraction, 0)}</span>{' '}
                  contraction (power {fmtPct(struct.binding_power, 0)}, target{' '}
                  {fmtPct(struct.power_target, 0)}) — the worst-identified claimed parameter.
                  Recommended estimator: a full structural refit with the experiment weeks
                  appended. This is an optimistic upper bound.
                </p>
              ) : (
                <p className="rounded-md bg-gold-100 px-3 py-2 text-xs text-gold-700">
                  This schedule doesn't move the structural parameters off their priors —
                  add spend levels and sharpen the pulses.
                </p>
              )}
              {(struct.n_clamped ?? 0) > 0 && (
                <p className="text-xs text-gold-700">
                  <span className="num">{struct.n_clamped}</span> test week(s) are fully
                  saturated (no marginal information there) — keep levels in the responsive
                  range.
                </p>
              )}
            </>
          ) : (
            <p className="rounded-md bg-cream-100 px-3 py-2 text-xs text-ink-600">
              {result.note ?? 'Structural design degenerate — no identifiable contrast.'}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function ParetoScatter({
  result,
  onSelect,
  selectedIndex,
}: {
  result: ExperimentOptimizationPayload;
  onSelect: (c: CandidateEval) => void;
  selectedIndex: number | null;
}) {
  // Label/units come from the backend's resolved tradeoff_label (derived from
  // the candidates' tradeoff_basis), not the global margin_known — which can
  // disagree with a candidate's actual basis.
  const tradeoffLabel = result.tradeoff_label ?? 'short-term cost';
  const isDollar = tradeoffLabel.includes('$');
  // Net-value axis: the backend's tradeoff is −net_value (lower-better Pareto
  // convention); plot the net value itself so up = better on the chart.
  const netAxis = result.net_value_axis === true;
  const yOf = (c: CandidateEval): number =>
    netAxis ? (c.net_value ?? -c.tradeoff) : c.tradeoff;
  const powerPct = (c: CandidateEval): string =>
    c.power != null ? `${Math.round(c.power * 100)}%` : 'n/a';
  const costText = (c: CandidateEval): string => {
    if (netAxis) {
      const nv = c.net_value ?? -c.tradeoff;
      return `net ${nv >= 0 ? '+' : '−'}$${fmtInt(Math.abs(nv))}`;
    }
    return `${isDollar ? '$' : ''}${fmtInt(c.tradeoff)}`;
  };
  const hover = (c: CandidateEval): string =>
    `${MODE_LABEL[c.mode]} · ${c.footprint} · ${fmtSignedPct(c.intensity_pct)} · ${
      c.duration
    }w<br>MDE ${fmtNum(c.mde_roas)} · power ${powerPct(c)} · ${costText(c)}${
      c.powered ? ' · powered' : ' · underpowered'
    }`;

  // All candidates — dominated points are muted; the front is emphasized; the
  // recommended design carries a distinct star + annotation.
  const dominated = result.candidates.filter((c) => !c.on_pareto && !c.is_recommended);
  const front = result.candidates.filter((c) => c.on_pareto && !c.is_recommended);
  const recommended = result.recommended;

  const data: Data[] = [
    {
      x: dominated.map((c) => c.mde_roas),
      y: dominated.map(yOf),
      text: dominated.map(hover),
      customdata: dominated.map((c) => c.index),
      type: 'scatter',
      mode: 'markers',
      name: 'dominated',
      marker: {
        size: 9,
        color: dominated.map((c) => c.duration),
        colorscale: 'YlGnBu',
        reversescale: true,
        opacity: 0.4,
        line: { width: 0 },
        colorbar: { title: { text: 'weeks' }, thickness: 10, len: 0.6 },
      },
      hovertemplate: '%{text}<extra></extra>',
    },
    {
      x: front.map((c) => c.mde_roas),
      y: front.map(yOf),
      text: front.map(hover),
      customdata: front.map((c) => c.index),
      type: 'scatter',
      mode: 'markers',
      name: 'Pareto front',
      marker: {
        size: 14,
        color: front.map((c) => c.duration),
        colorscale: 'YlGnBu',
        reversescale: true,
        opacity: 0.95,
        // under-powered front designs (below the power target) get a rust ring.
        line: {
          width: 2,
          color: front.map((c) => (c.powered ? COLORS.ink700 : COLORS.rust600)),
        },
        showscale: false,
      },
      hovertemplate: '%{text}<extra></extra>',
    },
  ];

  if (recommended) {
    data.push({
      x: [recommended.mde_roas],
      y: [yOf(recommended)],
      text: [hover(recommended)],
      customdata: [recommended.index],
      type: 'scatter',
      mode: 'markers',
      name: 'recommended',
      marker: {
        size: 20,
        symbol: 'star',
        color: COLORS.gold600,
        line: { width: 1.5, color: COLORS.ink900 },
      },
      hovertemplate: '%{text}<extra></extra>',
    });
  }

  // A ring around the currently-loaded candidate, so clicking a point gives
  // immediate feedback ON the chart (the table row also highlights below).
  const selected =
    selectedIndex != null
      ? result.candidates.find((c) => c.index === selectedIndex)
      : null;
  if (selected) {
    data.push({
      x: [selected.mde_roas],
      y: [yOf(selected)],
      type: 'scatter',
      mode: 'markers',
      name: 'selected',
      marker: {
        size: 24,
        symbol: 'circle-open',
        color: COLORS.steel600,
        line: { width: 3, color: COLORS.steel600 },
      },
      hoverinfo: 'skip',
      showlegend: false,
    });
  }

  const layout = mmmPlotlyLayout({
    height: 300,
    margin: { t: 14, l: 64, r: 20, b: 44 },
    xaxis: { title: { text: 'MDE (ROAS)' } },
    yaxis: { title: { text: tradeoffLabel } },
    showlegend: false,
    ...(recommended
      ? {
          annotations: [
            {
              x: recommended.mde_roas,
              y: yOf(recommended),
              text: 'recommended',
              showarrow: true,
              arrowhead: 0,
              ax: 28,
              ay: -28,
              font: { size: 10, color: COLORS.ink700 },
              bgcolor: COLORS.gold100,
              bordercolor: COLORS.gold300,
              borderpad: 2,
            },
          ],
        }
      : {}),
  });

  return (
    <div>
      <h4 className="mb-1 text-sm font-semibold text-ink-900">
        Pareto front{' '}
        <span className="font-normal text-ink-400">
          {netAxis
            ? '(upper-left is better — precise & net-positive to run)'
            : '(lower-left is better — precise & cheap)'}
        </span>
      </h4>
      <Plot
        data={data}
        layout={layout}
        config={PLOTLY_CONFIG as Partial<Config>}
        style={{ width: '100%' }}
        useResizeHandler
        onClick={(e: Readonly<PlotMouseEvent>) => {
          const p = e?.points?.[0];
          if (!p) return;
          // Map the clicked point back to its candidate by (trace, point) — the
          // trace order here mirrors `data` above: dominated, front, [recommended].
          const traces: CandidateEval[][] = [dominated, front];
          if (recommended) traces.push([recommended]);
          const pn = p.pointNumber ?? p.pointIndex;
          const c =
            traces[p.curveNumber]?.[pn] ??
            (typeof p.customdata === 'number'
              ? result.candidates.find((cc) => cc.index === p.customdata)
              : undefined);
          if (c) onSelect(c);
        }}
      />
      <p className="text-xs text-ink-400">
        Each point is a candidate design coloured by duration; the{' '}
        <span className="num">{result.pareto.length}</span> ringed points are
        non-dominated over {result.n_candidates} designs (MDE × statistical power ×
        short-term cost × duration). A{' '}
        <span className="text-rust-600">rust ring</span> marks a design below the{' '}
        <span className="num">{Math.round((result.power_target ?? 0.8) * 100)}%</span>{' '}
        power target. The gold star is the recommended setup (the powered knee).{' '}
        <span className="text-ink-600">Click any point to load it as the design.</span>
      </p>
    </div>
  );
}

function RecommendedCard({
  result,
  onSelect,
  selectedIndex,
  applying,
}: {
  result: ExperimentOptimizationPayload;
  onSelect: (c: CandidateEval) => void;
  selectedIndex: number | null;
  applying: boolean;
}) {
  const recommended = result.recommended!;
  // Show the SELECTED candidate's setup once one is loaded; otherwise the
  // recommended one. All fields below are per-candidate (CandidateEval).
  const selected =
    selectedIndex != null
      ? result.candidates.find((c) => c.index === selectedIndex)
      : undefined;
  const r = selected ?? recommended;
  const isRecommended = r.index === recommended.index;
  const isLoaded = selectedIndex === r.index;
  const cool = result.cooldown;
  const hasGroups = r.treatment_geos.length > 0 || r.control_geos.length > 0;
  return (
    <div
      className={`rounded-md border bg-white px-3 py-3 ${
        selected && !isRecommended ? 'border-steel-300' : 'border-line-200'
      }`}
    >
      <h4 className="mb-1.5 flex flex-wrap items-center gap-2 text-sm font-semibold text-ink-900">
        {selected && !isRecommended ? 'Selected setup' : 'Recommended setup'}
        <span className="rounded-full bg-cream-200 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-ink-600">
          {r.design_key} / {MODE_LABEL[r.mode]}
        </span>
        {selected && !isRecommended && (
          <span className="rounded-full bg-gold-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-gold-700">
            recommended: {fmtNum(recommended.mde_roas)} MDE
          </span>
        )}
        <span
          className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
            r.powered ? 'bg-sage-100 text-sage-800' : 'bg-rust-100 text-rust-600'
          }`}
        >
          {r.power != null
            ? `power ${Math.round(r.power * 100)}%${
                r.power_lower != null ? ` · lo95 ${Math.round(r.power_lower * 100)}%` : ''
              } · target ${Math.round((r.power_target ?? 0.8) * 100)}%`
            : r.powered
              ? 'powered'
              : 'underpowered'}
        </span>
      </h4>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Intensity" value={fmtSignedPct(r.intensity_pct)} />
        <Metric label="Duration (weeks)" value={`${r.duration}`} />
        <Metric label="MDE (ROAS)" value={fmtNum(r.mde_roas)} />
        <Metric
          label={`Power (≥${Math.round((r.power_target ?? 0.8) * 100)}%)`}
          value={r.power != null ? `${Math.round(r.power * 100)}%` : '—'}
        />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3">
        <Metric
          label={
            r.tradeoff_basis === 'net_value'
              ? 'Net value of testing ($)'
              : r.tradeoff_basis === 'net_dollar'
                ? 'Short-term cost ($)'
                : r.tradeoff_basis === 'spend_at_risk'
                  ? 'Budget at risk ($)'
                  : 'Forgone KPI'
          }
          value={
            r.tradeoff_basis === 'net_value'
              ? `${(r.net_value ?? -r.tradeoff) >= 0 ? '+' : '−'}${fmtInt(
                  Math.abs(r.net_value ?? -r.tradeoff),
                )}`
              : fmtInt(r.tradeoff)
          }
        />
        <Metric label="Spend at risk" value={fmtInt(r.spend_at_risk)} />
      </div>

      {r.power_breakdown && (
        <div className="mt-3 rounded-md bg-cream-100 px-3 py-2.5">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">
              Power by estimand — {r.power_breakdown.n_levels} spend levels
            </span>
            <span
              className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                r.power_breakdown.mroas_identified
                  ? 'bg-sage-100 text-sage-800'
                  : 'bg-gold-100 text-gold-700'
              }`}
            >
              {r.power_breakdown.mroas_identified
                ? 'curve identified'
                : 'on/off — no curve'}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center">
            <PowerStat
              label="Avg ROAS"
              value={r.power_breakdown.roas}
              lower={r.power_breakdown.roas_lower}
            />
            <PowerStat
              label="Contribution"
              value={r.power_breakdown.contribution}
              lower={r.power_breakdown.contribution_lower}
            />
            <PowerStat
              label="Marginal ROAS"
              value={r.power_breakdown.mroas}
              lower={r.power_breakdown.mroas_lower}
            />
          </div>
          {r.power_breakdown.contribution_lower != null && (
            <p className="mt-1.5 text-[11px] text-ink-400">
              <span className="font-medium">lo 95%</span> = power if the channel's
              true contribution is at the model's lower{' '}
              <span className="num">
                {Math.round((1 - (r.power_breakdown.lower_quantile ?? 0.025) * 2) * 100)}%
              </span>{' '}
              posterior bound (conservative; the headline averages over the
              posterior).
            </p>
          )}
          {!r.power_breakdown.mroas_identified && (
            <p className="mt-1.5 text-[11px] text-ink-400">
              A binary on/off pins the average ROAS but its marginal is a secant,
              not the curve — use ≥3 spend levels (widen the spend-variation range)
              to identify the saturation / marginal ROAS.
            </p>
          )}
        </div>
      )}

      <p className="mt-2 text-xs text-ink-400">
        {r.footprint} footprint
        {r.n_pairs != null && (
          <>
            {' '}· <span className="num">{r.n_pairs}</span> matched pair
            {r.n_pairs === 1 ? '' : 's'}
          </>
        )}{' '}
        · tradeoff basis:{' '}
        {r.tradeoff_basis === 'net_value'
          ? 'net value (reallocation gain − test loss)'
          : r.tradeoff_basis === 'net_dollar'
            ? 'net $ vs BAU'
            : r.tradeoff_basis === 'spend_at_risk'
              ? '$ budget committed'
              : 'forgone KPI'}
        {r.pct_of_window_kpi != null && (
          <> · {fmtPct(r.pct_of_window_kpi)} of the treated-window KPI</>
        )}
        .
      </p>

      {hasGroups ? (
        <div className="mt-2.5 grid grid-cols-2 gap-3 text-xs">
          <div className="rounded-md bg-sage-100/60 px-2.5 py-2">
            <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-sage-800">
              Treatment ({r.treatment_geos.length})
            </div>
            <div className="text-ink-700">{r.treatment_geos.join(', ') || '—'}</div>
          </div>
          <div className="rounded-md bg-cream-100 px-2.5 py-2">
            <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-ink-400">
              Control ({r.control_geos.length})
            </div>
            <div className="text-ink-700">{r.control_geos.join(', ') || '—'}</div>
          </div>
        </div>
      ) : r.schedule && r.schedule.length > 0 ? (
        <p className="mt-2.5 rounded-md bg-cream-100 px-2.5 py-2 text-xs text-ink-600">
          Budget-neutral flighting: {fmtSignedPct(r.intensity_pct)} pulses in{' '}
          <span className="num">{r.block_weeks ?? result.suggested_block_weeks}</span>-week
          blocks over <span className="num">{r.schedule.length}</span> test weeks (suggested
          block ≥ adstock memory: {result.suggested_block_weeks}w).
        </p>
      ) : null}

      <div className="mt-3 rounded-md border border-gold-300 bg-gold-100/60 px-3 py-2.5">
        <div className="flex items-baseline gap-2">
          <span className="font-display text-2xl font-semibold text-ink-900 num">
            {cool.cooldown_weeks}
          </span>
          <span className="text-sm font-medium text-ink-700">
            week{cool.cooldown_weeks === 1 ? '' : 's'} cool-down
          </span>
        </div>
        <p className="mt-0.5 text-xs text-ink-400">
          Carryover washes below {fmtPct(cool.threshold, 0)} of the impulse before the
          treated cells are back to BAU ({cool.basis}
          {cool.half_life != null && (
            <> · adstock half-life ≈ <span className="num">{fmtNum(cool.half_life, 1)}</span>w</>
          )}
          ). Also the minimum flighting block.
        </p>
      </div>

      <div className="mt-3 flex items-center gap-2">
        <Button onClick={() => onSelect(r)} disabled={applying || isLoaded}>
          {isLoaded ? (
            <>
              <Lock className="h-4 w-4" /> Loaded
            </>
          ) : applying ? (
            'Loading…'
          ) : (
            <>
              <Sparkles className="h-4 w-4" />{' '}
              {isRecommended ? 'Use this design' : 'Use selected design'}
            </>
          )}
        </Button>
        <span className="text-xs text-ink-400">
          loads it below to run the simulation or pre-register.
        </span>
      </div>
    </div>
  );
}

function ParetoTable({
  pareto,
  onSelect,
  selectedIndex,
  applying,
}: {
  pareto: CandidateEval[];
  onSelect: (c: CandidateEval) => void;
  selectedIndex: number | null;
  applying: boolean;
}) {
  return (
    <div>
      <h4 className="mb-1.5 text-sm font-semibold text-ink-900">
        Pareto front <span className="font-normal text-ink-400">(non-dominated designs)</span>
      </h4>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line-200 text-left text-xs uppercase tracking-wider text-ink-400">
            <th className="py-1.5">Design</th>
            <th>Footprint</th>
            <th className="text-right">Intensity</th>
            <th className="text-right">Weeks</th>
            <th className="text-right" title="Smallest detectable ROAS effect at 80% power">
              MDE
            </th>
            <th className="text-right" title="Statistical power to detect the model's expected effect">
              Power
            </th>
            <th
              className="text-right"
              title="With a margin: net value of testing (reallocation gain − test loss); else short-term cost (net $ / forgone KPI)"
            >
              Tradeoff
            </th>
            <th className="text-right">Powered</th>
            <th className="text-right">Load</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line-200">
          {pareto.map((c) => (
            <tr
              key={c.index}
              className={
                c.index === selectedIndex
                  ? 'bg-steel-100/70'
                  : c.is_recommended
                    ? 'bg-gold-100/40'
                    : undefined
              }
            >
              <td className="py-1.5 font-medium text-ink-900">
                {c.design_key} / {MODE_LABEL[c.mode]}
                {c.is_recommended && (
                  <span className="ml-1.5 rounded-full bg-gold-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-gold-700">
                    recommended
                  </span>
                )}
              </td>
              <td className="text-ink-700">{c.footprint}</td>
              <td className="text-right num text-ink-700">{fmtSignedPct(c.intensity_pct)}</td>
              <td className="text-right num text-ink-700">{c.duration}</td>
              <td className="text-right num text-ink-700">{fmtNum(c.mde_roas)}</td>
              <td
                className={`text-right num ${c.powered ? 'text-ink-700' : 'text-rust-600'}`}
              >
                {c.power != null ? `${Math.round(c.power * 100)}%` : '—'}
              </td>
              <td className="text-right num text-ink-700">
                {c.tradeoff_basis === 'net_value'
                  ? `${(c.net_value ?? -c.tradeoff) >= 0 ? '+' : '−'}$${fmtInt(
                      Math.abs(c.net_value ?? -c.tradeoff),
                    )}`
                  : `${
                      c.tradeoff_basis === 'net_dollar' ||
                      c.tradeoff_basis === 'spend_at_risk'
                        ? '$'
                        : ''
                    }${fmtInt(c.tradeoff)}`}
              </td>
              <td className="text-right">
                <span
                  className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                    c.powered ? 'bg-sage-100 text-sage-800' : 'bg-rust-100 text-rust-600'
                  }`}
                >
                  {c.powered ? 'yes' : 'no'}
                </span>
              </td>
              <td className="text-right">
                <button
                  onClick={() => onSelect(c)}
                  disabled={applying}
                  className="rounded-md border border-line-300 px-2 py-0.5 text-xs font-medium text-ink-700 transition-colors hover:bg-cream-100 disabled:opacity-50"
                >
                  {c.index === selectedIndex ? 'Loaded' : 'Use'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
