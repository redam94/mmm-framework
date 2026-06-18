import { useEffect, useMemo, useState } from 'react';
import { Calendar, Check, ChevronDown, ChevronRight, Layers, Zap } from 'lucide-react';
import { Badge } from '../common/Badge';
import { DashWidget } from '../common/DashWidget';
import { FLabel, iCls } from '../common/form';
import {
  ANY_DISTS, DIST_DEFS, POSITIVE_DISTS, PRIOR_DEFAULTS, UNIT_DISTS,
  computeDensity, initPriors,
} from '../../utils/priors';
import type { DistKey, PriorValue } from '../../utils/priors';
import { normalizeTrendType } from '../../utils/spec';

// --- DensitySparkline ---

function DensitySparkline({ x, y, color = '#6366f1' }: { x: number[]; y: number[]; color?: string }) {
  const valid = x.length >= 2 && y.length >= 2
    && x.every(v => isFinite(v)) && y.every(v => isFinite(v))
    && Math.max(...y) > 0;
  if (!valid) return <div className="h-[52px] bg-cream-50 rounded flex items-center justify-center text-[10px] text-ink-300">no preview</div>;
  const W = 280; const H = 52;
  const maxY = Math.max(...y, 1e-12);
  const minX = x[0]; const rangeX = x[x.length - 1] - x[0] || 1;
  const pts = x.map((xi, i) => {
    const px = ((xi - minX) / rangeX) * W;
    const py = H - (y[i] / maxY) * (H - 4) - 2;
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');
  const fill = `0,${H} ${pts} ${W},${H}`;
  const xFmt = (v: number) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
  return (
    <div className="w-full">
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="overflow-visible">
        <polyline points={fill} fill={color} fillOpacity={0.12} stroke="none" />
        <polyline points={pts} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="flex justify-between text-[9px] text-ink-300 mt-0.5 px-0.5">
        <span>{xFmt(x[0])}</span>
        <span>{xFmt(x[Math.floor(x.length / 2)])}</span>
        <span>{xFmt(x[x.length - 1])}</span>
      </div>
    </div>
  );
}

// --- PriorEditor ---

interface PriorEditorProps {
  label: string;
  hint: string;
  value: PriorValue;
  onChange: (v: PriorValue) => void;
  disabled?: boolean;
  allowed?: DistKey[];
  color?: string;
}

function PriorEditor({ label, hint, value, onChange, disabled, allowed = ANY_DISTS, color = '#6366f1' }: PriorEditorProps) {
  const dist = value.distribution as DistKey;
  const defn = DIST_DEFS[dist] ?? DIST_DEFS.half_normal;
  const density = useMemo(() => computeDensity(dist, value.params), [dist, value.params]);

  const changeDist = (newDist: string) => {
    const d = DIST_DEFS[newDist as DistKey];
    onChange({ distribution: newDist, params: Object.fromEntries(d.params.map(p => [p.key, p.default])) });
  };

  const changeParam = (key: string, val: number) =>
    onChange({ ...value, params: { ...value.params, [key]: val } });

  const selectCls = 'bg-cream-50 border border-line-200 rounded-lg px-2 py-1 text-xs text-ink-900 cursor-pointer focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition-all';

  return (
    <div className="bg-white rounded-xl border border-line-200 p-3 space-y-2">
      {/* Row 1: label + distribution select */}
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold text-ink-900 truncate">{label}</p>
        <select
          className={selectCls + ' w-36 shrink-0'}
          value={dist}
          disabled={disabled}
          onChange={e => changeDist(e.target.value)}
        >
          {allowed.map(k => <option key={k} value={k}>{DIST_DEFS[k].label}</option>)}
        </select>
      </div>

      {/* Row 2: hint */}
      <p className="text-[10px] text-ink-300 leading-snug">{hint}</p>

      {/* Row 3: parameter inputs */}
      <div className="flex gap-2 flex-wrap">
        {defn.params.map(p => (
          <div key={p.key} className="flex-1 min-w-[70px]">
            <FLabel>{p.label}</FLabel>
            <input className={iCls} type="number" min={p.min} max={p.max} step={p.step}
              value={value.params[p.key] ?? p.default}
              disabled={disabled}
              onChange={e => changeParam(p.key, Number(e.target.value))} />
          </div>
        ))}
      </div>

      {/* Row 4: density sparkline */}
      <DensitySparkline x={density.x} y={density.y} color={color} />
    </div>
  );
}

// ─── PriorConfigWidget ────────────────────────────────────────────────────────

interface PriorConfigWidgetProps { spec: any; editable: boolean; onApplySpec: (s: any) => void }

export function PriorConfigWidget({ spec, editable, onApplySpec }: PriorConfigWidgetProps) {
  const [priors, setPriors] = useState(() => initPriors(spec));
  const [tab, setTab] = useState<'media' | 'controls' | 'trend' | 'seasonality'>('media');
  const [openChannel, setOpenChannel] = useState<string | null>(null);
  const [openControl, setOpenControl] = useState<string | null>(null);

  // Re-sync when spec changes (new channels added via ModelSpecWidget)
  useEffect(() => { setPriors(initPriors(spec)); }, [spec?.media_channels?.length, spec?.control_variables?.length, spec?.trend?.type, spec?.seasonality?.yearly, spec?.seasonality?.monthly, spec?.seasonality?.weekly]);

  const setMediaPrior = (channel: string, key: string, val: PriorValue) =>
    setPriors((p: any) => ({ ...p, media: { ...p.media, [channel]: { ...p.media[channel], [key]: val } } }));

  const setControlPrior = (control: string, key: string, val: any) =>
    setPriors((p: any) => ({ ...p, controls: { ...p.controls, [control]: { ...p.controls[control], [key]: val } } }));

  const setTrendPrior = (key: string, val: any) =>
    setPriors((p: any) => ({ ...p, trend: { ...p.trend, [key]: val } }));

  const setSeasonalityPrior = (key: string, val: any) =>
    setPriors((p: any) => ({ ...p, seasonality: { ...p.seasonality, [key]: val } }));

  const handleApply = () => {
    const { _type, ...trendPriors } = priors.trend;
    // null overrides mean "inherit the shared sigma" — don't write them
    const seasPriors = Object.fromEntries(
      Object.entries(priors.seasonality).filter(([, v]) => v !== null)
    );
    onApplySpec({ ...spec, priors: { media: priors.media, controls: priors.controls, trend: trendPriors, seasonality: seasPriors } });
  };

  const channels = spec?.media_channels ?? [];
  const ctrls = spec?.control_variables ?? [];
  const trendType = normalizeTrendType(spec?.trend?.type);
  const seasonalityComponents = (['yearly', 'monthly', 'weekly'] as const)
    .filter(c => (spec?.seasonality?.[c] ?? 0) > 0);

  const TABS = [
    { key: 'media' as const,    label: `Media (${channels.length})` },
    { key: 'controls' as const, label: `Controls (${ctrls.length})` },
    { key: 'trend' as const,    label: 'Trend' },
    { key: 'seasonality' as const, label: 'Seasonality' },
  ];

  const adstockType = (chName: string) =>
    (spec?.media_channels?.find((c: any) => c.name === chName)?.adstock?.type ?? 'geometric').toLowerCase();
  const satType = (chName: string) =>
    (spec?.media_channels?.find((c: any) => c.name === chName)?.saturation?.type ?? 'hill').toLowerCase();

  const content = (
    <div className="space-y-3">
      {/* Tab bar */}
      <div className="flex gap-1.5 flex-wrap">
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${tab === t.key ? 'bg-fuchsia-600 text-white border-fuchsia-600' : 'bg-white text-ink-600 border-line-200 hover:bg-cream-100'}`}>
            {t.label}
          </button>
        ))}
        {editable && (
          <button onClick={handleApply}
            className="ml-auto flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg transition-colors">
            <Check size={12} /> Apply Priors
          </button>
        )}
      </div>

      {/* ── Media tab ─────────────────────────────────────────────────────── */}
      {tab === 'media' && (
        <div className="space-y-2">
          {channels.length === 0 && <p className="text-xs text-ink-300 italic py-2">No media channels configured yet.</p>}
          {channels.map((ch: any) => {
            const isOpen = openChannel === ch.name;
            const aSat = satType(ch.name);
            const aAds = adstockType(ch.name);
            const chPriors = priors.media[ch.name] ?? {};
            return (
              <div key={ch.name} className="border border-line-200 rounded-xl overflow-hidden">
                <button onClick={() => setOpenChannel(isOpen ? null : ch.name)}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-cream-50 hover:bg-cream-100 transition-colors text-left">
                  <Zap size={13} className="text-indigo-500 shrink-0" />
                  <span className="flex-1 text-xs font-bold text-ink-700">{ch.name}</span>
                  <div className="flex gap-1.5 mr-2">
                    <Badge label={aAds} color="indigo" />
                    <Badge label={aSat} color="blue" />
                  </div>
                  {isOpen ? <ChevronDown size={13} className="text-ink-300" /> : <ChevronRight size={13} className="text-ink-300" />}
                </button>
                {isOpen && (
                  <div className="px-4 py-3 space-y-3 bg-white">
                    <PriorEditor
                      label="Channel Coefficient" hint="Scale of this channel's contribution to the KPI. Use Half-Normal to enforce positivity."
                      value={chPriors.coefficient ?? PRIOR_DEFAULTS.media_coefficient}
                      onChange={v => setMediaPrior(ch.name, 'coefficient', v)}
                      disabled={!editable} allowed={POSITIVE_DISTS} color="#6366f1"
                    />
                    {aAds !== 'none' && (
                      <PriorEditor
                        label="Adstock Decay (α)" hint="Decay rate of advertising carryover. Beta(1,3) favours fast decay; Beta(3,1) favours slow decay."
                        value={chPriors.adstock_alpha ?? PRIOR_DEFAULTS.adstock_alpha}
                        onChange={v => setMediaPrior(ch.name, 'adstock_alpha', v)}
                        disabled={!editable} allowed={UNIT_DISTS} color="#10b981"
                      />
                    )}
                    {(aSat === 'hill' || aSat === 'logistic') && (
                      <PriorEditor
                        label="Saturation κ (half-saturation)" hint="Spend level at which 50% of max effect is reached, relative to observed range."
                        value={chPriors.saturation_kappa ?? PRIOR_DEFAULTS.sat_kappa}
                        onChange={v => setMediaPrior(ch.name, 'saturation_kappa', v)}
                        disabled={!editable} allowed={UNIT_DISTS} color="#f59e0b"
                      />
                    )}
                    {aSat !== 'none' && (
                      <PriorEditor
                        label="Saturation Slope (steepness)" hint="How steeply the response curve rises. Larger σ allows more extreme saturation shapes."
                        value={chPriors.saturation_slope ?? PRIOR_DEFAULTS.sat_slope}
                        onChange={v => setMediaPrior(ch.name, 'saturation_slope', v)}
                        disabled={!editable} allowed={POSITIVE_DISTS} color="#ef4444"
                      />
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Controls tab ──────────────────────────────────────────────────── */}
      {tab === 'controls' && (
        <div className="space-y-2">
          {ctrls.length === 0 && <p className="text-xs text-ink-300 italic py-2">No control variables configured yet.</p>}
          {ctrls.map((cv: any) => {
            const isOpen = openControl === cv.name;
            const cvPriors = priors.controls[cv.name] ?? {};
            const allowNeg = cvPriors.allow_negative ?? true;
            return (
              <div key={cv.name} className="border border-line-200 rounded-xl overflow-hidden">
                <button onClick={() => setOpenControl(isOpen ? null : cv.name)}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-cream-50 hover:bg-cream-100 transition-colors text-left">
                  <Layers size={13} className="text-ink-400 shrink-0" />
                  <span className="flex-1 text-xs font-bold text-ink-700">{cv.name}</span>
                  <Badge label={allowNeg ? 'any sign' : 'positive only'} color={allowNeg ? 'gray' : 'green'} />
                  {isOpen ? <ChevronDown size={13} className="text-ink-300" /> : <ChevronRight size={13} className="text-ink-300" />}
                </button>
                {isOpen && (
                  <div className="px-4 py-3 space-y-3 bg-white">
                    <div className="flex items-center gap-3 py-1">
                      <span className="text-xs text-ink-600 font-medium">Allow negative coefficient</span>
                      <button onClick={() => editable && setControlPrior(cv.name, 'allow_negative', !allowNeg)}
                        className={`relative w-9 h-5 rounded-full transition-colors ${allowNeg ? 'bg-indigo-500' : 'bg-gray-300'} ${!editable ? 'cursor-default opacity-60' : 'cursor-pointer'}`}>
                        <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${allowNeg ? 'translate-x-4' : 'translate-x-0.5'}`} />
                      </button>
                    </div>
                    <PriorEditor
                      label="Coefficient Prior" hint={allowNeg ? 'Normal prior centred at zero for controls with uncertain direction.' : 'Half-Normal for controls expected to have positive-only effects.'}
                      value={cvPriors.coefficient ?? PRIOR_DEFAULTS.control_coef}
                      onChange={v => setControlPrior(cv.name, 'coefficient', v)}
                      disabled={!editable} allowed={allowNeg ? ANY_DISTS : POSITIVE_DISTS} color="#6366f1"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Trend tab ─────────────────────────────────────────────────────── */}
      {tab === 'trend' && (
        <div className="space-y-3">
          {(trendType === 'linear' || trendType === 'none' || trendType === 'piecewise') && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <FLabel>Growth Prior μ</FLabel>
                  <input className={iCls} type="number" step={0.01}
                    value={priors.trend.growth_prior_mu}
                    disabled={!editable}
                    onChange={e => setTrendPrior('growth_prior_mu', Number(e.target.value))} />
                  <p className="text-[10px] text-ink-300 mt-0.5">Expected average growth rate (the base slope for piecewise). Use 0 for no expected trend.</p>
                </div>
                <div>
                  <FLabel>Growth Prior σ</FLabel>
                  <input className={iCls} type="number" step={0.01} min={0.001}
                    value={priors.trend.growth_prior_sigma}
                    disabled={!editable}
                    onChange={e => setTrendPrior('growth_prior_sigma', Number(e.target.value))} />
                  <p className="text-[10px] text-ink-300 mt-0.5">Uncertainty in growth rate. Smaller = tighter prior.</p>
                </div>
              </div>
              {(() => {
                const { x, y } = computeDensity('normal', { mu: priors.trend.growth_prior_mu, sigma: priors.trend.growth_prior_sigma });
                return (
                  <div className="bg-white rounded-xl border border-line-200 p-3">
                    <p className="text-[10px] text-ink-400 font-semibold mb-1.5">Growth Rate Prior Distribution</p>
                    <DensitySparkline x={x} y={y} color="#6366f1" />
                  </div>
                );
              })()}
            </div>
          )}

          {trendType === 'piecewise' && (
            <div>
              <FLabel>Changepoint Prior Scale</FLabel>
              <input className={iCls} type="number" step={0.001} min={0.001} max={1}
                value={priors.trend.changepoint_prior_scale}
                disabled={!editable}
                onChange={e => setTrendPrior('changepoint_prior_scale', Number(e.target.value))} />
              <p className="text-[10px] text-ink-300 mt-0.5">Controls how sharply the trend can change at each changepoint. Smaller = smoother.</p>
            </div>
          )}

          {trendType === 'spline' && (
            <div>
              <FLabel>Spline Coefficient Prior σ</FLabel>
              <input className={iCls} type="number" step={0.1} min={0.01}
                value={priors.trend.spline_prior_sigma}
                disabled={!editable}
                onChange={e => setTrendPrior('spline_prior_sigma', Number(e.target.value))} />
              <p className="text-[10px] text-ink-300 mt-0.5">Controls how far spline coefficients can deviate from zero. Larger = more flexible trend.</p>
            </div>
          )}

          {trendType === 'gaussian_process' && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <FLabel>Lengthscale μ</FLabel>
                  <input className={iCls} type="number" step={0.05} min={0.01}
                    value={priors.trend.gp_lengthscale_prior_mu}
                    disabled={!editable}
                    onChange={e => setTrendPrior('gp_lengthscale_prior_mu', Number(e.target.value))} />
                </div>
                <div>
                  <FLabel>Lengthscale σ</FLabel>
                  <input className={iCls} type="number" step={0.05} min={0.01}
                    value={priors.trend.gp_lengthscale_prior_sigma}
                    disabled={!editable}
                    onChange={e => setTrendPrior('gp_lengthscale_prior_sigma', Number(e.target.value))} />
                </div>
              </div>
              <div>
                <FLabel>Amplitude Prior σ</FLabel>
                <input className={iCls} type="number" step={0.05} min={0.01}
                  value={priors.trend.gp_amplitude_prior_sigma}
                  disabled={!editable}
                  onChange={e => setTrendPrior('gp_amplitude_prior_sigma', Number(e.target.value))} />
                <p className="text-[10px] text-ink-300 mt-0.5">Controls the overall magnitude of the GP trend component.</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Seasonality tab ───────────────────────────────────────────────── */}
      {tab === 'seasonality' && (
        <div className="space-y-3">
          {seasonalityComponents.length === 0 && (
            <p className="text-xs text-ink-300 italic py-2">
              No seasonality components are enabled — set a Fourier order (e.g. Seasonality → Yearly) in the model spec first.
            </p>
          )}
          {seasonalityComponents.length > 0 && (
            <>
              <div>
                <FLabel>Amplitude Prior σ (all components)</FLabel>
                <input className={iCls} type="number" step={0.05} min={0.01}
                  value={priors.seasonality.prior_sigma}
                  disabled={!editable}
                  onChange={e => setSeasonalityPrior('prior_sigma', Number(e.target.value))} />
                <p className="text-[10px] text-ink-300 mt-0.5">
                  Each Fourier coefficient gets Normal(0, σ) on standardized KPI — this bounds the seasonal swing.
                  Default 0.3 suits mild seasonality; raise toward 0.5–1.0 for strongly seasonal categories.
                </p>
              </div>
              <div className="grid grid-cols-3 gap-3">
                {seasonalityComponents.map(c => (
                  <div key={c}>
                    <FLabel>{c.charAt(0).toUpperCase() + c.slice(1)} override σ</FLabel>
                    <input className={iCls} type="number" step={0.05} min={0.01}
                      placeholder="inherit"
                      value={priors.seasonality[`${c}_prior_sigma`] ?? ''}
                      disabled={!editable}
                      onChange={e => setSeasonalityPrior(`${c}_prior_sigma`, e.target.value === '' ? null : Number(e.target.value))} />
                  </div>
                ))}
              </div>
              {(() => {
                const sigma = priors.seasonality.prior_sigma;
                const { x, y } = computeDensity('normal', { mu: 0, sigma });
                return (
                  <div className="bg-white rounded-xl border border-line-200 p-3">
                    <p className="text-[10px] text-ink-400 font-semibold mb-1.5">Fourier Coefficient Prior (shared σ)</p>
                    <DensitySparkline x={x} y={y} color="#d946ef" />
                  </div>
                );
              })()}
            </>
          )}
        </div>
      )}
    </div>
  );

  return (
    <DashWidget
      title="Prior Configuration"
      icon={<Calendar size={15} className="text-fuchsia-500 shrink-0" />}
      color="fuchsia"
      expandTitle="Prior Configuration"
      expandContent={<div className="max-w-3xl mx-auto py-2">{content}</div>}
    >
      {content}
    </DashWidget>
  );
}
