import React, { useEffect, useState } from 'react';
import {
  Activity, BarChart2, Calendar, Check, ChevronDown, ChevronRight, Layers,
  Lock, Pencil, Plus, RotateCcw, Trash2, TrendingUp, Unlock, X, Zap,
} from 'lucide-react';
import { Badge } from '../common/Badge';
import { FLabel, iCls, sCls } from '../common/form';
import { lockPathLabel, specWithDefaults } from '../../utils/spec';
import type { ModelSpec } from '../../types';

// The fully-defaulted, editable working model produced by specWithDefaults. Its
// leaves are concrete (numbers/strings/arrays), unlike the loose incoming spec.
type DraftSpec = ReturnType<typeof specWithDefaults>;

function SpecRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-center py-1.5 border-b border-line-200 last:border-0">
      <span className="text-xs text-ink-400 font-medium">{label}</span>
      <span className="text-xs text-ink-900 font-semibold text-right max-w-[60%]">{value}</span>
    </div>
  );
}

function SpecSection({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mb-3">
      <button onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-2 text-xs font-bold text-ink-600 uppercase tracking-wider mb-1.5 hover:text-ink-900 transition-colors">
        {icon}
        <span className="flex-1 text-left">{title}</span>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {open && <div className="bg-cream-50 rounded-lg px-3 py-1 border border-line-200">{children}</div>}
    </div>
  );
}

// ─── EditSection: collapsible form section ────────────────────────────────────

function EditSection({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border border-line-200 rounded-xl overflow-hidden">
      <button onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-2 px-4 py-3 bg-cream-50 hover:bg-cream-100 transition-colors text-left">
        <span className="text-ink-400">{icon}</span>
        <span className="flex-1 text-xs font-bold text-ink-700 uppercase tracking-wider">{title}</span>
        {open ? <ChevronDown size={13} className="text-ink-300" /> : <ChevronRight size={13} className="text-ink-300" />}
      </button>
      {open && <div className="px-4 py-3 space-y-3 bg-white">{children}</div>}
    </div>
  );
}

// ─── ModelSpecWidget ──────────────────────────────────────────────────────────

interface ModelSpecWidgetProps {
  spec: ModelSpec;
  editable: boolean;
  onApplySpec: (newSpec: ModelSpec) => void;
  lockedFields?: string[];
  onUnlock?: (path: string | string[]) => void;
}

export function ModelSpecWidget({ spec, editable, onApplySpec, lockedFields = [], onUnlock }: ModelSpecWidgetProps) {
  const [editMode, setEditMode] = useState(false);
  const [draft, setDraft] = useState(() => specWithDefaults(spec));
  const [newChannel, setNewChannel] = useState('');
  const [newControl, setNewControl] = useState('');

  // Re-sync draft when spec prop changes (e.g. agent updates it). View mode
  // already reads the live prop (displaySpec), so this only seeds the editable
  // draft; resetting it here is intentional and must not move to render (it
  // would clobber in-progress edits) — keep the effect.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- seed editable draft from the external spec; deferring to render would discard in-progress edits
    if (!editMode) setDraft(specWithDefaults(spec));
  }, [spec, editMode]);

  const setDraftField = (path: string[], value: unknown) =>
    setDraft((prev) => {
      const next = { ...prev } as Record<string, unknown>;
      let cur: Record<string, unknown> = next;
      for (let i = 0; i < path.length - 1; i++) {
        cur[path[i]] = { ...(cur[path[i]] as Record<string, unknown>) };
        cur = cur[path[i]] as Record<string, unknown>;
      }
      cur[path[path.length - 1]] = value;
      return next as DraftSpec;
    });

  const setChannel = (idx: number, field: string, subfield: string | null, value: unknown) =>
    setDraft((prev) => {
      const channels = prev.media_channels.map((ch, i) => {
        if (i !== idx) return ch;
        const chRec = ch as Record<string, unknown>;
        if (subfield) return { ...chRec, [field]: { ...(chRec[field] as Record<string, unknown>), [subfield]: value } };
        return { ...chRec, [field]: value };
      });
      return { ...prev, media_channels: channels } as DraftSpec;
    });

  // Measurement descriptor (impression-/click-measured media). Empty values are
  // DELETED, not written as undefined — absent keys mean "measured in dollars"
  // to the backend, and deleting keeps the lock-diff free of phantom leaves.
  const MEASUREMENT_KEYS = ['measurement_unit', 'spend_column', 'cpm', 'cpc'] as const;
  const patchChannelMeasurement = (idx: number, updates: Record<string, unknown>) =>
    setDraft((prev) => {
      const channels = prev.media_channels.map((ch, i) => {
        if (i !== idx) return ch;
        const next = { ...(ch as Record<string, unknown>), ...updates };
        for (const k of MEASUREMENT_KEYS) {
          if (next[k] === undefined || next[k] === null || next[k] === '') delete next[k];
        }
        return next;
      });
      return { ...prev, media_channels: channels } as DraftSpec;
    });

  const setChannelUnit = (idx: number, unit: string) => {
    if (unit === 'spend') {
      // dollars is the default — drop the whole descriptor
      patchChannelMeasurement(idx, {
        measurement_unit: undefined, spend_column: undefined, cpm: undefined, cpc: undefined,
      });
    } else {
      const ch = draft.media_channels[idx] as Record<string, unknown>;
      patchChannelMeasurement(idx, {
        measurement_unit: unit,
        // a cost-per rate only makes sense for its unit
        cpm: unit === 'impressions' ? ch.cpm : undefined,
        cpc: unit === 'clicks' ? ch.cpc : undefined,
      });
    }
  };

  const addChannel = () => {
    const name = newChannel.trim();
    if (!name) return;
    setDraft((prev) => ({
      ...prev,
      media_channels: [
        ...prev.media_channels,
        { name, adstock: { type: 'geometric', l_max: 8 }, saturation: { type: 'hill' } },
      ],
    }));
    setNewChannel('');
  };

  const removeChannel = (idx: number) =>
    setDraft((prev) => ({ ...prev, media_channels: prev.media_channels.filter((_, i) => i !== idx) }));

  const addControl = () => {
    const name = newControl.trim();
    if (!name) return;
    setDraft((prev) => ({ ...prev, control_variables: [...prev.control_variables, { name }] }));
    setNewControl('');
  };

  const setControlRole = (idx: number, role: string) =>
    setDraft((prev) => ({
      ...prev,
      control_variables: prev.control_variables.map((c, i) =>
        i === idx ? { ...c, role: role || undefined } : c,
      ),
    }));

  const removeControl = (idx: number) =>
    setDraft((prev) => ({ ...prev, control_variables: prev.control_variables.filter((_, i) => i !== idx) }));

  const handleApply = () => {
    // draft is the fully-defaulted working model; its leaves are typed `{}`
    // (from `unknown ?? default`) so we cast at the boundary to ModelSpec.
    // Spread the live spec underneath: PATCH /spec replaces model_spec
    // wholesale, so any key added by the agent while the editor was open
    // must survive an Apply it didn't touch.
    onApplySpec({ ...spec, ...(draft as unknown as ModelSpec) });
    setEditMode(false);
  };

  const handleDiscard = () => {
    setDraft(specWithDefaults(spec));
    setEditMode(false);
  };

  // ── View mode ──────────────────────────────────────────────────────────────
  const displaySpec = editMode ? draft : specWithDefaults(spec);
  const trendType = displaySpec.trend?.type ?? 'linear';
  const trendLabel = trendType.charAt(0).toUpperCase() + trendType.slice(1).replace('_', ' ');
  const seasonality = displaySpec.seasonality;
  const inference = displaySpec.inference;

  const viewContent = (
    <div className="space-y-3 pt-1">
      <SpecSection title="KPI & Data" icon={<BarChart2 size={13} />}>
        <SpecRow label="KPI Variable" value={displaySpec.kpi || '—'} />
        <SpecRow label="Level" value={(displaySpec.kpi_level || 'national').replace(/\b\w/g, (c: string) => c.toUpperCase())} />
        <SpecRow label="Granularity" value={displaySpec.time_granularity || 'weekly'} />
      </SpecSection>
      <SpecSection title="Inference" icon={<Activity size={13} />}>
        <SpecRow label="Chains" value={inference?.chains ?? 4} />
        <SpecRow label="Draws" value={inference?.draws ?? 1000} />
        <SpecRow label="Tune" value={inference?.tune ?? 1000} />
        <SpecRow label="Target Accept" value={inference?.target_accept ?? 0.85} />
        <SpecRow label="Seed" value={inference?.random_seed ?? 42} />
      </SpecSection>
      <SpecSection title="Trend" icon={<TrendingUp size={13} />}>
        <SpecRow label="Type" value={trendLabel} />
        {trendType === 'piecewise' && <><SpecRow label="Changepoints" value={displaySpec.trend?.n_changepoints ?? 5} /><SpecRow label="Range" value={`${((displaySpec.trend?.changepoint_range ?? 0.8) * 100).toFixed(0)}%`} /></>}
        {trendType === 'spline' && <><SpecRow label="Knots" value={displaySpec.trend?.n_knots ?? 5} /><SpecRow label="Degree" value={displaySpec.trend?.spline_degree ?? 3} /></>}
      </SpecSection>
      <SpecSection title="Seasonality" icon={<Calendar size={13} />}>
        <SpecRow label="Yearly" value={seasonality?.yearly ? `${seasonality.yearly} terms` : 'Off'} />
        <SpecRow label="Monthly" value={seasonality?.monthly ? `${seasonality.monthly} terms` : 'Off'} />
        <SpecRow label="Weekly" value={seasonality?.weekly ? `${seasonality.weekly} terms` : 'Off'} />
      </SpecSection>
      {displaySpec.media_channels?.length > 0 && (
        <SpecSection title="Media Channels" icon={<Zap size={13} />}>
          {displaySpec.media_channels.map((ch) => (
            <div key={ch.name} className="py-2 border-b border-line-200 last:border-0">
              <p className="text-xs font-semibold text-ink-900 mb-1">{ch.name}</p>
              <div className="flex flex-wrap gap-1.5">
                <Badge label={`${ch.adstock?.type ?? 'geometric'} adstock`} color="indigo" />
                <Badge label={`l_max=${ch.adstock?.l_max ?? 8}`} color="gray" />
                <Badge label={`${ch.saturation?.type ?? 'hill'} sat.`} color="blue" />
                {ch.measurement_unit && ch.measurement_unit !== 'spend' && (
                  <Badge label={`measured in ${ch.measurement_unit}`} color="amber" />
                )}
                {ch.spend_column && <Badge label={`spend col: ${ch.spend_column}`} color="gray" />}
                {ch.cpm != null && <Badge label={`CPM $${ch.cpm}`} color="gray" />}
                {ch.cpc != null && <Badge label={`CPC $${ch.cpc}`} color="gray" />}
              </div>
            </div>
          ))}
        </SpecSection>
      )}
      {displaySpec.control_variables?.length > 0 && (
        <SpecSection title="Controls" icon={<Layers size={13} />}>
          <div className="flex flex-wrap gap-1.5 py-1">
            {displaySpec.control_variables.map((c) => (
              <span key={c.name} className="flex items-center gap-1">
                <Badge label={c.name} />
                {c.role === 'confounder' ? (
                  <Badge label="confounder" color="indigo" />
                ) : c.role === 'precision' ? (
                  <Badge label="precision" color="gray" />
                ) : (
                  <Badge label="role?" color="amber" />
                )}
              </span>
            ))}
          </div>
          {displaySpec.control_variables.some((c) => !c.role) && (
            <p className="text-[11px] text-amber-700 py-1">
              Declare each control's causal role: confounders (drive both spend and the KPI)
              block back-door paths and must always stay in the model; precision controls only
              reduce noise. The distinction decides what may ever be shrunk or dropped.
            </p>
          )}
        </SpecSection>
      )}
    </div>
  );

  // ── Edit mode ──────────────────────────────────────────────────────────────
  const editForm = (
    <div className="space-y-3 pt-1">
      {/* KPI & Data */}
      <EditSection title="KPI & Data" icon={<BarChart2 size={13} />}>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <FLabel>KPI Variable</FLabel>
            <input className={iCls} value={draft.kpi}
              onChange={e => setDraftField(['kpi'], e.target.value)} placeholder="e.g. Sales" />
          </div>
          <div>
            <FLabel>Level</FLabel>
            <select className={sCls} value={draft.kpi_level} onChange={e => setDraftField(['kpi_level'], e.target.value)}>
              <option value="national">National</option>
              <option value="geo">Geo</option>
            </select>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <FLabel>Granularity</FLabel>
            <select className={sCls} value={draft.time_granularity} onChange={e => setDraftField(['time_granularity'], e.target.value)}>
              <option value="weekly">Weekly</option>
              <option value="daily">Daily</option>
              <option value="monthly">Monthly</option>
            </select>
          </div>
        </div>
      </EditSection>

      {/* Inference */}
      <EditSection title="Inference" icon={<Activity size={13} />}>
        <div className="grid grid-cols-3 gap-2">
          {([['Chains', 'chains', 1, 8], ['Draws', 'draws', 100, 10000], ['Tune', 'tune', 100, 5000]] as const).map(([label, key, min, max]) => (
            <div key={key}>
              <FLabel>{label}</FLabel>
              <input className={iCls} type="number" min={min} max={max}
                value={draft.inference[key]}
                onChange={e => setDraftField(['inference', key], Number(e.target.value))} />
            </div>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <FLabel>Target Accept</FLabel>
            <input className={iCls} type="number" min={0.5} max={0.99} step={0.01}
              value={draft.inference.target_accept}
              onChange={e => setDraftField(['inference', 'target_accept'], Number(e.target.value))} />
          </div>
          <div>
            <FLabel>Random Seed</FLabel>
            <input className={iCls} type="number"
              value={draft.inference.random_seed}
              onChange={e => setDraftField(['inference', 'random_seed'], Number(e.target.value))} />
          </div>
        </div>
      </EditSection>

      {/* Trend */}
      <EditSection title="Trend Model" icon={<TrendingUp size={13} />}>
        <div>
          <FLabel>Type</FLabel>
          <select className={sCls} value={draft.trend.type} onChange={e => setDraftField(['trend', 'type'], e.target.value)}>
            <option value="linear">Linear</option>
            <option value="piecewise">Piecewise Linear</option>
            <option value="spline">Spline</option>
            <option value="gaussian_process">Gaussian Process</option>
            <option value="none">None</option>
          </select>
        </div>
        {draft.trend.type === 'piecewise' && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <FLabel>Changepoints</FLabel>
              <input className={iCls} type="number" min={1} max={50}
                value={draft.trend.n_changepoints}
                onChange={e => setDraftField(['trend', 'n_changepoints'], Number(e.target.value))} />
            </div>
            <div>
              <FLabel>Changepoint Range (0–1)</FLabel>
              <input className={iCls} type="number" min={0.1} max={1} step={0.05}
                value={draft.trend.changepoint_range}
                onChange={e => setDraftField(['trend', 'changepoint_range'], Number(e.target.value))} />
            </div>
          </div>
        )}
        {draft.trend.type === 'spline' && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <FLabel>Knots</FLabel>
              <input className={iCls} type="number" min={2} max={50}
                value={draft.trend.n_knots}
                onChange={e => setDraftField(['trend', 'n_knots'], Number(e.target.value))} />
            </div>
            <div>
              <FLabel>Degree</FLabel>
              <select className={sCls} value={draft.trend.spline_degree}
                onChange={e => setDraftField(['trend', 'spline_degree'], Number(e.target.value))}>
                {[1, 2, 3, 4, 5].map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
          </div>
        )}
      </EditSection>

      {/* Seasonality */}
      <EditSection title="Seasonality (Fourier terms, 0 = off)" icon={<Calendar size={13} />}>
        <div className="grid grid-cols-3 gap-2">
          {(['yearly', 'monthly', 'weekly'] as const).map(period => (
            <div key={period}>
              <FLabel>{period.charAt(0).toUpperCase() + period.slice(1)}</FLabel>
              <input className={iCls} type="number" min={0} max={10}
                value={draft.seasonality[period]}
                onChange={e => setDraftField(['seasonality', period], Number(e.target.value))} />
            </div>
          ))}
        </div>
      </EditSection>

      {/* Media Channels */}
      <EditSection title="Media Channels" icon={<Zap size={13} />}>
        <div className="space-y-2">
          {draft.media_channels.map((ch, idx) => {
            const unit = ch.measurement_unit ?? 'spend';
            const hasSpendCol = !!ch.spend_column;
            const hasRate = ch.cpm != null || ch.cpc != null;
            return (
            <div key={idx} className="space-y-2 bg-cream-50 rounded-lg p-2 border border-line-200">
              <div className="flex gap-2 items-end">
                <div className="flex-1">
                  <FLabel>Name</FLabel>
                  <input className={iCls} value={ch.name}
                    onChange={e => setChannel(idx, 'name', null, e.target.value)} />
                </div>
                <div className="w-28">
                  <FLabel>Adstock</FLabel>
                  <select className={sCls} value={ch.adstock.type}
                    onChange={e => setChannel(idx, 'adstock', 'type', e.target.value)}>
                    <option value="geometric">Geometric</option>
                    <option value="weibull">Weibull</option>
                    <option value="delayed">Delayed</option>
                  </select>
                </div>
                <div className="w-16">
                  <FLabel>L-max</FLabel>
                  <input className={iCls} type="number" min={1} max={52}
                    value={ch.adstock.l_max}
                    onChange={e => setChannel(idx, 'adstock', 'l_max', Number(e.target.value))} />
                </div>
                <div className="w-32">
                  <FLabel>Saturation</FLabel>
                  <select className={sCls} value={ch.saturation.type}
                    onChange={e => setChannel(idx, 'saturation', 'type', e.target.value)}>
                    <option value="hill">Hill</option>
                    <option value="logistic">Logistic</option>
                    <option value="michaelis_menten">Michaelis-Menten</option>
                    <option value="tanh">Tanh</option>
                  </select>
                </div>
                <button onClick={() => removeChannel(idx)}
                  className="p-1.5 text-red-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors shrink-0" title="Remove channel">
                  <Trash2 size={14} />
                </button>
              </div>
              <div className="flex gap-2 items-end">
                <div className="w-32">
                  <FLabel>Measured in</FLabel>
                  <select className={sCls} value={unit} onChange={e => setChannelUnit(idx, e.target.value)}>
                    <option value="spend">Spend ($)</option>
                    <option value="impressions">Impressions</option>
                    <option value="clicks">Clicks</option>
                    <option value="other">Other units</option>
                  </select>
                </div>
                {unit !== 'spend' && (
                  <>
                    <div className="flex-1">
                      <FLabel>Spend column (optional)</FLabel>
                      <input className={iCls} value={ch.spend_column ?? ''} disabled={hasRate}
                        placeholder="external $ series in the dataset"
                        title={hasRate ? 'Clear the CPM/CPC first — one cost source per channel' : undefined}
                        onChange={e => patchChannelMeasurement(idx, { spend_column: e.target.value })} />
                    </div>
                    {unit === 'impressions' && (
                      <div className="w-24">
                        <FLabel>CPM ($)</FLabel>
                        <input className={iCls} type="number" min={0} step="any"
                          value={ch.cpm ?? ''} disabled={hasSpendCol}
                          title={hasSpendCol ? 'Clear the spend column first — one cost source per channel' : undefined}
                          onChange={e => patchChannelMeasurement(idx, { cpm: e.target.value === '' ? undefined : Number(e.target.value) })} />
                      </div>
                    )}
                    {unit === 'clicks' && (
                      <div className="w-24">
                        <FLabel>CPC ($)</FLabel>
                        <input className={iCls} type="number" min={0} step="any"
                          value={ch.cpc ?? ''} disabled={hasSpendCol}
                          title={hasSpendCol ? 'Clear the spend column first — one cost source per channel' : undefined}
                          onChange={e => patchChannelMeasurement(idx, { cpc: e.target.value === '' ? undefined : Number(e.target.value) })} />
                      </div>
                    )}
                  </>
                )}
              </div>
              {unit !== 'spend' && !hasSpendCol && !hasRate && (
                <p className="text-[11px] text-amber-700">
                  No cost source — ROI will report as <em>efficiency</em> (per{' '}
                  {unit === 'impressions' ? '1k impressions' : unit === 'clicks' ? 'click' : 'unit'},
                  break-even 0) instead of dollars. Add a spend column or a{' '}
                  {unit === 'clicks' ? 'CPC' : 'CPM'} to restore monetary ROI.
                </p>
              )}
            </div>
            );
          })}
        </div>
        <div className="flex gap-2 mt-2">
          <input className={iCls + ' flex-1'} placeholder="New channel name…"
            value={newChannel}
            onChange={e => setNewChannel(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addChannel()} />
          <button onClick={addChannel}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-medium rounded-lg border border-indigo-200 transition-colors">
            <Plus size={13} /> Add
          </button>
        </div>
      </EditSection>

      {/* Controls */}
      <EditSection title="Control Variables" icon={<Layers size={13} />}>
        <p className="text-[11px] text-ink-400 -mb-1">
          Confounders (drive both spend and the KPI) carry the identification — never shrink or
          drop one. Precision controls only soak up outcome noise and are safe to regularize.
        </p>
        <div className="space-y-1.5">
          {draft.control_variables.map((c, idx) => (
            <div key={idx} className="flex items-center gap-2 bg-cream-50 rounded-lg px-2.5 py-1.5 border border-line-200">
              <span className="flex-1 text-xs text-ink-700 truncate">{c.name}</span>
              <select
                className={sCls + ' w-44'}
                value={typeof c.role === 'string' ? c.role : ''}
                onChange={e => setControlRole(idx, e.target.value)}
              >
                <option value="">Role not declared</option>
                <option value="confounder">Confounder (identification)</option>
                <option value="precision">Precision control (noise)</option>
              </select>
              <button onClick={() => removeControl(idx)} className="text-ink-300 hover:text-red-500 transition-colors shrink-0">
                <X size={11} />
              </button>
            </div>
          ))}
        </div>
        <div className="flex gap-2 mt-1">
          <input className={iCls + ' flex-1'} placeholder="New control variable…"
            value={newControl}
            onChange={e => setNewControl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addControl()} />
          <button onClick={addControl}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-cream-50 hover:bg-cream-100 text-ink-700 text-xs font-medium rounded-lg border border-line-200 transition-colors">
            <Plus size={13} /> Add
          </button>
        </div>
      </EditSection>
    </div>
  );

  // ── Widget wrapper ─────────────────────────────────────────────────────────
  const headerActions = editMode ? (
    <div className="flex items-center gap-2">
      <button onClick={handleApply}
        className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg transition-colors">
        <Check size={13} /> Apply
      </button>
      <button onClick={handleDiscard}
        className="flex items-center gap-1.5 px-3 py-1.5 bg-cream-100 hover:bg-gray-200 text-ink-600 text-xs font-medium rounded-lg transition-colors border border-line-200">
        <RotateCcw size={13} /> Discard
      </button>
    </div>
  ) : editable ? (
    <button onClick={() => setEditMode(true)}
      className="flex items-center gap-1.5 px-2.5 py-1 bg-cream-50 hover:bg-cream-100 text-ink-400 hover:text-indigo-600 text-xs font-medium rounded-lg border border-line-200 transition-colors">
      <Pencil size={12} /> Edit
    </button>
  ) : null;

  return (
    <div className="bg-white rounded-2xl border border-line-200 shadow-sm hover:shadow-md transition-all overflow-hidden">
      <div className="flex items-center gap-3 px-5 py-4 border-b border-line-200">
        <Activity size={15} className="text-blue-500 shrink-0" />
        <span className="font-semibold text-sm text-blue-600 flex-1">Model Configuration</span>
        {editMode && (
          <span className="px-2 py-0.5 bg-amber-50 text-amber-600 text-[10px] font-bold uppercase tracking-wide rounded-full border border-amber-200">
            Editing
          </span>
        )}
        {headerActions}
      </div>
      {lockedFields.length > 0 && (
        <LockedFieldsSummary lockedFields={lockedFields} onUnlock={onUnlock} />
      )}
      <div className="px-5 py-4 max-h-[600px] overflow-y-auto">
        {editMode ? editForm : viewContent}
      </div>
    </div>
  );
}


// ─── Locked-fields summary ───────────────────────────────────────────────────
// A manual prior edit can lock dozens of leaf paths at once; rendering a pill
// per leaf buries the tab. Collapse to one quiet line, and group expanded
// pills by their parent (e.g. all of TV's prior leaves -> one pill).

function groupLockedPaths(paths: string[]): { label: string; paths: string[] }[] {
  const groups = new Map<string, string[]>();
  for (const p of paths) {
    const segs = p.split('.');
    const key = segs.slice(0, Math.min(3, segs.length)).join('.');
    const list = groups.get(key) ?? [];
    list.push(p);
    groups.set(key, list);
  }
  return [...groups.entries()]
    .map(([key, members]) => ({ label: lockPathLabel(key), paths: members }))
    .sort((a, b) => b.paths.length - a.paths.length || a.label.localeCompare(b.label));
}

function LockedFieldsSummary({ lockedFields, onUnlock }: {
  lockedFields: string[];
  onUnlock?: (path: string | string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const groups = groupLockedPaths(lockedFields);

  return (
    <div className="border-b border-line-200 bg-cream-100/60">
      <div className="flex items-center gap-2 px-5 py-2">
        <Lock size={11} className="shrink-0 text-gold-600" />
        <button
          onClick={() => setOpen(v => !v)}
          className="flex flex-1 items-center gap-1.5 text-left text-[11px] font-medium text-ink-600 hover:text-ink-900 transition-colors"
          title={open ? 'Hide locked fields' : 'Show locked fields'}
        >
          {lockedFields.length} field{lockedFields.length === 1 ? '' : 's'} locked by you
          <span className="text-ink-300">— the assistant asks before changing these</span>
          <ChevronDown size={12} className={`text-ink-300 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
        {onUnlock && (
          <button
            onClick={() => {
              if (confirm(`Unlock all ${lockedFields.length} fields? The assistant can change them again without asking.`)) {
                onUnlock([...lockedFields]);
              }
            }}
            className="shrink-0 text-[11px] font-medium text-ink-400 hover:text-rust-600 transition-colors"
          >
            Unlock all
          </button>
        )}
      </div>
      {open && (
        <div className="flex flex-wrap items-center gap-1.5 px-5 pb-2.5">
          {groups.map((g) => (
            <span
              key={g.label}
              className="flex items-center gap-1 rounded-full border border-line-200 bg-white px-2 py-0.5 text-[11px] text-ink-600"
              title={g.paths.map(lockPathLabel).join('\n')}
            >
              {g.label}
              {g.paths.length > 1 && (
                <span className="text-ink-300">({g.paths.length})</span>
              )}
              {onUnlock && (
                <button
                  onClick={() => onUnlock([...g.paths])}
                  title={g.paths.length > 1 ? `Unlock all ${g.paths.length} fields under ${g.label}` : 'Unlock — let the assistant change this'}
                  className="text-ink-300 hover:text-gold-700 transition-colors"
                >
                  <Unlock size={10} />
                </button>
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
