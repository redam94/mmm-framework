import { useEffect, useMemo, useState } from 'react';
import { Check, GitBranch, Layers, Share2, Target } from 'lucide-react';
import { Badge } from '../common/Badge';
import { DashWidget } from '../common/DashWidget';
import { FLabel, iCls, sCls } from '../common/form';
import type { ModelSpec } from '../../types';

// ─── Extension-model priors ───────────────────────────────────────────────────
// DAG-routed extension models (NestedMMM / MultivariateMMM / CombinedMMM) do NOT
// read the plain-model priors.media / priors.controls — those are rejected by the
// backend consumed-paths registry. Their priors come from the causal DAG node
// configs and ARE settable through three spec groups (validated 1:1 against the
// registry sets in agents/fitting.py):
//   priors.mediator["<variable_name>"]        (mediator node config)
//   priors.outcome["<variable_name>"]         (outcome node config)
//   priors.cross_effect["<sourceVar>__<targetVar>"] (cross-outcome edge)
// Mediator / outcome / cross-effect entries are all keyed by DAG *variable
// names* — the backend `_inject_extension_priors` resolves edge node-ids to
// variable_names and matches on that.

const EXTENSION_DAG_LABELS: Record<string, string> = {
  nested_mmm: 'Nested MMM (mediation)',
  multivariate_mmm: 'Multivariate MMM (multi-outcome)',
  combined_mmm: 'Combined MMM',
};

// The keys each group honors — mirrors _MEDIATOR/_OUTCOME/_CROSS_EFFECT_PRIOR_KEYS.
type MediatorPrior = {
  media_effect_sigma?: number;
  media_effect_constraint?: string; // none | positive | negative
  outcome_effect_sigma?: number;
  direct_effect_sigma?: number;
  observation_noise_sigma?: number;
  allow_direct_effect?: boolean;
};
type OutcomePrior = {
  intercept_prior_sigma?: number;
  media_effect_sigma?: number;
  include_trend?: boolean;
  include_seasonality?: boolean;
};
type CrossEffectPrior = {
  effect_type?: string; // cannibalization | halo | unconstrained
  prior_sigma?: number;
};

const MEDIATOR_KEYS: readonly (keyof MediatorPrior)[] = [
  'media_effect_sigma', 'media_effect_constraint', 'outcome_effect_sigma',
  'direct_effect_sigma', 'observation_noise_sigma', 'allow_direct_effect',
];
const OUTCOME_KEYS: readonly (keyof OutcomePrior)[] = [
  'intercept_prior_sigma', 'media_effect_sigma', 'include_trend', 'include_seasonality',
];
const CROSS_EFFECT_KEYS: readonly (keyof CrossEffectPrior)[] = ['effect_type', 'prior_sigma'];

// --- DAG parsing ---

interface DagNodeView { id?: string; variable_name?: string; node_type?: string; label?: string }
interface DagEdgeView { source?: string; target?: string; edge_type?: string }
interface CrossPair { key: string; sourceLabel: string; targetLabel: string; inferred: boolean }

const norm = (v: unknown) => String(v ?? '').toLowerCase().replace(/-/g, '_');

interface DerivedDag {
  hasDag: boolean;
  mediators: { name: string; label: string }[];
  outcomes: { name: string; label: string }[];
  crossPairs: CrossPair[];
}

function deriveDag(spec: ModelSpec): DerivedDag {
  const dagSpec = (spec as Record<string, unknown>).dag_spec as
    { nodes?: DagNodeView[]; edges?: DagEdgeView[] } | undefined;
  if (!dagSpec || !Array.isArray(dagSpec.nodes)) {
    return { hasDag: false, mediators: [], outcomes: [], crossPairs: [] };
  }
  const nodes = dagSpec.nodes;
  const edges = Array.isArray(dagSpec.edges) ? dagSpec.edges : [];

  const nameOf = (n: DagNodeView) => n.variable_name || n.label || n.id || '';
  const byId = new Map<string, DagNodeView>();
  for (const n of nodes) if (n.id) byId.set(n.id, n);
  const resolve = (idOrName?: string) => {
    if (!idOrName) return '';
    const n = byId.get(idOrName);
    return n ? nameOf(n) : idOrName;
  };

  const mediators = nodes
    .filter((n) => norm(n.node_type) === 'mediator')
    .map((n) => ({ name: nameOf(n), label: nameOf(n) }))
    .filter((m) => !!m.name);
  const outcomes = nodes
    .filter((n) => norm(n.node_type) === 'outcome')
    .map((n) => ({ name: nameOf(n), label: nameOf(n) }))
    .filter((o) => !!o.name);

  // Cross effects: prefer explicit cross_effect edges. The key is
  // "<sourceVar>__<targetVar>" (VARIABLE names) — the backend
  // `_inject_extension_priors` maps edge node-ids to variable_names and matches
  // on that. If no explicit cross_effect edges exist, infer directed pairs from
  // the outcome list (the KPI is the primary outcome, so it participates too).
  const crossMap = new Map<string, CrossPair>();
  for (const e of edges) {
    if (norm(e.edge_type) !== 'cross_effect') continue;
    if (!e.source || !e.target) continue;
    const sourceVar = resolve(e.source);
    const targetVar = resolve(e.target);
    const key = `${sourceVar}__${targetVar}`;
    if (!crossMap.has(key)) {
      crossMap.set(key, { key, sourceLabel: sourceVar, targetLabel: targetVar, inferred: false });
    }
  }
  if (crossMap.size === 0) {
    const outcomeNodes = nodes.filter((n) => ['outcome', 'kpi'].includes(norm(n.node_type)));
    if (outcomeNodes.length >= 2) {
      for (const a of outcomeNodes) {
        for (const b of outcomeNodes) {
          if (a === b) continue;
          const aName = nameOf(a);
          const bName = nameOf(b);
          if (!aName || !bName) continue;
          const key = `${aName}__${bName}`;
          if (!crossMap.has(key)) {
            crossMap.set(key, { key, sourceLabel: aName, targetLabel: bName, inferred: true });
          }
        }
      }
    }
  }

  return { hasDag: true, mediators, outcomes, crossPairs: [...crossMap.values()] };
}

// --- State init ---

interface ExtPriorState {
  mediator: Record<string, MediatorPrior>;
  outcome: Record<string, OutcomePrior>;
  cross_effect: Record<string, CrossEffectPrior>;
}

function pick(src: unknown, keys: readonly string[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (src && typeof src === 'object') {
    for (const k of keys) {
      const v = (src as Record<string, unknown>)[k];
      if (v !== undefined && v !== null) out[k] = v;
    }
  }
  return out;
}

function initState(spec: ModelSpec, dag: DerivedDag): ExtPriorState {
  const specPriors = ((spec as Record<string, unknown>).priors ?? {}) as Record<string, unknown>;
  const medSrc = (specPriors.mediator ?? {}) as Record<string, unknown>;
  const outSrc = (specPriors.outcome ?? {}) as Record<string, unknown>;
  const ceSrc = (specPriors.cross_effect ?? {}) as Record<string, unknown>;
  return {
    mediator: Object.fromEntries(
      dag.mediators.map((m) => [m.name, pick(medSrc[m.name], MEDIATOR_KEYS) as MediatorPrior]),
    ),
    outcome: Object.fromEntries(
      dag.outcomes.map((o) => [o.name, pick(outSrc[o.name], OUTCOME_KEYS) as OutcomePrior]),
    ),
    cross_effect: Object.fromEntries(
      dag.crossPairs.map((p) => [p.key, pick(ceSrc[p.key], CROSS_EFFECT_KEYS) as CrossEffectPrior]),
    ),
  };
}

// Drop unset leaves (undefined / '' / null) and empty entries — the "only emit
// set fields, never write nulls" convention. `false` and `0` are kept.
function cleanGroup<T extends object>(group: Record<string, T>): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [name, entry] of Object.entries(group)) {
    const kept = Object.fromEntries(
      Object.entries(entry).filter(([, v]) => v !== undefined && v !== '' && v !== null),
    );
    if (Object.keys(kept).length > 0) out[name] = kept as T;
  }
  return out;
}

// --- Field primitives (match the shared form kit) ---

function NumberField({ label, hint, value, onChange, disabled, min = 0, step = 0.05 }: {
  label: string; hint?: string; value?: number; onChange: (v?: number) => void;
  disabled?: boolean; min?: number; step?: number;
}) {
  return (
    <div>
      <FLabel>{label}</FLabel>
      <input className={iCls} type="number" min={min} step={step}
        placeholder="inherit"
        value={value ?? ''}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value === '' ? undefined : Number(e.target.value))} />
      {hint && <p className="text-[10px] text-ink-300 mt-0.5 leading-snug">{hint}</p>}
    </div>
  );
}

function ChoiceField({ label, hint, value, options, onChange, disabled }: {
  label: string; hint?: string; value?: string;
  options: readonly (readonly [string, string])[];
  onChange: (v?: string) => void; disabled?: boolean;
}) {
  return (
    <div>
      <FLabel>{label}</FLabel>
      <select className={sCls} value={value ?? ''} disabled={disabled}
        onChange={(e) => onChange(e.target.value === '' ? undefined : e.target.value)}>
        <option value="">Model default</option>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
      {hint && <p className="text-[10px] text-ink-300 mt-0.5 leading-snug">{hint}</p>}
    </div>
  );
}

function BoolField({ label, hint, value, onChange, disabled, trueLabel = 'Enabled', falseLabel = 'Disabled' }: {
  label: string; hint?: string; value?: boolean; onChange: (v?: boolean) => void;
  disabled?: boolean; trueLabel?: string; falseLabel?: string;
}) {
  const str = value === undefined ? '' : value ? 'true' : 'false';
  return (
    <div>
      <FLabel>{label}</FLabel>
      <select className={sCls} value={str} disabled={disabled}
        onChange={(e) => onChange(e.target.value === '' ? undefined : e.target.value === 'true')}>
        <option value="">Model default</option>
        <option value="true">{trueLabel}</option>
        <option value="false">{falseLabel}</option>
      </select>
      {hint && <p className="text-[10px] text-ink-300 mt-0.5 leading-snug">{hint}</p>}
    </div>
  );
}

// ─── ExtensionPriorConfigWidget ───────────────────────────────────────────────

interface ExtensionPriorConfigWidgetProps {
  spec: ModelSpec;
  editable: boolean;
  onApplySpec: (s: ModelSpec) => void;
}

export function ExtensionPriorConfigWidget({ spec, editable, onApplySpec }: ExtensionPriorConfigWidgetProps) {
  const dagModelType = norm((spec as Record<string, unknown>).dag_model_type);
  const dagLabel = EXTENSION_DAG_LABELS[dagModelType] ?? dagModelType;
  const dag = useMemo(() => deriveDag(spec), [spec]);

  const [state, setState] = useState<ExtPriorState>(() => initState(spec, dag));

  // Re-seed when the DAG's structure changes (mediator/outcome/cross-effect set),
  // not on every spec reference — mirrors PriorConfigWidget so in-progress edits
  // survive unrelated spec updates. Value-only external edits aren't re-pulled.
  const structureSig = [
    dagModelType,
    dag.mediators.map((m) => m.name).join(','),
    dag.outcomes.map((o) => o.name).join(','),
    dag.crossPairs.map((p) => p.key).join(','),
  ].join('|');
  // eslint-disable-next-line react-hooks/set-state-in-effect, react-hooks/exhaustive-deps
  useEffect(() => { setState(initState(spec, dag)); }, [structureSig]);

  const setMed = (name: string, key: keyof MediatorPrior, val: unknown) =>
    setState((s) => ({ ...s, mediator: { ...s.mediator, [name]: { ...s.mediator[name], [key]: val } } }));
  const setOut = (name: string, key: keyof OutcomePrior, val: unknown) =>
    setState((s) => ({ ...s, outcome: { ...s.outcome, [name]: { ...s.outcome[name], [key]: val } } }));
  const setCross = (key: string, field: keyof CrossEffectPrior, val: unknown) =>
    setState((s) => ({ ...s, cross_effect: { ...s.cross_effect, [key]: { ...s.cross_effect[key], [field]: val } } }));

  const handleApply = () => {
    const specPriors = ((spec as Record<string, unknown>).priors ?? {}) as Record<string, unknown>;
    // Preserve every other prior group (seasonality is honored by extensions;
    // media/controls are ignored but kept intact), then set the three extension
    // groups authoritatively so clearing a field actually takes effect.
    const priorsOut: Record<string, unknown> = { ...specPriors };
    const groups: [string, Record<string, object>][] = [
      ['mediator', cleanGroup(state.mediator)],
      ['outcome', cleanGroup(state.outcome)],
      ['cross_effect', cleanGroup(state.cross_effect)],
    ];
    for (const [k, cleaned] of groups) {
      if (Object.keys(cleaned).length > 0 || k in specPriors) priorsOut[k] = cleaned;
    }
    onApplySpec({ ...spec, priors: priorsOut } as ModelSpec);
  };

  const TABS = [
    { key: 'mediators' as const, label: `Mediators (${dag.mediators.length})` },
    { key: 'outcomes' as const, label: `Outcomes (${dag.outcomes.length})` },
    { key: 'cross' as const, label: `Cross-effects (${dag.crossPairs.length})` },
  ];
  // Default to the first tab that actually has entries.
  const firstNonEmpty = dag.mediators.length ? 'mediators'
    : dag.outcomes.length ? 'outcomes'
    : dag.crossPairs.length ? 'cross' : 'mediators';
  const [tab, setTab] = useState<'mediators' | 'outcomes' | 'cross'>(firstNonEmpty);

  const content = !dag.hasDag ? (
    <div className="py-6 text-center space-y-1.5">
      <Share2 size={22} className="text-ink-300 mx-auto" />
      <p className="text-xs text-ink-500 font-medium">This model&apos;s priors come from its causal DAG.</p>
      <p className="text-[11px] text-ink-300 max-w-sm mx-auto leading-snug">
        No DAG specification is attached to the spec yet — derive the model from a causal DAG
        (Causal tab) to inspect and set its mediator, outcome and cross-effect priors here.
      </p>
    </div>
  ) : (
    <div className="space-y-3">
      {/* Intro note: where these priors live */}
      <div className="flex items-start gap-2 bg-cream-50 rounded-lg px-3 py-2 border border-line-200">
        <GitBranch size={13} className="text-fuchsia-500 shrink-0 mt-0.5" />
        <div className="flex-1 text-[11px] text-ink-500 leading-snug">
          <span className="font-semibold text-ink-700">{dagLabel}</span> — priors are set on the
          causal DAG&apos;s nodes and edges. Media &amp; control priors come from the DAG itself; the
          model&apos;s trend, seasonality and likelihood live in Model Configuration. Leave a field
          blank to inherit the model default.
        </div>
      </div>

      {/* Tab bar + Apply */}
      <div className="flex gap-1.5 flex-wrap">
        {TABS.map((t) => (
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

      {/* ── Mediators ─────────────────────────────────────────────────────── */}
      {tab === 'mediators' && (
        <div className="space-y-2">
          {dag.mediators.length === 0 && (
            <p className="text-xs text-ink-300 italic py-2">This model has no mediator nodes.</p>
          )}
          {dag.mediators.map((m) => {
            const p = state.mediator[m.name] ?? {};
            return (
              <div key={m.name} className="border border-line-200 rounded-xl overflow-hidden">
                <div className="flex items-center gap-3 px-4 py-2.5 bg-cream-50">
                  <Layers size={13} className="text-fuchsia-500 shrink-0" />
                  <span className="flex-1 text-xs font-bold text-ink-700">{m.label}</span>
                  <Badge label="mediator" color="indigo" />
                </div>
                <div className="px-4 py-3 grid grid-cols-2 gap-3 bg-white">
                  <NumberField label="Media effect σ" value={p.media_effect_sigma}
                    onChange={(v) => setMed(m.name, 'media_effect_sigma', v)} disabled={!editable}
                    hint="Prior sd of media → mediator strength." />
                  <ChoiceField label="Media effect constraint" value={p.media_effect_constraint}
                    onChange={(v) => setMed(m.name, 'media_effect_constraint', v)} disabled={!editable}
                    options={[['none', 'None'], ['positive', 'Positive only'], ['negative', 'Negative only']]}
                    hint="Sign constraint on media → mediator." />
                  <NumberField label="Outcome effect σ" value={p.outcome_effect_sigma}
                    onChange={(v) => setMed(m.name, 'outcome_effect_sigma', v)} disabled={!editable}
                    hint="Prior sd of mediator → KPI strength." />
                  <NumberField label="Direct effect σ" value={p.direct_effect_sigma}
                    onChange={(v) => setMed(m.name, 'direct_effect_sigma', v)} disabled={!editable}
                    hint="Prior sd of media's direct (unmediated) KPI effect." />
                  <NumberField label="Observation noise σ" value={p.observation_noise_sigma}
                    onChange={(v) => setMed(m.name, 'observation_noise_sigma', v)} disabled={!editable}
                    hint="Prior sd of the mediator's observation noise." />
                  <BoolField label="Allow direct effect" value={p.allow_direct_effect}
                    onChange={(v) => setMed(m.name, 'allow_direct_effect', v)} disabled={!editable}
                    trueLabel="Yes — allow" falseLabel="No — fully mediated"
                    hint="May media also bypass the mediator and hit the KPI directly?" />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Outcomes ──────────────────────────────────────────────────────── */}
      {tab === 'outcomes' && (
        <div className="space-y-2">
          {dag.outcomes.length === 0 && (
            <p className="text-xs text-ink-300 italic py-2">
              This model has no secondary outcome nodes — the primary KPI&apos;s priors are set in Model Configuration.
            </p>
          )}
          {dag.outcomes.map((o) => {
            const p = state.outcome[o.name] ?? {};
            return (
              <div key={o.name} className="border border-line-200 rounded-xl overflow-hidden">
                <div className="flex items-center gap-3 px-4 py-2.5 bg-cream-50">
                  <Target size={13} className="text-blue-500 shrink-0" />
                  <span className="flex-1 text-xs font-bold text-ink-700">{o.label}</span>
                  <Badge label="outcome" color="blue" />
                </div>
                <div className="px-4 py-3 grid grid-cols-2 gap-3 bg-white">
                  <NumberField label="Intercept prior σ" value={p.intercept_prior_sigma}
                    onChange={(v) => setOut(o.name, 'intercept_prior_sigma', v)} disabled={!editable}
                    hint="Prior sd of this outcome's baseline intercept." />
                  <NumberField label="Media effect σ" value={p.media_effect_sigma}
                    onChange={(v) => setOut(o.name, 'media_effect_sigma', v)} disabled={!editable}
                    hint="Prior sd of media → this outcome." />
                  <BoolField label="Include trend" value={p.include_trend}
                    onChange={(v) => setOut(o.name, 'include_trend', v)} disabled={!editable}
                    hint="Give this outcome its own trend component." />
                  <BoolField label="Include seasonality" value={p.include_seasonality}
                    onChange={(v) => setOut(o.name, 'include_seasonality', v)} disabled={!editable}
                    hint="Give this outcome its own seasonality component." />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Cross-effects ─────────────────────────────────────────────────── */}
      {tab === 'cross' && (
        <div className="space-y-2">
          {dag.crossPairs.length === 0 && (
            <p className="text-xs text-ink-300 italic py-2">
              No cross-outcome effects — a single-outcome model (e.g. mediation) has none.
            </p>
          )}
          {dag.crossPairs.map((pair) => {
            const p = state.cross_effect[pair.key] ?? {};
            return (
              <div key={pair.key} className="border border-line-200 rounded-xl overflow-hidden">
                <div className="flex items-center gap-2 px-4 py-2.5 bg-cream-50">
                  <Share2 size={13} className="text-fuchsia-500 shrink-0" />
                  <span className="flex-1 text-xs font-bold text-ink-700">
                    {pair.sourceLabel} <span className="text-ink-300 font-normal">→</span> {pair.targetLabel}
                  </span>
                  {pair.inferred && <Badge label="inferred" color="amber" />}
                </div>
                <div className="px-4 py-3 grid grid-cols-2 gap-3 bg-white">
                  <ChoiceField label="Effect type" value={p.effect_type}
                    onChange={(v) => setCross(pair.key, 'effect_type', v)} disabled={!editable}
                    options={[['cannibalization', 'Cannibalization (−)'], ['halo', 'Halo (+)'], ['unconstrained', 'Unconstrained']]}
                    hint="Sign prior on the cross-outcome effect." />
                  <NumberField label="Prior σ" value={p.prior_sigma}
                    onChange={(v) => setCross(pair.key, 'prior_sigma', v)} disabled={!editable}
                    hint="Prior sd of the cross-effect magnitude." />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  return (
    <DashWidget
      title="Prior Configuration"
      icon={<Share2 size={15} className="text-fuchsia-500 shrink-0" />}
      color="fuchsia"
      expandTitle="Prior Configuration"
      expandContent={<div className="max-w-3xl mx-auto py-2">{content}</div>}
    >
      {content}
    </DashWidget>
  );
}
