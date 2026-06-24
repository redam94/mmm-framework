import { useMemo, useState } from 'react';
import { Boxes, Check, RotateCcw, Sliders } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { remarkPlugins, rehypePlugins, normalizeMath } from '../../../../lib/markdownMath';
import { DashWidget } from '../common/DashWidget';
import { Badge } from '../common/Badge';
import { FLabel, iCls, sCls } from '../common/form';
import { mdComponents } from '../common/markdown';
import { SchemaForm, type JsonSchema } from './SchemaForm';
import type { GardenModel } from '../../../../api/services/modelGardenService';
import type { ModelSpec } from '../../types';

const LIKELIHOOD_FAMILIES = [
  'normal', 'student_t', 'lognormal', 'binomial',
  'beta_binomial', 'poisson', 'negative_binomial', 'beta',
];

const KIND_COLOR: Record<string, 'indigo' | 'blue' | 'amber' | 'green' | 'gray'> = {
  mmm: 'indigo', cfa: 'blue', latent_class: 'amber', awareness: 'green',
};

const KIND_LABEL: Record<string, string> = {
  mmm: 'MMM', cfa: 'CFA', latent_class: 'Latent Class', awareness: 'Awareness',
};

/**
 * Model-tab configuration for a bespoke / non-MMM garden model: the model's
 * identity (name, version, family kind), its **bespoke ``model_params``** (rendered
 * from the manifest's ``config_schema`` JSON Schema), the **likelihood**, and —
 * for non-MMM families that don't get the MMM widgets — the inference settings.
 * Declared estimands + capabilities are shown as advisory chips. Edits round-trip
 * through ``onApplySpec`` (the same server-authoritative spec PATCH as the MMM tab).
 */
export function GardenModelConfigWidget({
  spec,
  gardenModel,
  modelKind,
  editable,
  fitted,
  onApplySpec,
  showInference,
}: {
  spec: ModelSpec;
  gardenModel: GardenModel | undefined;
  modelKind: string;
  editable: boolean;
  fitted: boolean;
  onApplySpec: (newSpec: ModelSpec) => void;
  showInference: boolean;
}) {
  const manifest = gardenModel?.manifest;
  const mdComps = useMemo(() => mdComponents(), []);
  const ref = spec.garden_ref;
  const configSchema = (manifest?.config_schema as JsonSchema | undefined) || undefined;
  const hasParams = !!configSchema?.properties && Object.keys(configSchema.properties).length > 0;
  const hasLikelihood = !!spec.likelihood;

  const [params, setParams] = useState<Record<string, unknown>>(spec.model_params || {});
  const [family, setFamily] = useState<string>(spec.likelihood?.family || 'normal');
  const [inference, setInference] = useState<Record<string, unknown>>(spec.inference || {});

  // Re-sync local drafts when the agent updates the spec. Done with the React
  // "storing information from previous renders" pattern (set-state during render
  // when a derived key changes) rather than an effect, so there's no extra commit.
  const specKey = JSON.stringify([spec.model_params, spec.likelihood?.family, spec.inference]);
  const [syncedKey, setSyncedKey] = useState(specKey);
  if (specKey !== syncedKey) {
    setSyncedKey(specKey);
    setParams(spec.model_params || {});
    setFamily(spec.likelihood?.family || 'normal');
    setInference(spec.inference || {});
  }

  const dirty = useMemo(() => {
    const a = JSON.stringify(params) !== JSON.stringify(spec.model_params || {});
    const b = hasLikelihood && family !== (spec.likelihood?.family || 'normal');
    const c = showInference && JSON.stringify(inference) !== JSON.stringify(spec.inference || {});
    return a || b || c;
  }, [params, family, inference, spec, hasLikelihood, showInference]);

  const apply = () => {
    const next: ModelSpec = { ...spec, model_params: params };
    if (hasLikelihood) next.likelihood = { ...(spec.likelihood || {}), family };
    if (showInference) next.inference = inference;
    onApplySpec(next);
  };
  const reset = () => {
    setParams(spec.model_params || {});
    setFamily(spec.likelihood?.family || 'normal');
    setInference(spec.inference || {});
  };

  const setInf = (k: string, v: unknown) => setInference((p) => ({ ...p, [k]: v }));
  const estimands = manifest?.default_estimands || [];
  const capabilities = manifest?.capabilities || [];

  return (
    <DashWidget title="Model" icon={<Boxes size={16} className="text-sage-700" />} color="sage">
      {/* Identity */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        {ref?.name && (
          <span className="text-sm font-bold text-ink-900">
            {ref.name}{ref.version != null && <span className="text-ink-400 font-medium"> v{ref.version}</span>}
          </span>
        )}
        <Badge label={KIND_LABEL[modelKind] || modelKind} color={KIND_COLOR[modelKind] || 'gray'} />
        {gardenModel?.status && gardenModel.status !== 'published' && (
          <Badge label={gardenModel.status} color="amber" />
        )}
        {manifest?.class_name && (
          <span className="text-[11px] font-mono text-ink-400">{manifest.class_name}</span>
        )}
      </div>
      {gardenModel?.docs && (
        <div className="prose prose-sm max-w-none text-xs text-ink-600 mb-3 max-h-48 overflow-y-auto rounded-lg border border-line-200 bg-cream-50 px-3 py-2">
          <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={mdComps}>
            {normalizeMath(gardenModel.docs)}
          </ReactMarkdown>
        </div>
      )}

      {/* Bespoke model parameters (from CONFIG_SCHEMA JSON Schema) */}
      {hasParams && configSchema && (
        <div className="mb-3 border border-line-200 rounded-xl overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-2 bg-cream-50 border-b border-line-200">
            <Sliders size={13} className="text-ink-400" />
            <span className="text-xs font-bold text-ink-700 uppercase tracking-wider">Model parameters</span>
          </div>
          <div className="px-3 py-3 bg-white">
            <SchemaForm schema={configSchema} values={params} editable={editable} onChange={setParams} />
          </div>
        </div>
      )}

      {/* Likelihood (only for families that read spec.likelihood, e.g. awareness) */}
      {hasLikelihood && (
        <div className="mb-3">
          <FLabel>Likelihood family</FLabel>
          <select className={sCls} disabled={!editable} value={family} onChange={(e) => setFamily(e.target.value)}>
            {LIKELIHOOD_FAMILIES.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
          {spec.likelihood?.link && (
            <p className="text-[10px] text-ink-300 mt-0.5">link: {spec.likelihood.link}</p>
          )}
        </div>
      )}

      {/* Inference (only when the MMM widgets aren't shown to cover it) */}
      {showInference && (
        <div className="mb-3 grid grid-cols-2 gap-2">
          {(['chains', 'draws', 'tune'] as const).map((k) => (
            <div key={k}>
              <FLabel>{k}</FLabel>
              <input
                type="number"
                className={iCls}
                disabled={!editable}
                value={inference[k] === undefined ? '' : Number(inference[k])}
                onChange={(e) => setInf(k, e.target.value === '' ? undefined : parseInt(e.target.value, 10))}
              />
            </div>
          ))}
          <div>
            <FLabel>target_accept</FLabel>
            <input
              type="number"
              className={iCls}
              step="0.01"
              disabled={!editable}
              value={inference.target_accept === undefined ? '' : Number(inference.target_accept)}
              onChange={(e) => setInf('target_accept', e.target.value === '' ? undefined : parseFloat(e.target.value))}
            />
          </div>
        </div>
      )}

      {/* Advisory: declared estimands + capabilities */}
      {(estimands.length > 0 || capabilities.length > 0) && (
        <div className="space-y-1.5">
          {estimands.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[10px] text-ink-400 uppercase tracking-wider font-semibold">Estimands</span>
              {estimands.map((e) => <Badge key={e} label={e} color="green" />)}
            </div>
          )}
          {capabilities.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[10px] text-ink-400 uppercase tracking-wider font-semibold">Capabilities</span>
              {capabilities.map((c) => <Badge key={c} label={c} color="gray" />)}
            </div>
          )}
        </div>
      )}

      {/* Apply / reset (server-authoritative spec PATCH) */}
      {editable && dirty && (
        <div className="flex items-center gap-2 mt-4 pt-3 border-t border-line-200">
          <button
            onClick={apply}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-sage-700 text-white hover:bg-sage-800 transition-colors"
          >
            <Check size={13} /> Apply
          </button>
          <button
            onClick={reset}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-ink-500 hover:bg-cream-100 transition-colors"
          >
            <RotateCcw size={13} /> Reset
          </button>
          {fitted && (
            <span className="text-[10px] text-ink-400 ml-1">
              Re-fit (<span className="font-mono">fit_mmm_model</span>) to apply.
            </span>
          )}
        </div>
      )}
    </DashWidget>
  );
}
