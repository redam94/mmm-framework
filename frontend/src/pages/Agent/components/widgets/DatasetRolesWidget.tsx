import { Tags } from 'lucide-react';
import { DashWidget } from '../common/DashWidget';
import { Badge } from '../common/Badge';
import type { DatasetRole, ModelSpec } from '../../types';
import type { GardenModel } from '../../../../api/services/modelGardenService';

type BadgeColor = 'blue' | 'indigo' | 'gray' | 'green' | 'amber' | 'red';

const ROLE_META: Record<DatasetRole, { label: string; color: BadgeColor }> = {
  target: { label: 'Target', color: 'green' },
  predictor: { label: 'Predictor', color: 'indigo' },
  control: { label: 'Control', color: 'blue' },
  indicator: { label: 'Indicator', color: 'amber' },
  group: { label: 'Group', color: 'gray' },
  time: { label: 'Time', color: 'gray' },
  offset: { label: 'Offset', color: 'gray' },
  weight: { label: 'Weight', color: 'gray' },
  trials: { label: 'Trials', color: 'amber' },
  auxiliary: { label: 'Auxiliary', color: 'gray' },
};

const ROLE_ORDER: DatasetRole[] = [
  'target', 'predictor', 'control', 'indicator',
  'trials', 'offset', 'weight', 'group', 'time', 'auxiliary',
];

/**
 * Data-tab view of how the dataset's columns map to model **roles** — the
 * "particulars of the dataset" for the flexible Dataset layer. The mapping comes
 * from the spec's explicit ``dataset`` role bindings (native non-MMM datasets) or
 * is synthesized from the MMM spec (kpi → target, media → predictor, controls →
 * control). When a garden model is loaded, the model's *required* roles /
 * capabilities (from its manifest) are surfaced so the user can see what data
 * shape it expects.
 */
export function DatasetRolesWidget({
  spec,
  gardenModel,
}: {
  spec: ModelSpec | undefined;
  gardenModel: GardenModel | undefined;
}) {
  const roleMap: Record<string, string[]> = {};
  const add = (role: string, name: string | undefined) => {
    if (!name) return;
    (roleMap[role] ||= []).push(name);
  };

  const bindings = spec?.dataset?.bindings;
  if (bindings && bindings.length > 0) {
    // Native role-tagged dataset.
    bindings.forEach((b) => add(b.role, b.name));
  } else if (spec) {
    // Synthesize the MMM role view from the standard spec fields.
    add('target', spec.kpi);
    (spec.media_channels || []).forEach((m) => add('predictor', m.name));
    (spec.control_variables || []).forEach((c) => add('control', c.name));
  }

  const ds = (gardenModel?.manifest?.dataset_schema || {}) as {
    required_roles?: string[];
    required_capabilities?: string[];
  };
  const requiredRoles = ds.required_roles || [];
  const requiredCaps = ds.required_capabilities || [];

  const hasMapping = Object.keys(roleMap).length > 0;
  if (!hasMapping && requiredRoles.length === 0 && requiredCaps.length === 0) return null;

  return (
    <DashWidget title="Column roles" icon={<Tags size={16} className="text-violet-600" />} color="violet">
      {hasMapping ? (
        <div className="space-y-2">
          {ROLE_ORDER.filter((r) => roleMap[r]?.length).map((role) => {
            const meta = ROLE_META[role];
            return (
              <div key={role} className="flex items-start gap-2">
                <span className="shrink-0 w-20 pt-0.5">
                  <Badge label={meta.label} color={meta.color} />
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {roleMap[role].map((c) => (
                    <span
                      key={c}
                      className="px-2 py-0.5 text-xs rounded-md bg-cream-50 border border-line-200 text-ink-700 font-mono"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-xs text-ink-400">
          No column roles assigned yet — configure the model (or load a dataset) to map columns to roles.
        </p>
      )}

      {(requiredRoles.length > 0 || requiredCaps.length > 0) && (
        <div className="mt-3 pt-3 border-t border-line-200 space-y-1.5">
          {requiredRoles.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[10px] text-ink-400 uppercase tracking-wider font-semibold">This model needs</span>
              {requiredRoles.map((r) => (
                <Badge key={r} label={ROLE_META[r as DatasetRole]?.label || r} color="amber" />
              ))}
            </div>
          )}
          {requiredCaps.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[10px] text-ink-400 uppercase tracking-wider font-semibold">Data capabilities</span>
              {requiredCaps.map((c) => <Badge key={c} label={c} color="gray" />)}
            </div>
          )}
        </div>
      )}
    </DashWidget>
  );
}
