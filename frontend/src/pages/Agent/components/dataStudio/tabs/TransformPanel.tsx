import { DashWidget } from '../../common/DashWidget';
import { DataTable } from '../../common/DataTable';
import { RolesEditor } from '../transform/RolesEditor';
import { StepBuilder } from '../transform/StepBuilder';
import { StepList } from '../transform/StepList';
import type { DataStudioState, StudioRole, TableSpec, TransformStep } from '../../../types';

function previewTable(state: DataStudioState): TableSpec {
  const cols = state.all_columns.length ? state.all_columns : state.columns;
  return {
    title: 'Preview',
    source: 'data_studio',
    columns: cols.map(c => ({ key: c, label: c })),
    rows: state.preview_rows,
    total_rows: state.n_rows,
    truncated: state.n_rows > state.preview_rows.length,
  };
}

function DiffPill({ label, before, after }: { label: string; before: number; after: number }) {
  const changed = before !== after;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs ${changed ? 'bg-indigo-50 text-indigo-700' : 'bg-cream-100 text-ink-500'}`}>
      {label}: {before}{changed ? ` → ${after}` : ''}
    </span>
  );
}

export function TransformPanel({
  state, busy, onSteps, onRoles,
}: {
  state: DataStudioState;
  busy: boolean;
  onSteps: (steps: TransformStep[]) => void;
  onRoles: (roles: Record<string, StudioRole>) => void;
}) {
  const addStep = (step: TransformStep) => onSteps([...state.steps, step]);
  const removeStep = (idx: number) => onSteps(state.steps.filter((_, i) => i !== idx));
  const reorder = (from: number, to: number) => {
    if (to < 0 || to >= state.steps.length) return;
    const next = [...state.steps];
    const [m] = next.splice(from, 1);
    next.splice(to, 0, m);
    onSteps(next);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="space-y-4">
        <DashWidget title="Column roles" dotColor="bg-emerald-500" color="emerald">
          <RolesEditor state={state} onChange={onRoles} disabled={busy} />
        </DashWidget>

        <DashWidget title="Cleaning pipeline" dotColor="bg-indigo-500" color="indigo">
          <div className="space-y-3">
            <StepList steps={state.steps} onReorder={reorder} onRemove={removeStep} disabled={busy} />
            <StepBuilder state={state} onAdd={addStep} disabled={busy} />
          </div>
        </DashWidget>
      </div>

      <DashWidget title="Preview" dotColor="bg-sky-500" color="sky">
        <div className="space-y-3">
          {state.diff && (
            <div className="flex flex-wrap gap-1.5">
              <DiffPill label="rows" before={state.diff.rows_before} after={state.diff.rows_after} />
              <DiffPill label="cols" before={state.diff.cols_before} after={state.diff.cols_after} />
            </div>
          )}
          {state.warnings.length > 0 && (
            <ul className="space-y-1">
              {state.warnings.map((w, i) => (
                <li key={i} className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2.5 py-1">{w}</li>
              ))}
            </ul>
          )}
          <DataTable table={previewTable(state)} maxHeight={420} />
        </div>
      </DashWidget>
    </div>
  );
}
