import { sCls } from '../../common/form';
import type { DataStudioState, StudioRole } from '../../../types';

const ROLES: { value: StudioRole; label: string }[] = [
  { value: 'ignore', label: '— ignore —' },
  { value: 'kpi', label: 'KPI (target)' },
  { value: 'media', label: 'Media' },
  { value: 'control', label: 'Control' },
  { value: 'date', label: 'Date' },
  { value: 'group', label: 'Group (geo)' },
];

const ROLE_DOT: Record<string, string> = {
  kpi: 'bg-emerald-500', media: 'bg-indigo-500', control: 'bg-violet-500',
  date: 'bg-sky-500', group: 'bg-amber-500', ignore: 'bg-line-400',
};

// Assign each column a model role. Roles drive the EDA (charts colour by role)
// and the commit (KPI/media/control → model spec; date → MFF Period; group → geo).
export function RolesEditor({
  state, onChange, disabled,
}: {
  state: DataStudioState;
  onChange: (roles: Record<string, StudioRole>) => void;
  disabled?: boolean;
}) {
  const setRole = (col: string, role: StudioRole) => {
    const next = { ...state.roles };
    if (role === 'ignore') delete next[col];
    else next[col] = role;
    onChange(next);
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {state.columns.map(col => {
        const role = state.roles[col] ?? 'ignore';
        return (
          <div key={col} className="flex items-center gap-2 bg-cream-50 border border-line-200 rounded-lg px-2.5 py-1.5">
            <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${ROLE_DOT[role]}`} />
            <span className="text-xs font-mono text-ink-700 truncate flex-1" title={col}>{col}</span>
            <select
              value={role}
              disabled={disabled}
              onChange={e => setRole(col, e.target.value as StudioRole)}
              className={`${sCls} w-32 shrink-0`}
            >
              {ROLES.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
          </div>
        );
      })}
    </div>
  );
}
