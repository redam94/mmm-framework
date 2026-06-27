import type { DataStudioState, StudioRole } from '../../../types';

// A fixed-width select — NOT the shared sCls, which carries `w-full` and would
// expand to the whole row and hide the column name beside it.
const roleSelectCls =
  'w-52 shrink-0 bg-cream-50 border border-line-200 rounded-lg px-2.5 py-1.5 ' +
  'text-xs text-ink-900 cursor-pointer focus:outline-none focus:ring-2 focus:ring-indigo-400';

const ROLES: { value: StudioRole; label: string }[] = [
  { value: 'ignore', label: 'Ignore (not used)' },
  { value: 'kpi', label: 'KPI — the outcome to explain' },
  { value: 'media', label: 'Media — a spend / activity channel' },
  { value: 'control', label: 'Control — a non-media driver' },
  { value: 'date', label: 'Date — the time period' },
  { value: 'group', label: 'Group — geo / segment' },
];

const ROLE_DOT: Record<string, string> = {
  kpi: 'bg-emerald-500', media: 'bg-indigo-500', control: 'bg-violet-500',
  date: 'bg-sky-500', group: 'bg-amber-500', ignore: 'bg-line-400',
};

function sample(state: DataStudioState, col: string): string {
  const v = state.preview_rows?.[0]?.[col];
  if (v == null) return '';
  const s = String(v);
  return s.length > 22 ? s.slice(0, 22) + '…' : s;
}

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
    <div className="space-y-2">
      <p className="text-xs text-ink-400">
        Tell the model what each column is. At minimum set one <span className="font-medium text-emerald-700">KPI</span>{' '}
        (the outcome) and your <span className="font-medium text-indigo-700">media</span> channels — the rest can stay <em>Ignore</em>.
      </p>
      <div className="divide-y divide-line-200 rounded-xl border border-line-200 overflow-hidden">
        {state.columns.map(col => {
          const role = state.roles[col] ?? 'ignore';
          const ex = sample(state, col);
          return (
            <div key={col} className="flex items-center gap-3 px-3 py-2 bg-white">
              <span className={`w-2 h-2 rounded-full shrink-0 ${ROLE_DOT[role]}`} />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-ink-900 font-mono truncate" title={col}>{col}</p>
                <p className="text-[11px] text-ink-300 truncate">
                  {state.dtypes?.[col] ?? ''}{ex ? ` · e.g. ${ex}` : ''}
                </p>
              </div>
              <select
                value={role}
                disabled={disabled}
                onChange={e => setRole(col, e.target.value as StudioRole)}
                className={roleSelectCls}
              >
                {ROLES.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
              </select>
            </div>
          );
        })}
      </div>
    </div>
  );
}
