import { useState } from 'react';
import { Plus } from 'lucide-react';
import { iCls, sCls, FLabel } from '../../common/form';
import type { DataStudioState, TransformStep } from '../../../types';

type OpKind = TransformStep['op'];

const OP_LABELS: { value: OpKind; label: string }[] = [
  { value: 'drop_columns', label: 'Drop columns' },
  { value: 'rename', label: 'Rename column' },
  { value: 'cast', label: 'Cast type' },
  { value: 'parse_date', label: 'Parse date' },
  { value: 'fill_missing', label: 'Fill missing' },
  { value: 'drop_duplicates', label: 'Drop duplicates' },
  { value: 'filter_rows', label: 'Filter rows' },
  { value: 'date_range', label: 'Date range' },
  { value: 'winsorize', label: 'Winsorize (cap)' },
  { value: 'impute', label: 'Impute value' },
];

const FILL = ['mean', 'median', 'zero', 'ffill', 'bfill', 'interpolate'];
const OPERATORS = ['==', '!=', '<', '<=', '>', '>=', 'notnull'];
const DTYPES = ['number', 'integer', 'string', 'category', 'datetime'];

export function StepBuilder({
  state, onAdd, disabled,
}: {
  state: DataStudioState;
  onAdd: (step: TransformStep) => void;
  disabled?: boolean;
}) {
  const [op, setOp] = useState<OpKind>('drop_columns');
  const cols = state.columns;
  const first = cols[0] ?? '';

  // op-specific draft fields
  const [column, setColumn] = useState(first);
  const [to, setTo] = useState('');
  const [dtype, setDtype] = useState('number');
  const [strategy, setStrategy] = useState('mean');
  const [operator, setOperator] = useState('==');
  const [value, setValue] = useState('');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [dropCols, setDropCols] = useState<string[]>([]);
  const [dayfirst, setDayfirst] = useState(false);

  const reset = () => { setTo(''); setValue(''); setStart(''); setEnd(''); setDropCols([]); setDayfirst(false); };

  const build = (): TransformStep | null => {
    switch (op) {
      case 'drop_columns': return dropCols.length ? { op, columns: dropCols } : null;
      case 'rename': return column && to ? { op, from: column, to } : null;
      case 'cast': return column ? { op, column, dtype: dtype as 'number' } : null;
      case 'parse_date': return { op, column: column || undefined, ...(dayfirst ? { dayfirst: true } : {}) } as TransformStep;
      case 'fill_missing': return { op, strategy: strategy as 'mean' };
      case 'drop_duplicates': return { op };
      case 'filter_rows': return column ? { op, column, operator, value: operator === 'notnull' ? undefined : value } : null;
      case 'date_range': return start || end ? { op, start: start || undefined, end: end || undefined } : null;
      case 'winsorize': return column && value !== '' ? { op, column, cap_value: Number(value) } : null;
      case 'impute': return column && value !== '' ? { op, column, value: Number(value) } : null;
      default: return null;
    }
  };

  const add = () => {
    const step = build();
    if (!step) return;
    onAdd(step);
    reset();
  };

  const colSelect = (
    <div>
      <FLabel>Column</FLabel>
      <select className={sCls} value={column} onChange={e => setColumn(e.target.value)}>
        {cols.map(c => <option key={c} value={c}>{c}</option>)}
      </select>
    </div>
  );

  return (
    <div className="bg-cream-50 border border-line-200 rounded-xl p-3 space-y-2.5">
      <div className="grid grid-cols-2 gap-2.5">
        <div>
          <FLabel>Operation</FLabel>
          <select className={sCls} value={op} onChange={e => { setOp(e.target.value as OpKind); reset(); }}>
            {OP_LABELS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>

        {op === 'rename' && (<>{colSelect}<div><FLabel>New name</FLabel><input className={iCls} value={to} onChange={e => setTo(e.target.value)} /></div></>)}
        {op === 'cast' && (<>{colSelect}<div><FLabel>Type</FLabel><select className={sCls} value={dtype} onChange={e => setDtype(e.target.value)}>{DTYPES.map(d => <option key={d}>{d}</option>)}</select></div></>)}
        {op === 'parse_date' && (<>{colSelect}<label className="flex items-center gap-1.5 text-xs text-ink-600 self-end pb-1.5"><input type="checkbox" checked={dayfirst} onChange={e => setDayfirst(e.target.checked)} /> Day-first (DD/MM/YYYY)</label></>)}
        {op === 'fill_missing' && (<div><FLabel>Strategy</FLabel><select className={sCls} value={strategy} onChange={e => setStrategy(e.target.value)}>{FILL.map(s => <option key={s}>{s}</option>)}</select></div>)}
        {op === 'filter_rows' && (<>{colSelect}<div><FLabel>Op</FLabel><select className={sCls} value={operator} onChange={e => setOperator(e.target.value)}>{OPERATORS.map(o => <option key={o}>{o}</option>)}</select></div>{operator !== 'notnull' && <div><FLabel>Value</FLabel><input className={iCls} value={value} onChange={e => setValue(e.target.value)} /></div>}</>)}
        {op === 'date_range' && (<><div><FLabel>Start</FLabel><input type="date" className={iCls} value={start} onChange={e => setStart(e.target.value)} /></div><div><FLabel>End</FLabel><input type="date" className={iCls} value={end} onChange={e => setEnd(e.target.value)} /></div></>)}
        {(op === 'winsorize' || op === 'impute') && (<>{colSelect}<div><FLabel>{op === 'winsorize' ? 'Cap value' : 'Value'}</FLabel><input type="number" className={iCls} value={value} onChange={e => setValue(e.target.value)} /></div></>)}
      </div>

      {op === 'drop_columns' && (
        <div>
          <FLabel>Columns to drop</FLabel>
          <div className="flex flex-wrap gap-1.5">
            {cols.map(c => {
              const on = dropCols.includes(c);
              return (
                <button key={c} onClick={() => setDropCols(p => on ? p.filter(x => x !== c) : [...p, c])}
                  className={`px-2 py-0.5 rounded-full text-xs border ${on ? 'bg-red-50 text-red-700 border-red-200' : 'bg-white text-ink-600 border-line-200 hover:border-red-300'}`}>
                  {c}
                </button>
              );
            })}
          </div>
        </div>
      )}

      <button onClick={add} disabled={disabled}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors disabled:opacity-50 self-start">
        <Plus size={13} /> Add step
      </button>
    </div>
  );
}
