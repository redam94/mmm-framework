import { useMemo, useState } from 'react';
import { Download } from 'lucide-react';
import { formatCell } from '../../utils/format';
import type { TableSpec } from '../../types';

// ─── DataTable ────────────────────────────────────────────────────────────────
// Dependency-free sortable table renderer for server-provided TableSpec payloads.
// Click a header to cycle asc → desc → none. Numeric columns sort numerically
// and right-align; everything else falls back to localeCompare.

type SortState = { key: string; dir: 'asc' | 'desc' } | null;

const NUMERIC_TYPES = new Set(['number', 'percent', 'currency']);

function isFiniteValue(v: unknown): boolean {
  if (typeof v === 'number') return Number.isFinite(v);
  if (typeof v === 'string' && v.trim() !== '') return Number.isFinite(Number(v));
  return false;
}

function csvEscape(v: unknown): string {
  const s = v == null ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function DataTable({ table, maxHeight = 360 }: { table: TableSpec; maxHeight?: number }) {
  const [sort, setSort] = useState<SortState>(null);

  // A column is "numeric" if its declared type says so, or every non-null value
  // in it parses as a finite number (with at least one value present).
  const numericCols = useMemo(() => {
    const set = new Set<string>();
    for (const col of table.columns) {
      if (col.type && NUMERIC_TYPES.has(col.type)) { set.add(col.key); continue; }
      if (col.type) continue; // declared non-numeric (string/date)
      const values = table.rows.map(r => r[col.key]).filter(v => v != null);
      if (values.length > 0 && values.every(isFiniteValue)) set.add(col.key);
    }
    return set;
  }, [table]);

  const sortedRows = useMemo(() => {
    if (!sort) return table.rows;
    const { key, dir } = sort;
    const numeric = numericCols.has(key);
    const sign = dir === 'asc' ? 1 : -1;
    return [...table.rows].sort((a, b) => {
      const va = a[key];
      const vb = b[key];
      if (va == null && vb == null) return 0;
      if (va == null) return 1; // nulls last regardless of direction
      if (vb == null) return -1;
      if (numeric) return (Number(va) - Number(vb)) * sign;
      return String(va).localeCompare(String(vb)) * sign;
    });
  }, [table, sort, numericCols]);

  const cycleSort = (key: string) =>
    setSort(prev => {
      if (!prev || prev.key !== key) return { key, dir: 'asc' };
      if (prev.dir === 'asc') return { key, dir: 'desc' };
      return null;
    });

  const downloadCsv = () => {
    const lines = [table.columns.map(c => csvEscape(c.label)).join(',')];
    for (const row of table.rows) {
      lines.push(table.columns.map(c => csvEscape(row[c.key])).join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(table.title || 'table').replace(/[^\w.-]+/g, '_').toLowerCase() || 'table'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const totalRows = table.total_rows ?? table.rows.length;

  return (
    <div>
      <div className="overflow-auto rounded-lg border border-line-200" style={{ maxHeight }}>
        <table className="w-full text-left text-sm border-collapse">
          <thead className="sticky top-0 bg-cream-50 z-10 text-ink-400 uppercase text-xs">
            <tr>
              {table.columns.map(col => {
                const numeric = numericCols.has(col.key);
                const active = sort?.key === col.key;
                return (
                  <th key={col.key} className="px-3 py-2.5 font-medium border-b border-line-200">
                    <button
                      onClick={() => cycleSort(col.key)}
                      className={`flex items-center gap-1 w-full uppercase transition-colors ${
                        numeric ? 'justify-end text-right' : 'justify-start text-left'
                      } ${active ? 'text-sage-800' : 'hover:text-ink-900'}`}
                      title="Sort column"
                    >
                      <span className="truncate">{col.label}</span>
                      <span className={`text-[9px] shrink-0 ${active ? 'text-sage-700' : 'text-ink-300'}`}>
                        {active ? (sort?.dir === 'asc' ? '▲' : '▼') : '↕'}
                      </span>
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-line-200">
            {sortedRows.map((row, ri) => (
              <tr key={ri} className="bg-white hover:bg-cream-100 transition-colors">
                {table.columns.map(col => (
                  <td
                    key={col.key}
                    className={`px-3 py-2 whitespace-nowrap ${
                      numericCols.has(col.key)
                        ? 'text-right tabular-nums text-ink-700'
                        : 'text-ink-700'
                    }`}
                  >
                    {formatCell(row[col.key], col.type)}
                  </td>
                ))}
              </tr>
            ))}
            {sortedRows.length === 0 && (
              <tr>
                <td colSpan={table.columns.length} className="px-3 py-6 text-center text-ink-300">
                  No rows
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between gap-3 pt-2">
        <p className="text-[11px] text-ink-300">
          {table.truncated
            ? `Showing ${table.rows.length} of ${totalRows} rows — full data available via CSV`
            : `${totalRows} row${totalRows === 1 ? '' : 's'}`}
        </p>
        <button
          onClick={downloadCsv}
          className="flex items-center gap-1 text-[11px] text-ink-300 hover:text-sage-800 transition-colors shrink-0"
          title="Download the visible rows as CSV"
        >
          <Download size={11} /> Download CSV
        </button>
      </div>
    </div>
  );
}
