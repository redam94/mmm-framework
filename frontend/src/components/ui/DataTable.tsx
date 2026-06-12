import { clsx } from 'clsx';
import type { ReactNode } from 'react';

export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => ReactNode;
  /** Right-align + mono (numbers) */
  numeric?: boolean;
  className?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  empty?: ReactNode;
  className?: string;
}

/** Dense editorial table: hairline rules, mono numerals, hover wash. */
export function DataTable<T>({ columns, rows, rowKey, onRowClick, empty, className }: DataTableProps<T>) {
  if (rows.length === 0 && empty) return <>{empty}</>;
  return (
    <div className={clsx('overflow-x-auto rounded-lg border border-line-200 bg-white shadow-sm', className)}>
      <table className="min-w-full divide-y divide-line-200">
        <thead>
          <tr className="bg-cream-100/70">
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={clsx(
                  'px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-ink-400',
                  c.numeric ? 'text-right' : 'text-left',
                  c.className,
                )}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line-200">
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={clsx('transition-colors', onRowClick && 'cursor-pointer hover:bg-cream-100/60')}
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={clsx(
                    'px-4 py-2.5 text-sm text-ink-700',
                    c.numeric && 'text-right num',
                    c.className,
                  )}
                >
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
