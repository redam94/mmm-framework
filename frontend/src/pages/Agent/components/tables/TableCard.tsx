import React from 'react';
import { useInView } from '../../hooks/useInView';
import { useTableSpec } from '../../hooks/useTableSpec';
import { DataTable } from '../common/DataTable';
import type { TableRef } from '../../types';

// React.memo: table refs are content-addressed by `id`, but each streaming
// update re-parses the dashboard_data JSON, giving every ref a NEW object
// identity. Compare by id so existing tables are not needlessly re-rendered
// (and re-sorted) on every SSE chunk while the agent streams. Combined with
// the viewport gate, this keeps the page responsive as tables accumulate.
export const TableCard = React.memo(function TableCard({
  tableRef,
  idx,
}: {
  tableRef: TableRef;
  idx: number;
}) {
  // The observed wrapper is ALWAYS in the DOM (even before reveal) so the
  // IntersectionObserver can fire; the payload fetch is deferred until in view.
  const [wrapRef, inView] = useInView<HTMLDivElement>();
  const spec = useTableSpec(tableRef.id, inView);

  const title = spec?.title || tableRef.title || `Table ${idx + 1}`;
  const source = spec?.source ?? tableRef.source;

  return (
    <div
      ref={wrapRef}
      className="rounded-xl border border-line-200 bg-white shadow-sm overflow-hidden min-h-[140px]"
    >
      <div className="flex items-center gap-2 px-4 pt-3 pb-2">
        <p className="text-xs text-ink-400 font-semibold truncate flex-1">{title}</p>
        {spec && (
          <span className="text-[10px] text-ink-300 shrink-0 tabular-nums">
            {(spec.total_rows ?? spec.rows.length)} × {spec.columns.length}
          </span>
        )}
        {source && (
          <span className="text-[10px] font-mono bg-cream-100 text-ink-400 border border-line-200 rounded-full px-2 py-0.5 shrink-0">
            {source}
          </span>
        )}
      </div>
      {inView && spec ? (
        <div className="px-4 pb-3">
          <DataTable table={spec} maxHeight={360} />
        </div>
      ) : (
        <div className="h-[100px] flex items-center justify-center text-sm text-ink-300">
          {inView ? 'Loading table…' : ''}
        </div>
      )}
    </div>
  );
}, (a, b) => a.idx === b.idx && a.tableRef.id === b.tableRef.id);
