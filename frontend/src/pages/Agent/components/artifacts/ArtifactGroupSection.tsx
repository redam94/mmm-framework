import { formatDistanceToNow } from 'date-fns';
import { FileDown } from 'lucide-react';
import { DashWidget } from '../common/DashWidget';
import { PlotCard } from '../plots/PlotCard';
import { TableCard } from '../tables/TableCard';
import { CodeSegmentCard } from './CodeSegmentCard';
import type { ArtifactGroup, ArtifactItem } from '../../utils/artifactGroups';

function countBadges(counts: ArtifactGroup['counts']): string {
  const parts: string[] = [];
  if (counts.plots > 0) parts.push(`${counts.plots} plot${counts.plots === 1 ? '' : 's'}`);
  if (counts.tables > 0) parts.push(`${counts.tables} table${counts.tables === 1 ? '' : 's'}`);
  if (counts.code > 0) parts.push(`${counts.code} code`);
  return parts.join(' · ');
}

function renderItem(item: ArtifactItem, idx: number) {
  if (item.kind === 'plot' && item.plot) {
    return <PlotCard key={`plot:${item.id}`} plot={item.plot} idx={idx} />;
  }
  if (item.kind === 'table' && item.tableRef) {
    return <TableCard key={`table:${item.id}`} tableRef={item.tableRef} idx={idx} />;
  }
  if (item.kind === 'code' && item.python) {
    return <CodeSegmentCard key={`code:${item.id}`} python={item.python} />;
  }
  return null;
}

/** One question's artifacts (mixed plots/tables/code, chronological) rendered
 *  as a collapsible DashWidget: truncated question title (full question in the
 *  tooltip), relative timestamp, count badges, and an optional per-question
 *  "Export as Python" header action. */
export function ArtifactGroupSection({ group, items, defaultOpen, exportUrl }: {
  group: ArtifactGroup;
  /** The (possibly type-filtered) items to render, in group order. */
  items: ArtifactItem[];
  defaultOpen: boolean;
  /** Turn-scoped export endpoint; null hides the button (legacy group / no session). */
  exportUrl: string | null;
}) {
  const badges = countBadges(group.counts);
  const rel = typeof group.ts === 'number' && group.ts > 0
    ? formatDistanceToNow(new Date(group.ts * 1000), { addSuffix: true })
    : null;

  return (
    <DashWidget
      title={group.title}
      titleTip={group.question ?? group.title}
      dotColor={group.qIdx === -1 ? 'bg-line-400' : 'bg-sage-600'}
      color={group.qIdx === -1 ? 'gray' : 'sage'}
      defaultOpen={defaultOpen}
      expandTitle={group.question ?? group.title}
      actions={
        <div className="flex items-center gap-2 shrink-0">
          {rel && <span className="text-[10px] text-ink-300 whitespace-nowrap hidden sm:inline">{rel}</span>}
          {badges && (
            <span className="text-[10px] text-ink-400 bg-cream-100 border border-line-200 rounded-full px-2 py-0.5 whitespace-nowrap">
              {badges}
            </span>
          )}
          {exportUrl && (
            <button
              onClick={() => window.open(exportUrl, '_blank')}
              className="flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium text-ink-400 hover:text-ink-900 hover:bg-cream-100 border border-line-200 transition-colors"
              title="Export as Python"
            >
              <FileDown size={12} /> <span className="hidden md:inline">Export as Python</span>
            </button>
          )}
        </div>
      }
    >
      <div className="space-y-4">
        {items.map((item, idx) => renderItem(item, idx))}
      </div>
    </DashWidget>
  );
}
