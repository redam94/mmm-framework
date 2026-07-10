import { useMemo, useState } from 'react';
import { Activity } from 'lucide-react';
import { API_BASE } from '../../constants';
import { EmptyTabState } from '../common/EmptyTabState';
import { ArtifactGroupSection } from './ArtifactGroupSection';
import type { ArtifactGroup } from '../../utils/artifactGroups';

type TypeFilter = 'all' | 'plot' | 'table' | 'code';

const FILTERS: { id: TypeFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'plot', label: 'Plots' },
  { id: 'table', label: 'Tables' },
  { id: 'code', label: 'Code' },
];

/** The Results-tab "Analysis timeline": artifacts grouped by the question they
 *  answer, newest question first, with an "Earlier work" bucket for refs that
 *  predate call_id provenance. A type filter narrows to plots/tables/code;
 *  groups emptied by the filter are hidden. */
export function GroupedArtifacts({ groups, threadId }: {
  groups: ArtifactGroup[];
  threadId: string | null;
}) {
  const [filter, setFilter] = useState<TypeFilter>('all');

  const visible = useMemo(
    () => groups
      .map(g => ({
        group: g,
        items: filter === 'all' ? g.items : g.items.filter(i => i.kind === filter),
      }))
      .filter(v => v.items.length > 0),
    [groups, filter],
  );

  const total = groups.reduce((n, g) => n + g.items.length, 0);
  if (total === 0) {
    return (
      <EmptyTabState
        icon={<Activity size={28} />}
        title="No analysis artifacts yet"
        hint="Ask the agent to plot, tabulate, or run Python — plots, tables, and code appear here grouped by the question they answer."
      />
    );
  }

  // Only the two newest groups start expanded (DashWidget captures defaultOpen
  // at mount); computed against the FULL group list so filtering doesn't
  // promote older groups.
  const openKeys = new Set(groups.slice(0, 2).map(g => g.key));

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-semibold uppercase tracking-wider text-ink-300 mr-1">
          Analysis timeline
        </span>
        {FILTERS.map(f => {
          const active = filter === f.id;
          return (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              className={`px-2.5 py-1 rounded-full text-[11px] font-medium border transition-colors ${
                active
                  ? 'bg-sage-700 border-sage-700 text-white'
                  : 'bg-white border-line-200 text-ink-400 hover:text-ink-900 hover:bg-cream-100'
              }`}
            >
              {f.label}
            </button>
          );
        })}
      </div>
      {visible.length === 0 ? (
        <p className="text-sm text-ink-300 px-1 py-4 text-center">
          Nothing matches this filter — try another artifact type.
        </p>
      ) : (
        visible.map(v => (
          <ArtifactGroupSection
            key={v.group.key}
            group={v.group}
            items={v.items}
            defaultOpen={openKeys.has(v.group.key)}
            exportUrl={
              threadId && v.group.qIdx >= 0
                ? `${API_BASE}/sessions/${threadId}/export?scope=turn:${v.group.qIdx + 1}`
                : null
            }
          />
        ))
      )}
    </div>
  );
}
