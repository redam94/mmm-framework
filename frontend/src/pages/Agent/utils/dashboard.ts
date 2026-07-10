import type { DashboardData, PlotRef, TableRef } from '../types';

/**
 * Merge helpers for streamed `dashboard_data` updates — the frontend mirror of
 * the backend state reducer (agents/state.py::_merge_dashboard/_union_refs).
 *
 * The ref-list keys (`plots`/`tables`) accumulate content-addressed refs across
 * the whole session, but a streamed `dashboard_update` payload may carry only a
 * SUBSET of them: a sub-agent (delegate_to_expert) folds back a list built from
 * an empty seed, and concurrent tools in one step each append to the same
 * pre-step list. A plain `{...prev, ...update}` spread replaces the array
 * wholesale, so previously-rendered results vanished from the Results tab until
 * a reload re-hydrated the (correctly unioned) checkpoint. Union by ref id
 * instead; every other key keeps last-write-wins semantics.
 */

/** Union two ref lists, deduped by ref id — first occurrence wins, so existing
 * refs keep their position and genuinely new ones append. Legacy inline
 * figures without an `id` are kept as-is. */
export function unionRefs<T extends { id?: string }>(
  a: T[] | undefined | null,
  b: T[] | undefined | null,
): T[] {
  const merged: T[] = [];
  const seen = new Set<string>();
  for (const item of [...(a ?? []), ...(b ?? [])]) {
    const rid = item && typeof item === 'object' ? item.id : undefined;
    if (rid != null) {
      if (seen.has(rid)) continue;
      seen.add(rid);
    }
    merged.push(item);
  }
  return merged;
}

/** Merge a streamed dashboard_data update into the accumulated dashboard.
 * Incoming keys win, EXCEPT `plots`/`tables` which union by id. An explicit
 * `null` clears a list (the backend's documented escape hatch). */
export function mergeDashboardData(
  prev: DashboardData,
  update: Record<string, unknown>,
): DashboardData {
  const merged = { ...prev, ...update } as DashboardData;
  if ('plots' in update) {
    merged.plots =
      update.plots == null ? undefined : unionRefs(prev.plots, update.plots as PlotRef[]);
  }
  if ('tables' in update) {
    merged.tables =
      update.tables == null ? undefined : unionRefs(prev.tables, update.tables as TableRef[]);
  }
  return merged;
}
