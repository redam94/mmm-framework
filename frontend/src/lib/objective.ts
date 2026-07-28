// Which allocator objective a portfolio point's `expected_uplift` is denominated in.
//
// `expected_uplift` means "left on the table versus the optimum" — a quantity
// defined BY the objective and the mode. Under the default it is KPI units;
// under a profit objective it is dollars of forgone profit. One line through
// both, or a period-over-period delta across the boundary, is arithmetic on two
// different units wearing one label (#221).
//
// Absence reads as the historical default, matching the server: every run
// recorded before run-metrics schema v3 was produced by `compute_run_metrics`,
// which has always called `optimize_budget` with no objective arguments.

export interface ObjectiveBearing {
  objective?: string | null;
  objective_label?: string | null;
  mode?: string | null;
  value_source?: string | null;
}

export function objectiveKey(p: ObjectiveBearing | undefined | null): string {
  if (!p) return '';
  return [p.objective ?? 'mean', p.mode ?? 'fixed', p.value_source ?? ''].join('|');
}

export function objectiveLabel(p: ObjectiveBearing | undefined | null): string {
  if (!p) return '';
  const label = p.objective_label ?? p.objective ?? 'expected KPI (posterior mean)';
  const mode = p.mode ?? 'fixed';
  const src = p.value_source ? `, valued from ${p.value_source}` : '';
  return `${label} (${mode} budget${src})`;
}

/** True when two points measure the same quantity and may be differenced. */
export function sameObjective(
  a: ObjectiveBearing | undefined | null,
  b: ObjectiveBearing | undefined | null,
): boolean {
  if (!a || !b) return true; // nothing to contradict
  return objectiveKey(a) === objectiveKey(b);
}

/** Group a series by objective, preserving order. One group → plot as usual;
 *  more than one → plot one trace per group rather than one misleading line. */
export function groupByObjective<T extends ObjectiveBearing>(
  points: T[],
): { key: string; label: string; points: T[] }[] {
  const out: { key: string; label: string; points: T[] }[] = [];
  for (const p of points) {
    const key = objectiveKey(p);
    const existing = out.find((g) => g.key === key);
    if (existing) existing.points.push(p);
    else out.push({ key, label: objectiveLabel(p), points: [p] });
  }
  return out;
}
