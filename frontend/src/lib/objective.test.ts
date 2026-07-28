import { describe, it, expect } from 'vitest';
import { groupByObjective, objectiveKey, objectiveLabel, sameObjective } from './objective';

// `expected_uplift` is "left on the table versus the optimum" — a quantity
// defined BY the allocator's objective and mode. Charting a profit-objective run
// and a KPI-uplift run as one line, or differencing them, is arithmetic on two
// different units wearing one label (#221).

const meanFixed = { objective: 'mean', mode: 'fixed', value_source: null };
const cvarFree = {
  objective: 'cvar5',
  objective_label: 'downside KPI (worst 5%)',
  mode: 'free',
  value_source: 'preference',
};

describe('objective provenance', () => {
  it('reads absence as the historical default, not as unknown', () => {
    // Runs predating run-metrics schema v3 carry no objective, but
    // compute_run_metrics has always optimized mean/fixed — so they ARE
    // comparable, and treating absence as unknown would refuse everything.
    expect(objectiveKey({})).toBe(objectiveKey(meanFixed));
    expect(sameObjective({}, meanFixed)).toBe(true);
  });

  it('separates runs optimized differently', () => {
    expect(sameObjective(meanFixed, cvarFree)).toBe(false);
  });

  it('treats a mode change alone as incomparable', () => {
    // Fixed-budget reallocation and fund-to-breakeven leave different amounts on
    // the table by construction, even under the same risk objective.
    expect(sameObjective(meanFixed, { objective: 'mean', mode: 'free' })).toBe(false);
  });

  it('treats a valuation-source change as incomparable', () => {
    expect(
      sameObjective(
        { objective: 'mean', mode: 'free', value_source: 'param' },
        { objective: 'mean', mode: 'free', value_source: 'preference' },
      ),
    ).toBe(false);
  });

  it('names the objective in a sentence a chart can render', () => {
    expect(objectiveLabel(cvarFree)).toBe(
      'downside KPI (worst 5%) (free budget, valued from preference)',
    );
    expect(objectiveLabel(meanFixed)).toContain('fixed budget');
  });

  it('groups a series so each objective is its own trace, preserving order', () => {
    const groups = groupByObjective([meanFixed, meanFixed, cvarFree, meanFixed]);
    expect(groups.map((g) => g.points.length)).toEqual([3, 1]);
    expect(groups).toHaveLength(2);
  });

  it('yields a single group for a homogeneous series', () => {
    expect(groupByObjective([meanFixed, {}, meanFixed])).toHaveLength(1);
  });
});
