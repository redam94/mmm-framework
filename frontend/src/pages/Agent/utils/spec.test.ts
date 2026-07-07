import { describe, it, expect } from 'vitest';
import { specWithDefaults, specLeafDiff, flattenLeaves } from './spec';

/** A realistic agent-built spec with keys the editor does NOT own. */
const RICH_SPEC = {
  kpi: 'Sales',
  media_channels: [
    {
      name: 'Display',
      adstock: { type: 'geometric', l_max: 6 },
      saturation: { type: 'hill' },
      measurement_unit: 'impressions',
      cpm: 5.5,
    },
    { name: 'TV' },
  ],
  control_variables: [{ name: 'price', role: 'confounder' }],
  priors: { media: { TV: { roi: { median: 1.2, sigma: 0.6 } } } },
  dataset: { path: 'data.csv' },
  garden_ref: { org: 'acme', name: 'custom_mmm' },
  likelihood: { family: 'student_t' },
  estimands: ['contribution_roi'],
  media_prior_mode: 'roi',
};

describe('specWithDefaults round-trip', () => {
  it('preserves top-level keys the editor does not own', () => {
    const d = specWithDefaults(RICH_SPEC) as Record<string, unknown>;
    expect(d.priors).toEqual(RICH_SPEC.priors);
    expect(d.dataset).toEqual(RICH_SPEC.dataset);
    expect(d.garden_ref).toEqual(RICH_SPEC.garden_ref);
    expect(d.likelihood).toEqual(RICH_SPEC.likelihood);
    expect(d.estimands).toEqual(RICH_SPEC.estimands);
    expect(d.media_prior_mode).toBe('roi');
  });

  it('preserves per-channel measurement descriptor keys', () => {
    const d = specWithDefaults(RICH_SPEC);
    const display = d.media_channels.find((c) => c.name === 'Display')!;
    expect(display.measurement_unit).toBe('impressions');
    expect(display.cpm).toBe(5.5);
    // and still materializes editor defaults
    expect(display.adstock).toEqual({ type: 'geometric', l_max: 6 });
    const tv = d.media_channels.find((c) => c.name === 'TV')!;
    expect(tv.measurement_unit).toBeUndefined();
    expect(tv.adstock).toEqual({ type: 'geometric', l_max: 8 });
  });

  it('preserves inference.method and metrics_draws through the round-trip', () => {
    // Regression: the editor rebuilds `inference` from its own field list, so
    // an agent-set approximate fit method (advi/pathfinder) used to be wiped
    // by any unrelated Apply.
    const spec = {
      ...RICH_SPEC,
      inference: { draws: 500, method: 'pathfinder', metrics_draws: 0 },
    };
    const d = specWithDefaults(spec);
    expect(d.inference.method).toBe('pathfinder');
    expect((d.inference as Record<string, unknown>).metrics_draws).toBe(0);
    expect(d.inference.draws).toBe(500);
    expect(d.inference.chains).toBe(4); // defaults still materialize
  });

  it('defaults inference.method to nuts when absent', () => {
    expect(specWithDefaults(RICH_SPEC).inference.method).toBe('nuts');
  });

  it('an untouched round-trip produces no leaf diff (no phantom locks)', () => {
    const baseline = specWithDefaults(RICH_SPEC);
    const again = specWithDefaults(RICH_SPEC);
    expect(specLeafDiff(baseline, again)).toEqual([]);
  });

  it('a measurement edit diffs only the touched leaves', () => {
    const baseline = specWithDefaults(RICH_SPEC);
    const edited = specWithDefaults(RICH_SPEC);
    const display = edited.media_channels.find((c) => c.name === 'Display')! as Record<
      string,
      unknown
    >;
    display.cpm = 7.25;
    const diff = specLeafDiff(baseline, edited);
    expect(diff).toEqual(['media_channels.Display.cpm']);
  });

  it('flattenLeaves keys named lists by item name', () => {
    const leaves = flattenLeaves(specWithDefaults(RICH_SPEC));
    expect(leaves['media_channels.Display.measurement_unit']).toBe('impressions');
    expect(leaves['control_variables.price.role']).toBe('confounder');
  });
});
