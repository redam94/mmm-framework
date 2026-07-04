// ROI-scale media priors: closed-form implied-ROI math, spec hydration, and
// the apply-payload serializer (exactly one of roi/coefficient per channel —
// the backend treats an explicit coefficient prior as overriding the ROI one).
import { describe, expect, it } from 'vitest';
import {
  initPriors,
  normalCdf,
  PRIOR_DEFAULTS,
  roiPriorStats,
  serializeMediaPriors,
} from './priors';
import type { MediaPriors } from './priors';

describe('normalCdf', () => {
  it('matches known values', () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 6);
    expect(normalCdf(1.6448536)).toBeCloseTo(0.95, 4);
    expect(normalCdf(-1.6448536)).toBeCloseTo(0.05, 4);
  });
});

describe('roiPriorStats', () => {
  it('default LogNormal(0,1) implies the documented [0.19x, 5.2x] / 50%', () => {
    const s = roiPriorStats(1.0, 1.0);
    expect(s.lower).toBeCloseTo(0.193, 2);
    expect(s.upper).toBeCloseTo(5.18, 1);
    expect(s.pAbove1).toBeCloseTo(0.5, 5);
  });

  it('an optimistic tight prior raises P(ROI>1) and narrows the band', () => {
    const s = roiPriorStats(2.0, 0.3);
    expect(s.lower).toBeGreaterThan(1.0);
    expect(s.upper).toBeLessThan(4.0);
    expect(s.pAbove1).toBeGreaterThan(0.95);
  });
});

const SPEC = {
  media_channels: [{ name: 'TV' }, { name: 'Digital' }],
  control_variables: [],
};

describe('initPriors — media prior mode + roi hydration', () => {
  it('defaults every channel to the ROI scale with the break-even prior', () => {
    const p = initPriors(SPEC);
    expect(p.media.TV.mode).toBe('roi');
    expect(p.media.TV.roi).toEqual(PRIOR_DEFAULTS.media_roi);
  });

  it('an explicit coefficient prior in the spec selects coefficient mode', () => {
    const p = initPriors({
      ...SPEC,
      priors: { media: { TV: { coefficient: { distribution: 'half_normal', params: { sigma: 0.5 } } } } },
    });
    expect(p.media.TV.mode).toBe('coefficient');
    expect(p.media.TV.coefficient.params.sigma).toBe(0.5);
    expect(p.media.Digital.mode).toBe('roi');
  });

  it('follows a coefficient-mode opt-out for untouched channels', () => {
    const p = initPriors({ ...SPEC, media_prior_mode: 'coefficient' });
    expect(p.media.TV.mode).toBe('coefficient');
    // …but a channel with an roi entry stays on the ROI scale.
    const p2 = initPriors({
      ...SPEC,
      media_prior_mode: 'coefficient',
      priors: { media: { TV: { roi: { median: 1.5, sigma: 0.4 } } } },
    });
    expect(p2.media.TV.mode).toBe('roi');
  });

  it('hydrates roi from {median, sigma} and converts a log-scale mu', () => {
    const p = initPriors({
      ...SPEC,
      priors: { media: { TV: { roi: { median: 1.5, sigma: 0.4 } }, Digital: { roi: { mu: Math.log(2), sigma: 0.6 } } } },
    });
    expect(p.media.TV.mode).toBe('roi');
    expect(p.media.TV.roi).toEqual({ median: 1.5, sigma: 0.4 });
    expect(p.media.Digital.roi.median).toBeCloseTo(2.0, 10);
    expect(p.media.Digital.roi.sigma).toBe(0.6);
  });
});

describe('serializeMediaPriors', () => {
  const base: Omit<MediaPriors, 'mode'> = {
    roi: { median: 1.2, sigma: 0.6 },
    coefficient: { distribution: 'half_normal', params: { sigma: 2.0 } },
    adstock_alpha: { distribution: 'beta', params: { alpha: 1, beta: 3 } },
    saturation_kappa: { distribution: 'beta', params: { alpha: 2, beta: 2 } },
    saturation_slope: { distribution: 'half_normal', params: { sigma: 1.5 } },
  };

  it('roi mode emits roi and omits coefficient (and vice versa)', () => {
    const out = serializeMediaPriors({
      TV: { ...base, mode: 'roi' },
      Digital: { ...base, mode: 'coefficient' },
    });
    expect(out.TV.roi).toEqual({ median: 1.2, sigma: 0.6 });
    expect(out.TV.coefficient).toBeUndefined();
    expect(out.TV.mode).toBeUndefined();
    expect(out.Digital.coefficient).toEqual(base.coefficient);
    expect(out.Digital.roi).toBeUndefined();
    // Shape/carryover priors ride along in both modes.
    expect(out.TV.adstock_alpha).toEqual(base.adstock_alpha);
    expect(out.Digital.saturation_slope).toEqual(base.saturation_slope);
  });
});
