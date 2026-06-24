export const STAGES: { t: string; name: string; detail: string }[] = [
  { t: 'T₀', name: 'Fit baseline', detail: 'Fit the MMM on history; document posterior widths.' },
  { t: 'T₁', name: 'Prioritize', detail: 'Score channels by EIG × EVOI; pick the experiment portfolio.' },
  { t: 'T₂', name: 'Run experiments', detail: 'Execute pre-registered geo-lift / pulse tests.' },
  { t: 'T₃', name: 'Calibrate', detail: 'Fold readouts into the next fit as likelihoods.' },
  { t: 'T₄', name: 'Allocate', detail: 'Budget from the calibrated posterior, with confidence tiers.' },
  { t: 'T₅', name: 'Re-evaluate', detail: 'Recompute priorities with tightened posteriors; loop.' },
];
