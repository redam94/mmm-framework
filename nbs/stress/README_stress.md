# The Stress Series — pressure-testing the MMM on data built to break it

Seven notebooks that fit the framework to synthetic worlds engineered around
real-world difficulties, grade every estimate against known causal ground
truth, and show — with measured numbers, not theory — **what goes wrong, which
diagnostics catch it (usually none), and how to pivot**.

The motivating worry: an MMM that converges cleanly and fits beautifully can
still be badly wrong about attribution, and the checks analysts trust (R-hat,
divergences, PPC, in-sample fit) validate the *computation*, not the *causal
claim*. The series makes that failure surface visible and rehearses the
recovery moves.

## The notebooks

| # | Notebook | Stress | Pivots demonstrated |
|---|----------|--------|---------------------|
| 00 | `stress_00_the_rosy_picture.ipynb` | Doctrine: the positive control recovers (7% err); an equally-green confounded fit is +111% wrong on Search | The estimand discipline; the silent-failure gate; the diagnostic toolkit tour (what each check can and cannot see) |
| 01 | `stress_01_carryover_and_shape.ipynb` | Delayed-Weibull carryover vs geometric-8 (med err 88%); Hill truth vs concave 1−exp (+40% total, all green); 15× spend spikes (coverage 0%) | Parametric Weibull adstock with informed priors + LOO kernel comparison; Hill via `mmm_extensions` components; the 30-second max/median outlier screen + winsorize |
| 02 | `stress_02_time_structure.ipynb` | Trend break + reactive media ramp (TV −63% under linear trend); evolving seasonality + holiday spikes scrambling the channel split (TV +33%, Display −70%); effectiveness drift hiding inside green totals | Piecewise trend (needs a *loosened* changepoint prior — defaults are too stiff), event/step controls (best), holiday dummies (med err → 9%); split-window refits as the drift diagnostic |
| 03 | `stress_03_confounding_and_selection.ipynb` | Demand-chasing spend (+110% on Search, all diagnostics green); a noisy proxy control leaving +35% residual bias; near-perfect collinearity; 25 candidate controls incl. media-tracking decoys | Causal roles as spec; horseshoe selection with the confounder exempt (matches the oracle); honest wide intervals on collinear splits; one ROAS lift test pulls Search +110% → +14% |
| 04 | `stress_04_extension_traps.ipynb` | Base model loses a mediated channel (TV ROAS 0.22 vs true 2.14); NestedMMM recovers TV but over-credits its shared-mediator sibling 2.6×; MultivariateMMM's ψ flips sign with the prior on the same data; a *wrong* mediator fits perfectly | The mediation ladder; counting data moments vs structural parameters; `compute_parameter_learning` + the contraction-≠-importance doctrine; mediator validity is an external causal assumption |
| 05 | `stress_05_the_gauntlet.ipynb` | Everything at once (the Aurora kitchen-sink world, fit demand-blind): ranking fully inverted, Search 8× overstated | The full workflow: EDA pre-flight → naive fit → triage → structural pivots (each measured: the demand control helps, the form pivot *reshuffles* bias) → experiment calibration (Search 5.59 → 0.65, true 0.66) → the symptom→cause→pivot decision table |
| 06 | `stress_06_geography_and_hierarchy.ipynb` | Multi-geo & geo×product panels: the cross-geo adstock-bleed bug (found + fixed by this build); per-geo effectiveness + performance-chasing budgets → national green/accurate while per-geo errors hit +330%, coverage 31%, and TV's regional ROI ranking comes out **fully inverted** (ρ = −1); aggregating geo data to national ≈ doubles interval widths | The impulse-isolation test for panel transforms; grading at the decision level (per-geo truth); the split-per-geo refit (med err 25%→13%, coverage 31%→88%); per-geo fit bands + contribution-path bands |

Read 00 first; 01–04 and 06 stand alone; 05 is the capstone. Companion
pieces: `../demos/mmm_walkthrough.ipynb` (the same workflow arc on the tamer
`realistic` world), `tests/synth/` (the scenario DGPs + scoring harness
behind every world here), `technical-docs/mmm-robustness-report.md`, and the
`math_0x` series for the underlying mathematics.

**Uncertainty-first visuals (added 2026-06-10).** The series now demonstrates
the posterior as pictures, not just verdicts: stress_00 §3.1 introduces three
fit-plot lenses (posterior-predictive bands; per-channel **contribution-over-
time bands** graded against the true weekly paths — the confounded fit's
Search band excludes the truth ~90% of weeks; and the **posterior correlation
of channel totals** as a separability report). stress_01 adds posterior
**kernel bands** (the Weibull pivot's band covers the true kernel at 92–100%
of lags while the naive geometric band is narrow around a shape that can't
bend) and **response-curve bands** (truth-coverage ~58% → ~86% when the family
flips to Hill). stress_03 adds the **posterior seesaw**: on the collinear
world every pairwise contribution correlation is negative (min ≈ −0.5) — the
joint posterior's own statement that the split is unidentified. stress_06
slices all three lenses per geography/cell.

## The data

All national worlds come from `tests/synth/dgp.py`. Each scenario starts from
a clean core world drawn from the model's exact structural family and injects
one (or, for the capstone, many) real-world violation; ground truth is the
model's own counterfactual zero-out estimand, so "true contribution" and
`compute_counterfactual_contributions` are directly comparable. The series
added three scenarios to the matrix: `trend_break`, `seasonality_misspec`,
and `dense_controls` — all three flag as silent failures. The recorded
scorecard (re-recorded 2026-06-10 on current code: **16 worlds, 8 silent
failures**, every one of them with r-hat ≤ 1.02 and zero divergences) lives
at `tests/synth/results/stress_matrix.md`.

Panel worlds (stress_06) come from `tests/synth/dgp_geo.py`: balanced
multi-geo and geo×product panels with the same estimand discipline, except
truth is recorded **per geography/cell** (`true_contribution_by_geo`), plus
`national_scenario()` (the aggregated view) and `geo_scenario(g)` (per-geo
slices for split refits). Three worlds: `geo_clean` (panel positive control —
level-shift heterogeneity only, the model's exact hierarchy),
`geo_heterogeneous` (per-geo effectiveness 0.3–1.8× with performance-chasing
budgets — outside the global-beta hypothesis space), and `geo_product`
(3 geos × 2 products positive control with product-tilted channel mixes).

## Headline lessons (all measured live in the notebooks)

1. **Convergence diagnostics never caught a single attribution failure.**
   Every silent failure in the matrix sampled cleanly (R-hat ≤ 1.02, 0
   divergences). PPC passes on the confounded fit (confounding is not a fit
   problem — and under the old starved trend prior it false-alarmed on the
   *clean* control instead); the confounding RV reads the same on clean and
   confounded fits.
2. **Misfit goes to whichever channel's spend co-moves with it** — and not in
   a predictable direction. The trend-break world *blames* the ramped
   channels; the holiday world scrambles the split while totals stay right.
   The EDA check is spend-vs-structure correlation, before any sampling.
3. **"We controlled for it" is not a fix.** A noisy proxy of the confounder
   leaves large residual bias with truth outside the HDI; only an experiment
   closes the gap, and calibration is surgical (untested channels stay wrong).
4. **Selection priors are for nuisance controls; confounders are exempt.**
   Horseshoe with role-aware exemption matched the oracle control set; the
   same machinery would reintroduce confounding bias if a confounder were
   shrunk. (Also measured: decoy controls tracking raw spend do *not* steal
   credit — the adstock/saturation pipeline separates them; their cost is
   precision.)
5. **Extensions are cure and disease.** NestedMMM recovers what the base
   model structurally cannot — and adds prior-determined splits (the
   shared-mediator allocation, the ψ/ρ split) that the data cannot identify.
   Count moments before trusting a new latent path; `parameter_learning`'s
   contraction tells you the data spoke, the posterior mean tells you what it
   said.
6. **The fix ladder is: estimand → EDA → structure → experiment.** Functional
   form pivots help only when form is the actual problem; identification
   problems (confounding, collinearity, mediation splits) are solved by
   design — roles, dummies, lift tests — not by sampling longer.

## Rebuilding / re-baking

Each notebook is authored by its own script (`build_stress_0X_*.py`,
nbformat pattern). Re-running a build script **overwrites baked outputs**, so
bake after building:

```bash
cd nbs
uv run python builders/build_stress_00_rosy_picture.py
PYTHONPATH=$(dirname $PWD) uv run jupyter nbconvert --to notebook --execute --inplace stress/stress_00_the_rosy_picture.ipynb
```

Fits use NumPyro (300–500 draws × 2 chains, seeded); whole-series bake is
roughly 25 minutes. Every computational cell ends in directional, seeded
asserts encoding the claim it demonstrates, so "executes clean" ⇒ "story
still true". Prose contains no hardcoded numbers — it points at live tables —
so re-bakes are seed-robust.

**When a framework default changes, re-probe every notebook's asserts** —
not just the notebook that motivated the change. The 2026-06-10 trend-prior
loosening halved the confounded-world bias at notebook fidelity; stress_00's
margins were recalibrated at the time but stress_03's (>50% chaser
overstatement, >30% collinear split drift) went stale and failed on the next
rebake. Asserts now carry the magnitude history in comments.

## Framework findings surfaced by the series

The notebooks found real framework issues; **six were fixed (2026-06-10) as a
direct result**, and the notebooks now demonstrate the fixed behavior with the
old behavior reproduced via explicit overrides (or, for the panel bug, in pure
NumPy):

- **FIXED — saturation config was silently ignored**: core `BayesianMMM` now
  honors per-channel `SaturationConfig` (hill / logistic / michaelis_menten /
  tanh / none) in-graph; the default flipped to `logistic()` so default models
  are bit-identical to the old behavior. On the Hill-truth world, the core
  Hill pivot cuts total-media bias ≈+40% → ≈+15%. (stress_01)
- **FIXED — Weibull adstock scale prior now scales with `l_max`**
  (mean `max(2, (l_max − 9)/2)`): the old fixed Gamma(2,1) default produced a
  divergence storm at `l_max=26` (reproduced in-notebook with the old prior
  passed explicitly); the new default samples cleanly. (stress_01)
- **FIXED — trend priors loosened**: linear growth sigma 0.1 → 0.5 (fitted
  trend now reaches ~95% of a true trend's amplitude vs 43% before);
  piecewise `changepoint_prior_scale` 0.05 → 0.5 (the default now tracks a
  structural break: trend corr 0.33 → 0.86 on the trend-break world).
  (stress_02)
- **FIXED — `CombinedMMM` cross-effects**: the duplicate-dims full ψ matrix
  (which crashed convergence checks + `compute_parameter_learning` and ignored
  the configured cross-effect) is replaced by the shared MultivariateMMM
  machinery — one sign-constrained RV per configured direction. Measured: 357
  → 0 divergences, r-hat 2.91 → 1.02. (stress_04)
- **FIXED — `NestedMMM` dead RV**: the never-used `beta_media_to_<mediator>`
  aggregate is no longer created on the ≥2-channel path; zero dead RVs by
  graph-ancestry test. (stress_04)
- A 0/0→NaN underflow in `transforms/adstock_pt.py` kernel normalization was
  guarded (epsilon) as part of this work. (stress_01)
- **FIXED — cross-geo adstock bleed on panel data**: both adstock paths
  (parametric in-graph and the legacy two-point blend) convolved straight
  down the stacked period-major observation vector, so on geo/geo×product
  panels an observation's "lag 1" was *a different geography at the same
  week* — a geography with zero spend in every week still received media
  contributions. Fixed with per-cell convolution
  (`transforms/adstock_pt.py::apply_adstock_panel_pt` + a per-cell legacy
  path); national graphs are bit-identical (the 1-D path is untouched when
  `n_cells == 1`). Regression-tested by impulse isolation in
  `tests/test_panel_adstock.py`; demonstrated live in stress_06 act 0.
  (stress_06)

Still open:

- With the legacy two-point blend adstock (`use_parametric_adstock=False`,
  the default) the *clean positive control* recovers at median ~28% error vs
  7% parametric — the default config underperforms on the model's own
  generative family. (stress_00)
- `validation/validator.py`'s holdout path re-computes saturation in numpy and
  falls back to a logistic curve for non-logistic channels — wrong holdout
  predictions (no crash) for Hill-fit models.
