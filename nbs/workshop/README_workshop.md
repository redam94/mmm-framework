# The Workshop Series — Bayesian MMM from zero

Six interactive notebooks that take an analyst with **no Bayesian background**
from "what is a prior?" to running a real `mmm_framework` model and turning
its posterior into budget decisions with honest uncertainty. Heavy on
visuals (plotly hover/zoom everywhere), every concept introduced through a
marketing example, every new term **bolded and glossed in plain English**,
and each notebook ends with a glossary of the vocabulary it introduced.

The arc: notebooks 00–02 build the Bayesian toolkit on small, transparent
examples (no framework, no MMM); 03–05 apply it to a real `BayesianMMM` fit
on a synthetic world whose causal truth is *known*, so every posterior claim
gets graded against the answer key.

## The notebooks

| # | Notebook | Teaches | The payoff moment |
|---|----------|---------|-------------------|
| 00 | `workshop_00_thinking_in_distributions.ipynb` | Probability as belief; Bayes' rule on a grid; priors sharpening into posteriors; credible intervals/HDI; Monte Carlo | "Is the new creative better?" answered as **P(B > A)** and a revenue-uplift *distribution* — your first derived quantity (pure numpy, runs in seconds) |
| 01 | `workshop_01_priors.ipynb` | Flat vs weakly-informative vs informative; the MMM prior zoo (Beta → adstock retention, Gamma/Exponential → saturation, HalfNormal → effects); prior predictive checks; the framework's real `PriorConfig`/builder API | A prior on a *parameter* is really a prior on *behavior*: slider-driven fans of implied adstock-decay and response curves, and a prior-predictive check that catches silly priors before any fitting |
| 02 | `workshop_02_sampling.ipynb` | Why grids die (curse of dimensionality); Monte Carlo; a 20-line Metropolis sampler, animated; chains, R-hat, ESS, autocorrelation; NUTS and divergences; the trace-plot reading clinic | Watch the walker explore the posterior live, break it with a step-size slider, then see R-hat and ESS catch every failure you just caused |
| 03 | `workshop_03_first_mmm.ipynb` | MMM terminology (adstock, saturation, contribution, ROAS, baseline, MFF); exploring panel data; `ModelConfig` choice-by-choice; the diagnostics gate; grading vs known truth | First real fit recovers the known answer at ~7% median error with all four channels' truth inside the 90% HDI — and you check it yourself |
| 04 | `workshop_04_reading_the_posterior.ipynb` | The posterior as a table of internally-consistent worlds; HDI vs equal-tailed; forest plots; joint correlations (why point estimates can't be mixed); prior→posterior contraction (`compute_parameter_learning`); PPC; decomposition with bands | β and saturation λ trade off at r ≈ −0.8 inside the posterior — the picture that proves derived quantities must be computed **per draw** |
| 05 | `workshop_05_from_draws_to_decisions.ipynb` | The doctrine: **compute per draw, summarize last** (Jensen demo); ROAS as a distribution; P(ROAS > t); marginal vs average ROAS; response curves with bands; what-if scenarios; reallocation under uncertainty | The average-ROAS ranking and the *next-dollar* ranking genuinely disagree, and the naive reallocation is a coin flip tilted to a loss while the marginal-driven one wins with P = 0.98 |

Do them in order — each assumes the previous. 00–02 run in seconds to a few
minutes (no MCMC beyond one tiny NUTS demo); 03–05 each run one fast fit
(numpyro, 500×2, ~10–15 s) so a full live workshop bakes end-to-end in well
under 15 minutes.

## Interactivity

Two layers, deliberately:

- **Plotly charts** (hover, zoom, pan, animation frames) are baked into the
  notebooks and work without a kernel — including in a static render.
- **🎛️ "Live exploration (run me!)" cells** use `ipywidgets` sliders
  (prior-strength, Metropolis step size, HDI probability, ROAS breakeven
  threshold, what-if budget changes…). These need a running kernel — that's
  the workshop part. Every slider cell is paired with a static panel showing
  representative settings, so the baked notebook still teaches on its own.

Plotly uses the `notebook_connected` renderer (loads plotly.js from a CDN);
if running fully offline, set `pio.renderers.default = "notebook"` at the top
of each notebook instead.

## The data

Notebooks 03–05 share one anchor world: `dgp.build("clean")` from
`tests/synth/dgp.py` — a 4-channel synthetic market whose true per-channel
contributions and ROAS are known exactly (the same positive-control world the
stress series uses). That's the honesty device of the whole series: learners
don't have to take the model's word for anything, because every estimate is
graded against the sealed answer key, and the framework earns trust by
passing.

## Rebuilding / re-baking

Each notebook is authored by its own build script (nbformat pattern shared
with the aurora/math/stress series). Re-running a build script **overwrites
baked outputs**, so execute after building:

```bash
cd nbs/
uv run python builders/build_workshop_03_first_mmm.py
PYTHONPATH=.. uv run jupyter nbconvert --to notebook --execute --inplace \
    workshop_03_first_mmm.ipynb --ExecutePreprocessor.timeout=2400
```

Every code cell ends with `assert`s encoding the claim it just made or
plotted (MCMC asserts are directional and seeded, never tight), so a clean
bake means the prose's claims were re-verified, not just re-rendered.

## Where to next

- `../aurora/00_overview.ipynb` … `../aurora/05_unified_workflow.ipynb` — the **Aurora** story:
  the full framework (causality, extensions, calibration, reporting) on one
  realistic brand narrative.
- `math_00`–`math_06` — the mathematics behind everything here (transforms,
  likelihood, calibration, extensions), LaTeX-first.
- `stress_00`–`stress_06` — the cold shower: worlds built to break the model,
  where green diagnostics and wrong answers coexist, and the pivots that
  recover the truth. Read `../stress/stress_00_the_rosy_picture.ipynb` right after this
  series — it is the caveat workshop_02 §9 and workshop_05 §8 point at.
- `../demos/mmm_walkthrough.ipynb` — the practitioner's v1→v3 modeling loop on the
  `realistic` world.
