# The Experimental Measurement Lifecycle

A seven-notebook walkthrough of how a marketing-mix model **earns the right to
allocate a budget** — by running the framework's own adaptive measurement loop
end-to-end on one synthetic brand.

> **The motivating worry.** A marketing-mix model fit to *observational* history
> can only report what **correlated** with sales, and marketers spend into demand:
> budgets rise exactly when the brand is already hot. So a naive read credits
> *spend* for sales that **demand** would have delivered anyway. A fitted MMM is a
> **hypothesis, not an answer.** The only way to settle it is to *create variation
> on purpose* — run an experiment — and fold what it measures back into the model.
> But you can't test everything, so the framework runs a **loop**: decide what's
> worth testing, design it so it's powered and affordable, calibrate the model to
> the result, re-allocate, and know *when the answer goes stale.*

This series is the fitted-MMM counterpart to the model-free
[`continuous_learning_story`](../continuous_learning/continuous_learning_story.ipynb) notebook: same loop,
but here it sits on top of a real `BayesianMMM` and the framework's `planning/`
and `calibration/` stacks.

## The loop, notebook by notebook

The brand is **Northwind Outfitters** (national outdoor apparel; channels
TV / Search / Social / Display; KPI weekly Sales). Every notebook shares one
brand, one palette, and one cached fit.

| # | Stage | Notebook | The question it answers | Payoff moment |
|---|-------|----------|-------------------------|---------------|
| 0 | — | [`lifecycle_00_overview`](lifecycle_00_overview.ipynb) | Why is a fit not enough, and what is the loop? | The confounded dashboard slope (~3×) vs the true causal slope |
| 1 | **T0 · Fit** | [`lifecycle_01_fit_baseline`](lifecycle_01_fit_baseline.ipynb) | What does the observational model believe — and how sure is it? | Every channel's ROI is **under-credited** and you can't see which from inside the model |
| 2 | **T1 · Prioritize** | [`lifecycle_02_prioritize`](lifecycle_02_prioritize.ipynb) | Of everything we're unsure about, what's worth *testing*? | The EIG × EVOI 2×2 — **Display is `test_now`, TV is only `monitor`** |
| 3 | **T2 · Design** | [`lifecycle_03_design`](lifecycle_03_design.ipynb) | How do we run it so it's *powered* and *affordable*? | A geo holdout detects an **~8× smaller** effect than national flighting can |
| 4 | **T3 · Calibrate** | [`lifecycle_04_calibrate`](lifecycle_04_calibrate.ipynb) | How does the readout update the model? | Display's ROI **slides onto the truth** (≈83% of the gap closed); untested channels barely move |
| 5 | **T4 · Allocate** | [`lifecycle_05_allocate`](lifecycle_05_allocate.ipynb) | What's the budget now — and how confident are we? | A reallocation with a **confidence band**; marginal ≠ average ROAS |
| 6 | **T5 · Re-evaluate** | [`lifecycle_06_reevaluate`](lifecycle_06_reevaluate.ipynb) | When does the answer go stale and trigger a re-test? | Information decay re-points the loop — **a stale channel flags for re-test** and we're back at T1 |

**The load-bearing idea:** you don't test what you're most *uncertain* about — you
test what most changes the *decision*. That distinction (information vs. value of
information) is T1, and it drives everything after it.

## The data (why it's honest)

The world is the `synth` **`"clean"`** scenario
(`mmm_framework.synth.mff.generate_mff`): the model's *exact* generative family
(geometric adstock, logistic saturation, additive Gaussian, exogenous well-pulsed
spend), so the true per-channel ROAS is **known** — a sealed answer key the analyst
never sees and the notebooks use only to *grade* the loop. The T2 geo-holdout demo
uses the same brand at DMA grain (`generate_mff("clean", geographies=[…])`).

Because recovery is honest, the series can show its punchlines truthfully: the
baseline fit systematically **under-credits** every channel (mean ROI ≈ 0.66 vs a
true ≈ 0.76 in the reference bake), and one experiment pulls the tested channel
back onto its truth.

## Shared machinery

Everything lives in **`nbs/builders/lifecycle_common.py`** (a plain module, imported as `L`,
in the same spirit as `aurora.py` / `charts_src.py`):

- the brand, channel list, palette, and `L.style(...)` plotting helper;
- `L.fit_baseline()` — the T0 fit, done **once** and cached to disk
  (`MMMSerializer` + a cloudpickled panel), so notebooks reload it in seconds
  instead of re-sampling;
- `L.fit_calibrated()` — the T3 refit (baseline + an in-graph ROAS likelihood);
- `L.national_csv()` / `L.geo_csv()` — MFF-long CSVs for the `planning.design` APIs;
- `L.FOCUS_CHANNEL` (`"Display"`, the T1 winner) threaded through T2–T6.

Delete the cache dir (`$MMM_LIFECYCLE_CACHE`, default a temp dir) to force a clean
re-fit. The fit is a deliberately fast **2-chain × 300-draw numpyro** posterior
(~10 s) — enough draws for the response-curve / EIG / EVOI machinery, but you'll
see benign R-hat advisories; bump the draws in `lifecycle_common.py` for a
convergence-clean headline run.

## Rebuilding / re-baking

Each notebook is generated by a `build_lifecycle_*.py` script (the source of truth;
it never carries outputs — **bake after building**). From `nbs/`:

```bash
# build one notebook, then execute it in place
uv run --with nbformat python builders/build_lifecycle_01_fit_baseline.py
TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
    jupyter nbconvert --to notebook --execute --inplace \
    lifecycle_01_fit_baseline.ipynb \
    --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3
```

Rebuild the whole series (the first notebook to run cold-fits and caches; the rest
reload it):

```bash
for n in 00_overview 01_fit_baseline 02_prioritize 03_design \
         04_calibrate 05_allocate 06_reevaluate; do
  uv run --with nbformat python build_lifecycle_$n.py
  TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
      jupyter nbconvert --to notebook --execute --inplace \
      lifecycle_$n.ipynb --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3
done
```

House conventions (shared with the `stress_*` / `workshop_*` series): **no number is
hardcoded in prose** (markdown points at the live tables/plots), and **every code
cell ends in a directional assert** encoding the claim it just made — so "it ran"
means "the story still holds."

## Headline lessons (measured, in the reference bake)

1. **A fit is a hypothesis.** The observational model under-credits every channel
   and can't tell you which — a credible interval is only honest about the
   uncertainty the *model* knows about, not the confounding in the history.
2. **Test for value, not curiosity.** Display and TV are both high-stakes, but the
   fit is already fairly precise on TV (→ `monitor`) while Display is uncertain
   *and* swings the budget (→ `test_now`). EIG ranks learning; **EVOI** ranks
   learning that changes the decision.
3. **Geometry beats brute force.** A budget-neutral national flighting test could
   only detect an implausibly huge effect (MDE ≈ 2.7× ROAS); a **geo holdout** on
   the same brand's DMA data detects ~**8× smaller** (MDE ≈ 0.35×).
4. **Naive significance lies under autocorrelation.** The analytic A/A false-positive
   rate runs ≈ **10× hot** (~48% vs a nominal 5%); a block-calibrated critical value
   restores the true size. Pre-register the estimand, power, and stop rule *before*
   the data.
5. **Calibration is surgical.** One experiment pulls the *tested* channel's ROI onto
   the truth (≈83% of the gap) and leaves the *untested* channels essentially where
   they were.
6. **Allocate under uncertainty.** The reallocation comes with a confidence band and
   a P(uplift > 0); average and marginal ROAS genuinely disagree (diminishing
   returns), and the *calibrated* plan is the tighter, defensible one.
7. **The loop never ends — it re-prioritizes.** Evidence decays at a channel-specific
   half-life; when the decayed value-of-a-test clears the bar (and a freshness floor),
   the loop re-points to whatever is now most worth learning.

## Where to next

- [`continuous_learning_story`](../continuous_learning/continuous_learning_story.ipynb) — the same loop run
  **model-free** (a geo response-surface bandit, no MMM).
- [`workshop_05_from_draws_to_decisions`](../workshop/workshop_05_from_draws_to_decisions.ipynb) —
  the marginal-vs-average-ROAS and reallocation-under-uncertainty pedagogy in depth.
- [`math_05_calibration`](../math/math_05_calibration.ipynb) — the calibration mathematics
  (prior and likelihood routes) this series applies.
- `technical-docs/experiment-economics.md` — the opportunity-cost and A/A·A/B
  simulation spec behind T2.
