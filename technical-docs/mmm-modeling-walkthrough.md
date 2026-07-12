# A Modeling Walkthrough: Many Noisy Factors, Done the Right Way

This is a worked example of fitting the MMM to **realistic, messy data** — many
candidate drivers, confounded spend, weak/collinear channels — following the
framework's documented [9-step scientific workflow](../docs/scientific-workflow-demo.html)
and treating modeling as what it actually is: **iterative**. You do not get the
answer from the first fit. Diagnostics, causal reasoning, and experiments drive
successive revisions.

Everything below is **executed, not narrated**: the data is
`src/mmm_framework/synth/dgp.make_realistic`, the iterations are
`tests/synth/realistic_walkthrough.py`, and every number comes from
`tests/synth/results/walkthrough.json` (156 weeks, numpyro, 700 draws × 4
chains). Because it is synthetic we know the **true** causal contribution of each
channel and can grade every version.

```bash
uv run python -m tests.synth.realistic_walkthrough          # reproduce (publication run)
uv run python -m tests.synth.realistic_walkthrough --quick  # fast arc check
```

> **Scope.** The adstock and saturation *forms* here are inside the model's
> family on purpose — this walkthrough is about the workflow (causal structure,
> identification, calibration), not functional-form misspecification. Those
> failure modes (and what they do to convergence and ROI) are catalogued in the
> companion [robustness report](mmm-robustness-report.md).

## The dataset: 20 candidate factors, low media SNR

| group | factors | role |
|---|---|---|
| **Media (7)** | TV, Search, Social, Display, Video, Radio, Print | the channels we want ROI for |
| **Confounders (2)** | `category_demand` (proxy for hidden demand), `distribution` | drive **both** spend and sales |
| **Precision (4)** | `price`, `competitor_promo`, `weather`, `holiday` | drive sales only |
| **Irrelevant (6)** | `noise_1` … `noise_6` | true effect **0** (the haystack) |
| **Mediator (1)** | `brand_awareness` | TV/Video build it; it drives sales — **post-treatment** |

The traps that make this realistic (none are functional-form errors):

* **Confounding.** A hidden demand signal drives sales *and* is chased by the TV
  and Search budgets (spend–demand correlation ≈ 0.5). Ignore it and those
  channels look far better than they are.
* **A tempting bad control.** `brand_awareness` is on the TV/Video → sales
  causal path. Adding it as a control *blocks* the path and under-counts those
  channels (post-treatment bias).
* **Unidentifiable media.** Radio and Print are always bought together (spend
  correlation ≈ 0.996), so observational data pins their *combined* effect but
  not the split.
* **Low signal-to-noise.** Observation noise is large relative to media, so weak
  effects cannot be teased out without help.

---

## Step 1 — Define the question

> *Given two years of weekly data and 20 candidate drivers, what is each media
> channel's incremental contribution (with honest uncertainty), so we can
> defend a budget reallocation?*

The bar is **defensibility**: a number we would stake a budget on, with an
interval that tells the truth about what we do and don't know.

## Step 2 — Tell the story of your data (the DAG)

Before any code, write down how the data was generated. This is the step that
turns "20 columns" into "which variables to include and how."

```
              category_demand (proxy of hidden DEMAND)
                 │  (confounder)        │
        chase ▼  │                      ▼
        TV, Search ─────────────────────────────►  SALES
        TV, Video ──► brand_awareness ───────────►  SALES   (mediated; awareness is POST-treatment)
        Social, Display, Radio, Print ───────────►  SALES   (direct)
        price, competitor, weather, holiday ─────►  SALES   (precision controls)
        distribution ──(≈spend)──────────────────►  SALES   (confounder)
        noise_1..6 ──╴(no edge)╶──────────────────  SALES   (irrelevant)
        Radio ≈ Print  (same flighting calendar → collinear)
```

The DAG **classifies** every candidate factor and tells you how to treat it:

| treatment | factors | why |
|---|---|---|
| **must include, do not shrink** | `category_demand`, `distribution` | confounders — omitting or shrinking them biases media |
| include (regularized) | `price`, `competitor_promo`, `weather`, `holiday`, `noise_1..6` | precision controls; the σ=0.5 prior soft-selects the irrelevant ones toward 0 |
| **must exclude** | `brand_awareness` | mediator (post-treatment) — controlling for it is a *bad control* |
| needs an experiment | `Radio`, `Print` | collinear — the split is unidentifiable from this data |

## Step 3 — Build the model (pre-specified)

Choices made **before** seeing any results (to avoid specification shopping):
additive contributions, **geometric adstock** (estimated per channel), concave
`1 − exp(−λx)` saturation, yearly Fourier seasonality, linear trend, positive
`Gamma(μ=1.5)` coefficients, and — the causal lever — **per-control prior
widths by role**: confounders get a wide `Normal(0, 2.0)`, everything else the
regularizing `Normal(0, 0.5)`. (The model *refuses* a control marked
mediator/collider, enforcing the DAG.)

## Step 4 — Prior predictive check

Do the priors generate plausible sales *before* touching the likelihood?

> Prior-predictive 90% KPI range **[623, 1696]** vs observed **[479, 1072]**. ✓

The observed data sits comfortably inside the prior's range (not absurdly wide,
not too tight to contain it). The priors are reasonable; proceed. *(If this had
spanned, say, [−10⁴, 10⁵], we would tighten priors and re-check — iteration
applies to priors too.)*

---

## Iteration 1 — v1, the naive model (Steps 5–7)

A first pass without the DAG: include the "obvious" controls and the tempting
`brand_awareness`, but **omit the demand confounders** — their back-door is
invisible until you draw the causal story.

**Step 6 (computational diagnostics):** r̂ = 1.005, **0 divergences**, healthy
ESS. **Step 7 (PPC):** passes. By every routine check, this model is *fine*.

It is not. Graded against truth:

| channel | true | v1 estimate | error | truth in 90% CI? |
|---|--:|--:|--:|:--:|
| **TV** | 8,382 | 13,249 | **+58%** | ✗ |
| **Search** | 2,887 | 4,693 | **+63%** | ✗ |
| **Video** | 7,194 | 3,177 | **−56%** | ✗ |
| Social | 4,534 | 5,880 | +30% | ✓ |
| Display | 2,909 | 2,993 | +3% | ✓ |
| Radio | 2,632 | 1,839 | −30% | ✓ |
| Print | 1,651 | 1,814 | +10% | ✓ |

**Median strong-channel error 56%, coverage 40%.** The pattern is exactly what
the DAG predicts: the demand-chasing channels (**TV, Search**) are massively
**over-credited** (open back-door), while **Video** is **under-credited** because
the `brand_awareness` mediator steals its mediated effect. Three of the largest
channels are wrong by more than half, with the truth *outside* their credible
intervals.

> **The lesson.** A model can converge cleanly, pass PPC, and be confidently
> wrong. Computational diagnostics check the *sampler*, not the *causal
> structure*. This is the silent-failure mode from the
> [robustness report](mmm-robustness-report.md) — and the reason Step 2 exists.

## Iteration 2 — v2, apply the causal structure (loop back to Step 2 → 5)

Revise using the DAG: **add the confounders** with the wide un-shrunk prior,
**drop the mediator** (the model would refuse it if marked), keep the precision
controls. (Wide-prior confounders on collinear data can strain the sampler; the
script raises `target_accept` if divergences appear — here none did, r̂ = 1.005.)

| channel | true | v1 → | **v2** | error | in CI? |
|---|--:|--:|--:|--:|:--:|
| Search | 2,887 | +63% | **3,086** | **+7%** | ✓ |
| Video | 7,194 | −56% | **7,032** | **−2%** | ✓ |
| Social | 4,534 | +30% | 3,656 | −19% | ✓ |
| Display | 2,909 | +3% | 3,254 | +12% | ✓ |
| TV | 8,382 | +58% | 11,790 | **+41%** | ✗ |

**Median strong-channel error 56% → 12%; coverage 40% → 80%.** Adding the
confounder and removing the bad control collapses the bias on Search (+63%→+7%)
and Video (−56%→−2%).

Two checks that confirm the structure is doing its job:

* **The confounder is now credited.** `category_demand` is the largest control
  coefficient (standardized 0.39) — the demand signal that was leaking into the
  chasers' ROI is now correctly attributed to demand.
* **The regularizing prior cleaned out the haystack.** All six irrelevant
  `noise_*` controls have posterior coefficients within ±0.14 of zero — the σ=0.5
  prior soft-selected them out without any explicit variable-selection step.

> **The lesson.** Causal structure — not more data or more tuning — is what
> fixes confounding and post-treatment bias. The DAG you drew in Step 2 is the
> model spec.

But v2 is not done. Two problems remain, and the model **tells you about one of
them and hides the other**:

1. **It admits Radio and Print are unidentifiable.** Their contribution credible
   intervals are enormous (wider than the estimate itself) — the honest signal of
   a collinear pair the data can't split.
2. **TV is still +41% wrong, and looks confident.** No observational diagnostic
   flags it: r̂ is fine, it isn't "fragile" by the robustness value, its interval
   is tight. It is a *confident, wrong* number — residual confounding from a
   **noisy** demand proxy that the control only partly removes.

## Iteration 3 — v3, bring in experiments (Step 8)

You cannot fix (1) or (2) with more observational modeling. You need
randomized evidence. Two triggers, both principled:

* **Unidentified channels** the model flagged (wide CI): **Radio, Print**.
* **Standard practice** — validate your **biggest** channels (**TV, Search**)
  with geo-lift tests, precisely because a confident-but-wrong estimate is
  invisible to every observational check.

Fold each lift test in as an experiment-calibrated coefficient prior
(`ExperimentCalibrator`, the framework's two-stage calibration) and refit.

**What calibration does cleanly — pins the coefficient:**

| channel | β posterior SD, v2 → v3 |
|---|--:|
| Radio | 0.69 → **0.08** |
| Print | 0.69 → **0.05** |
| Search | 0.72 → **0.14** |
| TV | 0.43 → **0.13** |
| *Social (uncalibrated)* | 0.70 → 0.69 |
| *Video (uncalibrated)* | 0.69 → 0.69 |

The calibrated channels' coefficients contract **5–10×**; the uncalibrated ones
don't move. Experiments identify what observational data cannot — the collinear
Radio/Print split, and the slope of every anchored channel.

**What calibration does *not* fully fix — and why that's the real lesson:**
TV's *contribution* only improves +41% → **+36%**. Pinning the coefficient is not
enough when the channel is heavily confounded by a **noisy** proxy: the design
factor that converts the lift into a coefficient prior inherits the residual
bias, so the contribution stays inflated. (Print's point estimate even drifts up
as its collinear partner is re-anchored.)

> **The lesson.** A single lift test pins a *slope*; it does not launder away
> residual confounding in the *contribution*. TV needs a better demand proxy or a
> cleaner, TV-specific experiment — and until then its number should be reported
> as provisional. This is the same conclusion the robustness report reaches from
> the other direction: **experiments are the one signal silent failures can't
> fake, but they must be designed for the channel you're trying to rescue.**

**Refutation suite (Step 8) on the final model:** negative-control, random-common
-cause, and data-subset all pass; placebo fires (it also fires on clean data —
see the robustness report — so it's not actionable here); no channel flagged
fragile.

---

## Step 9 — Communicate results (with honest uncertainty)

The deliverable is not a point estimate per channel — it's a recommendation that
states what we know, how well, and on what evidence.

| channel | contribution (best estimate) | evidence | confidence |
|---|--:|---|---|
| **Video** | ~7,000 | observational + DAG | **high** (−1%, covered) |
| **Search** | ~3,100 | observational + DAG (+lift) | **high** (+7%) |
| Social / Display | ~3,400 / ~3,200 | observational + DAG | medium (±15–25%) |
| **Radio / Print** | combined ~4,400 | **lift-test anchored** | coefficient identified; **report as a pair**, not individually |
| **TV** | ≥ 8,400, likely under-stated by the model | lift-anchored but **residual confounding** | **provisional — do not over-reallocate; commission a TV-specific experiment** |

Budget guidance that follows from this: confidently shift toward **Search** and
**Video** (well-identified, good ROI); treat **Radio/Print** as one line item
until a cleaner experiment separates them; **hold TV** pending a better demand
control or a dedicated lift test rather than trusting a tidy-looking but biased
number.

The framework's `MMMReportGenerator` turns the final fitted model into a
board-ready HTML report with these intervals and the causal-assumptions section.

---

## The arc, and the meta-lessons

| version | what changed | strong med\|err\| | coverage | headline |
|---|---|--:|--:|---|
| **v1 naive** | everything in, no DAG | **56%** | 40% | converges & passes PPC, but confidently wrong |
| **v2 causal** | add confounders, drop mediator | **12%** | 80% | structure fixes confounding + bad control |
| **v3 calibrated** | lift tests on unidentified + major channels | 14%\* | 80% | coefficients pinned; residual TV bias exposed |

\* The strong-median ticks up slightly at v3 because calibration trades a little
point accuracy on already-good channels for **identification** (much tighter
coefficients) and for **exposing** TV's residual bias — the honest outcome, not a
regression.

1. **Modeling is iterative.** No version was right on the first fit. The DAG
   (Step 2) drove v2; the model's own uncertainty and standard experimental
   practice drove v3.
2. **Diagnostics check the sampler, not the truth.** v1 was green on r̂, ESS,
   and PPC while being 56% wrong. Convergence is necessary, not sufficient.
3. **Causal structure is the spec.** The single biggest improvement came from
   classifying controls (confounder / precision / bad control), not from tuning.
4. **Let the model tell you what it can't do.** Wide credible intervals
   (Radio/Print) are a feature — they route you to the right experiment.
5. **Experiments identify, but must be designed for the problem.** Calibration
   pinned every anchored coefficient, yet TV's contribution stayed biased — a
   reminder that a noisy confounder proxy is not fully cured by one lift test.

## Reproduce

```bash
uv run python -m tests.synth.realistic_walkthrough           # publication run -> results/walkthrough.json
uv run python -m tests.synth.realistic_walkthrough --quick   # fast arc check
uv run pytest tests/synth/test_dgp.py -q                      # generator self-checks
```

Companion docs: [robustness report](mmm-robustness-report.md) (convergence,
fit-time, failure modes) and `tests/synth/README.md` (the structural-violation
study these tools grew out of).
