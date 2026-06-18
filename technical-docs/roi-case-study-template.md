# ROI Case-Study Template — Design-Partner Engagement

> **Status: TEMPLATE, NOT A CASE STUDY.** This document is a **fill-in-the-blanks
> skeleton** for turning a real design-partner engagement into a credible,
> defensible ROI case study. **There is no real customer yet.** Every value in
> `[BRACKETS]` is a **placeholder** to be filled from a real engagement — client
> names, dollar figures, ROAS/SE numbers, lift estimates, and measured outcomes
> are **to be supplied from actual data, never invented.** Do not circulate any
> rendering of this template that still contains placeholders, and do not replace
> a placeholder with a plausible-looking made-up number to "show what it could
> look like."
>
> The only numbers in this document that are **real today** are the
> **platform-validation figures** in the *"How we keep this honest"* callout
> (§7). Those are measured, citable, and carry their own honesty labels — they
> describe the *tool*, not any client. Everything that describes the *engagement*
> is a placeholder.
>
> The MMM Framework is a **causal, agent-operated Bayesian marketing-mix-modeling
> platform**. Its pitch is **credibility**: pre-registered analyses, experiments
> folded into the model's posterior, a refutation suite, and intervals that say
> when we don't know. A case study that overstates a result destroys the only
> thing that distinguishes this product. Fill it in honestly or not at all.

---

## 0. How to use this template

1. Run a real engagement through the **adaptive measurement loop** (T₀–T₅ — see
   `docs/platform-overview.html` and the project CLAUDE.md). Each stage below maps
   to a loop stage, so the case study writes itself as the engagement progresses.
2. Replace **every** `[PLACEHOLDER]` with a value sourced from the engagement.
   Keep the uncertainty intervals — a point estimate without its interval is not
   a publishable result for this product.
3. Keep the §7 honesty callout **verbatim** (update only if the underlying
   validation evidence changes). It is what makes the case study survive
   diligence.
4. Have the client approve the final numbers and the pull-quote before
   publication. The pull-quote (§8) is the client's words, not ours.

> **Naming convention for placeholders.** `[CLIENT]`, `[CHANNEL_A]`,
> `[$FIGURE]`, `[N%]`, `[ROAS ± SE]`, `[DATE]`, `[MEASURED_OUTCOME]`. If you
> cannot fill a placeholder from real data, **leave it bracketed and flag the
> section as incomplete** — do not approximate.

---

## 1. Client context  `[PLACEHOLDER]`

- **Client:** `[CLIENT]` — `[one line: category, business model, geography]`
- **Scale:** `[$ANNUAL_MEDIA_SPEND]` annual measured media across `[N]` channels;
  `[$REVENUE or N units]` annual KPI.
- **KPI modeled:** `[KPI — e.g., national weekly revenue / units / new accounts]`,
  `[kpi_kind: revenue | units | other]`.
- **Data:** `[N]` periods at `[weekly | daily]` cadence, `[national | geo |
  geo×product]` panel, `[N]` channels, `[list controls: price, promo, distribution,
  macro]`.
- **Why they engaged:** `[the trigger — budget review, a planned channel cut, a
  CFO who stopped trusting last-click, a privacy-driven attribution gap]`.

> *Fill from the engagement intake / project onboarding brief
> (`POST /projects/{id}/onboarding` → `project_brief.md`).*

---

## 2. The decision at stake

State the **specific dollar decision** the model was hired to inform. A case study
without a decision is a demo.

- **The question:** `[e.g., "Can we cut [$FIGURE] from [CHANNEL_A] without losing
  sales?" or "Where does the next [$FIGURE] of budget earn the most?"]`
- **The stakes:** `[$BUDGET_AT_STAKE]` of annual spend was on the line; the wrong
  call costs roughly `[$DOWNSIDE]`.
- **The deadline / cadence:** `[annual planning cycle | quarterly reallocation |
  a one-time channel-cut decision]`.

---

## 3. The baseline — how they allocated and measured *before*

Establish the "before" honestly. The case study's value is the **delta** against
this baseline, so do not strawman it.

| Dimension | Before the engagement |
|---|---|
| Allocation method | `[last-click attribution | last year ± a % | gut + agency recommendation | a legacy MMM refreshed annually]` |
| Measurement | `[platform-reported ROAS | last-touch | a spreadsheet model | none]` |
| Uncertainty | `[typically none — single point estimates, no intervals]` |
| Experiment history | `[none | ad-hoc geo tests not folded back into the model]` |
| Known pain | `[e.g., channels double-counting conversions; no way to defend a number to finance]` |

> **The honest baseline framing:** the prior approach gave `[a point estimate with
> no interval / a platform-reported ROAS of [X]]`, which `[over-credited [CHANNEL]
> / could not be audited / inverted under scrutiny]`. The engagement's job was to
> replace that with a **causally identified estimate carrying genuine uncertainty.**

---

## 4. What the MMM found (T₀ fit → T₁ EIG/EVOI)

### 4.1 The fitted decomposition

After the **T₀ fit**, the model decomposed `[KPI]` into baseline + per-channel
contributions with full posteriors.

| Channel | Posterior contribution share | ROI / ROAS (mean) | 80% credible interval |
|---|---|---|---|
| `[CHANNEL_A]` | `[N%]` | `[ROAS]` | `[lo, hi]` |
| `[CHANNEL_B]` | `[N%]` | `[ROAS]` | `[lo, hi]` |
| `[CHANNEL_C]` | `[N%]` | `[ROAS]` | `[lo, hi]` |
| `[…]` | `[N%]` | `[ROAS]` | `[lo, hi]` |
| **Baseline** | `[N%]` | — | — |

> *Source: the fitted `MMMResults` / agent decomposition report. Report ROAS with
> its interval — a bare point estimate is not a deliverable for this product.*

### 4.2 Where uncertainty was most *expensive* (EIG / EVOI)

This is the differentiator. The model didn't just produce numbers — it priced its
own ignorance. The **T₁ EIG/EVOI engine** ranked channels by **how many dollars the
remaining uncertainty was costing** the allocation decision.

- **Most decision-expensive uncertainty:** `[CHANNEL_X]`. Its ROAS posterior was
  wide enough (`[lo, hi]`) that the optimal allocation flipped across the interval —
  the **expected value of resolving that uncertainty (EVOI)** was `[$EVOI_FIGURE]`,
  the largest in the portfolio.
- **Cheap-to-leave-alone:** `[CHANNEL_Y]` — `[narrow interval / small budget; not
  worth an experiment]`.
- **The priority call:** the agent recommended spending measurement budget on
  `[CHANNEL_X]` first, because `[$EVOI_FIGURE]` of decision value was unresolved
  there versus `[$SMALLER]` elsewhere.

> *Source: `compute_experiment_priorities` / the `/projects/{id}/experiment-priorities`
> view. EIG ranks information gain; EVOI converts it to dollars at the decision.*

---

## 5. The experiment (T₂ pre-registration → readout)

The model's claim about `[CHANNEL_X]` was **tested**, not assumed.

### 5.1 Design

- **Type:** `[randomized matched-pair geo lift | matched-market DiD | budget-neutral
  national flighting]` (from `design_experiment_plan` / the DesignStudio).
- **Treated vs control:** `[N]` `[geos | flight weeks]`, `[scale-up +N% | go-dark
  −100%]` on `[CHANNEL_X]`.
- **Duration:** `[N]` weeks + `[N]`-week cool-down (adstock washout).
- **Power verdict before running:** `[powered | underpowered | overpowered]` to
  detect an incremental ROAS of `[MDE]` at `[N]%` assurance — from the
  model-anchored economics (`experiment_economics`).
- **Opportunity cost:** running the test was projected to `[forgo [$FIGURE] of KPI |
  *save* [$FIGURE] in spend — a net-positive holdout]` with `[uncertainty]`.

### 5.2 Pre-registration

The design, the estimand, the analysis method, and the success threshold were
**pre-registered before any data was seen** (`preregister_experiment` →
lifecycle `draft → planned → running`). This is the researcher-degrees-of-freedom
lock: the readout could not be reverse-engineered to a convenient answer.

- **Pre-registered estimand:** `[incremental ROAS for [CHANNEL_X] over the test
  window]`.
- **Pre-registered decision rule:** `[reallocate iff readout ROAS interval clears
  [threshold]]`.
- **Registered:** `[DATE]`, `[experiment_id]`.

### 5.3 Readout

| Quantity | Value |
|---|---|
| Estimated incremental ROAS | `[ROAS] ± [SE]` |
| Method | `[pooled DiD | per-pair DiD | synthetic-control geo | national on/off]` |
| Placebo / A-A check | `[passed — empirical false-positive rate [N]% | flagged]` |
| Verdict vs pre-registered rule | `[met | not met]` |

> *Source: `record_experiment_readout`. Report **ROAS ± SE** — the standard error
> is not optional. If the test came back underpowered or inconclusive, **say so**
> and stop; an inconclusive experiment is a real (and publishable) outcome.*

---

## 6. Calibrated reallocation + projected lift (T₃ refit → T₄ allocate)

### 6.1 The calibrated refit

The experiment readout entered the model as an **in-graph likelihood** (T₃), jointly
updating `[CHANNEL_X]`'s coefficient, its saturation curve, **and** its adstock
kernel — not a post-hoc override. The channel moved from
`[observational ROAS [old] [lo, hi]]` to `[calibrated ROAS [new] [lo, hi]]`, and the
posterior **tightened** by `[N%]` on the decision-relevant interval.

> *Source: `apply_experiment_calibration`; `fit_mmm_model` auto-marks the model
> calibrated. The coverage map shows `[CHANNEL_X]` flipping from observational to
> experiment-backed.*

### 6.2 The recommended reallocation

| Channel | Before `[$]` | After `[$]` | Change |
|---|---|---|---|
| `[CHANNEL_X]` | `[$]` | `[$]` | `[+/− N%]` |
| `[CHANNEL_A]` | `[$]` | `[$]` | `[+/− N%]` |
| `[CHANNEL_B]` | `[$]` | `[$]` | `[+/− N%]` |
| **Total** | `[$BUDGET]` | `[$BUDGET]` | `[budget-neutral | +N%]` |

### 6.3 Projected lift — *with uncertainty*

- **Projected KPI lift from the reallocation:** `[+N% / +$FIGURE]`, **80% credible
  interval `[lo, hi]`**.
- **Probability the reallocation beats the status quo:** `[N%]`.
- **Honest caveat:** the projection assumes `[structural stationarity — the response
  curves hold over the reallocation horizon]`; it is a model forecast, to be
  confirmed by the measured outcome in §6.4.

> **Never report the projected lift as a point estimate.** The interval and the
> probability-of-beating-status-quo are the deliverable.

### 6.4 Measured outcome  `[PLACEHOLDER]`

The reallocation was implemented on `[DATE]`. After `[N]` weeks:

- **Measured `[KPI]` change vs the pre-registered counterfactual:** `[MEASURED_OUTCOME
  ± interval]`.
- **Did it land inside the projected interval?** `[yes / no — and what that means]`.
- **Follow-up (T₅ re-test):** `[information decay triggered a re-test of [CHANNEL] on
  [DATE] | the program moved to the next-highest-EVOI channel]`.

> *This section can only be filled **after** the measured outcome exists. Until then,
> the case study is **provisional** and must be labeled as such. Do not publish a
> "measured outcome" that is actually a projection.*

---

## 7. How we keep this honest  *(keep this section verbatim)*

> The point of this product is a number a CFO can defend. The following are the
> guardrails that make the case study above credible — and the honest limits of the
> validation evidence. **Do not soften these labels.**

- **Pre-registration reduces researcher degrees of freedom.** The experiment's
  estimand, analysis method, and decision rule were fixed **before** the readout
  (§5.2). This is the single biggest defense against the failure mode that burns MMM
  buyers: a model quietly tuned until it agrees with what someone already wanted to
  do.

- **The refutation suite.** Every fitted model is stress-tested by a **4-test causal
  refutation suite** — placebo treatment, negative control, random common cause, and
  data-subset refutation. A passing suite is **not** an all-clear (it cannot prove the
  causal assumptions); a failing suite is a **stop**. The case study should state
  which tests were run and their results.

- **We report intervals, and we say when we don't know.** Every ROAS, contribution,
  and projected lift above carries a credible interval. When the data cannot identify
  an effect, the model reports a wide interval rather than a confident wrong number.

- **The validation evidence — labeled honestly (these are the *only* real numbers in
  this template):**
  - The **headline backtest accuracy — 3.0% MAPE out-of-time, roughly one-quarter the
    error of a "copy last year" naïve baseline (13.8%), with 80%/95% interval coverage
    at 90.4% / 94.2%** — is measured on a **synthetic benchmark world** (156 weeks, 7
    channels), **not a client engagement.** Present it as a benchmark result, never as
    a customer result.
  - The one **genuinely external, real dataset** is **Lydia Pinkham (54 years,
    1907–1960)**, the most-studied series in advertising econometrics. There it is a
    **causal-anchor win** — total-effect **ROAS 1.48 [1.12, 1.85]**, inside the bracket
    six decades of published estimates produced — but an **honest forecasting non-win:
    MASE 1.36, i.e., no skill over naïve persistence**, while keeping intervals
    **calibrated (94% coverage).** We do **not** claim to beat naïve forecasting on real
    data; we explicitly did not, and we say so. "A backtest that cannot fail cannot
    validate."
  - **Speed (measured):** **NumPyro NUTS ~6.0× faster** than stock PyMC at equal draws
    (not 7.9×); a national fit completes in **~17 s**. Use these exact figures.
  - **SOC 2 is readiness, not certification** (see
    `technical-docs/soc2-readiness.md`). No audit has occurred; do not imply one has.

---

## 8. Pull-quote template  *(client's words — approve before use)*

> "`[Before the engagement, we allocated [$FIGURE] across [N] channels on
> [last-click / gut / last year], and we couldn't defend any of it to finance. The
> MMM told us where our uncertainty was actually costing us money, we ran the
> experiment to settle it, and the calibrated reallocation [moved [$FIGURE] from
> [CHANNEL] to [CHANNEL]] — and the measured result came in [at/inside [interval]].
> For the first time we have a number with an honest error bar attached.]`"
>
> — `[NAME, TITLE, [CLIENT]]`

> *The pull-quote must be the client's own words, reviewed and approved by the
> client. Do not draft a quote and attribute it. The bracketed text above is a
> *prompt* for the conversation, not a quote to put in someone's mouth.*

---

## 9. Metrics summary table  `[PLACEHOLDER cells]`

| Metric | Before | After (calibrated) | Source |
|---|---|---|---|
| Allocation method | `[baseline]` | Causal MMM + experiment-calibrated posterior | §3, §6 |
| `[CHANNEL_X]` ROAS | `[old point estimate, no interval]` | `[ROAS] [lo, hi]` | §4.1, §6.1 |
| Decision-expensive uncertainty (EVOI) | `[unpriced]` | `[$EVOI_FIGURE]` resolved by experiment | §4.2 |
| Experiment incremental ROAS | `[n/a]` | `[ROAS] ± [SE]` (pre-registered) | §5.3 |
| Posterior interval width on `[CHANNEL_X]` | `[wide — [lo, hi]]` | `[tightened [N%] — [lo, hi]]` | §6.1 |
| Reallocation | `[n/a]` | `[+/− N%]` across `[channels]` | §6.2 |
| Projected KPI lift | `[n/a]` | `[+N%]`, 80% CI `[lo, hi]` | §6.3 |
| **Measured KPI outcome** | `[n/a]` | `[MEASURED_OUTCOME ± interval]` | §6.4 |
| Refutation suite | `[n/a]` | `[N/4 passed]` | §7 |

> Every cell in the "After" column must trace to a numbered section above and to a
> real artifact from the engagement (a fitted model, a registered experiment, a
> readout, a measured outcome). **A blank or bracketed cell means the case study is
> not finished — it does not mean "estimate it."**

---

## 10. Pre-publication checklist

- [ ] Every `[PLACEHOLDER]` replaced with a real, sourced value (or the section is
      explicitly marked incomplete).
- [ ] Every ROAS / contribution / lift carries its interval.
- [ ] The §6.4 measured outcome is a **measured** outcome, not a projection.
- [ ] The §7 honesty callout is present and unmodified.
- [ ] Synthetic-benchmark figures (3.0% MAPE) are labeled synthetic; Pinkham is
      labeled a causal win **and** a forecasting non-win (MASE 1.36); speed is 6.0×
      / ~17 s; SOC 2 is readiness.
- [ ] The pull-quote is the client's approved words.
- [ ] The client has signed off on all numbers and the publication.
