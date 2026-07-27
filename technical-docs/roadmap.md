# Roadmap

Where the framework is going, quarter by quarter, and — as importantly — what it
already does, so that neither planning nor contribution proposes work that shipped.

Milestones on GitHub mirror this file; each quarter's theme is an epic issue.
Dates are *targets for a release*, not commitments for individual issues.

| Quarter | Milestone | Theme | Epic | Status |
|---|---|---|---|---|
| Q3 2026 | `v1.3` | Frequentist estimation | [#180](https://github.com/redam94/mmm-framework/issues/180) | **shipped** v1.3.0 |
| Q4 2026 | `v1.4` | Finance-grade planning | [#195](https://github.com/redam94/mmm-framework/issues/195) | **in flight** · #215–#228 |
| Q1 2027 | `v1.5` | Competition & dynamic baselines | [#191](https://github.com/redam94/mmm-framework/issues/191) | planned |
| Q2 2027 | `v1.6` | Media modeling breadth | [#192](https://github.com/redam94/mmm-framework/issues/192) | planned |
| Q3 2027 | `v1.7` | Measurement operations & data layer | [#196](https://github.com/redam94/mmm-framework/issues/196) | planned |
| Q4 2027 | `v1.8` | Portfolio scale | [#197](https://github.com/redam94/mmm-framework/issues/197) | planned |
| Q1 2028 | `v2.0` | Contract cleanup & scale | [#193](https://github.com/redam94/mmm-framework/issues/193) | planned |

### Re-sequenced 2026-07-25

Three quarters were added and two moved, after a gap analysis run from a CMO, a
marketing-data-scientist and a CFO perspective against the shipped surface. Per
this file's own rule, the reasons rather than a quiet retcon:

- **Finance-grade planning (#195) was inserted at Q4 2026**, ahead of everything
  except the in-flight frequentist work. The gap analysis found the framework
  estimates and optimizes well but cannot produce a number a CFO commits to —
  no forward forecast on a fiscal calendar, no variance to plan, no payback
  horizon, and an optimizer that ignores the price and promo levers the model
  already estimates. Most of it is wiring machinery that exists to the right
  consumer, which makes it unusually cheap for its value. — *That last sentence
  did not survive decomposition on 2026-07-26; see the Q4 2026 section for what
  verification found and why the quarter is front-loaded with bug fixes.*
- **Competitive / share-of-voice moved out of #192 and into #191**, which
  becomes the bias-fix quarter. #192 itself identified competition as "a
  confounding problem rather than a missing feature" and then scheduled it
  behind two quarters that fix no bias. Competition and the fitted-not-filtered
  baseline are the framework's two remaining structural bias sources; they now
  share a quarter.
- **Media breadth (#192) slipped one quarter to Q2 2027**, keeping cost drift —
  a wrong ROI denominator on long panels — as its lead item, since v1.4's
  margin and payback numbers divide by it.
- **Measurement operations (#196) and portfolio scale (#197) are new**, at
  Q3/Q4 2027.
- **`v2.0` moved from Q2 2027 to Q1 2028.** It is a holding pen for deferred
  breaking changes with an entry rule of "nothing lands here for being tidier";
  four quarters of feature work legitimately precede it.

Only the in-flight quarter has its issues broken out (#195 · #215–#228; the
shipped v1.3 quarter was #180 · #181–#189). Future quarters carry a *Proposed
issues* list in the epic and get decomposed when they start.

---

## The bar

Every entry below is subject to one rule, which is the framework's actual
differentiator: **a feature ships with evidence that it recovers a planted truth,
or it does not ship.** The synthetic worlds in `mmm_framework.synth` carry answer
keys precisely so that "we added a knob" can be distinguished from "we added a
knob that helps".

Two corollaries that have already changed decisions in this codebase:

- If a capability cannot be identified from the data a typical panel carries,
  the honest deliverable is a *refusal plus an explanation*, not a parameter.
  (See the extension-prior registry, which tells you a write will not apply
  rather than accepting it silently.)
- If a number reads as more trustworthy than it is, that is a bug of the same
  severity as a wrong number. The v1.1.0 spec-curve and robustness-value fixes,
  and the v1.2.0 sampler fixes, were all of this kind.

---

## Q3 2026 — `v1.3`, Frequentist estimation — **shipped 2026-07-26**

**Epic [#180](https://github.com/redam94/mmm-framework/issues/180) · issues #181–#189 · released as v1.3.0**

Delivered all three justifications below. The headline result is a *negative*
one, and it is stated in the docs rather than buried: graded against planted
truth on five synthetic worlds, ridge wins only on the positive control and is
worse everywhere else — most starkly under unobserved confounding, where it
over-credits media by **+41.6%** against MAP's +5.9%, because the shipped media
priors shrink effects in a way a data-selected L2 penalty does not. Reach for the
frequentist path for a **hard constraint**, a fast second opinion, or iteration
speed — not to publish intervals, and not on data you suspect of confounding.

`InferenceMethod.FREQUENTIST_RIDGE` and `FREQUENTIST_CVXPY` had been declared
since early in the project and never implemented; until v1.2.0 they silently fit
Bayesian NUTS instead ([#181](https://github.com/redam94/mmm-framework/issues/181),
now fixed — they refuse).

The design question that governs the quarter: **ridge regression is MAP
estimation under Gaussian priors**, so `fit(method="map")` is already ridge in
substance. Three things justify the work, and the epic delivers them or shrinks:

1. **Transform hyperparameters by search, not by prior.** Fix per-channel
   (α, λ) and solve the resulting *linear* problem in closed form — a different
   estimator with different failure modes, and a fast one.
2. **Frequentist intervals.** Bootstrap, not a posterior — with the
   autocorrelation and post-selection traps handled honestly rather than papered
   over.
3. **Hard constraints.** A prior makes a negative coefficient unlikely; a convex
   program makes it impossible. This is the one capability with no Bayesian
   equivalent.

Ridge needs **no new dependency** (closed form, numpy/scipy). Only CVXPY does,
and it goes in an optional `[frequentist]` extra — the lean-core invariant
(`tests/test_lean_imports.py`) holds.

**Riskiest issues**: [#186](https://github.com/redam94/mmm-framework/issues/186)
(a plausible bootstrap that silently under-covers) and
[#188](https://github.com/redam94/mmm-framework/issues/188) (rendering a
confidence interval as a credible one across a reporting stack that assumes a
posterior everywhere).

---

## Q4 2026 — `v1.4`, Finance-grade planning

**Epic [#195](https://github.com/redam94/mmm-framework/issues/195) · issues #215–#228**

The framework estimates well and, since #139, optimizes well. It cannot produce
a number a CFO will **commit to**: no forward forecast on a fiscal calendar, no
variance to plan, no payback horizon, and an optimizer that ignores the price and
promo levers the model already estimates.

### The premise changed when we decomposed it

This quarter was scoped on the belief that **the machinery mostly exists and is
wired to the wrong consumer**, which made it look cheap for its value. A
six-area survey with an adversarial verification pass on each found that much of
the machinery exists and is **wrong**, and that two wrong numbers ship to users
today. Recorded here rather than quietly re-planned, per this file's own rule:

- **The forecaster is structurally incomplete.** `PosteriorForecaster.forecast()`
  sums 5 of the fitted `mu`'s 10 terms — it silently drops the product, event
  (#143), cross-channel interaction (#142) and price/promo lever (#138)
  contributions, substitutes a time-averaged beta for a time-varying channel,
  forecasts reach on raw reach, and `_clone_for_prefix` downcasts any garden or
  custom model class to plain `BayesianMMM`. Nothing raises. Those numbers ship
  today through `run_backtest`, the `cross_validation` agent tool, a Validation-tab
  REST job and a client artifact. Same defect class as #202 and #171: a code path
  that reads as complete and silently does nothing for one configuration.
- **The carryover reader is family-blind**, builds `alpha_mean ** lags`
  unconditionally, collapses the posterior *before* a transform convex in alpha,
  and always reports `l_max = 8` because it probes a field name the panel does
  not have. A payback horizon cannot be built on it — so "payback is arithmetic
  on top of `compute_adstock_weights`" was wrong, and the reader is fixed first.
- **The Planner already emits a fabricated dollar figure.** "Fund to breakeven"
  never sends `value_per_kpi`; the server defaults it to `1.0`, and the free-mode
  objective then funds every channel until marginal KPI equals one KPI-unit per
  dollar. On a KPI denominated in thousands, the recommended budget is ~1000×
  off — rendered with credible intervals.
- **The same fitted model already gives two opposite recommendations.** The PPTX
  deck resolves break-even as `1/margin` from saved project economics; the Augur
  HTML report takes the `1.0` default. At a 40% margin a channel with ROI 1.8 is
  *Scale* in one artifact and *Reduce* in the other. Margin is not missing — it
  is present twice and inconsistent, so the work is convention reconciliation.
- **Plan and delivery do not join by period.** Actual rows are emitted in
  lexicographic label order (`P1, P10, P11, …, P2`) and aligned *positionally*
  against the plan. Per-channel totals survive because sums are order-invariant,
  which is why it went unnoticed; the per-period series a user reads are shuffled
  once there are more than nine periods.
- **There is no realized-KPI store.** `delivery` records spend only, so there is
  no variance to compute — only a forecast restated under actual spend.
- **The synthetic worlds cannot grade a forecast.** `Scenario` keeps window
  totals and discards the per-period noiseless mean, so an interval graded
  against realized `y` conflates model error with irreducible noise.

Four items were cut to non-goals on identifiability grounds: a **price
recommendation** (the framework's own published measurement recovers 39% of a
true elasticity from a typical panel, so price ships as a what-if evaluator that
refuses to recommend), **NPV as a headline** (the repo measured the discounting
correction at 0.33–2.4% against posterior intervals of ±30–50%), **cash timing**
(no payment-terms concept exists to declare, and emitting one from undeclared
zero lags would assert we pay and collect instantly), and **per-product profit
ROI** (media betas are shared across products, so a per-product scalar margin
rescales every channel equally and cannot reorder them).

### The shape of the quarter

Six `priority:P0` correctness issues land first — three shared foundations
(#215 valuation, #216 calendar, #217 the planted-truth substrate) and three
standing bugs (#218 carryover reader, #219 forecaster audit, #220 decomposition
closure) — followed by #221, this quarter's integration-risk issue and the
analogue of v1.3's #188: one value basis and one break-even convention, gated at
every render site. The features build on top: #223 forward forecast, #224
payback, #225 plan of record, #226 promo-depth optimization, #227 actuals and
variance.

The standing "reads as more trustworthy than it is" rule bites hardest here, and
it forced the quarter's governing split. A forward interval is knowingly too
narrow twice — a spline trend is held flat beyond the panel, and observation
noise is iid on residuals the framework routinely finds autocorrelated — and
both fixes are v1.5 modelling work (state-space baselines). Rather than
disclose-and-commit-anyway, v1.4 separates the two: a forecast is always
**computable** with its caveats attached (#223), and **committable** only through
a gate that can refuse (#225).

---

## Q1 2027 — `v1.5`, Competition & dynamic baselines

**Epic [#191](https://github.com/redam94/mmm-framework/issues/191)**

The framework's two remaining sources of *structural bias*, deliberately paired
in one quarter.

**Competition / share-of-voice** (moved here from #192). There is no way to tell
the model a competitor exists — `competitive` and `share_of_voice` appear only in
an intervention helper, a stationarity utility and an example dataset. Competitor
spend moves your KPI *and* correlates with your own, since both follow category
seasonality and respond to each other; omit it and the media coefficients absorb
it. The control path can carry a competitor series, but nothing models
share-of-voice structure, and nothing warns that the omission is a back-door
path. Scope includes the causal-role treatment, so `find_backdoor_paths` and the
identification verdict know about it.

**Scoping note on the second half, because it is easy to get wrong: dynamic
coefficients already ship.** `MediaChannelConfig.time_varying` (#137) fits a smooth random walk on
`log(beta_t)` with a time-average summary and report trajectories, and the AR(1)
state machinery in `mmm_extensions/components/latent_states.py` already powers
structural mediator states and latent factors. This quarter is not "add TVP".

The real gap is that **the baseline is fitted, not filtered**. Every trend option
— linear, piecewise, spline, HSGP — is a basis-function regression over a fixed
design matrix. There is no local level, no local linear trend, no Kalman
marginalization. That costs three things:

- **Extrapolation.** A spline has no principled forward behaviour; a state does.
  This is what the backtest/forecast path needs to mean something beyond the panel.
- **An explicit signal-to-noise statement.** A wiggly basis trend and a media
  effect compete for the same variance, with the winner decided implicitly by
  knot placement. A state-space formulation makes it a variance ratio you set
  and can argue about.
- **Speed and geometry.** Marginalizing states with a Kalman filter integrates
  out the largest parameter block rather than adding hundreds of correlated
  latents.

Also in scope: a **drift-is-real check** for TVP. A fitted random walk always
produces a wiggle; without an evidence check on the innovation variance, a
trajectory chart will be read as "effectiveness is declining" when it is noise.

**The state-space half ships as an option, not a default**, unless it beats the
existing trends on a world with a planted drifting baseline. The competition half
has no such escape: an unmodelled confounder is a bias, and the deliverable is
either the model or an identification verdict that says the estimate is not
defensible.

---

## Q2 2027 — `v1.6`, Media modeling breadth

**Epic [#192](https://github.com/redam94/mmm-framework/issues/192)**

With competition promoted to v1.5, four gaps remain — correctness first, then
capability:

1. **Media cost drift** — `cpm` is a scalar, so a channel's cost basis cannot
   change across a three-year panel, making ROI denominators wrong on any long
   panel with real cost inflation. A correctness bug, already logged as deferred,
   and now also a dependency: every margin and payback number in v1.4 divides by
   this.
2. **Creative-level modeling and wear-out** — "is this creative worn out or is
   the channel saturating?" is currently unanswerable, and the two have opposite
   remedies. Depends on the time-varying-saturation scoping question in #191.
3. **Reach & frequency depth** — the config landed in #141; optimal frequency,
   capping and incremental-reach curves did not.
4. **Cross-channel synergy reconciliation** — the continuous-learning module
   treats pairwise synergy as first-class with sign-informed priors and off-axis
   designs to identify it; the MMM path does not. Synergy should mean one thing
   across the codebase.

---

## Q3 2027 — `v1.7`, Measurement operations & data layer

**Epic [#196](https://github.com/redam94/mmm-framework/issues/196)**

Everything in the codebase assumes a human sitting in front of a model they just
fit. Nothing watches a model that shipped three months ago, and nothing brings
data in on a schedule.

1. **Live-model monitoring.** Nothing checks a fitted model against the weeks
   that arrive *after* it was fit. The adjacent pieces are all subtly different:
   `reexperiment_due` decays *experimental evidence*, `portfolio_benchmark`
   reports model *age*, `scorecard` compares predicted ROI to *experiment
   readouts*. None is a posterior-predictive check against incoming actuals. The
   hard part is not the check — it is separating *the world changed* from *the
   model was always wrong* from *this is noise*, and reporting uncertainty rather
   than firing a false alarm when it cannot.
2. **Platform / MTA data as a calibrated measurement.** `triangulation.py` puts
   MMM, experiment and platform estimates side by side and labels divergence — a
   report; the model never sees the platform number. `calibration/likelihood.py`
   already folds an external estimate into the graph on a named estimand.
   Extending it to observational platform conversions **with an explicit bias
   term** would let the model use that data at its correct (low) evidential
   weight instead of arguing with it in a table. The bias term is the point:
   self-attribution inflation is real, and the honest model of it is a
   measurement with a bias parameter, not a calibration target.
3. **Mixed-frequency panels.** `time_granularity` is one choice. Real inputs are
   daily media, weekly sell-through, monthly finance. "Aggregate up" discards
   exactly the daily variation that identifies adstock.
4. **Live connectors.** `integrations/ad_platforms/` is three documented stubs
   that raise `AdPlatformNotImplemented`. The warehouse-transfer recommendation
   is right; the scheduled path on top of it is not built. Time-to-first-model is
   a purchase criterion for every persona.

**Not** in scope: auto-refitting without a human. A trigger is a recommendation
with evidence attached, not a silent model swap.

---

## Q4 2027 — `v1.8`, Portfolio scale

**Epic [#197](https://github.com/redam94/mmm-framework/issues/197)**

One statistical idea at two levels, for advertisers running the same question
across many entities with very different data depth.

1. **Multi-brand / multi-market hierarchy.** There is no `brand_idx` or
   `country_idx` in the model; the hierarchy stops at geo.
   `platform/portfolio_benchmark.py` aggregates *finished results* — it ranks a
   brand after every model has been fit independently, which is a dashboard, not
   pooling. A brand with eight quarters gets nothing from eleven siblings with
   five years. Includes the hard half: **when pooling is wrong**, since two
   brands in one category can genuinely differ and a confident average describes
   neither.
2. **Meta-analytic priors — the cold-start problem.** A new brand gets a generic
   `LogNormal(0, 1)` ROI prior, correct in the absence of information. The
   information is not absent: every past fit's per-channel ROI posterior is
   already in `run_metrics`, and today it only draws charts. Turning it into
   empirical-Bayes priors by category and channel is the highest-value use of
   data the platform already stores — with leakage exclusion enforced at fit
   time, provenance disclosed at the point of reading, cross-tenant aggregation
   governed, and a calibrated experiment always dominating.

---

## Q1 2028 — `v2.0`, Contract cleanup & scale

**Epic [#193](https://github.com/redam94/mmm-framework/issues/193)**

A holding pen for what strict SemVer defers, kept visible so it is not
rediscovered: retained-but-dead enum values, the vestigial `TrendBuilder`, the
legacy adstock path, the `tests/synth` shim, the legacy sessions-DB fallback.

Entry rule: nothing lands here for being *tidier*. It lands because keeping it
costs something measurable — a real bug, a real support burden, or a real
performance floor.

Scale work is parked here too, with the same discipline: **profile before
optimizing.** The most promising lever is already identified (Kalman
marginalization, #191), which is a reason to sequence it after that quarter
rather than guess at bottlenecks now. The portfolio hierarchy (#197) lands the
quarter before and changes the panel sizes worth profiling, which is a second
reason this sits last.

---

## What this roadmap deliberately does not contain

- **Regime-switching / Markov-switching models.** No evidence of demand.
- **Replacing the Bayesian path as the default.** The frequentist work is an
  alternative for speed and constraints, not a new centre of gravity.
- **Frequentist support for the extension models.** Their bespoke graphs are not
  linear given fixed transforms.
- **Deep-learning attribution.** Out of scope for a framework whose value is
  interpretable causal structure with honest uncertainty.
- **A benchmark data product.** The portfolio priors in #197 improve *a client's
  own* estimates. Publishing anyone's numbers, even aggregated, is a different
  business with a different consent model.
- **User-level / MTA modeling.** #196 folds platform-reported figures in as a
  biased *measurement*; it does not rebuild the framework around person-level
  data it cannot causally identify from.
- **A financial planning system.** #195 produces the marketing block of a plan.
  It does not own the plan, the ledger, or the close.

## Keeping this current

Update this file in the same PR that opens or closes a theme epic. If a quarter
slips, move the milestone date and say why here — a roadmap that quietly
retcons its own dates is worth less than no roadmap.
