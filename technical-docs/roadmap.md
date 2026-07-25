# Roadmap

Where the framework is going, quarter by quarter, and — as importantly — what it
already does, so that neither planning nor contribution proposes work that shipped.

Milestones on GitHub mirror this file; each quarter's theme is an epic issue.
Dates are *targets for a release*, not commitments for individual issues.

| Quarter | Milestone | Theme | Epic |
|---|---|---|---|
| Q3 2026 | `v1.3` | Frequentist estimation | [#180](https://github.com/redam94/mmm-framework/issues/180) |
| Q4 2026 | `v1.4` | Finance-grade planning | [#195](https://github.com/redam94/mmm-framework/issues/195) |
| Q1 2027 | `v1.5` | Competition & dynamic baselines | [#191](https://github.com/redam94/mmm-framework/issues/191) |
| Q2 2027 | `v1.6` | Media modeling breadth | [#192](https://github.com/redam94/mmm-framework/issues/192) |
| Q3 2027 | `v1.7` | Measurement operations & data layer | [#196](https://github.com/redam94/mmm-framework/issues/196) |
| Q4 2027 | `v1.8` | Portfolio scale | [#197](https://github.com/redam94/mmm-framework/issues/197) |
| Q1 2028 | `v2.0` | Contract cleanup & scale | [#193](https://github.com/redam94/mmm-framework/issues/193) |

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
  consumer, which makes it unusually cheap for its value.
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

Only the in-flight quarter has its issues broken out (#180 · #181–#189).
Future quarters carry a *Proposed issues* list in the epic and get decomposed
when they start.

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

## Q3 2026 — `v1.3`, Frequentist estimation

**Epic [#180](https://github.com/redam94/mmm-framework/issues/180) · issues #181–#189**

`InferenceMethod.FREQUENTIST_RIDGE` and `FREQUENTIST_CVXPY` have been declared
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

**Epic [#195](https://github.com/redam94/mmm-framework/issues/195)**

The framework estimates well and, since #139, optimizes well. It cannot produce
a number a CFO will **commit to**. The pattern across all five gaps is the same,
and is the reason this quarter is cheap for its value: **the machinery mostly
exists and is wired to the wrong consumer.**

1. **Plan of record → forward forecast → variance.**
   `validation/backtest.py::PosteriorForecaster.forecast()` already takes future
   media and returns original-scale draws with carryover handled across the
   boundary — reachable only through the rolling-origin backtest, a validation
   tool. Missing is the finance-shaped wrapper: a locked plan, a forecast on the
   *fiscal* calendar, and dollar variance as actuals land. `pacing.py` (#107)
   closes the loop on spend; nothing closes it on the outcome anyone committed to.
2. **Payback period and NPV.** Carryover means a Q4 dollar returns across Q1 — a
   working-capital fact with no representation. `discount_rate_annual` exists
   only in `planning/experiment_value.py`. The return profile is already computed
   by `compute_adstock_weights`; a payback horizon is arithmetic on top of it. A
   CFO comparing brand against performance is implicitly comparing payback
   horizons the framework can compute and does not show.
3. **Optimization across levers, not just media.** `PriceConfig` / `PromoConfig`
   (#138) and `EventsConfig` (#143) ship as first-class model terms; `budget.py`
   optimizes media spend curves only. So the model knows the price elasticity and
   the promo lift, and the optimizer cannot say whether the next dollar belongs
   in TV or in promo depth — the actual planning question for CPG, retail and DTC.
4. **Margin as a first-class dimension.** `value_per_kpi` is a scalar knob;
   `opportunity_cost.py` reads a project-level `gross_margin`. Profit ROI and
   breakeven CPA are not native to the estimand or reporting stack, so a
   revenue-optimal recommendation can be margin-destructive and the report will
   not say so.
5. **A bridge to the booked P&L.** Decomposition sums to the modeled KPI, which
   is rarely the finance line — gross vs net, returns, trade spend, price/mix.
   Low science, high trust: without a stated reconciliation, finance treats the
   model as marketing's number rather than the company's.

The standing "reads as more trustworthy than it is" rule bites hardest here. A
forecast under a plan is a counterfactual and inherits every caveat the model
carries; it ships saying so.

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
