# Roadmap

Where the framework is going, quarter by quarter, and — as importantly — what it
already does, so that neither planning nor contribution proposes work that shipped.

Milestones on GitHub mirror this file; each quarter's theme is an epic issue.
Dates are *targets for a release*, not commitments for individual issues.

| Quarter | Milestone | Theme | Epic |
|---|---|---|---|
| Q3 2026 | `v1.3` | Frequentist estimation | [#180](https://github.com/redam94/mmm-framework/issues/180) |
| Q4 2026 | `v1.4` | Dynamic & state-space models | [#191](https://github.com/redam94/mmm-framework/issues/191) |
| Q1 2027 | `v1.5` | Media modeling breadth | [#192](https://github.com/redam94/mmm-framework/issues/192) |
| Q2 2027 | `v2.0` | Contract cleanup & scale | [#193](https://github.com/redam94/mmm-framework/issues/193) |

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

## Q4 2026 — `v1.4`, Dynamic & state-space models

**Epic [#191](https://github.com/redam94/mmm-framework/issues/191)**

**Scoping note, because it is easy to get wrong: dynamic coefficients already
ship.** `MediaChannelConfig.time_varying` (#137) fits a smooth random walk on
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

**Ships as an option, not a default**, unless it beats the existing trends on a
world with a planted drifting baseline.

---

## Q1 2027 — `v1.5`, Media modeling breadth

**Epic [#192](https://github.com/redam94/mmm-framework/issues/192)**

Audited against what exists, five gaps remain. Ordered by whether they are *bias*
or *capability*, because that is the order they should be fixed in:

1. **Competitive effects / share-of-voice** — the largest gap, and a
   **confounding** problem rather than a missing feature. There is currently no
   way to tell the model a competitor exists. Competitor spend moves your KPI and
   correlates with your own; omit it and the media coefficients absorb it. This
   undermines the causal claims the framework is built to make, so it leads the
   quarter.
2. **Media cost drift** — `cpm` is a scalar, so a channel's cost basis cannot
   change across a three-year panel, making ROI denominators wrong on any long
   panel with real cost inflation. A correctness bug, already logged as deferred.
3. **Creative-level modeling and wear-out** — "is this creative worn out or is
   the channel saturating?" is currently unanswerable, and the two have opposite
   remedies.
4. **Reach & frequency depth** — the config landed in #141; optimal frequency,
   capping and incremental-reach curves did not.
5. **Cross-channel synergy reconciliation** — the continuous-learning module
   treats pairwise synergy as first-class with sign-informed priors and off-axis
   designs to identify it; the MMM path does not. Synergy should mean one thing
   across the codebase.

---

## Q2 2027 — `v2.0`, Contract cleanup & scale

**Epic [#193](https://github.com/redam94/mmm-framework/issues/193)**

A holding pen for what strict SemVer defers, kept visible so it is not
rediscovered: retained-but-dead enum values, the vestigial `TrendBuilder`, the
legacy adstock path, the `tests/synth` shim, the legacy sessions-DB fallback.

Entry rule: nothing lands here for being *tidier*. It lands because keeping it
costs something measurable — a real bug, a real support burden, or a real
performance floor.

Scale work is parked here too, with the same discipline: **profile before
optimizing.** The most promising lever is already identified (Kalman
marginalization, #191), which is a reason to sequence it after Q4 rather than
guess at bottlenecks now.

---

## What this roadmap deliberately does not contain

- **Regime-switching / Markov-switching models.** No evidence of demand.
- **Replacing the Bayesian path as the default.** The frequentist work is an
  alternative for speed and constraints, not a new centre of gravity.
- **Frequentist support for the extension models.** Their bespoke graphs are not
  linear given fixed transforms.
- **Deep-learning attribution.** Out of scope for a framework whose value is
  interpretable causal structure with honest uncertainty.

## Keeping this current

Update this file in the same PR that opens or closes a theme epic. If a quarter
slips, move the milestone date and say why here — a roadmap that quietly
retcons its own dates is worth less than no roadmap.
