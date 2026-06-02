# Critique: MMM Framework as a *Causal* Marketing Mix Modeling Platform

**Scope.** This document critiques `src/mmm_framework/` specifically against the goal of being a **causal** marketing MMM framework — not a generic code review. It credits what exists, identifies what is missing, and ends with a prioritized task list (P0/P1/P2) anchored to specific files.

**Date:** 2026-06-01 · **Reviewed against:** `main` working tree.

---

## 1. Executive summary

The framework is a **well-engineered Bayesian regression MMM with an unusually rich layer of causal *scaffolding* sitting on top of an estimation engine that is blind to it.** The central problem is not "this isn't causal" — *no* observational MMM (Robyn, Meridian, Nielsen, etc.) is causal absent identification assumptions plus experimental calibration. The problem is the **gap between the causal machinery the framework advertises and what the likelihood actually does**:

- A real backdoor-identification module exists (`dag_model_builder/identification.py`) and computes adjustment sets — but **the model-fitting path never reads them**. Identification is logged as an assumption, not enforced in estimation.
- Lift/incrementality tests are supported **only as a post-hoc Pass/Fail calibration check** (`validation/validator.py::_run_calibration`), not as **informative priors** — which is the single most important mechanism for making an MMM causal.
- Variables are typed as `KPI/MEDIA/CONTROL/AUXILIARY` with **no distinction between confounder, mediator, collider, or precision control**, so the framework cannot prevent the classic "bad control" bias it documents in its own README.

Crucially, **the core confounder in MMM is *unobserved demand*** (marketers spend more when they expect more sales). No adjustment set can fix an unobserved confounder, so even fully wiring the DAG into fitting would **not** make the framework causal in the way that matters. The route to causal validity runs through **experiment-calibrated priors and quasi-experimental geo variation**, not through more DAG plumbing.

**Bottom line:** the framework has *more* causal infrastructure than most MMM tools, but it is currently decorative. The work ahead is to **connect the scaffolding to the engine** and to **anchor effects in experimental evidence**.

---

## 2. What the framework already gets right (credit where due)

These are genuine strengths and should be preserved:

| Strength | Where | Why it matters causally |
|---|---|---|
| Real backdoor identification (collider logic, path enumeration, adjustment-set proposal) | `dag_model_builder/identification.py` | Most MMM tools have **no** identification layer at all. The graph logic is correct (d-separation, collider rules). |
| Mediation models (media → mediator → KPI) with direct/indirect decomposition | `mmm_extensions/models/nested.py` | Mediation is a genuinely causal construct; front-door-style reasoning is possible here. |
| Assumption logging / versioning + a canonical "scientific workflow" | `agents/causal_tools.py`, `agents/graph.py` | Auditability and pre-registration *intent* are real differentiators. |
| Bayesian uncertainty, hierarchical pooling, PPC, convergence diagnostics | `model/base.py`, `validation/` | Honest uncertainty quantification is a precondition for credible causal claims. |
| Lift-test data structures + calibration *check* | `validation/validator.py:1588`, `validation/results.py` (`LiftTestComparison`) | The hooks to ingest experiments exist — they just need to be promoted from "check" to "prior." |
| Parametric adstock (geometric/delayed/Weibull) with continuous estimation | `transforms/adstock_pt.py`, `model/base.py` | Flexible carryover shapes reduce one source of functional-form misspecification. |
| Documented awareness of confounder-vs-precision-control bias (with a proof) | `README` | The conceptual understanding is present; only the enforcement is missing. |

---

## 3. Detailed critique

### 3.1 The estimation engine is blind to the causal graph *(architectural, high severity)*

`identification_report` and `adjustment_set` are referenced **only** in the agent layer (`agents/causal_tools.py:35–36, 360–391`) and defined in `dag_model_builder/identification.py`. A search of the fitting path (`model/`, `mmm_extensions/models/`) finds **zero** references to identification or adjustment sets. The DAG is translated to an `MFFConfig` (node lists + per-node priors) by `dag_model_builder/config_translator.py`. Edges *do* survive enough to drive model-type resolution and mediator wiring (`model_type_resolver.py`, `dag_to_nested_config`), but **for the base model's confounding adjustment the edge topology is effectively discarded** — the adjustment set that the identification module computes is never consulted by the likelihood.

Consequences:
- The computed adjustment set is recorded as an assumption (`record_assumption(..., category="identification")`) but never constrains which variables enter the regression.
- Nothing stops a user from including a **mediator** or **collider** as a "control," which *induces* bias rather than removing it — the exact error the README warns against.
- `identifiable: True/False` is reported to the user as if it characterizes the fitted model, when the fitted model never used the result.

This is the highest-leverage architectural gap: the framework's most distinctive causal feature is currently inert.

### 3.2 The dominant MMM confounder — unobserved demand — is unaddressed *(conceptual, high severity)*

The core threat to MMM causality is **simultaneity / targeting**: budgets are set in anticipation of demand, so spend is correlated with the unobserved "expected demand" that also drives sales. The model (`model/base.py`) treats all media as **exogenous**: media enters the likelihood after adstock/saturation with a positive `Gamma` coefficient prior, and controls enter as ordinary linear regressors (`beta_controls ~ Normal(0, 0.5)`). There is:

- No instrument, no control-function/2SLS, no proxy for latent demand;
- No feedback term (lagged outcome → spend) and no acknowledgment that such feedback violates the likelihood;
- No sensitivity analysis bounding how strong an unobserved confounder would need to be to overturn a conclusion (e.g., E-values, Cinelli–Hazlett sensitivity).

The DAG layer implicitly assumes **no unobserved confounding (ignorability)** — which in MMM is usually false. This should be stated loudly to users, and it reframes the priority list: *adjustment alone cannot rescue identification here.*

### 3.3 Experiments are validated against, not learned from *(the #1 missing lever, high severity)*

`_run_calibration` (`validation/validator.py:1588`) compares fitted estimates to external lift tests and reports a coverage/`calibrated: Pass/Fail` verdict. This is useful but backwards relative to the state of the art (Meridian, Robyn): the gold standard is to fold **geo-lift / incrementality results into informative priors on channel ROI/β before or during fitting**, so the observational model is *anchored* to randomized evidence. As implemented, experiments can tell you the model is wrong but cannot make it right. This is the single highest-value causal feature to add, and it directly mitigates §3.2 (a randomized lift test estimates the causal effect *despite* unobserved demand).

### 3.4 No causal role typing → "bad control" cannot be prevented *(design, medium-high severity)*

`VariableRole` (`config/enums.py`) is `KPI/MEDIA/CONTROL/AUXILIARY`. There is no way to mark a control as **confounder** (must always be included, never shrunk), **precision control** (include for efficiency; safe to select/shrink), **mediator** (must *not* be conditioned on for a total-effect estimate), or **collider** (must *not* be conditioned on, ever). `ControlVariableConfig` has only `allow_negative` and `use_shrinkage`. Because all controls become one undifferentiated `beta_controls` block (`model/base.py`), the framework cannot:
- guarantee confounders are included with un-shrunk priors (the README proves shrinking a confounder biases β by `(1−s)·γ·Cov(X,C)/Var(X)`), or
- warn when a user conditions on a mediator/collider.

The knowledge exists in prose; it needs to become a type the system can enforce.

### 3.5 Weak identification from channel collinearity *(statistical-causal, medium-high severity, under-recognized)*

Brands tend to spend on all channels simultaneously and scale budgets together, so individual channel coefficients are often **weakly identified regardless of confounding** — the data cannot separate near-collinear channels. The framework has `channel_diagnostics` (VIF/correlation) but treats this as a fit diagnostic, not an identification threat, and offers no remedy (e.g., hierarchical shrinkage across related channels, informative priors, or explicit reporting of pairwise non-identifiability). This compounds with §3.6.

### 3.6 Adstock / saturation / β equifinality *(identifiability, medium severity)*

Long-carryover-weak-saturation and short-carryover-strong-saturation can produce nearly identical fits; baseline-vs-media is similarly entangled. Priors on adstock, saturation, and coefficients are independent, with no joint regularization or identifiability-oriented parameterization. **Note this interacts with recent changes:** `normalize=True` on the new parametric adstock folds total magnitude into the coefficient, so adstock shape, saturation, and β trade off even more freely. Worth documenting and, ideally, constraining (e.g., anchor saturation `kappa` to data percentiles, which the config already hints at via `kappa_bounds_percentiles`).

### 3.7 Geo variation is treated as nuisance, not signal *(missed opportunity, medium severity)*

Cross-geo variation in spend is **quasi-experimental** identification signal and is the backbone of Meridian-style geo MMM. Here, geo effects are modeled as hierarchical *nuisance* offsets (`geo_sigma`, `geo_offset` in `model/base.py`) for pooling, not exploited as a source of identifying variation (e.g., geo-level treatment heterogeneity, geo-based holdout structure, or differential-spend identification). Leaning on geo variation is one of the few credible observational routes to causal effects.

### 3.8 Validation tests fit, not causal validity *(rigor, medium severity)*

`validation/` is comprehensive on **statistical** grounds (R̂, ESS, divergences, PPC, residuals, LOO/WAIC, CV, prior-sensitivity, stability) but has **no causal refutation suite**: no placebo treatment, no negative-control outcome, no random-common-cause injection, no data-subset refutation, no E-value/sensitivity-to-unobserved-confounding. Good fit + tight intervals + passing PPC can all coexist with severe confounding bias. The DoWhy "refute" pattern is the natural model to emulate.

### 3.9 Counterfactual semantics are valid only under un-tested assumptions *(interpretation, medium severity)*

`compute_counterfactual_contributions` (delegating to the model) and `what_if_scenario` compute predictions under altered spend and **do** carry HDIs on contributions — good. But this equals a genuine `do(spend)` intervention *only if* the model is correctly specified with no unobserved confounding — precisely what §3.2 says is usually false. The marginal-ROAS path is also weaker than the contribution path: `MarginalAnalysisResult` (`analysis.py:48`) stores `marginal_roas` and `marginal_contribution` as **bare floats with no credible interval**, so the headline efficiency number is reported without uncertainty. Counterfactuals should be labeled as model-conditional and marginal ROAS should carry posterior intervals.

### 3.10 Pre-specification is rhetorical, not enforced *(process, medium severity)*

The README and `agents/causal_tools.py` (`define_research_question`, `define_analysis_plan`) emphasize pre-registration to reduce researcher degrees of freedom, but assumptions are **logged, not locked**: a user can register a plan, see the result, edit the spec, and refit, with no diff/warning and no record that the reported model diverged from the pre-registered one. Without a freeze + diff mechanism, the anti-"garden-of-forking-paths" claim is aspirational.

### 3.11 Reporting does not disclose causal assumptions *(communication, medium severity)*

Reports (`reporting/sections.py`) communicate uncertainty well but present channel ROI/contributions without a **causal-assumptions section**: which variables were assumed to be confounders, what identification strategy is claimed, the no-unobserved-confounding/SUTVA caveats, and sensitivity bounds. For a tool branded as causal, stakeholders need to see the assumptions the number rests on, not just its credible interval.

---

## 4. What is missing (summary)

1. **Experiment-calibrated priors** (geo-lift/incrementality → priors on ROI/β). *Absent — only post-hoc checking.*
2. **DAG → estimation wiring** (use the adjustment set; block bad controls). *Absent — identification is inert.*
3. **Causal role typing** for variables (confounder/mediator/collider/precision). *Absent.*
4. **Sensitivity to unobserved confounding** (E-values, partial-R² bounds). *Absent.*
5. **Causal refutation suite** (placebo, negative control, random common cause, subset). *Absent.*
6. **Endogeneity tooling** (instruments / control-function / proxy-for-demand). *Absent.*
7. **Geo-based identification** (treat cross-geo spend variation as signal; geo holdouts). *Partial — geo is nuisance only.*
8. **Weak-identification handling** for collinear channels (hierarchical priors, explicit reporting). *Partial.*
9. **Pre-spec lock + diff** to make pre-registration enforceable. *Absent.*
10. **Causal-assumptions reporting section** + marginal-ROAS uncertainty. *Absent / partial.*
11. **Frontdoor / IV identification** in the DAG layer (currently backdoor-only, per `identification.py:4`). *Absent.*

---

## 5. Prioritized tasks

Effort: **S** ≈ days, **M** ≈ 1–2 weeks, **L** ≈ multi-week. Priority reflects *causal impact*, not ease.

### P0 — Without these, "causal" is not defensible

| # | Task | Files / anchors | Effort |
|---|---|---|---|
| P0-1 | **Experiment-calibrated priors.** Promote lift tests from a post-hoc check to informative priors on channel ROI/β (and ideally saturation). Support a two-stage flow: fit → set priors from incrementality results → refit; or a single fit with experiment-derived priors. | new `calibration/` module; consume existing `LiftTestComparison`; wire into `model/base.py` media block + `config/` (e.g., `MediaChannelConfig.roi_prior`) | **L** |
| P0-2 | **Unobserved-confounding sensitivity + honest framing.** Add E-values / partial-R² (Cinelli–Hazlett) sensitivity for each channel; surface a prominent "identification rests on no-unobserved-confounding" caveat in reports and the agent workflow. | `validation/`, `reporting/sections.py`, `agents/graph.py` | **M** |
| P0-3 | **Causal refutation suite.** Implement placebo treatment, negative-control outcome, random-common-cause injection, and data-subset refutation; fail loudly when a "null" test is non-null. | `validation/validator.py` (add `_run_causal_refutation`), `validation/results.py` | **M** |

### P1 — Connect the scaffolding; prevent foot-guns

| # | Task | Files / anchors | Effort |
|---|---|---|---|
| P1-1 | **Wire the adjustment set into fitting.** Pass the identification result through `config_translator.py` into the model; *enforce* it: include the adjustment set, and **refuse/warn** when a mediator or collider is used as a control. Reframe value as bad-control avoidance, not "now it's causal." | `dag_model_builder/config_translator.py`, `dag_model_builder/builder.py`, `model/base.py` | **M** |
| P1-2 | **Causal role typing.** Extend `VariableRole`/`ControlVariableConfig` with confounder / precision-control / mediator / collider; route confounders to an un-shrunk prior block and exclude them from selection; block conditioning on mediators/colliders. | `config/enums.py`, `config/variables.py`, `model/base.py` (`beta_controls` block) | **M** |
| P1-3 | **Marginal-ROAS uncertainty.** Make `MarginalAnalysisResult` carry posterior intervals (propagate saturation/coefficient uncertainty), not point estimates. | `analysis.py:24–48`, model marginal computation | **S** |
| P1-4 | **Equifinality guardrails.** Document the `normalize=True` adstock↔β trade-off and add a joint adstock/saturation identifiability note + optional weakly-informative joint priors. For channels using Hill saturation, anchor `kappa` to data percentiles (`SaturationConfig` already exposes `kappa_bounds_percentiles`). Note: the **core `model/base.py` uses logistic saturation** (`sat_lam`, no `kappa`), so the percentile-anchoring applies to the Hill path, not the default fit. | `model/base.py`, `config/transforms.py`, docs | **S–M** |

### P2 — Strengthen identification and process

| # | Task | Files / anchors | Effort |
|---|---|---|---|
| P2-1 | **Geo-based identification.** Exploit cross-geo spend variation as identifying signal; support geo holdout structures and report geo-level treatment heterogeneity rather than pooling it away. | `model/base.py` (geo effects), new geo-experiment config | **L** |
| P2-2 | **Weak-identification handling.** Detect collinear channel clusters; offer hierarchical/grouped priors and explicitly report pairwise non-identifiability instead of overconfident per-channel ROIs. | `validation/channel_diagnostics.py`, `model/base.py`, `reporting/` | **M** |
| P2-3 | **Pre-spec lock + diff.** Freeze the registered config and warn/annotate when the fitted/reported model diverges from the pre-registered spec. | `agents/causal_tools.py`, `config/` | **S–M** |
| P2-4 | **Causal-assumptions report section.** Add a section listing assumed confounders, identification strategy, SUTVA/no-unobserved-confounding caveats, and the P0-2 sensitivity bounds. | `reporting/sections.py`, `reporting/extractors/` | **S** |
| P2-5 | **Frontdoor / IV in the DAG layer.** Extend `identification.py` beyond backdoor (front-door for the mediation models; IV when an instrument is declared). | `dag_model_builder/identification.py` | **M–L** |
| P2-6 | **Optional: integrate a vetted causal library** (DoWhy/EconML/pgmpy or `networkx` d-separation) to cross-check the hand-rolled graph logic rather than maintain it alone. | `dag_model_builder/`, `pyproject.toml` | **M** |

---

## 6. A note on sequencing

Do **P0-1 (experiment calibration)** first. It is the only item that directly attacks the dominant confounder (unobserved demand), it leverages infrastructure that already exists (`LiftTestComparison`), and it is the feature that distinguishes credible modern causal MMM (Meridian/Robyn) from regression-with-adstock. P1-1 (adjustment-set wiring) is valuable but should be framed honestly as **bad-control prevention** — it does not, by itself, make the framework causal. Everything in P2 raises the ceiling once the P0/P1 foundation makes causal claims defensible.

---

*Claims above are anchored to the code as of review. Architectural findings (identification not wired into fitting; calibration is post-hoc; no causal role typing) were verified against source; implementation-level specifics are phrased "as implemented / based on current code" so they self-correct if the code changes.*
