# Documentation Review: Selling the MMM Framework to Media Measurement Consultancies

**Date:** 2026-06-12
**Scope:** All ~57 pages under `docs/` (entry/positioning, methodology, workflows, the four notebook-guide series, example report, site infrastructure), reviewed from the perspective of a media measurement consultancy doing technical and commercial due diligence — the analytics director, the PhD methodologist, the practice lead, the training lead, and the senior data scientist who would each evaluate this before recommending adoption for client engagements.

---

## 1. Executive Summary

This is one of the most methodologically serious MMM documentation sets in the open-source space. The pressure-testing program (16 synthetic worlds with known causal truth, published silent failures, the framework's own bugs disclosed with before/after numbers), the EIG/EVOI calibration loop with real math and a correct submodularity proof, and the consistent "green diagnostics validate the computation, not the causal claim" doctrine clear a bar that Robyn and Meridian documentation does not attempt. The honest-uncertainty voice is consistent, the audience tiering (business / analyst / methodologist) is deliberate, and the writing quality is high.

**The credibility risks are almost entirely self-inflicted polish failures, not methodology failures.** They cluster into five categories:

1. **Docs–code drift.** Multiple prominently displayed code blocks call APIs that do not exist. For a framework whose brand is reproducibility, this is the first thing a due-diligence engineer finds.
2. **Internal contradictions.** The same quantity is defined differently on sibling pages (saturation functions, default priors, contribution-sum conventions, MFF format, headline error numbers), and there is at least one genuine statistical error and one 1000× units bug.
3. **Unsourced numbers on pages that preach sourcing.** The site's entire thesis is "don't report numbers you can't defend," yet several load-bearing pages report invented or uncited statistics typeset as facts.
4. **The evaluating-buyer persona has no page.** License, version, PyPI status, maintainer, security posture, runtime, data requirements, and support expectations must currently be assembled from five pages, two of which contradict each other.
5. **No real-data path and no competitive benchmark.** Every workflow starts from a synthetic world with an answer key; no page shows messy-client-data onboarding, no forecast-accuracy evidence exists anywhere, and the single most persuasive artifact possible — running Robyn/Meridian through the existing stress harness — has not been built.

All of these are fixable without changing the methodology. The prioritized plan is in §8.

---

## 2. What the Framework's Docs Do Well (Keep and Amplify)

These are genuine differentiators a consultancy can resell. Do not dilute them while fixing the rest.

- **The pressure-testing program is the strongest sales asset on the site.** Publishing "8 silent failures — wrong attribution, green diagnostics, r-hat ≤ 1.02 on every one," disclosing the cross-geo adstock bug the harness itself found, and labeling still-open defects (legacy adstock default ~28% error vs 7% parametric; Hill holdout fallback) is adversarial self-evaluation no vendor does. The estimand discipline (truth and estimate graded on the same counterfactual zero-out) pre-empts the standard "you graded the model on a question it wasn't asked" defense.
- **The honest competitor framing on the homepage** names Robyn, Meridian, and PyMC-Marketing specifically, characterizes them fairly, and differentiates concretely (Robyn's Pareto-front selection as institutionalized specification shopping; Meridian has calibration but not the planning loop). This is exactly the comparison a consultant must repeat to a CMO.
- **The measurement-loop thesis (T₀–T₅)** gives the product a narrative beyond "another Bayesian MMM library": the model prices which experiment to run next, pre-registers it, and folds the result back in. The calibration math (conjugate update, EIG closed form, complete submodularity proof with the Nemhauser–Wolsey–Fisher greedy guarantee, delta-method power ceiling) is real and mostly correct.
- **Pearl-correct causal exposition** (SCM, d-separation, do-calculus, backdoor/frontdoor) tied to MMM operations, with real references. Far above industry norm.
- **The variable-roles doctrine** (confounders always, mediators never, colliders never; selection priors on precision controls only) including a `ValueError` when a mediator is misused — a differentiator no major MMM vendor ships.
- **The geo experiment design section** (information-yield geo ranking, Mahalanobis matching on MMM posteriors, saturation-slope-aware spend increments, simulation-based power, spillover exclusion) reads like a real test-design memo — the most "real client work"-shaped artifact on the site.
- **Provenance discipline where it exists:** "Measured (from baked notebook)" vs "Illustrative (seeded, in-browser)" labels, seeded asserts encoding quoted numbers, a recorded scorecard file. This convention should become site law (see §5.3).
- **Pedagogy.** The Workshop series is best-in-class for analysts who know regression but not Bayes (live R-hat/ESS on chains the reader just broke; Jensen's-inequality plug-in-vs-per-draw demo). The Aurora series carries one client engagement with a sealed answer key across six chapters and teaches failure first. The two series are complementary, not redundant.
- **Maturity signaling in the changelog:** API-stability tiers by module (Stable/Beta/Experimental) and a pre-1.0 warning box are exactly what an engineering reviewer wants and few OSS projects provide.
- **Honest-uncertainty client language** throughout: confidence tiers (validated > narrow-unvalidated > wide-unvalidated), the traffic-light CI framework, "a tight number is not a true number," objection-handling scripts in the FAQ.

---

## 3. Critical Defects (Fix First — These Actively Damage Credibility)

### 3.1 Code samples that don't run (docs–code drift)

A diligence engineer will paste these into a session within the first hour. Verified-missing against `src/mmm_framework`:

| Page | Fictional API | Reality |
|---|---|---|
| `causal-inference.html` | `model.compute_contributions(method="counterfactual", baseline="zero")` | Actual API is `compute_counterfactual_contributions` (`analysis.py`) |
| `bayesian-workflow.html` | `SaturationConfigBuilder().with_lam_prior(...)`, `panel.split(holdout_periods=8)`, `model.compute_coverage(...)`, `model.compute_channel_lift("tv", scenario=...)` | None exist; this is the validation code block — the part most likely to be tried |
| `modeling-guide.html` | `panel.missing_summary()`, `results.roi(channel)`, `results.saturation_level(channel)`, `mmm.sample_prior_predictive` | Not on those objects (live only in `agents/model_ops.py` / reporting helpers / extractors) |

**Fix:** CI-test every documentation code block against the installed package (extend the `nbs/` seeded-assert pattern to doc snippets), or explicitly mark blocks as pseudocode. This is the cheapest, highest-ROI fix in the entire review.

By contrast, `getting-started.html`, `causal-features-showcase.html`, and the workflow-channel-effectiveness/budget snippets were verified real — proof the team can do this; it just isn't enforced.

### 3.2 Mathematical and statistical errors

1. **"Logistic saturation" means three different things across the site.** `technical-guide.html` Eq. (4) defines "logistic" as f(x) = 1 − exp(−λx) (this is exponential saturation); `bayesian-workflow.html` uses the shifted sigmoid 1/(1+e^{−λx}) − 0.5; `math-06` uses the unshifted sigmoid 1/(1+e^{−λz}). The config name `SaturationConfig.logistic()` is itself the misnomer. Pick one taxonomy, footnote the legacy name, reconcile all pages.
2. **The prior-elicitation worked example is unsolvable under its own formula** (`bayesian-workflow.html`): with the shifted logistic (range [0, 0.5]), f(1M) = 0.7 is impossible; the code solves the unshifted form. The page meant to teach prior elicitation contradicts itself algebraically.
3. **AR(1)/compound-symmetry design-effect error** (`measurement-calibration.html`, Step 4): declares ρ an AR(1) coefficient but applies the exchangeable design effect (1+(T−1)ρ). Under AR(1) the inflation for a T-week mean is ≈ (1+ρ)/(1−ρ) at large T; the stated formula badly overstates required test duration. Anyone who knows clustered SEs catches this immediately.
4. **math-06 contradicts math-05 on a load-bearing fact:** the extensions' shared transform 1/(1+e^{−λz}) gives f(0) = 0.5 ≠ 0, while math-05 §2 makes sat(0) = 0 the foundation of the contribution estimand ("that one fact powers everything below"). Either the extensions shift/rescale the sigmoid (document it) or the estimand carries an offset — say which.
5. **Aurora-04 units bug (1000×):** executive summary quotes "$14.7K (80% CI [$5.6K–$25.2K]) … 16.9% of $87.3K total revenue" where every other page uses $87.3M. A board-deck chapter with a thousand-fold units slip is the worst place for one.
6. **Beta(3, 1.5) "mode at ~0.7"** (`modeling-guide.html`) — the mode is 0.8. Small, but it sits in the section teaching prior care.
7. **EVOI linearization (Eq. 8, measurement-calibration)** is asserted with no derivation, no error bounds, and is dimensionally loose; Eq. (4)'s experiment-variance approximation is similarly underived while in tension with the serial-correlation discussion two steps later.
8. **Proportion mediated (technical-guide Eq. 9)** can exceed 1 or go negative with opposite-signed effects — no caveat.
9. **math-00 prior table** glosses β ~ Gamma as "encourages ROI > 1×," conflating the standardized max-contribution ceiling with ROI — a line a stats reviewer will poke first.

### 3.3 Internal contradictions across pages

| Contradiction | Pages | Resolution |
|---|---|---|
| Contributions "sum to 100% of the modeled outcome" vs. counterfactual contributions of saturating channels "typically exceed total minus baseline — expected and correct" | interpreting-results vs. causal-inference | Define the decomposition convention once (normalized vs. raw counterfactual + interaction residual); state it both places |
| Default adstock prior: Beta(3,3) vs Beta(1,3) vs Beta(3,1.5)/Beta(2,3) examples | bayesian-workflow vs technical-guide vs modeling-guide | Generate prior tables from `config.py` at bake time — single source of truth |
| Prior tables describe a continuous learned α while the shipped default is a blend weight over a fixed decay bank (and pressure-testing says the default underperforms 28% vs 7%) | bayesian-workflow vs technical-guide vs pressure-testing | Either make parametric adstock the default or document why not, and make the reference tables describe what actually ships |
| MFF format: long format (VariableName/VariableValue rows) vs. wide format ("columns for date, outcome, media spend per channel") | getting-started vs glossary | Long format is correct per the code; fix the glossary |
| Confounded-Search headline: +111% / +110% / +69% / ~+44% / +153% for the same world across pages and eras | stress-00, stress-03, stress matrix, notebooks | Canonicalize: one current-defaults number per world in hero blocks, history demoted to footnotes |
| NumPyro speedup "4–10×" vs "5–10×" | getting-started vs glossary | Pick one (with a stated baseline) |
| Cannibalization narrative: aurora-00/05 say ch. 03 "exposed Cold Brew cannibalizing Original"; aurora-03's actual finding is ψ ≈ −0.0003, negligible direct effect, shared demand wave | aurora-00/03/05 | Rewrite 00's table cell and 05's recap to match 03's (correct, honest) verdict |
| Competitor media: blanket "confounder" vs "precision control (if it doesn't affect your spending)" | variable-selection vs modeling-guide | State the conditional explicitly in both |
| Holiday as selection-eligible control with a PIP vs. holidays as core confounder structure | variable-selection vs modeling-guide | Clarify minor-holiday vs structural-holiday distinction |
| Radio "robustly profitable (1.6)" vs Radio "likely unprofitable (0.82)" — different fictional datasets, never flagged | scientific-workflow-demo vs mmm-example-report | Unify the fictional dataset, or badge each page's world |
| Example report: $116M attributed / blended ROI 1.26 in the exec summary vs $127M / 1.38 in its own simulator at current allocation | mmm-example-report | Reconcile — clients catch report-internal arithmetic on first read |
| Negative-ROI FAQ advises "check whether the estimate is robust across model specifications" — the exact behavior the site forbids | interpreting-results | Reword to "pre-specified sensitivity analysis" |
| Report footer "Model version 2.1.0 \| Report template 1.0" vs package version 0.1.0 | mmm-example-report | Fix the fictional version string |

### 3.4 Unsourced numbers on pages that preach sourcing

The site's central doctrine makes these stand out more, not less:

- **business-stakeholders.html metric cards:** "30–50% ROI Overestimation," "5–15% Budget Misallocation," "<50% Actual Reliability" — presented as facts, zero citations. The page whose thesis is "don't report numbers you can't defend" reports four numbers it can't defend. The decorative risk meters ("Client trust erosion 85%") compound it.
- **scientific-modeling.html:** "Industry studies suggest two-thirds of uncalibrated MMM estimates require significant adjustment" — uncited.
- **measurement-calibration.html / workflow-calibration-decisions.html:** "Mahalanobis matching reduces post-period control-group variance by 20–40% … in typical retail and DTC datasets"; "EIG flattens after 6–10 weeks"; the −92%/+25pp loop-benefit panel is simulated but typeset like measured results.
- **interpreting-results / modeling-guide:** "TV: 6–8 weeks carryover," "industry meta-analysis showing 4–8 week half-lives" — the meta-analysis is never named.
- **Calibration decay half-lives** (digital ~6 months, TV ~1 year) stated as empirical fact on business-stakeholders, as λ defaults on platform-overview — reconcile and label.

**Fix:** adopt the pressure-testing **Measured / Simulated / Heuristic** labeling convention site-wide; source what can be sourced (the lift-vs-MMM calibration literature exists), reframe the rest as illustrative or derived-from-pressure-tests with links, and delete what can't be defended.

### 3.5 Deployment-broken and stale infrastructure

- **`demos.html` report-template gallery links to `../examples/…` (7 links)** — outside the published root if the site deploys from `/docs` (the canonical URLs imply GitHub Pages), so all seven 404 in production, for exactly the deliverables-focused visitor that section targets. Copy the reports into `docs/`.
- **`docs/api/` contains no built API reference** — only the Sphinx skeleton. "Build it yourself" on the API card is a maturity red flag; publish the built HTML.
- **Stale `tests/synth/` paths** across mmm-walkthrough, workshops 03–05, and the stress pages — the package moved to `src/mmm_framework/synth/` (tests/synth is a shim).
- **`sitemap.xml` lastmod dates contradict actual file edit dates** (several pages edited 2026-06-12 list January lastmods).
- **~430 lines of commented-out dead CSS** shipped on `index.html`.
- **FAQ meta/OG description belongs to a different page.**
- **Unescaped `<` characters** in mmm-example-report source (`(ROI < 1.0)`).
- **"Pin `mmm-framework==0.1.0`"** (changelog) is an instruction that cannot be followed — the package is not on PyPI, and getting-started never acknowledges this.
- **Zero hyperlinks to the notebooks** from any of the 26 guide pages (`href="nbs/` → 0 matches): the runnable material the site constantly references is unreachable from the published site.

---

## 4. The Missing Buyer: Gaps for a Consultancy Evaluator

### 4.1 No "evaluator's page" (procurement diligence)

A consultancy comparing this against Robyn/Meridian/an incumbent vendor needs, on one URL: **license** (never named anywhere on the site), version + release date (the only release is undated), PyPI/packaging status, maintainer and governance (the site reads as a solo project but never says so — concealment-by-omission is worse than disclosure with mitigations), support expectations, security posture, and a roadmap to 1.0. Today this must be assembled from five pages, two of which contradict each other.

### 4.2 No data requirements specification

The first question every consultancy asks — minimum history length, granularity (weekly/daily), channel-count guidance, geo counts needed for hierarchical models, spend vs. impressions as the media input, MFF column dictionary beyond the toy example — is answered nowhere. (Fragments exist: modeling-guide's "104+ weeks, <50% zero-spend" checklist; the contradictory MFF definitions.) This page also resolves the glossary/getting-started format contradiction.

### 4.3 No runtime/scalability evidence

The computational-scaling table gives multipliers with no baseline ("what is 1×?"). No wall-clock benchmarks by data shape × model type × sampler, no hardware guidance, no "this quickstart fits in ~N minutes on a laptop." The walkthrough's 83-second compute-budget note proves the team can do this.

### 4.4 No real-data path

Every demonstrated workflow starts from a synthetic world with an answer key. Missing entirely: messy CSV → MFF assembly (`MFFLoader`) → `mmm_framework.eda` QA → fit → report, with a sidebar on **which checks substitute for ground truth when none exists** (holdout fit, lift-test agreement, split-window refit stability, the contraction/overlap/shift audit). The site also never preempts the obvious objection ("of course it works on data you generated") with the one-paragraph defense of known-truth evaluation.

### 4.5 No forecast/backtest accuracy evidence

`workflow-forecasting.html` recommends tracking forecast accuracy but presents none — no rolling-origin backtest, no holdout MAPE, no interval-coverage check anywhere in the docs. Clients ask "how accurate were last year's forecasts?" and the docs have no answer. Worse, the page's code shows in-sample/what-if re-prediction while the narrative promises out-of-time forecasting. The canonical modeling-guide Phase 3 ("Validate") contains **no out-of-sample holdout step at all**.

### 4.6 No competitive benchmark

The harness to run Robyn/Meridian/vanilla PyMC-Marketing through the 16 synthetic worlds already exists, and its absence is conspicuous. Even a partial run (3–4 worlds) would be the single most persuasive artifact possible, converting the stress series from "a list of this framework's wounds" into a moat ("these failures are generic to additive MMMs; we're the only ones who ship the harness"). At minimum, argue the genericity claim explicitly in one hub paragraph.

### 4.7 No security / AI-governance documentation

The platform ships an AI agent that executes code and calls LLMs, and the docs are silent on: which LLM providers see what data, the local-model (LM Studio) privacy implication, the sandboxed-kernel guarantees, audit logging, authentication status (platform-overview quietly admits roles "are not gatekeeping logins"), multi-client isolation, data residency. This is the client InfoSec questionnaire, unanswered. Much of the substance already exists in `technical-docs/` (hosted profile, kernel sandbox, env-scrub) — it needs a public-facing page.

### 4.8 No screenshots of the platform

`platform-overview.html` is a product tour of a web app with **zero images of the app**. Prose claims ("a stage ring shows…") are unverifiable; an evaluator can't gauge polish, and an unillustrated tour reads as vaporware. Annotated screenshots or a 2-minute video is the single biggest fix on that page. Also: the Beta label from the changelog never appears on the platform page.

### 4.9 No engagement playbook artifacts

The pieces a practice lead needs to run a paid engagement off these docs: an engagement timeline (which doc/step lands in which week of a 10–12 week engagement), per-phase deliverable artifacts (pre-registration memo template — §10 of the calibration page computes everything needed for one; EDA QA sign-off; robustness appendix; readout deck; one-page exec summary), experiment cost/duration benchmarks, and a data-onboarding checklist. The walkthrough's pre-modeling checklist and stress-05's 15-row decision table are 80% of a downloadable QA one-pager that currently exists only as HTML mid-page.

### 4.10 Training-program infrastructure

For the 26 guide pages (workshop/aurora/math/stress): no time-to-complete estimates (realistic: Workshop ≈ 6–7.5 h, Aurora ≈ 3.5–4.5 h reading-only), no up-front learning objectives (the takeaway boxes are 90% of the content, framed as exits not entries), no exercises/solutions, no assessment path (the sealed-answer-key device is a ready-made grading mechanism — "refit on the confounded world and explain the failure" — and it's unused), no prerequisites banners (Aurora assumes Workshop-level Bayes literacy and never says so; the only ordering signal lives on demos.html), and no notebook links (§3.5).

---

## 5. Page-Group Findings (Condensed)

### 5.1 Entry & positioning (index, about, getting-started, business-stakeholders, platform-overview, faq, glossary, demos, changelog)

- **index.html:** Sharp lead message and the fairest competitor section in the category — but no license, no version badge, no PyPI, no "who is behind this," and the comparison is prose-only (add the feature-by-feature table an evaluator can paste into a vendor-assessment deck). The "Contributions" demo tab shows hardcoded fake data unconnected to the simulation — wire it or cut it. Hero CTA underwhelms relative to the buried best assets (pressure-test scorecard, Aurora arc, example report).
- **about.html:** Honest and specific, but reads as an anonymous solo manifesto. Add "who's behind this" (name, background), an explicit sustainability/bus-factor statement with mitigations, a roadmap to 1.0, and citation guidance. De-duplicate the "industry inflection point" paragraph shared verbatim with business-stakeholders.
- **getting-started.html:** The strongest "this is real software" page (verified-runnable quickstart, diagnostics-first). Needs: install-status honesty (not on PyPI), expected fit runtime, a data-requirements section, hardware guidance; trim the drift-prone hand-maintained file tree. The example config's `analysis_period="Jan 2023 - Dec 2025"` vs 104 weeks of 2023 data is sloppy.
- **business-stakeholders.html:** Best client-education page (specification-shopping practices table, the 7 questions accordion, the confounding and calibration demos) — undermined by the unsourced metric cards and decorative risk meters (§3.4). Replace the invented stats with the Aurora case-study box ($11.9M/yr on the same budget, clearly labeled synthetic) — better evidence, already on the site. Attribute or cut the self-citation pull-quote.
- **platform-overview.html:** Concrete and credible on the loop and agent tooling; fatally unillustrated (§4.8); silent on deployment/auth/LLM data flow; missing the Beta label.
- **faq.html:** The best objection-handling page — add an "Adoption & Operations" block (license, install, runtime, data needs, support, migration path — the glossary's excellent parallel-run OLS-migration guidance belongs here); fix the wrong meta/OG description; the comparison table check-marks itself on every row and omits Robyn/Meridian — add them or drop the table for the (better) prose.
- **glossary.html:** Comprehensive and opinionated (the OLS-migration entry is gold and hidden). Fix the A–Z jump bar over a non-alphabetical list, the wrong MFF definition, and add ROAS, EVPI, baseline/contribution, halo.
- **demos.html:** The proof hub. Fix the seven deploy-broken template links; add a "recommended paths by audience and time budget" router ("skeptical evaluator with 30 min: scorecard → Gauntlet → Aurora 05"); add the one-paragraph "why synthetic ground truth" defusal; label the Q4 2025 report as synthetic.
- **changelog.html:** Excellent stability tiers; but one undated release proves no cadence, the pip-pin instruction is impossible, NestedMMM is marketed on the homepage without its Experimental badge, and the page isn't in the nav — the page selling maturity is the hardest to find.

### 5.2 Methodology (causal-inference, scientific-modeling, bayesian-workflow, measurement-calibration, variable-selection, pressure-testing, interpreting-results, technical-guide, modeling-guide)

The through-line is coherent and mutually reinforcing — causal question → declared roles → Bayesian workflow → demonstrated silent failures → experiments as corrective → confidence-tiered communication — with pressure-testing as the keystone that converts rhetoric into measured evidence. Beyond the §3 defects:

- **The identification assumptions are demonstrated but never stated formally.** Add a numbered "Identification Assumptions" box/page — conditional exchangeability given declared confounders, positivity/overlap (channels that never go dark have no support for do(X=0); extrapolation is prior-driven), SUTVA/no-interference for geo designs, correct functional form, sequential ignorability for mediation — each labeled testable/untestable with the framework feature that addresses it, linked from every methodology page. The site teaches the concepts but never signs the contract.
- **Mediation and cross-effects are sold harder than they're identified.** The nested section never states sequential ignorability; stress-04 shows the splits are prior-determined and technical-guide doesn't say so. The multivariate model regresses contemporaneous outcomes on each other — a simultaneity problem with no discussion.
- **Missing standard caveats:** LOO for autocorrelated weekly data (recommend leave-future-out); predictive adequacy ≠ causal adequacy needs one sentence in every predictive-checking section of scientific-modeling; "the model is built to credit your media only with the sales it actually caused" (interpreting-results) overclaims against pressure-testing's own +110% demonstration — add the assumptions conditional.
- **Missing modern tools the repo already has or gestures at:** SBC/parameter-recovery on the synth worlds; the prior→posterior contraction/overlap/shift diagnostic surfaced beyond one line; the "unobserved-confounding robustness value" used in the silent-failure gate is never defined anywhere; the lift-test→prior design-factor math (the framework's most defensible technical asset) is buried in math-05 instead of summarized on the calibration page.
- **Citation poverty outside causal-inference:** variable-selection has zero citations (Carvalho/Polson/Scott, Piironen & Vehtari, Westreich & Greenland for the Table-2 fallacy it correctly handles); measurement-calibration — the page with the most novel math — has no references block (Lindley 1956, Raiffa–Schlaifer, Chaloner–Verdinelli 1995, NWF 1978); the stress series has zero (Cinelli & Hazlett for RV, Jin et al. 2017, Chan & Perry 2017, Vaver & Koehler 2011); the math series has exactly one. ~15 citations total would cover everything.
- **interpreting-results:** define the saturation-percentage metric (80% of what?); fix the ROI > 1.0 = "profitable" revenue/profit conflation; foreground marginal vs average ROI (the suite's own signature lesson).
- **modeling-guide:** Phase 3 has no holdout step; the EDA module that exists precisely for Phase 2 is never mentioned; "Sampler: NumpyRo" typo in the flagship pre-registration artifact.
- **pressure-testing:** add a plain-HTML row-per-world table (currently hover-only on a JS chart); state explicitly that the 7% clean-world number is an upper bound (the DGP is drawn from the model's own structural family); stamp the page with the scorecard commit/date.

### 5.3 Workflows & deliverables (mmm-walkthrough, scientific-workflow-demo/-simple, causal-features-showcase, mmm-example-report, workflow-{budget,calibration,channel,forecasting})

- **mmm-walkthrough is the single most convincing page** (v1 fails with green diagnostics → DAG recovery → calibration; every number measured from a baked notebook; per-channel confidence labels in the deliverable table; an actual compute budget). Extend, don't dilute: add "on real data, this check becomes…" callouts, experiment cost/duration, and a sized budget-guidance table.
- **scientific-workflow-demo is the inverse:** nine steps, zero framework code, every number decorative and unlabeled as such, every gate passes (no failure branch), and no EDA step despite EDA being the walkthrough's headline lesson. Add the illustrative banner, a Step 0 (Data & EDA), one real code block per step, and "when this gate fails" boxes.
- **workflow-budget-optimization never shows an optimizer.** The shipped `planning/budget.py:optimize_budget()` — with per-channel floors/caps, total-budget changes, uplift HDI, P(positive uplift), allocation stability — is exactly what consultancies need and is undocumented on the page titled Budget Optimization. Documenting it with a `bounds={"TV": (0.8, 1.2)}` example is the highest-ROI single edit in the workflow set. Also: the marginal-ROI table has no intervals, violating site doctrine where it's most consequential; no halo/cross-effects discussion where it matters most.
- **workflow-calibration-decisions is the most differentiated workflow page** (misallocation cost quadratic in σ, λ-tradeoff portfolios, the CTV geo-design memo) but names none of the code that implements it. Add per-section "In the framework" boxes (`planning/eig.py`, `evoi.py`, `priority.py`, `design.py`; agent tools `compute_experiment_priorities` → `design_experiment_plan` → `preregister_experiment` → `record_experiment_readout` → `apply_experiment_calibration`) and a downloadable pre-registration memo template generated from §10's outputs.
- **workflow-channel-effectiveness:** good 2×2 decision table and verified APIs, but shallow causally — no "before you trust these numbers" box linking the walkthrough EDA checks, and no evidence-tier column. Its two illustrative output tables don't cohere with each other.
- **workflow-forecasting:** good target-probability framing; disqualifying absence of accuracy evidence (§4.5); the shown code is in-sample re-prediction, not out-of-time forecasting — demonstrate a genuine future-period forecast or scope the page honestly.
- **mmm-example-report:** right bones (CIs on every headline, validation-required section, proposed experiments by quarter) but: the internal $116M/$127M contradiction, the fictional 2.1.0 footer, no causal-assumptions section (which causal-features-showcase §13 claims reports "now carry," citing this page), "captures 94% of weekly variance" presented as a virtue against the site's own R²-proves-nothing doctrine, and no statement of whether it's generator output or a hand-built mock. Add evidence-tier labels per channel (lift-anchored / observational+DAG / provisional) — the walkthrough's best idea, absent from the actual example deliverable.
- **causal-features-showcase:** strong, verified, honest — needs a "standard engagement battery" sidebar (which of §1–13 run on every client model) and a rendered excerpt of the §13 report section.
- **The five workflow pages use five contradictory fictional datasets.** Unify on one "Acme" engagement whose numbers agree across demo, report, budget, effectiveness, and forecasting — making the suite read as one case study.

### 5.4 Guide series (workshop, aurora, math, stress)

Covered substantially in §3 and §4.10. Additional specifics:

- **Aurora:** flag the simulated-experiment circularity (calibration values set to ground truth — `value=float(aurora.true_roas[ch])`); footnote the cross-chapter refit drift (ch. 2's 600-draw vs ch. 4's 400-draw numbers); add "Prerequisites: Workshop 00–05" banners.
- **Workshop:** split or tier workshop-02 (the NUTS internals are a 2-hour page for beginners; mark §6 optional); reconcile mean/median ROAS presentation in workshop-05 §2 vs §4.
- **Math:** add the missing chapter — **math-07: panels & hierarchy** (geo/product offsets, non-centered parameterization, what intercept-only pooling does and doesn't buy) — the math companion to stress-06's worst decision-level failure; add a likelihood-alternatives note (log-KPI, heavy tails, heteroskedasticity — the most standard stats-team question, absent from the whole series); put the parametric-adstock priors in the math-00 reference card (it currently documents only the discouraged legacy configuration); comment on math-04's 2 uncommented divergences.
- **Stress:** consolidate the scattered bug confessions into one tracked **"Known open findings / limitations & roadmap"** page with remediation status — honesty without a timeline reads as risk, not rigor; extract stress-05's decision table + EDA pre-flight + fix ladder into a downloadable one-pager; link "buy a lift test" prescriptions to the framework's own experiment-design machinery; justify (or at least exhibit the false-alarm record behind) the 25%/50%/75% silent-failure gate thresholds.

---

## 6. Information Architecture & Site Mechanics

- **Flat 13-item nav omits key pages** — Changelog, Causal Inference, Bayesian Workflow, Variable Selection, Pressure Testing, Scientific Modeling are unreachable from global nav (orphan-discoverable only via homepage cards). Move to grouped dropdowns: **Learn / Methodology / Platform / Proof / Project**, and add Changelog to the footer.
- **No persistent role-based entry point** despite good audience tiering on individual pages. Add a "choose your path" strip (business sponsor / analyst / methodologist / evaluator) on the homepage and demos.
- **Near-duplicate pages dilute the path** (scientific-modeling vs scientific-workflow-simple vs scientific-workflow-demo) — differentiate their roles explicitly or merge.
- **Staleness is structural:** dozens of hardcoded measured numbers per page across stress/aurora/workshop/demos with nothing binding HTML to the recorded scorecard. Auto-generate measured numbers from `stress_matrix.md` / baked-notebook outputs at bake time and stamp pages with commit + date; add a CI check that page constants match notebook outputs so "Measured" stays true.

---

## 7. New Pages/Artifacts to Create

| Artifact | Purpose | Raw material that already exists |
|---|---|---|
| **Evaluator's page** (license, version, PyPI status, maintainer, governance, support, security posture, roadmap to 1.0) | Procurement diligence on one URL | changelog tiers, about.html |
| **Data requirements & runtime reference** (MFF dictionary, min history, channel/geo guidance, wall-clock benchmarks) | First question in every RFP | modeling-guide checklist, walkthrough compute budget |
| **Real-data end-to-end guide** (messy CSV → MFF → EDA → fit → report, with "what replaces the truth column" sidebar) | Closes the synthetic-only objection | `MFFLoader`, `mmm_framework.eda`, walkthrough Part 1 |
| **Competitive benchmark** (Robyn/Meridian/PyMC-Marketing on 3–4 stress worlds + a comparison table under index#compare) | The bake-off question every consultancy asks | the stress harness |
| **Security & AI-governance page** (LLM data flow, local-model option, kernel sandbox, audit logging, auth status) | Client InfoSec questionnaire | `technical-docs/agent-session-kernels-*`, hosted profile docs |
| **Identification Assumptions page/box** (numbered, testable vs untestable, mitigation per assumption) | The methodology contract, signed | causal-inference + pressure-testing content |
| **Known open findings / roadmap page** (legacy adstock default, Hill holdout fallback, intercept-only hierarchy — with status) | Converts scattered confessions into governance | stress/math disclosures |
| **Downloadable consultant artifacts:** diagnostic checklist one-pager, pre-registration memo template, data-onboarding checklist, one-page exec summary template, engagement timeline map | Engagement playbook | stress-05 decision table, calibration §10, walkthrough checklist, analysis_plan.py docstring |
| **math-07: panels & hierarchy** | Biggest stats-team question with no answer | stress-06, model/base.py hierarchy code |
| **Backtest/accuracy section** (rolling-origin MAPE + interval coverage, even on the walkthrough world) | The first proof point procurement requests | synth worlds, PredictionResults |

---

## 8. Prioritized Action Plan

### P0 — Credibility bleeding; fix this week (small, surgical)

1. Fix the **aurora-04 $K/$M units bug** and the **example report's $116M/$127M + version-footer** contradictions.
2. Fix or pseudocode-label the **fictional API code blocks** (§3.1); start CI-testing doc snippets.
3. Resolve the **"logistic saturation" naming collision** and the unsolvable prior-elicitation example; fix the **AR(1) design-effect error** and the math-05/math-06 sat(0) contradiction.
4. **Source, relabel, or delete every unsourced number** (§3.4) — adopt Measured/Simulated/Heuristic site-wide.
5. Fix deploy-level breakage: **report-template links into docs/**, stale `tests/synth/` paths, sitemap lastmods, FAQ meta description, dead CSS on index, pip-pin-vs-no-PyPI contradiction.
6. **Name the license** on index + footer; date the 0.1.0 release; badge Experimental features where marketed.
7. Reconcile the cross-page contradictions table (§3.3) — most are one-line edits.

### P1 — The buyer's missing pages; fix this month

8. **Evaluator's page** + security/AI-governance page + data-requirements & runtime reference (§7).
9. **Platform screenshots or a 2-minute video** on platform-overview; carry the Beta label.
10. **Document `optimize_budget()`** with constraints on the budget page; add CIs to its tables.
11. **Add the backtest/accuracy section** to forecasting + the example report; add a holdout step to modeling-guide Phase 3.
12. **Identification Assumptions box/page**, linked from every methodology page.
13. **Notebook links** (GitHub/nbviewer/Colab) on every "Run it yourself" box; learning objectives + time estimates + prerequisites banners on all 26 guide pages.
14. **Nav restructure** (grouped dropdowns, Changelog surfaced, role-based router on home/demos).
15. **References blocks** (~15 citations cover the whole site).
16. **Canonicalize stress headline numbers**; auto-bake measured numbers from `stress_matrix.md` with commit stamps.

### P2 — The moat; fix this quarter

17. **Run Robyn/Meridian/PyMC-Marketing through 3–4 stress worlds** and publish the head-to-head — the single most persuasive artifact available, and the harness already exists.
18. **Real-data end-to-end guide** (§4.4) — closes the biggest workflow gap.
19. **Unify the fictional dataset** across the workflow pages into one coherent engagement; add evidence-tier labels to the example report; add the §13 causal-assumptions section it's claimed to contain.
20. **Ship the consultant artifacts** (checklists, templates, engagement timeline) as downloadables.
21. **Known-open-findings/roadmap page**; make parametric adstock the default (or document why not) and align all prior tables with shipped code via bake-time generation.
22. **math-07 (hierarchy)** + likelihood-alternatives note; exercises/assessment path for the training series.
23. Wire the calibration workshop to the planning APIs and agent tools; pre-registration memo template generated from the geo-design outputs.

---

## 9. Bottom Line

The framework's documentation already contains the hardest things to fake: real math, real self-criticism, and a coherent causal doctrine demonstrated with measured evidence. What it lacks is almost everything a consultancy's *non-methodologist* evaluators need — the procurement page, the data spec, the runtime numbers, the screenshots, the real-data story, the accuracy evidence, the competitive benchmark — plus the internal consistency (one saturation function, one set of defaults, one headline number per world, code that runs) that the site's own honesty doctrine demands of itself. The P0 list is days of work and removes the self-inflicted wounds; the P1 list creates the missing buyer-facing surface; the P2 list (especially the competitor benchmark on the existing stress harness) converts the site's radical honesty from a curiosity into a moat no competing framework can cheaply copy.
