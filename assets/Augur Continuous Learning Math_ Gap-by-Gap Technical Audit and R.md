# Technical Audit — *Continuous Learning · Mathematical Foundations* (Augur / mmm-framework)

**Prepared for:** Matthew Reda
**Scope:** Gap-by-gap audit of `continuous-learning-math.html` against the live page, the root site, and the public GitHub repo `redam94/mmm-framework`, with sourced literature and concrete fixes.

## TL;DR

- The page is mathematically sound and unusually well-referenced for a product doc; most of the nine candidate gaps are **real but documentation-level** omissions (missing error bounds, diagnostics, and estimation procedures) rather than errors in the derived equations. The two most serious are the **un-bounded/undiagnosed Laplace EIG error** (Gaps a/b) and the **single-seed misspecification numbers reported without Monte Carlo standard errors** (Gap g).
- One candidate gap is **largely a misconception**: the ENBS optional-stopping concern (Gap e) does not apply to a Bayesian decision-theoretic stopping rule the way it applies to frequentist repeated significance testing (likelihood principle; Berger & Wolpert, Edwards-Lindman-Savage). It should be *addressed with a one-paragraph clarification*, not a redesign — but with a caveat about model misspecification.
- The remaining gaps (grouped-budget KKT shadow prices, KG transform/PSD guarantees, decay half-life estimation, cost-per-bit objective, within-wave non-stationarity) are genuine and each closable with 2–4 well-established references.

## Gap summary table

| # | Gap | Severity | Real gap? | Key source(s) |
|---|-----|----------|-----------|---------------|
| a | Laplace/Gaussian EIG error not bounded | **High** | Yes | Long, Scavino, Tempone & Wang (2013); Long, Motamed & Tempone (2015); Beck, Dia, Espath, Long & Tempone (2018) |
| b | No divergence diagnostic for Laplace vs. true EIG (skew/multimodality) | **High** | Yes | Vehtari, Simpson, Gelman, Yao & Gabry (PSIS / k̂); Spokoiny (computable KL bound, arXiv:1711.08911); Kuss & Rasmussen (2005) |
| c | KG covariance V, ThetaMap transform & PSD after Schur complement | Medium | Partial (repo mitigates transform) | Frazier, Powell & Dayanik (2008); Wu & Frazier (2016); Higham (1988, 2002) |
| d | Grouped-budget KKT lacks per-group shadow prices | Medium | Yes | Boyd & Vandenberghe (2004), §5.5; Fischer, Albers, Wagner & Frie (2011) |
| e | ENBS sequential stopping / optional stopping bias | **Low (mostly misconception)** | No (with caveats) | Edwards, Lindman & Savage (1963); Berger & Wolpert (1988); Rouder (2014); Ramdas et al. (2023) |
| f | Decay half-life h not estimated / not seasonality-dependent | Medium | Yes | West & Harrison (1997), §6.3; Fry, Broadbent & Dixon (2000) |
| g | Misspecification profit gaps reported without Monte Carlo SE | **High** | Yes | Morris, White & Crowther (2019) |
| h | No unified "information per unit dollar" objective | Medium | Yes | Lee, Perrone, Archambeau & Seeger (2020, CArBO); Kleinegesse & Gutmann (2020) |
| i | No within-wave non-stationarity / structural-break detection | Medium–High | Yes | Vaver & Koehler (2011); Kerman, Wang & Vaver (2017); Brodersen et al. (2015); Adams & MacKay (2007) |

---

## What the document actually says (verified against the live page)

The live page's equation numbering matches the review's references. Confirmed key equations: **Eq. 3** (NMC EIG estimator, explicit O(1/M) bias and O(C^{−1/3}) RMSE wall); **Eq. 13** (Fisher information Λ(ξ) = σ⁻²·Σ w_c (g_c − ḡ)(g_c − ḡ)ᵀ with intercept residualization); **Eq. 14** (Σ_post = (Σ⁻¹+Λ)⁻¹); **Eq. 15** (D-/Ds-optimal EIG as ½ log-det difference, with the Ds Schur complement Λ_{γ|rest}); **Eq. 16** (V(ξ) = Σ − Σ_post, the pre-posterior spread of the mean); **Eq. 17** (fantasy-sampling KG, θ_m ∼ N(μ, V)); **Eq. 18** (Thompson allocation, acknowledged non-concave when γ_cc' < 0); **Eq. 19** (funding rule / marginal-ROAS KKT with a single λ); **Eq. 19a** (grouped-budget equality constraints, one global λ plus group equalities); **Eq. 21** (ENBS, stop ⇔ ENBS ≤ 0, "lineage of Wald's SPRT"); **Eq. 22** (σ²_eff(t) = σ²_post·e^{λt}, λ = ln2/h). The page explicitly gives the NMC O(1/M) bias but gives **no** analogous error statement for the Laplace surrogate — the asymmetry the review flags is real.

**Repo check.** The repo (`redam94/mmm-framework`, Apache-2.0, v0.1.0 released Jun 24 2026) maps equations to modules: `acquisition.py` (Eqs. 12–17), `planner.py` (Eqs. 18–21), `arms.py` (Eq. 19a), `loop.py` (Eqs. 9, 22), with the spec in `technical-docs/continuous-learning.md`. The `ThetaMap` unconstrained reparameterisation (log for β,κ,λ, scaled logit for α and the mixture weight, sign-aware log for synergies) **is implemented** and documented on the page — so part of Gap c (mapping bounded params to a valid space) is already handled; the *numerical PSD guarantee* portion is not documented. The misspecification study lives in `nbs/continuous_learning.ipynb §14`. The page does not document Monte Carlo SEs, decay-h estimation, cost-per-bit acquisition, within-wave break detection, or Laplace error bounds — confirming those gaps at the documentation level. (GitHub blocked automated fetch of the individual source files and notebooks, so implementation-level claims are inferred from the page's function mappings and module list, not confirmed line-by-line.)

---

## Gap A — Laplace/Gaussian EIG approximation error is neither bounded nor diagnosed

**Restatement.** The page derives the closed-form Gaussian EIG (Eq. 15) via a Laplace linearisation of the response surface at the posterior mean μ, but — unlike the explicit O(1/M) NMC bias in Eq. 3 — offers no statement of how the linearisation error scales with the curvature of R(s) or the distance |θ − μ|, and no bound on the resulting EIG error.

**Assessment: real, high severity.** This is the single largest internal asymmetry in the document. The framework's central selling point is that it replaces Monte-Carlo EIG with a closed-form ½ log-det; honesty about the NMC error demands matching honesty about the Laplace error. The good news: the error is well-characterised in the BOED literature and is provably small in the low-noise / high-replication regime the framework operates in.

**Sources.**
1. **Long, Q., Scavino, M., Tempone, R. & Wang, S. (2013). "Fast estimation of expected information gains for Bayesian experimental designs based on Laplace approximations." *Computer Methods in Applied Mechanics and Engineering*, 259, 24–39. DOI 10.1016/j.cma.2013.02.017.** The origin result: EIG via Laplace has error that vanishes asymptotically proportional to O(1/M) in the number of repeated/independent measurements M, with the leading correction expressible from the posterior covariance and the log-likelihood derivatives.
2. **Long, Q., Motamed, M. & Tempone, R. (2015). "Fast Bayesian optimal experimental design for seismic source inversion." *CMAME*, 291, 123–145 (arXiv:1502.07873).** Extends the bound to the *non-repeatable* experiment case. §3 states directly: *"if we are able to carry out M repetitive experiments and if M is large, the expected information gain can be estimated by Laplace approximation with a diminishing error asymptotically proportional to M⁻¹ … the error of the Laplace approximation also decreases when the number of receivers and the measurement time increase."* This is exactly analogous to Matt's replication weight w_c = geos × test weeks.
3. **Beck, J., Dia, B. M., Espath, L., Long, Q. & Tempone, R. (2018). "Fast Bayesian experimental design: Laplace-based importance sampling for the expected information gain." *CMAME*, 334, 523–553 (arXiv:1710.03500).** Provides the practical bias/variance decomposition and a Laplace-importance-sampling correction; their finding is precise — *"MCLA is unable to achieve an accuracy better than the Laplace bias; accordingly, the Laplace bias decreases linearly with N_e"* — i.e., the residual floor is the Laplace bias, which is the natural upgrade path if the closed form proves too coarse.

**Fix.** Add a short subsection to §"the surrogate" stating that the Laplace EIG error scales like the third-order Taylor remainder of R about μ, formally O(1/M) in the effective replication M = Σ_c w_c (Long et al. 2013; Long, Motamed & Tempone 2015), and is therefore smallest exactly where the framework has the most data. Where a design's cells sit far from μ or on a high-curvature part of the Hill/spline, flag that the surrogate ordering should be spot-checked against `planner.knowledge_gradient` (the NUTS-refit reference the page already ships). Offer the Beck et al. (2018) Laplace-IS correction as an optional high-accuracy mode, noting its residual floor is the Laplace bias itself.

---

## Gap B — No diagnostic for when the Gaussian/Laplace surrogate diverges from the true EIG

**Restatement.** The page acknowledges non-concavity from negative synergy γ_cc' < 0 in the Thompson section but never connects that (or posterior skew / multimodality) to the *validity of the Gaussian EIG surrogate*. There is no runnable check that the moment-matched N(μ,Σ) is actually close to the carried posterior.

**Assessment: real, high severity — and cheap to close.** The NUTS posterior is already computed each wave, so the raw material for a diagnostic exists; only the check is missing. This pairs naturally with Gap A: A bounds the error in theory, B detects it in practice.

**Sources.**
1. **Vehtari, A., Simpson, D., Gelman, A., Yao, Y. & Gabry, J. (2024). "Pareto Smoothed Importance Sampling," *JMLR* (arXiv:1507.02646).** The **k̂ (Pareto-k) diagnostic**: use the Gaussian surrogate as an importance proposal for the true posterior and read off k̂. For moderate sample size (S > 2000), k̂ ≤ 0.7 indicates the PSIS estimate is reliable (with the sample-size-dependent threshold min(1 − 1/log₁₀(S), 0.7)); as the Stan `loo` documentation puts it, *"If 0.7 ≤ k < 1, the PSIS estimate and the corresponding Monte Carlo standard error have large bias and are not reliable."* Directly implementable from existing NUTS draws.
2. **Spokoiny, V. (2017). "Computing the quality of the Laplace approximation," arXiv:1711.08911.** Gives a *computable* upper bound on the KL divergence between a log-concave posterior and its Laplace approximation, driven by the third derivative of the log-density — a direct "is the Gaussian good enough here?" number. Its log-concavity assumption is exactly what γ_cc' < 0 ridges can violate, which is itself the useful alarm. (Preprint; verify the published version before formal citation.)
3. **Kuss, M. & Rasmussen, C. E. (2005). "Assessing Approximations for Gaussian Process Classification," *JMLR* 6, 1679–1704.** Classic empirical demonstration of *how* and *where* Laplace fails (skewed/truncated posteriors: it misplaces mass and understates skew) versus moment-matching (EP) — the intuition for why negative-synergy ridges break the Gaussian.
4. **Rue, H., Martino, S. & Chopin, N. (2009). "Approximate Bayesian Inference for Latent Gaussian Models… (INLA)," *JRSS-B* 71(2), 319–392.** Shows how to *correct* a Gaussian posterior approximation for location error and lack of skewness via a series expansion — the upgrade if the diagnostic frequently fires.

**Fix.** Add a "surrogate validity" diagnostic to `acquisition.py`: compute the Pareto-k̂ of the Gaussian moment-match against the NUTS draws each wave (Vehtari et al.), and/or the Spokoiny KL bound; surface per-parameter skew/kurtosis flags. When k̂ crosses 0.7 — expected precisely on channels with active γ_cc' < 0 ridges — fall back to the NUTS-refit KG for that wave and record it in `diagnostics`. Cite Kuss & Rasmussen for the failure-mode intuition and INLA as the correction path.

---

## Gap C — KG fantasy-covariance V, transform to bounded parameters, and PSD guarantees

**Restatement.** KG (Eqs. 16–17) samples fantasy means θ_m ∼ N(μ, V), V = Σ − Σ_post, which assumes Gaussianity of the posterior mean in parameter space, and forms the Ds-optimal criterion via a Schur complement (Eq. 15). Two concerns: (i) positive/bounded parameters must be handled so fantasies map to valid θ; (ii) V and the Schur complement must remain positive semi-definite numerically.

**Assessment: partly mitigated in the repo, partly a real documentation gap.** The `ThetaMap` unconstrained reparameterisation already ensures every fantasy sample "maps back to a valid parameter vector by construction," and the page argues EIG differences are invariant to the fixed reparameterisation — so concern (i) is addressed. Concern (ii) — the numerical PSD guarantee after finite-precision inversion — is **not documented** and is a genuine robustness gap (V = Σ − Σ_post is PSD in exact arithmetic because Λ ⪰ 0, but floating-point subtraction and a near-singular Λ_rr can produce tiny negative eigenvalues in V or the Schur complement).

**Sources.**
1. **Frazier, P. I., Powell, W. B. & Dayanik, S. (2008). "A Knowledge-Gradient Policy for Sequential Information Collection," *SIAM J. Control Optim.* 47(5), 2410–2439 / (2009) "…for Correlated Normal Beliefs," *INFORMS J. Computing* 21(4), 599–613.** Origin of the KG / pre-posterior-variance object; establishes V = Σ − Σ_post as the covariance of the updated mean and the Gaussian fantasy construction — the exact object in Eqs. 16–17.
2. **Wu, J. & Frazier, P. I. (2016). "The Parallel Knowledge Gradient Method for Batch Bayesian Optimization," *NeurIPS* 29 (arXiv:1606.04414).** Modern treatment of KG with correlated Gaussian beliefs and the Cholesky-based linear algebra for the fantasy step — the reference for a robust (and batched) implementation.
3. **Higham, N. J. (2002). "Computing the nearest correlation matrix — a problem from finance," *IMA J. Numer. Anal.* 22(3), 329–343 (DOI 10.1093/imanum/22.3.329);** and **Higham, N. J. (1988). "Computing a nearest symmetric positive semidefinite matrix," *Linear Algebra Appl.* 103, 103–118.** The canonical nearest-PSD projection (alternating projections / eigenvalue clipping) to repair a V that has drifted indefinite.

**Fix.** Document that (a) the ThetaMap makes Gaussian fantasies valid by construction (already true), and (b) V and the Ds Schur complement are symmetrised and passed through a PSD safeguard before sampling — either Cholesky-with-jitter (add εI until the factorisation succeeds) or a Higham nearest-PSD projection (eigenvalue clipping), citing Higham (1988, 2002). Reference Wu & Frazier for the batch/parallel Cholesky machinery if KG is ever batched across candidate waves.

---

## Gap D — Grouped-budget KKT needs per-group shadow prices, not one global λ

**Restatement.** Eq. 19 gives the single-budget KKT stationarity condition value·∂R/∂a_c = λ (equalise marginal ROAS across funded channels). Eq. 19a adds per-group equality constraints Σ_{i∈g} s_i = B_g, but the KKT stationarity story is not re-derived for the multi-constraint case, which requires a **separate Lagrange multiplier (shadow price) μ_g per group** in addition to the global budget λ.

**Assessment: real, medium severity.** The `arms.py` SLSQP solver enforces the constraints correctly in code, so allocations are right — but the *documentation's* economic interpretation ("marginal ROAS equalises across all funded channels") is only true *within* a group at the parent-fixed optimum. Across groups, marginal ROAS equalises only up to the per-group shadow price μ_g; a binding parent budget drives a wedge. This matters because the flat-funding-line intuition the page sells is subtly wrong across arm groups.

**Sources.**
1. **Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*, Cambridge Univ. Press, §5.5 (KKT conditions) and §11 (equality-constrained minimization / KKT matrix).** The standard reference: with equality constraints h_g(x)=0 the stationarity condition is ∇f + Σ_g ν_g∇h_g + λ∇(budget) = 0; each ν_g is that group's shadow price. The "water-filling" marginal-equalization rule generalises to equalization *net of* each active group's multiplier.
2. **Fischer, M., Albers, S., Wagner, N. & Frie, M. (2011). "Practice Prize Winner — Dynamic Marketing Budget Allocation Across Countries, Products, and Marketing Activities," *Marketing Science* 30(4), 568–585 (DOI 10.1287/mksc.1100.0627).** The canonical marketing treatment of *hierarchical/nested* budget allocation (country × product × activity) — exactly the arm-group structure — with the marginal-return-equalization logic under nested constraints.

**Fix.** In §"Arms & grouped budgets," re-derive Eq. 19a's stationarity: introduce μ_g for each group constraint so that at the optimum value·∂R/∂a_c = λ + μ_g for every funded channel c in group g. State the corrected interpretation: *marginal ROAS equalises within a group, and across groups only after netting each group's shadow price; a binding parent budget (μ_g ≠ 0) means that group's channels run at a different marginal ROAS than the global line.* Cite Boyd & Vandenberghe §5.5 and Fischer et al. (2011).

---

## Gap E — ENBS stopping rule and "optional stopping" — mostly a misconception

**Restatement.** ENBS (Eq. 21) is evaluated every wave and the loop stops at the first ENBS ≤ 0. The concern: sequentially peeking at a stopping statistic is the classic optional-stopping / repeated-significance problem, and the page invokes Wald's SPRT lineage without discussing peeking correction.

**Assessment: NOT a real gap for the Bayesian decision rule as stated — but worth an explicit paragraph, with a caveat.** ENBS is a Bayesian decision-theoretic stopping rule (stop when the expected value of continuing is negative), not a frequentist significance test. Under the **likelihood principle**, Bayesian posterior / expected-utility quantities do not depend on the stopping rule, so there is no α to "spend." The alpha-spending machinery (O'Brien-Fleming, Pocock) the review gestures at is the *frequentist* fix for a problem the Bayesian rule does not have. The important caveat is that this immunity assumes a correctly specified model — and the page's own §Misspecification shows credible intervals can be "narrow and wrong," which *is* the real residual risk.

**Sources.**
1. **Edwards, W., Lindman, H. & Savage, L. J. (1963). "Bayesian statistical inference for psychological research," *Psychological Review* 70(3), 193–242.** The classic statement (verbatim): *"the rules governing when data collection stops are irrelevant to data interpretation, and it is entirely appropriate to collect data until a point has been proven or disproven."*
2. **Berger, J. O. & Wolpert, R. L. (1988). *The Likelihood Principle*, 2nd ed., IMS Lecture Notes–Monograph Series 6.** The authoritative formal treatment of why stopping rules are irrelevant to likelihood-based (Bayesian) inference.
3. **Rouder, J. N. (2014). "Optional stopping: No problem for Bayesians," *Psychonomic Bulletin & Review* 21(2), 301–308** (with Wagenmakers 2007 appendix). Simulation-based confirmation that the interpretation of Bayesian quantities is invariant to the stopping rule; also fairly presents the dissent (Yu et al. 2013; de Heide & Grünwald on when *default-prior* Bayesian error rates can still be affected).
4. **Ramdas, A., Grünwald, P., Vovk, V. & Shafer, G. (2023). "Game-Theoretic Statistics and Safe Anytime-Valid Inference," *Statistical Science* 38(4), 576–597.** If Matt ever wants a *frequentist* guarantee on the stopping decision (e.g., a client demands a bounded false-stop rate), e-processes / confidence sequences give continuously-monitorable validity [Rutgers Research](https://www.researchwithrutgers.com/en/publications/game-theoretic-statistics-and-safe-anytime-valid-inference) — the modern, correct tool, superior to alpha-spending here. **Berry, S. M., Carlin, B. P., Lee, J. J. & Müller, P. (2010). *Bayesian Adaptive Methods for Clinical Trials*** is the applied reference showing decision-theoretic Bayesian stopping in practice.

**Fix.** Add a short "Why ENBS peeking is safe" note citing the likelihood principle (Berger & Wolpert 1988; Edwards, Lindman & Savage 1963; Rouder 2014) to state that the Bayesian expected-value stopping rule is not subject to frequentist optional-stopping inflation. Then add the honest caveat: this holds under correct specification; because §Misspecification shows intervals can be miscalibrated, a robustness guard (an anytime-valid confidence sequence on the realised profit gap per Ramdas et al. 2023, or simply requiring two consecutive ENBS ≤ 0 waves) is a reasonable belt-and-suspenders measure.

---

## Gap F — Information-decay half-life h is neither estimated nor made category/seasonality-dependent

**Restatement.** Eq. 22 inflates posterior variance as σ²_eff(t) = σ²_post·e^{λt}, λ = ln2/h, but h is treated as an exogenous constant with no guidance on how to choose/estimate it, and no dependence on category dynamics or seasonality.

**Assessment: real, medium severity.** h drives the re-test cadence (`due_for_retest`), so a mis-set h either wastes budget re-testing stable channels or lets stale beliefs govern spend. There is a mature literature on choosing exactly this kind of forgetting/discount parameter.

**Sources.**
1. **West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*, 2nd ed., Springer, §6.3 (discount factors) and §4.3.6.** The discount/forgetting-factor framework: model each component as losing a constant fraction of information per period (δ), with *separate discount factors per block/component* [Duke University](http://www2.stat.duke.edu/~mw/MWextrapubs/WestAFMSbook2012.pdf) — directly the "category-dependent decay" the review asks for. δ maps to Matt's h.
2. **Dynamic linear models with adaptive discounting** (e.g., the *International Journal of Forecasting* 2023 treatment building on West & Harrison, and Raftery et al.-style online forgetting-factor estimation). Shows how to let the data set the forgetting rate online rather than fixing it a priori, and that forecast performance is highly sensitive to δ — motivating estimation over a hard-coded h.
3. **Advertising-adstock half-life literature: Broadbent (1979/1984); Fry, T. R. L., Broadbent, S. & Dixon, J. M. (2000). "Estimating Advertising Half-life and the Data Interval Bias," *Journal of Targeting, Measurement & Analysis for Marketing* 8, 314–334** (Monash working-paper version 1999-6). Empirical half-lives are category-specific — industry FMCG figures cluster near 2–2.5 weeks vs. academic estimates of 7–12 weeks — concrete evidence that a single global h is inappropriate and that h should be tied to the channel's measured carryover.

**Fix.** Reframe Eq. 22's h as a per-channel (or per-category) discount factor in the West-Harrison sense (§6.3), defaulting h to a transformation of each channel's estimated adstock half-life (the model already fits carryover), and allowing seasonality by letting λ vary across the calendar. Offer an adaptive-discounting option that estimates h from how quickly successive waves' posteriors actually drift. Cite West & Harrison (1997) and Fry, Broadbent & Dixon (2000).

---

## Gap G — Misspecification profit gaps (0.9% / 1.4%) reported as single-cycle point estimates without Monte Carlo error

**Restatement.** The §Misspecification study reports single-cycle numbers (0.9% profit gap for single-Hill-on-mixture, 1.4% for logistic; coverage like 3/4, 2/4, 4/4; sequential convergence 0.9→0.5→0.2→0.3%) with no confidence intervals or variance across simulation seeds.

**Assessment: real, high severity for credibility.** These numbers are load-bearing — they back the page's central "the decision is robust" claim. A single-seed point estimate cannot distinguish a genuine 0.9% vs. 1.4% difference from Monte-Carlo noise, and "3/4 coverage" from 4 marginal-ROAS reads carries enormous binomial uncertainty. This is precisely the reporting failure the simulation-methods literature targets.

**Sources.**
1. **Morris, T. P., White, I. R. & Crowther, M. J. (2019). "Using simulation studies to evaluate statistical methods," *Statistics in Medicine* 38(11), 2074–2102 (DOI 10.1002/sim.8086).** The standard reference. Its **ADEMP** framework and §5/Table 6 give explicit **Monte Carlo standard error** formulas for every performance measure. Coverage MCSE = √(p(1−p)/n_sim), maximal at p = 0.5 — so n_sim = 4 gives a worst-case MCSE of √(0.25/4) = 0.25 (25 percentage points), making "3/4" essentially uninformative, whereas n_sim = 1000 gives a worst-case coverage SE ≈ 0.016 (< 2%), the benchmark the paper uses for "keep Monte Carlo SE below 0.5%."
2. **White, I. R. (2010). "simsum: Analyses of simulation studies including Monte Carlo error," *The Stata Journal* 10(3), 369–385;** and **Gasparini, A. (2018). `rsimsum`, *JOSS* 3(26), 739.** Ready-made tooling that computes performance metrics with MCSEs [Ellessenne](https://ellessenne.github.io/rsimsum/) — a drop-in for reporting the study.

**Fix.** Re-run the misspecification study over many seeds (n_sim chosen so the profit-gap MCSE is ≪ the 0.5% differences being claimed, and coverage estimated from many more than 4 reads), and report every headline number as **estimate ± Monte Carlo SE** (Morris, White & Crowther 2019). Present coverage as a proportion with a binomial CI. This converts the section from "anecdote" to "evidence" and directly strengthens the robustness argument.

---

## Gap H — No unified "information per unit dollar" cost objective

**Restatement.** ENBS nets a dollar cost against dollar-valued regret, but the *acquisition* functions (D-/Ds-optimal EIG, KG) rank designs by information/value alone, not by information *per unit cost*, even though experimental cells/arms grow the cost (the page notes ~3 cells per arm + 2 per probed pair, quadratic in arms under all-pairs probing).

**Assessment: real, medium severity.** Because arms and pairs make design cost heterogeneous and roughly quadratic in arms, a cost-blind EIG will over-favour large, expensive central-composite designs. A cost-per-bit acquisition is the standard fix and is conceptually already implied by ENBS.

**Sources.**
1. **Lee, E. H., Perrone, V., Archambeau, C. & Seeger, M. (2020). "Cost-aware Bayesian Optimization" (ICML AutoML workshop; arXiv:2003.10870, "CArBO").** Introduces **expected improvement per unit cost (EIpu)** and a cost-effective initial design; [arXiv](https://arxiv.org/pdf/2003.10870) the canonical cost-aware BO reference and the source of the EIpu acquisition (also implemented in BoTorch's cost-aware tutorial with an α∈[0,1] cost-decay knob).
2. **Kleinegesse, S. & Gutmann, M. U. (2020). "Bayesian Experimental Design for Implicit Models by Mutual Information Neural Estimation," *ICML* (PMLR 119, 5316–5326);** and **Kleinegesse, Drovandi & Gutmann (2020), "Sequential Bayesian Experimental Design for Implicit Models via Mutual Information," *Bayesian Analysis* (DOI 10.1214/20-BA1225).** BOED framed explicitly as optimising information subject to experiment cost/performance — the design-side analogue of Matt's dollar-valued ENBS, and the bridge to a cost-normalised EIG/KG.
3. **Rainforth, T., Foster, A., Ivanova, D. R. & Bickford Smith, F. (2023). "Modern Bayesian Experimental Design," *Statistical Science*** (already cited on the page) — its cost/performance framing supports a cost-aware acquisition.

**Fix.** Add a cost-aware acquisition mode: rank candidate waves by EIG(ξ)/cost(ξ) or KG(ξ)/cost(ξ) (EIpu, Lee et al. 2020), where cost(ξ) is the page's own cell/arm/pair count times per-cell geo cost. Note the relationship to ENBS: the per-wave cost-normalised acquisition is the greedy myopic version of the ENBS cost-benefit trade, making the two coherent. Cite Lee et al. (2020) and Kleinegesse & Gutmann (2020).

---

## Gap I — No detection of adversarial / non-stationary drift *within* a wave

**Restatement.** Eq. 22 models information decay *between* waves, but nothing guards against a competitor action, promo, or seasonality shock *during* an active wave that silently violates the stationarity assumption underlying the geo-week likelihood (Eq. 7) and the summary-observation "structural stationarity" assumption (Eq. 7a).

**Assessment: real, medium–high severity.** The framework's causal identification rests on between-cell variation within a stable test window; a mid-test regime change contaminates exactly the contrast the design is built to measure. This is a known, actively-managed problem in geo-experiment practice, with strong, directly-applicable methods.

**Sources.**
1. **Vaver, J. & Koehler, J. (2011). "Measuring Ad Effectiveness Using Geo Experiments," Google Research (pub38355);** and **Kerman, J., Wang, P. & Vaver, J. (2017). "Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework," Google Research (pub45950; open-source `GeoexperimentsResearch`).** GBR and TBR use a contemporaneous randomized control set so common mid-test shocks largely cancel; a treatment-only shock shows up as anomalous second-stage residuals / counterfactual divergence — a built-in detection signal. (Technical reports — cite by Google Research publication URL, no DOI.)
2. **Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N. & Scott, S. L. (2015). "Inferring causal impact using Bayesian structural time-series models," *The Annals of Applied Statistics* 9(1), 247–274 (DOI 10.1214/14-AOAS788; software google.github.io/CausalImpact).** CausalImpact builds a control-based counterfactual with explicit local-trend + seasonality + spike-and-slab covariate components; posterior predictive intervals widen when the series departs from the counterfactual for non-treatment reasons — a natural within-test break alarm.
3. **Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian Online Changepoint Detection," arXiv:0710.3742;** and **Fearnhead, P. & Liu, Z. (2007). "On-line Inference for Multiple Changepoint Problems," *JRSS-B* 69(4), 589–605 (DOI 10.1111/j.1467-9868.2007.00601.x).** Run online on the treatment-minus-control residual series to flag the exact time step of a mid-test regime change (single or multiple), letting the loop censor or down-weight the contaminated segment.
4. **Ben-Michael, E., Feller, A. & Rothstein, J. (2021). "The Augmented Synthetic Control Method," *JASA* 116(536), 1789–1803 (DOI 10.1080/01621459.2021.1929245).** The synthetic-control basis of Meta GeoLift; its pre-period fit metric (L2 imbalance) is a direct diagnostic that the control-based baseline has stopped tracking — i.e., a stationarity violation during the window.

**Fix.** Add a within-wave stationarity guard: (a) monitor the treatment-minus-control (or observed-minus-counterfactual) residual with Bayesian online changepoint detection (Adams & MacKay 2007) during the wave; (b) at readout, run a CausalImpact-style counterfactual check (Brodersen et al. 2015) and flag if the control-based prediction diverged for non-treatment reasons; (c) surface a break flag in `diagnostics` and, if fired, censor the affected periods or extend/repeat the wave rather than folding a contaminated readout into the posterior. Note that the framework's randomized holdout/shut-off cells already provide the contemporaneous control that makes common-shock cancellation (Vaver & Koehler 2011; Kerman et al. 2017) work — the gap is only the *detection and censoring* step.

---

## Recommendations (staged)

**Stage 1 — highest credibility-per-effort (do first).**
1. **Gap G:** Re-run the misspecification study over many seeds and report every headline number with Monte Carlo SEs and binomial CIs on coverage (Morris, White & Crowther 2019). This defends the load-bearing "≳99% of profit" claim. *Threshold to change plan: if the profit-gap MCSE is not ≪ the 0.5% inter-family differences, increase n_sim (n_sim ≈ 1000 puts worst-case coverage SE < 2%).*
2. **Gap A/B (paired):** Add the Laplace-EIG error statement (O(1/M) in effective replication; Long et al. 2013; Long, Motamed & Tempone 2015) *and* a runnable surrogate-validity diagnostic (Pareto-k̂ against the existing NUTS draws; Vehtari et al.). *Threshold: if k̂ ≥ 0.7 fires on more than a small fraction of waves, wire the automatic fallback to NUTS-refit KG.*
3. **Gap E:** Add the one-paragraph likelihood-principle clarification (Berger & Wolpert 1988; Rouder 2014) with the misspecification caveat. Near-zero code cost.

**Stage 2 — correctness and robustness.**
4. **Gap D:** Re-derive Eq. 19a's KKT with per-group shadow prices μ_g and correct the funding-line interpretation (Boyd & Vandenberghe §5.5; Fischer et al. 2011).
5. **Gap C:** Document the PSD safeguard (Cholesky jitter or Higham nearest-PSD) on V and the Schur complement (Higham 1988/2002).
6. **Gap I:** Add the within-wave changepoint/counterfactual guard and break-flag (Adams & MacKay 2007; Brodersen et al. 2015; Vaver & Koehler 2011).

**Stage 3 — features / research.**
7. **Gap F:** Tie h to per-channel adstock half-life and allow adaptive discounting (West & Harrison §6.3; Fry, Broadbent & Dixon 2000).
8. **Gap H:** Add a cost-aware EIpu / KG-per-cost acquisition and connect it to ENBS (Lee et al. 2020; Kleinegesse & Gutmann 2020).

## Caveats

- This audit is based on the live `continuous-learning-math.html`, the root site, and the public repo README/tree; GitHub blocked automated fetch of the individual source files (`acquisition.py`, `planner.py`, `arms.py`) and the notebooks, so implementation-level claims (e.g., whether a PSD safeguard or a k̂ diagnostic already exists in code but is undocumented) are inferred from the page's function mappings and module list, not confirmed line-by-line. Several gaps may already be partially handled in code — the ThetaMap for Gap c is a documented example; in those cases the recommendation is at minimum to *document* them.
- One preprint is cited with mild attribution uncertainty: the "Computing the quality of the Laplace approximation" bound (arXiv:1711.08911) is a preprint whose journal version may differ; verify before formal citation.
- The Google geo-experiment papers (Vaver & Koehler 2011; Kerman, Wang & Vaver 2017) are technical reports without journal DOIs — cite by Google Research publication URL.
- Severity ratings reflect impact on the document's *credibility and correctness of interpretation*, not the probability that Augur produces wrong allocations in practice (the SLSQP/constraint code appears to enforce the right constraints even where the prose under-explains them). The single genuine "not a gap" verdict is Gap e (ENBS optional stopping), which is a misconception under the likelihood principle but merits an explicit clarifying paragraph plus a misspecification caveat.