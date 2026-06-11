"""Methodological reference for the Bayesian workflow, adapted to MMM.

The system prompt carries the 9-step tool recipe (WHAT to call, in which
order); this module carries the METHODOLOGY behind it (why each step exists,
what to look at, the decision thresholds, and what to do when a check fails).
Exposed to the agent via the ``bayesian_workflow_reference`` tool in
``tools.py``, topic-filterable by ``## `` section like ``library_reference``.

Grounding: Gelman et al. (2020) "Bayesian Workflow" (arXiv:2011.01808),
Betancourt's diagnostics writings, and this framework's own stress-test
findings (tests/synth — the silent-failure modes a converged, well-fitting
MMM can still have).
"""

BAYESIAN_WORKFLOW_REFERENCE = """\
# Bayesian workflow reference (methodology — why, what to check, what to do on failure)

The workflow is ITERATIVE, not linear: a failed check sends you back to priors,
structure, or data — and every revision is logged with `record_assumption` so the
final answer carries its assumption stack. Model criticism is the product, not an
obstacle. (Gelman et al. 2020, arXiv:2011.01808.)

## Principles
- Pre-specify before peeking: the research question, the DAG, and the priors come
  before fitting. Choices made after seeing results are researcher degrees of
  freedom — log them as revisions, never silently.
- A converged model is not a correct model. R-hat/ESS check the SAMPLER;
  posterior predictive checks check the FIT; neither checks IDENTIFICATION
  (confounding, collinearity, functional form). Each needs its own evidence.
- Honest uncertainty beats tight intervals. If channels co-move (collinearity),
  wide ROI posteriors are the truth — do not tighten priors merely to shrink them.
- Prefer fixing the model over fixing the sampler. Divergences and poor mixing
  usually mean a mis-specified model or priors fighting the data.

## Question & causal structure (steps 1–2)
- The estimand comes first: incremental effect of WHICH treatments on WHICH
  outcome, over what scope. Without it, no check can tell you if the model is fit
  for purpose (`define_research_question`).
- The DAG is the identification argument. Backdoor-validate it
  (`validate_causal_identification`); the adjustment set becomes the control list.
- Latent baseline drivers (trend, seasonality) are MODEL COMPONENTS, not dataset
  regressors — name DAG proxies "Trend"/"Seasonality" and they map onto the
  built-in components. They proxy unobserved demand: say so in the assumptions.
- An unidentified effect cannot be rescued by fitting harder. Surface the open
  backdoor path and either add the measured confounder or record the assumption
  that it is negligible.

## Priors & prior predictive check (steps 3–4)
- Priors encode real knowledge (ROI plausibly 0–5, adstock decay under ~8 weeks,
  saturation reached within observed spend) — not vagueness. Flat priors in a
  weakly-identified MMM let the likelihood wander into absurd ROI space.
- `prior_predictive_check` runs PRE-FIT from the active spec. Ask: do simulated
  KPIs live on the plausible scale? Flags: fraction of negative draws > 5%; an
  implied 5–95% range orders of magnitude wider (prior too vague — the data will
  be forced to do all the work) or narrower (prior too tight — it will dominate)
  than the observed KPI range. NOTE: values are reported in the model's
  standardized scale; compare against the standardized KPI, not raw units.
- Iterate priors → re-check → `record_assumption` (category: prior) until
  implications are plausible. Only then fit.
- Spec-settable baseline priors (`update_model_setting`): `priors.trend.*`
  (growth mu/sigma — base slope for linear AND piecewise; changepoint scale;
  spline sigma; GP lengthscale/amplitude) and `priors.seasonality.prior_sigma`
  (+ `yearly/monthly/weekly_prior_sigma` overrides) — the seasonal amplitude
  prior: Normal(0, sigma) per Fourier coefficient on standardized y, default
  0.3. Strongly seasonal KPI → raise it, or the seasonal signal leaks into
  trend and media.

## Fitting & computational diagnostics (steps 5–6)
- Gates, in order (`get_model_diagnostics`): divergences = 0; max R-hat < 1.01;
  min bulk AND tail ESS > 400. Fail → STOP; do not interpret a single number.
- Remedies, cheapest first: raise `inference.target_accept` (0.9 → 0.95 → 0.99)
  for few divergences; more `tune`; tighten or re-shape the offending priors
  (funnel-prone scales); simplify (fewer changepoints, lower Fourier order);
  reconsider the functional form. Re-fit after each change — never hand-tune and
  report the old run.
- ESS marginally above the gate makes tail quantities (interval endpoints)
  noisy; prefer more draws when decisions hinge on interval edges.

## Posterior predictive check (step 7)
- Compare fitted vs observed KPI over TIME (`execute_python`), not just R².
  Look for structure in residuals: systematic seasonal misfit (raise Fourier
  order), an unmodeled level shift or trend break (changepoints), holiday spikes
  (add the control), heteroscedastic errors.
- A model that cannot reproduce the data it was trained on has no business
  making ROI claims. A model that reproduces it perfectly can still be
  causally wrong — PPC is necessary, not sufficient.

## Did the data update the priors? (parameter learning)
- After fitting, check prior→posterior contraction
  (`mmm.compute_parameter_learning()` via `execute_python`, or
  `from mmm_framework.diagnostics import parameter_learning`). Low contraction +
  high prior/posterior overlap = the posterior re-states the prior; the data did
  not speak for that parameter.
- This is COMMON for saturation shape and adstock decay in weekly national data.
  It is not a bug — but a prior-dominated ROI must be communicated as
  "assumption-driven", and prior sensitivity (step 8) becomes mandatory for it.

## Sensitivity analysis & model comparison (step 8)
- Genuine sensitivity = RE-FIT with the perturbed spec (priors halved/doubled,
  different adstock l_max, alternative saturation form, controls in/out) and
  compare the decision-relevant quantities (ROI ranking, budget shifts).
  `leave_one_out_decomposition` only reweights the existing posterior — quick,
  but it is NOT a refit and must not be reported as sensitivity having "passed".
- If a conclusion flips under a defensible alternative spec, report the range,
  not the favorite.
- Model comparison: prefer expansion guided by PPC failures over leaderboard
  selection. Time-series cross-validation caveat: pointwise LOO ignores temporal
  dependence; treat in-sample information criteria with suspicion.

## MMM-specific silent failure modes (all checks can pass while these bite)
- Confounded media: an unobserved demand driver moves spend AND sales → inflated
  ROI. Only the DAG + adjustment discipline addresses this.
- Collinear channels: co-moving spends make individual ROIs unidentifiable;
  posteriors stay wide or split arbitrarily. Check VIF in EDA (`run_eda`);
  consider aggregating channels or bringing external lift evidence.
- Saturation form mismatch: forcing Hill on logistic-shaped response (or vice
  versa) biases marginal ROAS even with good fit — test both forms (re-fit).
- Spend outliers: one ~15x data-entry spike corrupts that channel's
  max-normalization and flattens its curve (`detect_outliers` pre-fit).
- Noisy proxy controls: a poorly-measured confounder proxy leaves residual
  confounding while LOOKING adjusted-for.
- Lift-test calibration: when an experiment exists, fold it in
  (`ExperimentMeasurement` → `add_experiment_calibration` before fit) rather
  than informally reconciling afterwards.
- Average vs marginal ROAS: average ROAS ranks PAST spend; budget decisions need
  MARGINAL ROAS (`run_marginal_analysis`) — under saturation the two routinely
  disagree, and the flip is the finding.

## Communicating results (step 9)
- Report intervals, not points: ROI as posterior median + credible interval.
- State the conditioning: adjustment set, functional forms, priors that the data
  did not update (from parameter learning), and the assumption stack
  (`list_assumptions`).
- Causal language discipline: "incremental effect, identified under the DAG's
  assumptions" — name the assumptions that would break it.

## From learnings to decisions (budget + next experiment)
- Budget: `run_budget_optimizer` allocates on the model's posterior response
  curves (greedy marginal allocation — exact for saturating curves), bounded to
  the observed spend range (beyond it the curves are extrapolation). It
  re-optimizes under every posterior draw: a channel whose optimal share has a
  wide 90% range is a decision the data does NOT support — report the range,
  don't just hand over the point allocation.
- The optimizer maximizes contribution given the model; it inherits every
  upstream assumption. Confounded ROI in → confidently wrong allocation out.
- Experiments: `recommend_lift_experiments` ranks channels by spend share ×
  ROAS uncertainty × allocation instability — the value-of-information logic:
  test where better information would actually change the decision, not where
  the posterior is merely wide. Designs include duration (adstock window + 4)
  and a target SE (≤ half the posterior ROAS sd, so calibration genuinely
  shrinks it).
- Close the loop: a finished experiment enters the NEXT fit as an
  `ExperimentMeasurement` likelihood term (`add_experiment_calibration`), never
  as an informal reconciliation. MMM → experiment → calibrated MMM is the
  flywheel that converts correlational fits into defensible causal estimates.
"""
