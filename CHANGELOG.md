# Changelog

All notable changes to `mmm-framework` are documented here. The project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html): the major version changes when a
frozen public contract breaks, and the contract itself is pinned by
`tests/test_api_contracts.py` and `tests/test_lean_imports.py`.

## [1.2.0] — 2026-07-25

Sampler selection worked in one place and silently failed in four. Fixing that required new
public API — a third NUTS backend that was a declared dependency no code path could reach —
so this is a minor rather than a patch bump. It also raises an install floor and marks two
inference methods as deprecated, neither of which is patch-safe.

Plus a reporting correctness fix: cross-effect tables in multi-outcome reports listed pairs
the model never estimated.

The theme is knobs that read as configuration and were no-ops. If you set `target_accept`,
selected a frequentist inference method, or read a multi-outcome cross-effect table, your
results change.

### Fixed

- **`ModelConfig.target_accept` is now honored by `fit()`.** The fallback went straight to a
  literal `0.9`, so `ModelConfigBuilder().with_target_accept(0.95)` was a silent no-op — the
  first knob the sampling-failure playbook tells you to reach for, and the one that does
  nothing about divergences if it never reaches `pm.sample`. Precedence is now explicit
  argument → config → `0.9`. The field's own default is `0.9`, so an untouched config
  samples byte-identically to 1.1.0; if you *had* set it, your fits change.

- **`fit(nuts_sampler=...)` no longer raises on the core model.** `BayesianMMM.fit()` had no
  such parameter, so the keyword — which `BaseExtendedMMM.fit()` does accept, and which four
  places in the docs recommended — fell into `**kwargs` and collided with the explicit
  argument already passed to `pm.sample`, raising a `TypeError` that named `pm.sample` rather
  than anything the caller wrote. The same line worked on an extension model and failed on a
  plain one.

- **`frequentist_ridge` / `frequentist_cvxpy` no longer silently fit Bayesian NUTS.**
  Both are declared `InferenceMethod` values with builder methods advertising "Ridge
  regression (fast, frequentist)" and "CVXPY for constrained optimization". Neither has ever
  been implemented: `fit()` dispatches on `FitMethod`, never on `InferenceMethod`, and the
  package depends on neither `scikit-learn` nor `cvxpy`. Selecting one did not raise — it
  fitted a **full Bayesian posterior** via the `"pymc"` fallback in `ModelConfig.nuts_sampler`,
  so you asked for a fast frequentist point estimate, waited out MCMC, and got a posterior
  with no indication the request had been ignored.

  `fit()` now raises `NotImplementedError` naming the supported alternative, and constructing
  such a config emits a `DeprecationWarning`. **The alternative is `fit(method="map")`**: under
  Gaussian coefficient priors, maximum a posteriori estimation *is* ridge regression, so the
  capability the name promised has effectively been available all along.

  The enum values are **retained** — stored configs still parse and the frozen-enum contract
  stays green. The Excel template parser rejects the strings at parse time. The config fields
  that exist only for this path (`ridge_alpha`, `bootstrap_samples`, `optim_maxiter`) are
  likewise inert and now marked as such; they are the reserved surface for the real
  implementation, tracked in
  [#180](https://github.com/redam94/mmm-framework/issues/180).

- **Cross-effect summaries no longer report structurally-zero outcome pairs as estimated.**
  `reporting/helpers/mediated.py::compute_cross_effects` probed for `get_cross_effect_summary`
  (singular); both `MultivariateMMM` and `CombinedMMM` spell it `get_cross_effects_summary`,
  so the `hasattr` check never matched and every report fell through to the manual branch.
  That branch walks every off-diagonal `psi` entry, and the matrix starts from zeros with
  only declared specs filled — so undeclared pairs appeared alongside the real ones, and
  `effect_type` was dropped. If you have a multi-outcome report with cross-effect rows you
  did not declare, that is this bug; re-run to get the declared set.

### Added

- `InferenceMethod.BAYESIAN_NUTPIE` — the `nutpie` Rust NUTS sampler has been a core
  dependency since 1.0.0, but sampler choice was the binary `use_numpyro` with no
  representable value for it. Existing enum values are unchanged.
- `ModelConfigBuilder.bayesian_nutpie()` and `DAGModelBuilder.bayesian_nutpie()`, alongside
  the existing `.bayesian_pymc()` / `.bayesian_numpyro()`. The three sample the same graph;
  only the NUTS implementation differs. Verified end-to-end — a real MMM fit through nutpie
  agrees with the pymc backend.
- `ModelConfig.nuts_sampler` — the single resolver from inference method to the
  `pm.sample(nuts_sampler=...)` string. Non-Bayesian methods report `"pymc"` rather than
  raising, so a caller reading it on a frequentist config still gets the historical default.
- `BayesianMMM.fit(nuts_sampler=...)` — explicit argument, defaulting to the config. Accepts
  `"pymc"`, `"numpyro"`, `"nutpie"` or `"blackjax"`; ignored by the non-NUTS fit methods.
- `"bayesian_nutpie"` is accepted by the Excel config parser's inference-method mapping.

### Changed

- **`nutpie>=0.16.4` → `>=0.16.10`.** pymc 6.0 raises `ImportError` below `0.16.10`, so the
  old pin shipped a sampler that could not have started even if it had been reachable. A
  locked environment resolving the old floor needs to update.

### Deprecated

- `InferenceMethod.FREQUENTIST_RIDGE` and `FREQUENTIST_CVXPY`, and the corresponding
  `ModelConfigBuilder.frequentist_ridge()` / `.frequentist_cvxpy()`. They are unimplemented
  and now refuse rather than falling through to NUTS (see Fixed). They are **not** removed:
  removal would break the frozen-enum contract and is reserved for a major version. They will
  become live — not removed — when [#180](https://github.com/redam94/mmm-framework/issues/180)
  lands.

### Notes

- Extension models (`BaseExtendedMMM` — Nested / MV / Combined / Structural) keep their
  `nuts_sampler="pymc"` default deliberately. Their bespoke graphs are not all JAX-traceable,
  so inheriting a numpyro config would break fits rather than speed them up.
- If a pipeline of yours currently runs with a frequentist inference method, it will now fail
  loudly. That is the point: it was returning a Bayesian posterior for a frequentist request.
  Switch to `fit(method="map")` for the fast penalized estimate, or to an explicit Bayesian
  method to keep exactly what you were already getting.
- Internal only, no package surface: the docs code-snippet gate now covers Markdown
  (`README.md`, `CLAUDE.md`, `technical-docs/*.md`) as well as HTML, docs navigation
  registration is gated, and `docs/tools/build_seo.py` is idempotent.

## [1.1.0] — 2026-07-25

Two methodological fixes in `validation/`. Both corrected numbers that read as **more
trustworthy than they were**, so both change output you may have quoted.

A minor rather than a patch bump: the fixes are accompanied by new public API (a `weighting`
argument, new result fields, two new exported names), and one of them changes a default so
that a re-run returns a different number than 1.0.0 did.

### Fixed

- **Spec-curve model averaging no longer weights a causal estimand by predictive skill.**
  `run_spec_curve` applied LOO-stacking weights to per-channel ROI. Stacking
  (Yao et al. 2018) maximizes expected *predictive* utility — it answers "which mixture
  forecasts held-out `y` best?" — while a spec curve averages a *causal* estimand. The two
  objectives come apart exactly where MMM specs differ, because two specs can predict the
  KPI equally well while splitting that same fitted mean very differently between media and
  baseline. Worse, the direction inverts: a spec that overfits the confounder block often
  predicts *better* while being *less* trustworthy for the causal contrast, so stacking
  systematically upweighted the specs a causal analyst should trust least.

  The default is now **equal weights** over the pre-registered set, on the reasoning that
  pre-registration already asserted every variant is defensible, so there is no post-hoc
  predictive ground to promote one.

- **The unobserved-confounding robustness value is no longer inflated by tight priors.**
  `robustness_value` is strictly increasing in `|t|`, and `t = posterior_mean / posterior_sd`.
  Tightening a prior shrinks the posterior sd, which *raised* the reported robustness with no
  new evidence — so the most prior-dominated channel could report the most robust value, and
  robustness values were not comparable across channels with differently tight priors.

  Prior contraction is now computed per channel where a prior group exists. A channel below
  the `PRIOR_DOMINATED_CONTRACTION` threshold (0.20) renders as **"Not assessable
  (prior-driven)"** rather than a green "Robust", with a footnote explaining the inversion.
  "Could not check" is now reported distinctly from "checked and passed".

### Added

- `run_spec_curve(..., weighting=...)` — `"equal"` (default) or `"stacking"`. Requesting
  stacking opts back into the old behavior and logs a warning; it is defensible only when
  every spec in the set is identified for the same estimand and you genuinely want a
  predictive mixture. An unrecognized value raises `ValueError`.
- `SpecCurveResult.weighting` and `.predictive_weights` — the stacking weights are still
  computed and reported, deliberately, as a *diagnostic* rather than an input: divergence
  from uniform says predictive fit discriminates between your specifications, which is worth
  seeing and not worth acting on. `to_dict()` also emits `weighting_caveat`.
- `validation.spec_curve.WEIGHTING_CAVEAT` — the caveat text keyed by weighting mode, so
  reports and payloads state which weighting produced a number.
- `ChannelRobustness.prior_contraction`, `.is_prior_dominated` and `.rv_is_quotable`, plus
  `validation.sensitivity_unobserved.prior_inflation_warning()` and
  `PRIOR_DOMINATED_CONTRACTION`.

### Changed

- `_stacking_weights` returns `{}` on every fallback path instead of fabricating a uniform
  vector, so `predictive_weights` honestly distinguishes "unavailable" from "uniform".
- Report surfaces relabelled to match: the spec-curve blend prose is now derived from the
  weighting actually used, the weights table separates the applied weight from the
  (unapplied) predictive weight, and a prior-driven channel can no longer render green.

### Notes

If you have a stored `SpecCurveResult` or a report produced by 1.0.0, its model-averaged ROI
was stacking-weighted. Re-run to get the equal-weight number, or read the new
`predictive_weights` field to see how far the two diverge on your set.

## [1.0.0] — 2026-07-24

First stable release: the package was split into a lean modeling core and optional
application layers, the public contracts were audited and frozen, and the project adopted
strict semantic versioning. See the
[v1.0.0 release notes](https://github.com/redam94/mmm-framework/releases/tag/v1.0.0) for the
full list of breaking packaging changes, the contract freeze, and the audit fixes.
