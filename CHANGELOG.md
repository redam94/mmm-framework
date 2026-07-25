# Changelog

All notable changes to `mmm-framework` are documented here. The project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html): the major version changes when a
frozen public contract breaks, and the contract itself is pinned by
`tests/test_api_contracts.py` and `tests/test_lean_imports.py`.

## [1.0.1] — 2026-07-24

Two methodological fixes in `validation/`. Both corrected numbers that read as **more
trustworthy than they were**, so both change output you may have quoted.

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
