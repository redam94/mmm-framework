# Adstock / Saturation / β Equifinality

*Companion note to `mmm-methodology.tex`. Addresses critique.md §3.6.*

## The problem

For a single media channel the additive model contributes

```
contribution_t = β · sat( adstock(x_t ; θ_adstock) ; θ_sat )
```

Two structural choices make the three parameter groups —
`θ_adstock` (decay shape), `θ_sat` (saturation strength), and `β` (coefficient)
— **trade off against one another**, so distinct combinations produce nearly
identical in-sample fits (*equifinality*):

1. **Normalized adstock** (`AdstockConfig.normalize=True`, the default). The
   weight kernel is rescaled to sum to 1, so the *level* of carryover is removed
   from the kernel and absorbed into `β`. A higher-decay kernel and a lower `β`
   can reproduce the same fitted series as a lower-decay kernel and a higher `β`.

2. **Carryover ↔ saturation entanglement.** A long carryover with weak
   saturation spreads and dampens the response much like a short carryover with
   strong saturation. Baseline-vs-media is similarly entangled when a flexible
   trend can absorb slow-moving media effects.

Because the priors on adstock, saturation, and `β` are independent, nothing in
the likelihood *prefers* one decomposition over another. Good convergence, tight
posteriors, and a passing PPC can all coexist with a decomposition that is an
artifact of the prior rather than the data.

## Why this is not "a bug"

Equifinality is a property of the model class, shared by Robyn, Meridian, and
every additive carryover/saturation MMM. The fitted **total** media effect over
a period is far better identified than its split into "shape" vs "level." The
danger is reporting per-channel decay/half-saturation/`β` as if each were pinned
by the data.

## Remedies, in order of leverage

1. **Experiment-calibrated coefficient priors — the real fix.**
   `mmm_framework.calibration` turns a geo-lift / incrementality result into an
   informative prior on `β`. Pinning `β` collapses the trade-off: once the level
   is anchored to randomized evidence, the remaining decay/saturation parameters
   are identified by the data's dynamics. This is the only remedy that also
   addresses the dominant confounder (unobserved demand).

2. **Data-anchored half-saturation (Hill path).** Compute bounds from the
   observed (adstocked) spend with
   `SaturationConfig.compute_kappa_bounds_from_data(x, percentiles)` and pass them
   as `kappa_lower`/`kappa_upper` to `create_saturation_prior`, which then bounds
   the Hill `kappa` to a `Uniform` over that range so the curve's "elbow" cannot
   drift outside the data's support. Opt-in (the default prior is unchanged unless
   you pass bounds); the core logistic model has no `kappa` and is unaffected.

3. **Weakly-informative priors on decay/saturation.** Tightening
   `alpha_prior` / saturation priors toward plausible ranges reduces — but does
   not remove — the entanglement.

## What the framework does and does not claim

The model reports honest posterior uncertainty on contributions and (now) on
marginal ROAS, but a narrow interval on a *decomposition* parameter reflects the
prior as much as the data. Treat per-channel decay and saturation as
weakly-identified nuisance shape parameters unless a channel is anchored by
experimental calibration. See the module docstring in `model/base.py` and the
kernel-level discussion in `transforms/adstock.py::adstock_weights`.
