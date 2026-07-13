# Time-varying media coefficients (TVP) — #137

Opt-in per channel. Lets a channel's effectiveness **drift over time** (creative
fatigue, a channel maturing, a structural break) instead of forcing one scalar
`beta` for the whole window.

## Method

Turn it on per channel with `MediaChannelConfig.time_varying=True` (builder
`.with_time_varying()`). When on, the channel's coefficient becomes a smooth
random walk on the log scale:

```
log(beta_{c,t}) = level_c + tvp_sigma_c · (RW_t − mean RW)
beta_{c,t}      = exp(log(beta_{c,t}))          # positive, one value per period
contribution    = beta_{c,t}[obs] · sat_c(adstock_c(x))
```

- `level_c ~ Normal(log 1.5, 0.5)` — the **average** log-coefficient. The random
  walk is **demeaned**, so `level_c` is identifiable separately from the drift
  *shape* (the walk carries only deviations from the average).
- `RW_t = cumsum(z_t)`, `z_t ~ Normal(0,1)` — a non-centered Gaussian random walk
  (clean sampling with many periods).
- `tvp_sigma_c ~ HalfNormal(tvp_innovation_sigma)` — the per-step innovation
  scale, default `0.15`. **Smaller ⇒ closer to a constant beta**; the model
  collapses to today's scalar-beta model as `tvp_sigma_c → 0`.

Emitted variables (naming contract, R0.4): `beta_{c}` is the **time-average**
coefficient (a scalar Deterministic, so every existing consumer — reporting,
sensitivity, estimands — keeps working and reads the pooled effect);
`beta_tv_{c}` is the full per-period trajectory (surfaced for reporting as
`bundle.time_varying_betas`). `channel_contributions` is populated per obs as
usual.

## Off by default / regression

`time_varying` defaults to `False` and the whole TVP block is gated on it, so a
model with no time-varying channel has **no** TVP random variables and is
byte-identical to today (R0.1/R0.2). Verified structurally in
`tests/test_time_varying_betas.py`.

## Identifiability caveats

- **TVP trades off against trend/seasonality and the baseline.** A slow secular
  drift in effectiveness and a slow trend in the baseline are only weakly
  separable from observational data. The small default innovation prior
  regularizes toward a constant coefficient; loosen `tvp_innovation_sigma` only
  when you have a real reason (and enough signal) to expect drift.
- **The trajectory's *shape/sign* is more trustworthy than its level.** On the
  planted-drift stress world (`synth/dgp.make_time_varying_beta`: TV fatigues
  1.4→0.6, Search steps up at the midpoint) the recovered `beta_tv_` trajectories
  recover the **direction** of the drift; the absolute standardized-coefficient
  levels are not the DGP multipliers. Read it as "effectiveness rose / fell / broke
  here," not "beta was exactly X in week t."
- **It is not a substitute for structure you can model.** If the drift is
  actually seasonality, model seasonality; if it is a known creative change, a
  step regressor is cleaner. TVP is for genuinely smooth, unexplained drift.
- **Precedence:** TVP takes precedence over grouped priors (DF-2) and the per-geo
  hierarchy for that channel, and it is ignored under the multiplicative
  specification (which uses a max-lift, not an additive coefficient).

## Tests

`tests/test_time_varying_betas.py` — off-has-no-TVP-RVs, on-structure + contract,
grouping-exclusion, drift recovery on the stress DGP (NUTS), reporting trajectory.
