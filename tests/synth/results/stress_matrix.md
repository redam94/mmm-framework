# MMM structural-violation stress matrix

`med|err|` / `max|err|` = median / worst |relative error| on per-channel total contribution. `cover` = fraction of channels whose true contribution lands in the 90% credible interval. A **silent failure** is a *representable* scenario where recovery is wrong (median err > 25%, worst channel > 50%, or coverage < 75%) yet **the checks an analyst acts on are green**: MCMC convergence (r-hat < 1.05, no divergences) and the unobserved-confounding robustness value. `ppc` and `refut` are reported but **excluded from the gate** — both refit/resample at low fidelity and fire on the clean control (false positives), so they are not a reliable all-clear. (Where `ppc`=✗ on a flagged row, the failure is silent only to an analyst who has learned to ignore PPC's crying-wolf.)

| scenario | assumption broken | med\|err\| | max\|err\| | total-media err | cover | rhat | div | ppc | refut | fragile | verdict |
|---|---|--:|--:|--:|--:|--:|--:|:--:|:--:|---|---|
| clean | — (control) | 7% | 9% | -4% | 100% | 1.01 | 0 | ✗ | ✗ placebo_treatment | — | 🟢 ok |
| unobserved_confounding | no unobserved confounding (exogeneity of spend) | 41% | 153% | +22% | 50% | 1.01 | 0 | ✓ | ✗ placebo_treatment | — | 🔴 **SILENT FAILURE** |
| reverse_causality | no unobserved confounding (exogeneity of spend) | 14% | 21% | -2% | 100% | 1.01 | 0 | ✗ | ✗ placebo_treatment | — | 🟢 ok |
| multicollinearity | identifiability (independent spend variation) | 29% | 80% | -2% | 100% | 1.01 | 0 | ✗ | ✗ placebo_treatment | — | 🔴 **SILENT FAILURE** |
| adstock_misspec | adstock functional form / carryover window | 77% | 93% | -62% | 50% | 1.01 | 0 | ✗ | ✗ placebo_treatment | Search | 🟢 ok |
| saturation_misspec | saturation functional form (concavity / no threshold) | 46% | 63% | +39% | 25% | 1.01 | 0 | ✓ | ✓ | — | 🔴 **SILENT FAILURE** |
| time_varying_beta | time-invariant coefficients (stationarity) | 11% | 19% | -8% | 100% | 1.01 | 0 | ✗ | ✓ | — | 🟢 ok |
| heavy_tailed_noise | Gaussian, homoscedastic likelihood | 3% | 38% | -11% | 75% | 1.01 | 0 | ✗ | ✗ placebo_treatment | — | 🟢 ok |
| synergy | additive separability of channels | 6% | 6% | -3% | 100% | 1.01 | 0 | ✓ | ✗ placebo_treatment | — | 🟢 ok |
| spend_outliers | robustness of per-channel max-normalization | 46% | 75% | -47% | 0% | 1.01 | 0 | ✗ | ✗ data_subset | — | 🔴 **SILENT FAILURE** |
| negative_effect | positivity of media effects (Gamma prior, beta >= 0) | 19% | 101% | +83% | 50% | 1.01 | 0 | ✗ | ✗ placebo_treatment | Display | ⚪ expected (unrepresentable) |
| confounding_controlled | no unobserved confounding (exogeneity of spend) | 11% | 79% | +12% | 75% | 1.01 | 0 | ✓ | ✗ placebo_treatment | — | 🔴 **SILENT FAILURE** |
| aurora_kitchen_sink | multiple (confounding + mediation + non-1-exp saturation) | 88% | 560% | -16% | 25% | 1.11 | 108 | ✗ | ✗ placebo_treatment | TV | ⚪ expected (unrepresentable) |

## Per-channel detail (silent failures & unrepresentable)

### unobserved_confounding — no unobserved confounding (exogeneity of spend)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,584 | 4,289 | -23% | ✓ | 0.19 | strong |
| Search | 1,750 | 4,430 | +153% | ✗ | 0.24 | weak |
| Social | 2,751 | 4,358 | +58% | ✗ | 0.20 | moderate |
| Display | 2,617 | 2,387 | -9% | ✓ | 0.14 | moderate |

### multicollinearity — identifiability (independent spend variation)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 8,828 | 5,715 | -35% | ✓ | 0.13 | moderate |
| Search | 5,573 | 4,265 | -23% | ✓ | 0.12 | moderate |
| Social | 5,147 | 6,300 | +22% | ✓ | 0.14 | moderate |
| Display | 3,485 | 6,260 | +80% | ✓ | 0.13 | moderate |

### saturation_misspec — saturation functional form (concavity / no threshold)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 2,658 | 4,344 | +63% | ✗ | 0.23 | weak |
| Search | 3,868 | 4,314 | +12% | ✓ | 0.21 | moderate |
| Social | 3,106 | 4,861 | +56% | ✗ | 0.24 | weak |
| Display | 2,456 | 3,342 | +36% | ✗ | 0.19 | moderate |

### spend_outliers — robustness of per-channel max-normalization

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,668 | 1,967 | -65% | ✗ | 0.30 | strong |
| Search | 4,531 | 3,430 | -24% | ✗ | 0.38 | strong |
| Social | 4,017 | 2,964 | -26% | ✗ | 0.37 | strong |
| Display | 2,958 | 747 | -75% | ✗ | 0.16 | strong |

### negative_effect — positivity of media effects (Gamma prior, beta >= 0)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,668 | 3,898 | -31% | ✗ | 0.18 | moderate |
| Search | 4,531 | 4,426 | -2% | ✓ | 0.18 | moderate |
| Social | 4,017 | 4,311 | +7% | ✓ | 0.19 | strong |
| Display | -7,281 | 61 | +101% | ✗ | 0.09 | moderate |

### confounding_controlled — no unobserved confounding (exogeneity of spend)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,584 | 5,141 | -8% | ✓ | 0.21 | moderate |
| Search | 1,750 | 3,134 | +79% | ✗ | 0.21 | strong |
| Social | 2,751 | 2,989 | +9% | ✓ | 0.16 | moderate |
| Display | 2,617 | 2,940 | +12% | ✓ | 0.15 | strong |

### aurora_kitchen_sink — multiple (confounding + mediation + non-1-exp saturation)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 13,053 | 692 | -95% | ✗ | 0.10 | weak |
| Search | 2,193 | 14,480 | +560% | ✗ | 0.36 | weak |
| Social | 1,508 | 2,723 | +81% | ✓ | 0.13 | weak |
| Display | 6,037 | 1,139 | -81% | ✗ | 0.12 | moderate |
