# MMM structural-violation stress matrix

`med|err|` / `max|err|` = median / worst |relative error| on per-channel total contribution. `cover` = fraction of channels whose true contribution lands in the 90% credible interval. A **silent failure** is a *representable* scenario where recovery is wrong (median err > 25%, worst channel > 50%, or coverage < 75%) yet **the checks an analyst acts on are green**: MCMC convergence (r-hat < 1.05, no divergences) and the unobserved-confounding robustness value. `ppc` and `refut` are reported but **excluded from the gate** — both are fit-level checks whose verdicts have proven config-sensitive across recordings (under the pre-2026-06-10 trend prior both false-alarmed on the clean control; under current defaults they pass on clean and on several wrong-attribution worlds), so neither is a reliable all-clear for the *causal* claim.

| scenario | assumption broken | med\|err\| | max\|err\| | total-media err | cover | rhat | div | ppc | refut | fragile | verdict |
|---|---|--:|--:|--:|--:|--:|--:|:--:|:--:|---|---|
| clean | — (control) | 7% | 7% | -4% | 100% | 1.01 | 0 | ✓ | ✓ | — | 🟢 ok |
| unobserved_confounding | no unobserved confounding (exogeneity of spend) | 23% | 69% | +4% | 75% | 1.01 | 0 | ✓ | ✗ placebo_treatment | — | 🔴 **SILENT FAILURE** |
| reverse_causality | no unobserved confounding (exogeneity of spend) | 13% | 19% | -6% | 100% | 1.01 | 0 | ✓ | ✗ placebo_treatment | — | 🟢 ok |
| multicollinearity | identifiability (independent spend variation) | 39% | 60% | -5% | 100% | 1.01 | 0 | ✓ | ✗ placebo_treatment | — | 🔴 **SILENT FAILURE** |
| adstock_misspec | adstock functional form / carryover window | 80% | 96% | -68% | 50% | 1.01 | 0 | ✗ | ✗ placebo_treatment | Search,Display | 🟢 ok |
| saturation_misspec | saturation functional form (concavity / no threshold) | 43% | 65% | +38% | 25% | 1.01 | 0 | ✓ | ✓ | — | 🔴 **SILENT FAILURE** |
| time_varying_beta | time-invariant coefficients (stationarity) | 13% | 20% | -10% | 100% | 1.01 | 0 | ✓ | ✓ | — | 🟢 ok |
| heavy_tailed_noise | Gaussian, homoscedastic likelihood | 4% | 37% | -12% | 75% | 1.01 | 0 | ✗ | ✗ placebo_treatment | — | 🟢 ok |
| synergy | additive separability of channels | 4% | 6% | -3% | 100% | 1.01 | 0 | ✓ | ✓ | — | 🟢 ok |
| spend_outliers | robustness of per-channel max-normalization | 39% | 70% | -41% | 0% | 1.01 | 0 | ✗ | ✗ data_subset | — | 🔴 **SILENT FAILURE** |
| negative_effect | positivity of media effects (Gamma prior, beta >= 0) | 19% | 101% | +78% | 50% | 1.02 | 0 | ✗ | ✗ placebo_treatment | Display | ⚪ expected (unrepresentable) |
| trend_break | smooth global trend (no structural breaks) | 27% | 41% | -21% | 100% | 1.02 | 0 | ✗ | ✗ placebo_treatment,data_subset | — | 🔴 **SILENT FAILURE** |
| seasonality_misspec | static low-order Fourier seasonality | 37% | 104% | +11% | 75% | 1.02 | 0 | ✗ | ✗ placebo_treatment,data_subset | — | 🔴 **SILENT FAILURE** |
| dense_controls | a parsimonious, correctly-chosen control set | 16% | 52% | -22% | 75% | 1.01 | 0 | ✗ | ✓ | — | 🔴 **SILENT FAILURE** |
| confounding_controlled | no unobserved confounding (exogeneity of spend) | 10% | 50% | +5% | 100% | 1.02 | 0 | ✓ | ✗ placebo_treatment | — | 🔴 **SILENT FAILURE** |
| aurora_kitchen_sink | multiple (confounding + mediation + non-1-exp saturation) | 92% | 223% | -44% | 25% | 1.02 | 11 | ✗ | ✗ placebo_treatment,data_subset | — | ⚪ expected (unrepresentable) |

## Per-channel detail (silent failures & unrepresentable)

### unobserved_confounding — no unobserved confounding (exogeneity of spend)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,584 | 4,368 | -22% | ✓ | 0.20 | strong |
| Search | 1,750 | 2,960 | +69% | ✗ | 0.22 | weak |
| Social | 2,751 | 3,396 | +23% | ✓ | 0.19 | moderate |
| Display | 2,617 | 2,487 | -5% | ✓ | 0.15 | strong |

### multicollinearity — identifiability (independent spend variation)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 8,828 | 5,498 | -38% | ✓ | 0.12 | moderate |
| Search | 5,573 | 3,660 | -34% | ✓ | 0.11 | moderate |
| Social | 5,147 | 7,252 | +41% | ✓ | 0.15 | moderate |
| Display | 3,485 | 5,580 | +60% | ✓ | 0.13 | moderate |

### saturation_misspec — saturation functional form (concavity / no threshold)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 2,658 | 4,397 | +65% | ✗ | 0.24 | weak |
| Search | 3,868 | 4,294 | +11% | ✓ | 0.22 | strong |
| Social | 3,106 | 4,767 | +53% | ✗ | 0.25 | weak |
| Display | 2,456 | 3,271 | +33% | ✗ | 0.18 | moderate |

### spend_outliers — robustness of per-channel max-normalization

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,668 | 2,866 | -49% | ✗ | 0.39 | strong |
| Search | 4,531 | 3,228 | -29% | ✗ | 0.40 | strong |
| Social | 4,017 | 3,113 | -23% | ✗ | 0.41 | strong |
| Display | 2,958 | 902 | -70% | ✗ | 0.18 | strong |

### negative_effect — positivity of media effects (Gamma prior, beta >= 0)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,668 | 3,892 | -31% | ✗ | 0.19 | moderate |
| Search | 4,531 | 4,249 | -6% | ✓ | 0.18 | moderate |
| Social | 4,017 | 4,168 | +4% | ✓ | 0.18 | strong |
| Display | -7,281 | 53 | +101% | ✗ | 0.09 | moderate |

### trend_break — smooth global trend (no structural breaks)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 4,702 | 2,953 | -37% | ✓ | 0.17 | strong |
| Search | 4,531 | 3,764 | -17% | ✓ | 0.16 | strong |
| Social | 4,017 | 4,256 | +6% | ✓ | 0.18 | moderate |
| Display | 2,505 | 1,482 | -41% | ✓ | 0.11 | moderate |

### seasonality_misspec — static low-order Fourier seasonality

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,668 | 4,728 | -17% | ✓ | 0.18 | strong |
| Search | 4,531 | 3,447 | -24% | ✓ | 0.14 | moderate |
| Social | 2,185 | 4,465 | +104% | ✗ | 0.21 | moderate |
| Display | 2,958 | 4,423 | +50% | ✓ | 0.14 | moderate |

### dense_controls — a parsimonious, correctly-chosen control set

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 4,256 | 3,735 | -12% | ✓ | 0.16 | moderate |
| Search | 2,162 | 1,870 | -13% | ✓ | 0.16 | moderate |
| Social | 3,999 | 3,292 | -18% | ✓ | 0.15 | moderate |
| Display | 2,402 | 1,149 | -52% | ✗ | 0.11 | moderate |

### confounding_controlled — no unobserved confounding (exogeneity of spend)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 5,584 | 4,950 | -11% | ✓ | 0.21 | moderate |
| Search | 1,750 | 2,630 | +50% | ✓ | 0.21 | moderate |
| Social | 2,751 | 2,886 | +5% | ✓ | 0.17 | moderate |
| Display | 2,617 | 2,826 | +8% | ✓ | 0.15 | strong |

### aurora_kitchen_sink — multiple (confounding + mediation + non-1-exp saturation)

| channel | true | est | rel err | in 90% CI | RV | learning |
|---|--:|--:|--:|:--:|--:|---|
| TV | 13,053 | 1,057 | -92% | ✗ | 0.12 | moderate |
| Search | 2,193 | 7,091 | +223% | ✗ | 0.23 | weak |
| Social | 1,508 | 2,899 | +92% | ✓ | 0.14 | prior-dominated |
| Display | 6,037 | 1,718 | -72% | ✗ | 0.12 | moderate |
