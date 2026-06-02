# 📐 The Mathematics of `mmm_framework`

A seven-notebook companion to the [Aurora story set](README_aurora.md) that opens up the
**math and statistics** under the hood. Where the Aurora notebooks tell *what* a causal MMM
decides, these derive *how* the model is built — every transform, prior, and likelihood term —
with heavy LaTeX and a lot of graphs. Every formula is grounded in the framework's real code
(`mmm_framework.transforms`, `model/base.py`, `mmm_extensions`, `mmm_framework.calibration`), and
most cells end with `assert` statements that encode the math identity they just plotted, so
**"executes clean" means "the math is correct."**

## Read order

| # | Notebook | What it derives | Fits a model? |
|---|----------|-----------------|---------------|
| 0 | `math_00_overview.ipynb` | The MMM as a **generative model**: the full additive `μ_t`, the Normal likelihood, every prior, a pipeline schematic, and one simulated draw from the prior. The index for the set. | No (pure numpy) |
| 1 | `math_01_adstock.ipynb` | **Carryover**: the geometric recurrence (IIR) vs normalized FIR kernels, geometric / delayed / Weibull shapes, the `1/(1−α)` multiplier, and the default model's two-α `Beta(2,2)` blend. | No (pure numpy) |
| 2 | `math_02_saturation.ipynb` | **Diminishing returns**: the core exponential curve `1−e^(−λx)` (half-saturation `ln2/λ`), Hill as the alternative, marginal returns, normalization, and `β·f`. | No (pure numpy) |
| 3 | `math_03_seasonality_trend.ipynb` | The **baseline structure**: Fourier seasonality (harmonics, order as a bias–variance dial) and three trend families (linear, piecewise/Prophet, B-spline partition-of-unity). | No (pure numpy) |
| 4 | `math_04_bayesian_model.ipynb` | The **full Bayesian model**: priors → prior predictive → NUTS → convergence (R̂, ESS, trace) → posterior intervals → posterior-predictive check → decomposition. | Yes (MCMC) |
| 5 | `math_05_calibration.ipynb` | **Identifiability & experiment calibration**: the equifinality trap, the prior route (design factor `K_c`, Gamma moment-matching) and the likelihood route (contribution/ROAS/mROAS estimands), demoed on Aurora. | Yes (MCMC) |
| 6 | `math_06_extensions.ipynb` | **The extension models**: `NestedMMM` mediation (direct vs indirect, `Σ β_{c→m}·γ_m + δ_c`, proportion mediated), `MultivariateMMM` cannibalization (the `ψ` cross-effect + correlated residuals via an MvNormal likelihood), and `CombinedMMM` as their routed synthesis. | Yes (MCMC) |

Notebooks 0–3 bake in seconds; notebooks 4–6 fit small PyMC models (a few minutes each).

## What you'll be able to read off the code afterwards

- Why the core model's saturation is **exponential `1−e^(−λu)`**, not Hill — and why that curve is
  *strictly concave*, not S-shaped (notebook 2 corrects the docstring).
- Why the default adstock parameter `adstock_<channel> ~ Beta(2,2)` is a **blend weight between two
  fixed-α geometric adstocks**, not a decay rate (notebook 1).
- How the priors (`β ~ Gamma(μ=1.5,σ=1)`, `sat_lam ~ Exponential(0.5)`, `σ ~ HalfNormal(0.5)`, …)
  shape the prior predictive (notebooks 0, 4).
- Why **observational adjustment alone is insufficient** (the decay/saturation/β trade-off) and how a
  randomized experiment folded into the likelihood pulls Search's ROAS from an overstated **2.8 down
  to ≈0.66**, its true value (notebook 5).
- How `NestedMMM` recovers that **TV and Display are ~fully mediated** (their effect flows through
  awareness, `proportion_mediated ≈ 1.0` vs the known truth) and how `MultivariateMMM` proves **Cold
  Brew cannibalizes Original** (`ψ` posterior entirely below zero) while their residuals still share a
  demand wave (notebook 6).
- How **piecewise trends** actually work: each changepoint changes the *slope* (not the level), via
  `trend(t) = (k + A·δ)·t + (m + A·γ)` with `γ = −s·δ` keeping it continuous (notebook 3).

## Files

- `build_math_0X_*.py` — each notebook is authored from a self-contained `nbformat` build script
  (same pattern as `build_aurora_notebooks.py`). Regenerate one with
  `uv run python build_math_0X_*.py`, then bake it (below). **Re-running a build script overwrites
  executed outputs**, so execute *after* building.
- `aurora.py` — the math notebooks import the brand palette (and, for notebooks 4–5, the Aurora
  dataset with its known ground truth) from here.

## Running

```bash
cd nbs
# regenerate one notebook from source (optional — they ship with outputs)
uv run python build_math_01_adstock.py
# bake it
PYTHONPATH=$PWD uv run jupyter nbconvert --to notebook --execute --inplace math_01_adstock.ipynb
```

**Notes**
- Notebooks 4–5 use small draw counts (300–400) and `cores=1` to stay fast and crash-free on macOS.
  For real analyses use ≥4 chains and ≥1000 draws/tune.
- Assertions on MCMC results are *directional and seeded* (`random_seed=0`) — they check that the
  estimate moves the right way, never tight equality on a 400-draw fit.
