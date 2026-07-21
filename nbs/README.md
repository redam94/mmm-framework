# Notebooks

Baked (pre-executed) notebook series, grouped by topic. Every notebook is
authored programmatically by a script in `builders/` — edit the builder, not
the `.ipynb`.

## Layout

| Folder | Series | Guide |
|---|---|---|
| `aurora/` | Aurora Coffee Co. framework tour (00–05: overview, causality, base MMM, extended MMM, reporting, unified workflow) | `aurora/README_aurora.md` |
| `math/` | Mathematics companion (math_00–06: generative model, adstock, saturation, seasonality/trend, Bayesian model, calibration, extensions) | `math/README_math.md` |
| `stress/` | Pressure-test series (stress_00–06: silent failure modes, confounding, extension traps, geo hierarchy) | `stress/README_stress.md` |
| `workshop/` | Beginner Bayesian workshop (workshop_00–05) | `workshop/README_workshop.md` |
| `lifecycle/` | Experimental Measurement Lifecycle, T0→T5 (lifecycle_00–06) | `lifecycle/README_lifecycle.md` |
| `continuous_learning/` | Model-free sequential learning loop (engine walkthrough + "Nomi" story) | — |
| `validation/` | Measured evidence: rolling-origin backtest, runtime benchmark, Pinkham real-data pressure test | — |
| `demos/` | Standalone walkthroughs: modeling workflow, messy-data onboarding, causal features showcase, PPTX deck demo, every fit method compared (MAP/Laplace/ADVI/Pathfinder/SMC vs NUTS), experiment-planning playbook (method registry → power → EIG/EVOI → net-value Pareto optimizer → calibration), LTV problems & the Atelier models (BG/NBD+Gamma-Gamma CLV, binomial awareness, CFA, LCA, long-term brand — each on synthetic data with a known answer key) | — |
| `extensions/` | Bespoke/extension models: structural nested MMM, breakout weighting (+ `_recovery/` nested-mediation recovery harness) | — |
| `legacy/` | Old scratch notebooks and generated outputs (untracked HTML/xlsx/csv) | — |
| `builders/` | All `build_*.py` authoring scripts + shared runtime modules (`aurora.py`, `charts_src.py`, `lifecycle_common.py`, `validate_chart_cells.py`) | — |
| `artifacts/` | Shared baked outputs (JSON metrics the docs quote, GIF animations, saved models, reports). Do not move — `.gitignore`, `scripts/`, and the docs site pin these paths. | — |

## Conventions

- **Author** (from `nbs/`): `uv run python builders/build_<name>.py` — writes
  `<topic>/<name>.ipynb` (unexecuted).
- **Execute** (from `nbs/`):
  `uv run jupyter nbconvert --to notebook --execute --inplace <topic>/<name>.ipynb`
  — the kernel's working directory is the notebook's topic folder, so emitted
  cells reference `../artifacts/`, add `../builders` to `sys.path` for shared
  modules (`aurora`, `lifecycle_common`, sibling builders), and reach the repo
  root via `Path.cwd().parents[1]`.
- Each builder's docstring carries its exact bake command (timeouts included).
- Notebooks baked before the 2026-07-10 reorg contain the old flat-layout
  paths in their cells; they render fine, but re-executing one requires
  re-authoring it first (run its builder, then nbconvert).
