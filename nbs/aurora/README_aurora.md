# ☕ The Aurora Coffee Co. Showcase Notebooks

A six-notebook tour of `mmm_framework`, told as one story: **Aurora Coffee Co.**, a
direct-to-consumer coffee brand, has to decide where next quarter's marketing budget
goes. A dashboard would answer with correlations — and be **expensively wrong**. These
notebooks show how a *causal* marketing-mix model gets it right.

Every notebook runs on one shared synthetic world (`aurora.py`) built with a **known
ground truth**, so the model can be graded against the real answer. The data is seeded
with the exact pathologies a causal MMM exists to handle: **unobserved-demand
confounding**, **mediation** (TV → awareness → sales), and **product cannibalization**.

## Run order

| # | Notebook | The question it answers |
|---|----------|--------------------------|
| 0 | `00_overview.ipynb` | What's the trap? *(the hook — naive correlation vs true ROAS)* |
| 1 | `01_causality.ipynb` | Can we trust the number? *(DAG, identification, bad-control detection, **experiment calibration**)* |
| 2 | `02_base_mmm.ipynb` | What is each channel worth? *(fit, contributions & ROAS with **credible intervals**, marginal ROAS, what-if, diagnostics)* |
| 3 | `03_extended_mmm.ipynb` | What is TV *really* doing? *(NestedMMM mediation; MultivariateMMM cannibalization)* |
| 4 | `04_reporting.ipynb` | How do we present this? *(board-ready themed **HTML report**)* |
| 5 | `05_unified_workflow.ipynb` | Put it together → **a defensible budget reallocation** worth real money |

Read them in order — the story builds. Each is also self-contained (regenerates the
data and fits what it needs), so you can dip into any one.

## The punchline

- The dashboard's hero channel (**Search**, correlation ≈ 0.85) is actually a
  **demand-chasing mirage** (true ROAS 0.66). Its naive ROI of ~2.8 collapses to ~0.68
  once a **geo-lift experiment** is folded into the likelihood.
- **TV and Display** look weak to the base model but are Aurora's **brand engines** —
  the NestedMMM shows their effect flows almost entirely through **awareness**.
- Acting on the dashboard would shift budget exactly the wrong way. The
  experiment-anchored causal plan is worth **~$12M/year more** than the naive plan on
  the same budget (notebook 5).

## Files

- `aurora.py` — the shared synthetic world + brand palette + framework adapters
  (`generate_aurora()`, `AuroraData.base_panel()`, `AuroraData.extension_inputs()`).
- `build_aurora_notebooks.py` — regenerates all six notebooks from source
  (`uv run python builders/build_aurora_notebooks.py`), then execute with
  `jupyter nbconvert --execute`.
- `charts_src.py` — matplotlib code for the inline diagnostic/decision charts, keyed by
  name; the build script bakes these strings into notebook cells. Each block reuses
  objects the notebook already fitted (no extra MCMC) and the Aurora palette.
- `validate_chart_cells.py` — fast regression harness: runs every chart in
  `charts_src.py` against tiny 50-draw fits and `savefig`s them, catching chart-code
  errors in seconds before the expensive full bake
  (`PYTHONPATH=$PWD uv run python validate_chart_cells.py`).
- `artifacts/` — saved models and generated HTML reports (created on run).

## Running

```bash
cd nbs
# regenerate the notebooks from source (optional — they're committed with outputs)
uv run python builders/build_aurora_notebooks.py
# execute one
PYTHONPATH=$PWD uv run jupyter nbconvert --to notebook --execute --inplace aurora/01_causality.ipynb
```

**Notes**
- Fits use small draw counts (300–600) and `cores=1` to stay fast and crash-free on
  macOS. For real analyses use ≥4 chains and ≥1000 draws/tune.
- Report charts load Plotly from a CDN, so rendering the HTML needs a network
  connection; the file is otherwise self-contained.
