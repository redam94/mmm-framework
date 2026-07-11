# Causal Inference in Practice — the notebook series

Eleven notebooks that treat marketing measurement as a **causal-inference
problem**, walked up a ladder of evidence on one fictional brand — **Veranda
Home**, a national home & garden retailer — whose dashboard looks the same in
every notebook while the *hidden world behind it* changes.

| # | notebook | rung |
|---|----------|------|
| 00 | `causal_00_the_ladder.ipynb` | Why dashboards (and good MMMs) can't settle causal questions |
| 01 | `causal_01_confounding_and_adjustment.ipynb` | Back-door adjustment: what a control buys, and the disease it can't touch |
| 02 | `causal_02_mmm_as_causal_model.ipynb` | Estimands, bad controls, refutation, honest "I don't know" (realistic 7-channel world) |
| 03 | `causal_03_structural_mediation.ipynb` | The brand funnel from surveys: `StructuralNestedMMM` (full NUTS) |
| 04 | `causal_04_latent_confounders.ipynb` | The economy as a latent confounder: `LatentFactorMMM` vs indicators-as-controls |
| 05 | `causal_05_measuring_one_experiment.ipynb` | Geo design, A/A false-positive calibration, injected-truth power, leaderboard |
| 06 | `causal_06_calibrating_the_model.ipynb` | Prior route vs likelihood route; estimand discipline |
| 07 | `causal_07_many_experiments.ipynb` | Evidence portfolios: ridge resolution, off-panel windows, tension checks |
| 08 | `causal_08_designing_the_next_experiment.ipynb` | Model-anchored power, opportunity cost, Pareto designs, curve identification |
| 09 | `causal_09_planning_the_measurement_series.ipynb` | EIG/EVOI grid, EVPI ceiling, decay, the 12-month calendar |
| 10 | `causal_10_the_closed_loop.ipynb` | Capstone: 3 loop cycles converge on the sealed truth — parameters AND decisions |

## The series' epistemics

Every world is synthetic (`mmm_framework.synth`) and ships a **sealed answer
key** (true per-channel causal contributions, structural parameters, latent
paths). The analyst-voice never sees the key; the narrator unseals it only to
**grade** claims. The grading machinery is itself verified: `C.check_truth(sc)`
re-derives each world's truth from the DGP's own response constants and fails
the bake loudly if the DGP ever drifts. Simulated experiment readouts are
always *truth + sampling noise at the design's SE* — honest experiments, never
oracles. Every headline number is executed as an `assert`.

## Shared module & cache

All notebooks import `causal_common` (`nbs/builders/causal_common.py`): brand
constants and palette, world builders, DGP truth helpers
(`true_media_term` / `true_readout` / `check_truth`), the keyed fit cache, and
chart helpers. Observational fits are fast-but-real numpyro NUTS
(2 chains × 500 draws) cached via `MMMSerializer` under
`$MMM_CAUSAL_CACHE` (default: `<tempdir>/mmm_causal_cache`); the two
structural notebooks (03, 04) run genuine 4-chain NUTS and cache derived
tables. Delete the cache dir to force clean re-fits.

## Rebuild

Notebooks are **generated** — edit the builder, not the `.ipynb`. From `nbs/`:

```bash
# author (per notebook)
uv run --with nbformat python builders/build_causal_00_the_ladder.py

# execute in place (timeouts: 1800 no-fit, 2400 cached-fit, 3600 for 03/04)
TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
    jupyter nbconvert --to notebook --execute --inplace \
    causal/causal_00_the_ladder.ipynb --ExecutePreprocessor.timeout=2400 \
    --ExecutePreprocessor.kernel_name=python3
```

Bake order 00→10 (the shared fit cache warms on first use; the first bake of
03 is the long one, ~12 min of NUTS). Measured-number artifacts land in
`../artifacts/causal_*.json` — the docs-site companion pages
(`docs/causal-NN-*.html`) quote them and must be kept in sync after a re-bake.
