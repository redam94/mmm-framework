"""Author the runtime / scalability benchmark notebook (run from ``nbs/``).

    uv run python builders/build_runtime_benchmark.py
    uv run jupyter nbconvert --to notebook --execute --inplace \
        validation/runtime_benchmark.ipynb --ExecutePreprocessor.timeout=3600

Measures the wall-clock numbers the docs site quotes but never sourced:
fit time by data shape (weeks x channels), draws scaling, the PyMC-vs-NumPyro
sampler gap (docs claim "4-10x"; the walkthrough measured ~2x — this settles
it on the record), geo-panel cost, and the extension-model multipliers
(mediator / multi-outcome) that anchor the technical guide's "computational
scaling" table to a stated baseline. Results land in
``artifacts/runtime_benchmark.json``.

Authored as md/code cells via nbformat (pattern: ``build_mmm_walkthrough.py``).
Timing asserts are deliberately loose (machine noise); the value is the
recorded table, not a pass/fail gate.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


CELLS = [
    md(r"""
# Runtime & Scalability — Measured, On the Record

The docs make runtime claims that were never sourced: a "4–10× speedup" for
NumPyro, a computational-scaling table of multipliers with no stated
baseline, and no answer at all to "how long does a quickstart fit take on a
laptop?" This notebook measures all of it, on disclosed hardware, with the
exact configurations printed beside every number.

Ground rules:

- **Wall-clock, as a practitioner experiences it.** Every timing covers the
  full `fit()` call including model compilation (JAX tracing for NumPyro,
  PyTensor compilation for PyMC). Nobody's first fit skips compilation.
- **One machine, one snapshot.** These numbers transfer as *ratios and
  orders of magnitude*, not as guarantees. Hardware and library versions are
  recorded in the artifact.
- **No convergence gate.** This notebook measures cost, not quality —
  R-hat is recorded per fit for context only.

Measured here: §1 fit time by data shape (weeks × channels), §2 PyMC vs
NumPyro, §3 draws scaling, §4 geo-panel (hierarchical) cost, §5 the
extension-model multipliers (NestedMMM mediator, MultivariateMMM outcomes)
that the technical guide's scaling table asserts.

> Library entry points timed: `BayesianMMM.fit`, `NestedMMM.fit`,
> `MultivariateMMM.fit`. Artifact: `nbs/artifacts/runtime_benchmark.json`.
"""),
    code(r"""
import sys, pathlib, json, time, platform, subprocess, warnings, logging
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
for _n in ("pymc", "numpyro", "jax", "arviz", "pytensor"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)
try:
    from loguru import logger as _loguru
    _loguru.disable("mmm_framework")
except ImportError:
    pass
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # repo root (run from nbs/)

import contextlib, os
@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn), \
            contextlib.redirect_stderr(_dn):
        yield

import pymc, numpyro, jax
from mmm_framework.synth import dgp, dgp_geo

ARTIFACTS = pathlib.Path("../artifacts"); ARTIFACTS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""

HARDWARE = {
    "platform": platform.platform(),
    "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
    "cores": os.cpu_count(),
    "ram_gb": round(int(_sysctl("hw.memsize") or 0) / 2**30) or None,
    "python": platform.python_version(),
    "pymc": pymc.__version__,
    "numpyro": numpyro.__version__,
    "jax": jax.__version__,
}
print(json.dumps(HARDWARE, indent=2))
assert HARDWARE["pymc"] and HARDWARE["numpyro"]
print("✓ hardware + versions recorded — every number below is conditional on this box")
"""),
    md(r"""
## 1 — Fit time by data shape

The workhorse configuration everywhere on this site: parametric (in-graph)
geometric adstock, logistic saturation, linear trend, yearly seasonality,
**NumPyro NUTS, 4 chains × 500 draws (500 tune)**. We fit the realistic
world at two history lengths (104, 156 weeks) and three channel counts
(3, 5, 7 — column subsets of the same spend matrix, so only model size
varies), plus a repeat of the largest fit to show run-to-run variance.

Channels are the expensive dimension: each channel adds its own adstock
kernel, saturation curve, and coefficient inside every gradient
evaluation.
"""),
    code(r"""
import dataclasses
from mmm_framework.config import InferenceMethod, ModelConfig
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

def subset_world(n_weeks, n_channels):
    sc = dgp.build("realistic", n_weeks=n_weeks)
    chs = sc.channels[:n_channels]
    return dataclasses.replace(sc, spend=sc.spend[chs])

def time_fit(panel, *, sampler="numpyro", draws=500, tune=500, chains=4,
             seed=0, **fit_kwargs):
    method = (InferenceMethod.BAYESIAN_NUMPYRO if sampler == "numpyro"
              else InferenceMethod.BAYESIAN_PYMC)
    cfg = ModelConfig(inference_method=method, n_chains=chains, n_draws=draws,
                      n_tune=tune, use_parametric_adstock=True, optim_seed=seed)
    mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
    t0 = time.perf_counter()
    with quiet():
        fit = mmm.fit(random_seed=seed, progressbar=False, **fit_kwargs)
    return time.perf_counter() - t0, fit.diagnostics

GRID = []
for n_weeks in (104, 156):
    for n_ch in (3, 5, 7):
        sc = subset_world(n_weeks, n_ch)
        secs, diag = time_fit(sc.panel())
        GRID.append({"weeks": n_weeks, "channels": n_ch, "seconds": secs,
                     "rhat_max": diag["rhat_max"]})
        print(f"  {n_weeks}w x {n_ch}ch: {secs:5.1f}s  (rhat {diag['rhat_max']:.3f})")

# run-to-run variance on the largest config
secs2, _ = time_fit(subset_world(156, 7).panel(), seed=1)
GRID.append({"weeks": 156, "channels": 7, "seconds": secs2, "rhat_max": np.nan})
print(f"  156w x 7ch (repeat): {secs2:5.1f}s")

grid = pd.DataFrame(GRID)
base = grid[(grid.weeks == 156) & (grid.channels == 7)]["seconds"].mean()
small = float(grid[(grid.weeks == 104) & (grid.channels == 3)]["seconds"].iloc[0])
print(f"\nBASELINE (156w x 7ch, NumPyro 4x500/500): {base:.0f}s mean of 2 runs")

# CLAIM: the full production-size national fit is interactive-scale (well
# under a minute on a laptop), and cost grows from the smallest config.
assert base < 60, f"baseline fit took {base:.0f}s (>60s)"
assert grid["seconds"].max() < 120
assert float(grid[(grid.weeks == 156) & (grid.channels == 7)]["seconds"].min()) > small * 0.8
print("✓ grid measured: production-size national fits are sub-minute on this laptop")
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for n_weeks, gg in grid.groupby("weeks"):
    g = gg.groupby("channels")["seconds"].mean()
    ax.plot(g.index, g.values, "o-", label=f"{n_weeks} weeks")
ax.set(xlabel="media channels", ylabel="fit seconds",
       title="Wall-clock fit time (NumPyro, 4 chains x 500/500)")
ax.set_xticks([3, 5, 7]); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("✓ scaling curve rendered")
"""),
    md(r"""
## 2 — PyMC vs NumPyro: settling the "4–10×" claim

The docs have claimed a "4–10× speedup" from `nuts_sampler="numpyro"`; the
walkthrough's robustness study measured ~2×. Same model, same data
(104 weeks × 7 channels), same 4 × 500/500 budget, both samplers. The PyMC
fit runs chains sequentially (`cores=1` — multiprocessing inside notebook
kernels on macOS is unreliable, and sequential is what a notebook user
typically gets), while NumPyro vectorizes chains in one process; that
end-to-end difference *is* the practitioner-relevant comparison, and the
configuration is disclosed so nobody mistakes it for a per-gradient
microbenchmark.
"""),
    code(r"""
sc_cmp = subset_world(104, 7)
pymc_secs, pymc_diag = time_fit(sc_cmp.panel(), sampler="pymc", cores=1)
numpyro_secs, numpyro_diag = time_fit(sc_cmp.panel(), sampler="numpyro")
speedup = pymc_secs / numpyro_secs

cmp_tbl = pd.DataFrame({
    "sampler": ["pymc (cores=1)", "numpyro"],
    "seconds": [round(pymc_secs, 1), round(numpyro_secs, 1)],
    "rhat_max": [round(pymc_diag["rhat_max"], 3), round(numpyro_diag["rhat_max"], 3)],
}).set_index("sampler")
print(cmp_tbl.to_string())
print(f"\nmeasured NumPyro speedup on this config: {speedup:.1f}x")

# CLAIM: NumPyro is materially faster end-to-end on this model family.
# (The docs' "4-10x" is graded against this measurement in §6.)
assert speedup > 1.2, f"NumPyro speedup only {speedup:.2f}x"
print(f"✓ measured: NumPyro {speedup:.1f}x faster wall-clock at equal draws")
"""),
    md(r"""
## 3 — Draws scaling

Doubling draws should roughly double sampling time, but the fixed costs
(compilation, tuning warm-up) dilute it — useful when budgeting a
production fit (2000+ draws) from a quick exploratory one.
"""),
    code(r"""
sc_full = subset_world(156, 7)
secs_1000, _ = time_fit(sc_full.panel(), draws=1000)
draws_ratio = secs_1000 / base
print(f"156w x 7ch: 500 draws {base:.1f}s -> 1000 draws {secs_1000:.1f}s "
      f"({draws_ratio:.2f}x)")

# CLAIM: more draws cost more, sub-linearly (fixed compile+tune overhead).
assert secs_1000 > base * 0.9, "1000 draws not slower than 500?"
assert draws_ratio < 3.0
print(f"✓ draws scaling measured: 2x draws -> {draws_ratio:.2f}x wall-clock")
"""),
    md(r"""
## 4 — Geo-panel (hierarchical) cost

The geo-clean world: 4 geographies × 130 weeks × 4 channels = 520 panel
rows, with partial pooling across geos and per-cell adstock. Its national
twin (the clean world at 130 weeks, 4 channels, 130 rows) isolates what the
panel structure costs: 4× the rows plus the hierarchy. Both raw wall-clock
and the per-row view are reported — hierarchy overhead and data volume are
different things and the scaling table in the technical guide must not
conflate them.
"""),
    code(r"""
sc_nat = dgp.build("clean", n_weeks=130)
nat_secs, nat_diag = time_fit(sc_nat.panel())

geo_sc = dgp_geo.build("geo_clean")
geo_secs, geo_diag = time_fit(geo_sc.panel())

geo_mult = geo_secs / nat_secs
per_row_mult = (geo_secs / len(geo_sc.y)) / (nat_secs / len(sc_nat.y))
print(f"national  130 rows x 4ch: {nat_secs:.1f}s (rhat {nat_diag['rhat_max']:.3f})")
print(f"geo panel 520 rows x 4ch: {geo_secs:.1f}s (rhat {geo_diag['rhat_max']:.3f})")
print(f"raw multiplier {geo_mult:.1f}x; per-row multiplier {per_row_mult:.1f}x")

# CLAIM: the geo panel costs more than its national twin in wall-clock.
assert geo_secs > nat_secs
print(f"✓ hierarchy cost measured: {geo_mult:.1f}x wall-clock for 4 geos "
      f"({per_row_mult:.1f}x per row)")
"""),
    md(r"""
## 5 — Extension-model multipliers

The technical guide's scaling table claims ~1.3× for an added mediator and
~1.5× for an added outcome — previously with no baseline stated. Here the
baseline is the core `BayesianMMM` on the Aurora world (104 weeks ×
4 channels), and the extensions fit **the same data with the same budget**
(PyMC, 2 chains × 500/500, `cores=1` — the extension models sample via
PyMC):

- **NestedMMM** — one partially-observed mediator (awareness survey),
  two channels routed through it;
- **MultivariateMMM** — two outcomes (Original, Cold Brew) plus one
  cross-effect.
"""),
    code(r"""
import sys
sys.path.insert(0, "../builders")  # aurora.py (shared world/palette) lives with the builders
from aurora import generate_aurora, CHANNELS as A_CHANNELS
from mmm_framework.mmm_extensions.models import NestedMMM, MultivariateMMM
from mmm_framework.mmm_extensions.builders import (
    NestedModelConfigBuilder, MediatorConfigBuilder,
    MultivariateModelConfigBuilder, OutcomeConfigBuilder, cannibalization_effect)

aurora = generate_aurora()
X = aurora.media_matrix()
BRAND = ["TV", "Display"]

# baseline: core BayesianMMM, same data, same sampler family + budget
cfg_base = ModelConfig(inference_method=InferenceMethod.BAYESIAN_PYMC,
                       n_chains=2, n_draws=500, n_tune=500,
                       use_parametric_adstock=True, optim_seed=0)
mmm_base = BayesianMMM(aurora.base_panel(), cfg_base, TrendConfig(type=TrendType.LINEAR))
t0 = time.perf_counter()
with quiet():
    mmm_base.fit(random_seed=0, progressbar=False, cores=1)
base_ext = time.perf_counter() - t0

nested_cfg = (NestedModelConfigBuilder()
              .add_mediator(MediatorConfigBuilder("awareness")
                            .partially_observed(observation_noise=0.1)
                            .with_positive_media_effect(sigma=1.0)
                            .with_direct_effect(sigma=0.5)
                            .build())
              .map_channels_to_mediator("awareness", BRAND)
              .build())
nested = NestedMMM(X, aurora.sales_total, list(A_CHANNELS), nested_cfg,
                   mediator_data={"awareness": aurora.awareness_survey},
                   index=aurora.weeks)
t0 = time.perf_counter()
with quiet():
    nested.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)
nested_secs = time.perf_counter() - t0

_, outcomes = aurora.extension_inputs()
mv_cfg = (MultivariateModelConfigBuilder()
          .add_outcome(OutcomeConfigBuilder("sales_original", column="sales_original")
                       .with_positive_media_effects(sigma=0.5).build())
          .add_outcome(OutcomeConfigBuilder("sales_coldbrew", column="sales_coldbrew")
                       .with_positive_media_effects(sigma=0.5).build())
          .add_cross_effect(cannibalization_effect(source="sales_coldbrew",
                                                   target="sales_original"))
          .build())
mv = MultivariateMMM(X, outcomes, list(A_CHANNELS), mv_cfg, index=aurora.weeks)
t0 = time.perf_counter()
with quiet():
    mv.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)
mv_secs = time.perf_counter() - t0

nested_mult, mv_mult = nested_secs / base_ext, mv_secs / base_ext
ext_tbl = pd.DataFrame({
    "model": ["BayesianMMM (baseline)", "NestedMMM (+1 mediator)",
              "MultivariateMMM (2 outcomes + cross-effect)"],
    "seconds": [round(base_ext, 1), round(nested_secs, 1), round(mv_secs, 1)],
    "multiplier": [1.0, round(nested_mult, 2), round(mv_mult, 2)],
}).set_index("model")
print(ext_tbl.to_string())

# CLAIM: the extensions cost real multiples of the base model on the same
# data -- the scaling table's qualitative story holds; §6 grades the values.
assert base_ext > 0 and nested_secs > 0 and mv_secs > 0
print(f"✓ extension multipliers measured: mediator {nested_mult:.1f}x, "
      f"multi-outcome {mv_mult:.1f}x (claimed: ~1.3x, ~1.5x)")
"""),
    md(r"""
## 6 — Verdict on the docs' claims, and the artifact

Each previously-unsourced docs claim, graded against today's measurement.
The artifact records everything; docs pages must quote it (and say so).
"""),
    code(r"""
verdict = pd.DataFrame([
    {"claim": "NumPyro '4-10x' speedup (getting-started, technical-guide)",
     "measured": f"{speedup:.1f}x wall-clock at equal draws (104w x 7ch)",
     "holds": bool(4 <= speedup <= 10)},
    {"claim": "scaling table: +1 mediator ~1.3x",
     "measured": f"{nested_mult:.1f}x (NestedMMM vs base, same data/budget)",
     "holds": bool(abs(nested_mult - 1.3) <= 0.4)},
    {"claim": "scaling table: +1 outcome ~1.5x",
     "measured": f"{mv_mult:.1f}x (MultivariateMMM, 2 outcomes + cross-effect)",
     "holds": bool(abs(mv_mult - 1.5) <= 0.6)},
    {"claim": "scaling table: hierarchical pooling ~1.5x (no baseline stated)",
     "measured": f"{per_row_mult:.1f}x per row / {geo_mult:.1f}x raw (4 geos)",
     "holds": bool(abs(per_row_mult - 1.5) <= 0.6)},
])
print(verdict.to_string(index=False))

results = {
    "hardware": HARDWARE,
    "config": "parametric adstock, logistic saturation, linear trend, yearly seasonality",
    "grid_numpyro_4x500": grid.round(2).to_dict(orient="records"),
    "baseline_156w_7ch_seconds": round(float(base), 1),
    "sampler_comparison_104w_7ch": {
        "pymc_cores1_seconds": round(pymc_secs, 1),
        "numpyro_seconds": round(numpyro_secs, 1),
        "speedup": round(speedup, 2),
    },
    "draws_scaling_156w_7ch": {"500": round(float(base), 1),
                               "1000": round(secs_1000, 1),
                               "ratio": round(draws_ratio, 2)},
    "geo_panel": {"national_130w_4ch_seconds": round(nat_secs, 1),
                  "geo_4x130w_4ch_seconds": round(geo_secs, 1),
                  "raw_multiplier": round(geo_mult, 2),
                  "per_row_multiplier": round(per_row_mult, 2)},
    "extensions_aurora_104w_4ch_pymc_2x500": {
        "base_seconds": round(base_ext, 1),
        "nested_seconds": round(nested_secs, 1),
        "nested_multiplier": round(nested_mult, 2),
        "multivariate_seconds": round(mv_secs, 1),
        "multivariate_multiplier": round(mv_mult, 2),
    },
    "docs_claims_verdict": verdict.to_dict(orient="records"),
}
out = ARTIFACTS / "runtime_benchmark.json"
out.write_text(json.dumps(results, indent=2))
assert out.exists() and out.stat().st_size > 500
print(f"\n✓ measured numbers recorded -> {out}")
"""),
    md(r"""
## Takeaways

1. **A production-size national fit is interactive.** 156 weeks × 7
   channels, 4 chains × 500 draws on a laptop: see the §1 baseline. Runtime
   is not a barrier to the refit-heavy workflows this site recommends
   (backtests, sensitivity, refutation suites).
2. **Quote the measured sampler gap, not folklore.** The "4–10×" NumPyro
   claim is graded in §6 against an end-to-end measurement with disclosed
   settings; the docs now carry the measured number.
3. **The scaling table has a baseline now.** Multipliers for mediators,
   extra outcomes, and geo hierarchy are measured on named worlds against a
   stated reference fit — and per-row vs raw multipliers are kept distinct.
4. **Ratios travel, absolutes don't.** On different hardware, expect the
   ratios to hold approximately and the absolute seconds to move.

> **Run it yourself.** This notebook is authored by
> `nbs/builders/build_runtime_benchmark.py`; re-baking refreshes
> `nbs/artifacts/runtime_benchmark.json`, which the docs quote. If your
> numbers differ materially, the docs are stale — re-bake and update them.
"""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    path = "validation/runtime_benchmark.ipynb"
    with open(path, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
