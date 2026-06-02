"""Cheap validation harness: fit tiny models, set up the SAME namespace the
notebooks have, and execute every new chart code block with savefig. Reports
pass/fail per chart so all 14 are shaken out against 50-draw fits in seconds
before the expensive full bake. Throwaway."""
import os, sys, traceback, warnings
warnings.filterwarnings("ignore")
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="ERROR")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aurora import generate_aurora, CHANNELS, PRODUCTS, PALETTE, CHANNEL_COLORS

ACCENT, INK, MUTED = PALETTE["accent"], PALETTE["ink"], PALETTE["muted"]
aurora = generate_aurora()

OUT = "/tmp/chartval"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Build the namespace the notebooks have at each chart's insertion point.
# ---------------------------------------------------------------------------
from mmm_framework import (BayesianMMM, ModelConfigBuilder, SeasonalityConfigBuilder,
                           TrendConfig, TrendType)
from mmm_framework.analysis import MMMAnalyzer

panel = aurora.base_panel(control_demand=True)
cfg = (ModelConfigBuilder().bayesian_pymc().with_chains(2).with_draws(50).with_tune(50)
       .with_target_accept(0.9)
       .with_seasonality_builder(SeasonalityConfigBuilder().with_yearly(order=2)).build())
mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
results = mmm.fit(draws=50, tune=50, chains=2, cores=1, random_seed=0)

contrib = mmm.compute_counterfactual_contributions(compute_uncertainty=True, hdi_prob=0.9)
roi = MMMAnalyzer(mmm).compute_channel_roi().set_index("Channel").loc[list(CHANNELS)]
marg = mmm.compute_marginal_contributions(spend_increase_pct=10, compute_uncertainty=True, hdi_prob=0.9)
scenario = mmm.what_if_scenario({"Search": 0.8, "TV": 1.2})

# nb_01 synthetic stand-ins (just to exercise the plotting code)
roi_ctrl = MMMAnalyzer(mmm).compute_channel_roi().set_index("Channel")["ROI"]
roi_cal = roi_ctrl * 0.6
comp = pd.DataFrame({"true ROAS": aurora.true_roas, "demand-blind": roi_ctrl * 1.4,
                     "demand-controlled": roi_ctrl}).loc[list(CHANNELS)]

# nb_05 synthetic stand-ins
recovered = pd.DataFrame({"true ROAS": aurora.true_roas.loc[list(CHANNELS)],
                          "model ROAS": roi_ctrl.loc[list(CHANNELS)]})
spend = aurora.spend[list(CHANNELS)].sum()
causal_delta = pd.Series({"TV": 120.0, "Display": 120.0, "Search": -120.0, "Social": -120.0})

# extensions
from mmm_framework.mmm_extensions.models import NestedMMM, MultivariateMMM
from mmm_framework.mmm_extensions.builders import (
    MediatorConfigBuilder, NestedModelConfigBuilder,
    MultivariateModelConfigBuilder, OutcomeConfigBuilder, cannibalization_effect)

X = aurora.media_matrix()
nested_cfg = (NestedModelConfigBuilder()
    .add_mediator(MediatorConfigBuilder("awareness")
                  .partially_observed(observation_noise=0.1)
                  .with_positive_media_effect(sigma=1.0)
                  .with_direct_effect(sigma=0.5).build())
    .map_channels_to_mediator("awareness", ["TV", "Display"]).build())
nested = NestedMMM(X, aurora.sales_total, list(CHANNELS), nested_cfg,
                   mediator_data={"awareness": aurora.awareness_survey}, index=aurora.weeks)
nested.fit(draws=50, tune=50, chains=2, cores=1, random_seed=0)
med = nested.get_mediation_effects().set_index("channel")     # matches nb_03
brand = ["TV", "Display"]

_, outcomes = aurora.extension_inputs()
mv_cfg = (MultivariateModelConfigBuilder()
    .add_outcome(OutcomeConfigBuilder("sales_original", column="sales_original")
                 .with_positive_media_effects(sigma=0.5).build())
    .add_outcome(OutcomeConfigBuilder("sales_coldbrew", column="sales_coldbrew")
                 .with_positive_media_effects(sigma=0.5).build())
    .add_cross_effect(cannibalization_effect(source="sales_coldbrew", target="sales_original"))
    .build())
mv = MultivariateMMM(X, outcomes, list(CHANNELS), mv_cfg, index=aurora.weeks)
mv.fit(draws=50, tune=50, chains=2, cores=1, random_seed=0)
ce = mv.get_cross_effects_summary()

NS = dict(globals())   # the shared namespace handed to each chart block

# ---------------------------------------------------------------------------
# The 14 chart code blocks (verbatim what goes into the notebooks).
# ---------------------------------------------------------------------------
from charts_src import CHARTS   # single source of truth

# ---------------------------------------------------------------------------
# Run each, savefig, report.
# ---------------------------------------------------------------------------
fails = []
for key, code in CHARTS.items():
    ns = dict(NS)
    _saved = {"n": 0}

    def _save_show(*a, **k):
        plt.gcf().savefig(f"{OUT}/{key}.png", dpi=80, bbox_inches="tight")
        plt.close("all")
        _saved["n"] += 1

    ns["plt"].show = _save_show
    try:
        exec(code, ns)
        ok = _saved["n"] > 0 and os.path.exists(f"{OUT}/{key}.png")
        print(f"  {'OK  ' if ok else 'NOFIG'}  {key}")
        if not ok:
            fails.append((key, "no figure saved (missing plt.show?)"))
    except Exception as e:
        print(f"  FAIL  {key}: {type(e).__name__}: {e}")
        fails.append((key, traceback.format_exc()))
    finally:
        plt.close("all")

print("\n" + "=" * 60)
if fails:
    print(f"{len(fails)} FAILED:\n")
    for key, tb in fails:
        print(f"----- {key} -----\n{tb}\n")
    sys.exit(1)
print(f"ALL {len(CHARTS)} CHARTS OK  ->  {OUT}/")
