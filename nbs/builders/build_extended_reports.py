"""Refit the nested + multivariate demo models (post-standardization) and
regenerate their reports."""

import warnings, sys
warnings.filterwarnings("ignore")
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="WARNING")

import numpy as np
from aurora import generate_aurora, CHANNELS

from mmm_framework.mmm_extensions.models import NestedMMM, MultivariateMMM
from mmm_framework.mmm_extensions.builders import (
    MediatorConfigBuilder, NestedModelConfigBuilder,
    MultivariateModelConfigBuilder, OutcomeConfigBuilder, cannibalization_effect)
from mmm_framework.reporting import ReportBuilder
from mmm_framework.reporting.extractors import create_extractor

aurora = generate_aurora()
X = aurora.media_matrix()

# --- Nested (nb03 Part A config) ---
nested_cfg = (NestedModelConfigBuilder()
    .add_mediator(MediatorConfigBuilder("awareness")
                  .partially_observed(observation_noise=0.1)
                  .with_positive_media_effect(sigma=1.0)
                  .with_direct_effect(sigma=0.5).build())
    .map_channels_to_mediator("awareness", ["TV", "Display"])
    .build())
nested = NestedMMM(X, aurora.sales_total, list(CHANNELS), nested_cfg,
                   mediator_data={"awareness": aurora.awareness_survey},
                   index=aurora.weeks)
nested.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)
nested.save("artifacts/aurora_nested")

b = create_extractor(nested).extract()
print("\n--- nested ---")
print("fit:", {k: round(v, 3) for k, v in (b.fit_statistics or {}).items()})
stacked = np.sum(list(b.component_time_series.values()), axis=0)
print("stacked mean %.1f vs predicted mean %.1f vs actual mean %.1f" % (
    stacked.mean(), b.predicted["mean"].mean(), np.asarray(b.actual).mean()))
print("pathways:", {ch: round(p["_total"]["mean"], 2)
                    for ch, p in (b.mediator_pathways or {}).items()})
print("attributed:", b.marketing_attributed_revenue)
print("blended roi:", b.blended_roi)

(ReportBuilder().with_data(b)
    .with_title("Aurora — Nested (Mediation) Model").with_client("Aurora Coffee Co.")
    .enable_all_sections().with_credible_interval(0.8).build()
 ).to_html("artifacts/aurora_nested_report.html")
print("wrote artifacts/aurora_nested_report.html")

# --- Multivariate (nb03 Part B config) ---
_, outcomes = aurora.extension_inputs()
mv_cfg = (MultivariateModelConfigBuilder()
    .add_outcome(OutcomeConfigBuilder("sales_original", column="sales_original")
                 .with_positive_media_effects(sigma=0.5).build())
    .add_outcome(OutcomeConfigBuilder("sales_coldbrew", column="sales_coldbrew")
                 .with_positive_media_effects(sigma=0.5).build())
    .add_cross_effect(cannibalization_effect(source="sales_coldbrew", target="sales_original"))
    .build())
mv = MultivariateMMM(X, outcomes, list(CHANNELS), mv_cfg, index=aurora.weeks)
mv.fit(draws=500, tune=500, chains=2, cores=1, random_seed=0)
mv.save("artifacts/aurora_multivariate")

b = create_extractor(mv).extract()
print("\n--- multivariate ---")
print("fit by outcome:", {k: round(v["r2"], 3)
                          for k, v in (b.fit_statistics_by_product or {}).items()})
stacked = np.sum(list(b.component_time_series.values()), axis=0)
print("stacked mean %.1f vs predicted mean %.1f vs actual mean %.1f" % (
    stacked.mean(), b.predicted["mean"].mean(), np.asarray(b.actual).mean()))
print("cannibalization:", b.cannibalization_matrix)
print("roi:", {k: round(v["mean"], 2) for k, v in (b.channel_roi or {}).items()})
sat = b.saturation_curves["Social"]
print("Social sat curve: x to %.0f, response to %.2f | current spend %.0f" % (
    sat["spend"].max(), sat["response"].max(), b.current_spend["Social"]))

(ReportBuilder().with_data(b)
    .with_title("Aurora — Multivariate (Cannibalization) Model")
    .with_client("Aurora Coffee Co.")
    .enable_all_sections().with_credible_interval(0.8).build()
 ).to_html("artifacts/aurora_multivariate_report.html")
print("wrote artifacts/aurora_multivariate_report.html")
