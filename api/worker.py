"""
ARQ Worker for async model fitting tasks.

Run with: arq api.worker.WorkerSettings
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add the parent directory containing mmm_framework to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from arq import cron
from arq.connections import RedisSettings

from config import get_settings
from schemas import JobStatus
from storage import get_storage
import logging

logging.basicConfig(level=logging.DEBUG)


# =============================================================================
# Task Functions
# =============================================================================


async def fit_model_task(
    ctx: dict,
    model_id: str,
    data_id: str,
    config_id: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Async task to fit a Bayesian MMM model.

    Parameters
    ----------
    ctx : dict
        ARQ context with Redis connection.
    model_id : str
        Model ID for tracking.
    data_id : str
        ID of the dataset to use.
    config_id : str
        ID of the configuration to use.
    overrides : dict, optional
        Parameter overrides (n_chains, n_draws, etc.).

    Returns
    -------
    dict
        Results summary.
    """
    import redis.asyncio as aioredis
    import logging

    logger = logging.getLogger(__name__)

    storage = get_storage()
    redis_client: aioredis.Redis = ctx["redis"]

    # DEBUG: Log storage path and model existence
    logger.info(f"Worker storage base path: {storage}")
    logger.info(f"Looking for model_id: {model_id}")
    logger.info(f"Model exists check: {storage.model_exists(model_id)}")

    async def update_status(
        status: JobStatus, progress: float = 0.0, message: str | None = None, **extra
    ):
        """Update job status in Redis."""
        job_key = f"mmm:job:{model_id}"
        await redis_client.hset(
            job_key,
            mapping={
                "status": status.value,
                "progress": str(progress),
                "progress_message": message or "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **{k: str(v) if v is not None else "" for k, v in extra.items()},
            },
        )

    try:
        # Update status to running
        await update_status(JobStatus.RUNNING, 0.0, "Initializing...")

        # Update model metadata
        storage.update_model_metadata(
            model_id,
            {
                "status": JobStatus.RUNNING.value,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Load data
        await update_status(JobStatus.RUNNING, 5.0, "Loading data...")
        df = storage.load_data(data_id)

        # Load config
        await update_status(JobStatus.RUNNING, 10.0, "Loading configuration...")
        config = storage.load_config(config_id)
        mff_config_dict = config["mff_config"]
        model_settings_dict = config["model_settings"]

        # Apply overrides
        if overrides:
            for key in ["n_chains", "n_draws", "n_tune", "random_seed"]:
                if key in overrides and overrides[key] is not None:
                    model_settings_dict[key] = overrides[key]

        # Import MMM framework
        await update_status(JobStatus.RUNNING, 15.0, "Building model configuration...")

        from mmm_framework import (
            BayesianMMM,
            TrendConfig,
            TrendType,
            load_mff,
            MFFConfigBuilder,
            KPIConfigBuilder,
            MediaChannelConfigBuilder,
            ControlVariableConfigBuilder,
            DimensionAlignmentConfigBuilder,
            ModelConfigBuilder,
            HierarchicalConfigBuilder,
            SeasonalityConfigBuilder,
            TrendConfigBuilder,
        )
        import pytensor

        pytensor.config.exception_verbosity = "high"
        pytensor.config.mode == "NUMBA"
        pytensor.config.cxx = ""
        # Build MFF config from dict
        mff_builder = MFFConfigBuilder()

        # KPI
        kpi_dict = mff_config_dict["kpi"]
        kpi_builder = KPIConfigBuilder(kpi_dict["name"])
        if kpi_dict.get("dimensions") == ["Period"]:
            kpi_builder.national()
        elif "Geography" in kpi_dict.get("dimensions", []):
            if "Product" in kpi_dict.get("dimensions", []):
                kpi_builder.by_geo_and_product()
            else:
                kpi_builder.by_geo()
        elif "Product" in kpi_dict.get("dimensions", []):
            kpi_builder.by_product()

        if kpi_dict.get("log_transform"):
            kpi_builder.multiplicative()

        mff_builder.with_kpi_builder(kpi_builder)

        # Media channels
        for media_dict in mff_config_dict.get("media_channels", []):
            media_builder = MediaChannelConfigBuilder(media_dict["name"])

            # Handle dimensions
            media_dims = media_dict.get("dimensions", ["Period"])
            if media_dims == ["Period"]:
                media_builder.national()
            elif "Geography" in media_dims and "Product" in media_dims:
                media_builder.by_geo_and_product()
            elif "Geography" in media_dims:
                media_builder.by_geo()
            elif "Product" in media_dims:
                media_builder.by_product()

            # Handle adstock type
            adstock_config = media_dict.get("adstock", {})
            l_max = adstock_config.get("l_max", 8)
            adstock_type = adstock_config.get("type", "geometric")

            if adstock_type == "geometric":
                media_builder.with_geometric_adstock(l_max)
            elif adstock_type == "weibull":
                media_builder.with_weibull_adstock(l_max)
            elif adstock_type == "delayed":
                media_builder.with_delayed_adstock(l_max)
            elif adstock_type == "none":
                pass  # No adstock

            # Handle saturation type
            sat_config = media_dict.get("saturation", {})
            sat_type = sat_config.get("type", "hill")

            if sat_type == "hill":
                media_builder.with_hill_saturation()
            elif sat_type == "logistic":
                media_builder.with_logistic_saturation()
            elif sat_type == "michaelis_menten":
                media_builder.with_michaelis_menten_saturation()
            elif sat_type == "tanh":
                media_builder.with_tanh_saturation()
            elif sat_type == "none":
                pass  # No saturation

            # Handle display name
            if media_dict.get("display_name"):
                media_builder.with_display_name(media_dict["display_name"])

            # Handle parent channel for hierarchical
            if media_dict.get("parent_channel"):
                media_builder.with_parent_channel(media_dict["parent_channel"])

            mff_builder.add_media_builder(media_builder)

        # Controls
        for control_dict in mff_config_dict.get("controls", []):
            control_builder = ControlVariableConfigBuilder(control_dict["name"])

            # Handle dimensions
            control_dims = control_dict.get("dimensions", ["Period"])
            if control_dims == ["Period"]:
                control_builder.national()
            elif "Geography" in control_dims and "Product" in control_dims:
                control_builder.by_geo_and_product()
            elif "Geography" in control_dims:
                control_builder.by_geo()
            elif "Product" in control_dims:
                control_builder.by_product()

            # Handle sign constraint
            if control_dict.get("allow_negative", True):
                control_builder.allow_negative()
            else:
                control_builder.positive_only()

            # Handle display name
            if control_dict.get("display_name"):
                control_builder.with_display_name(control_dict["display_name"])

            # Handle shrinkage
            if control_dict.get("use_shrinkage", False):
                control_builder.with_shrinkage()

            mff_builder.add_control_builder(control_builder)

        # Alignment
        alignment_dict = mff_config_dict.get("alignment", {})
        align_builder = DimensionAlignmentConfigBuilder()
        geo_alloc = alignment_dict.get("geo_allocation", "equal")
        if geo_alloc == "equal":
            align_builder.geo_equal()
        elif geo_alloc == "sales":
            align_builder.geo_by_sales()
        elif geo_alloc == "population":
            align_builder.geo_by_population()
        mff_builder.with_alignment_builder(align_builder)

        mff_config = mff_builder.build()

        # Load panel
        await update_status(JobStatus.RUNNING, 20.0, "Loading panel data...")
        panel = load_mff(df, mff_config)

        # Build model config
        await update_status(JobStatus.RUNNING, 25.0, "Building model...")

        model_builder = ModelConfigBuilder()

        inference = model_settings_dict.get("inference_method", "bayesian_pymc")
        if inference == "bayesian_numpyro":
            model_builder.bayesian_numpyro()
        else:
            model_builder.bayesian_pymc()

        model_builder.with_chains(model_settings_dict.get("n_chains", 4))
        model_builder.with_draws(model_settings_dict.get("n_draws", 1000))
        model_builder.with_tune(model_settings_dict.get("n_tune", 1000))
        model_builder.with_target_accept(model_settings_dict.get("target_accept", 0.9))

        # Seasonality
        season_dict = model_settings_dict.get("seasonality", {})
        season_builder = SeasonalityConfigBuilder()
        yearly_order = season_dict.get("yearly", 2)
        if yearly_order and yearly_order > 0:
            season_builder.with_yearly(yearly_order)
        model_builder.with_seasonality_builder(season_builder)

        # Hierarchical
        hier_dict = model_settings_dict.get("hierarchical", {})
        if hier_dict.get("enabled", True):
            hier_builder = HierarchicalConfigBuilder().enabled()
            if hier_dict.get("pool_across_geo", True):
                hier_builder.pool_across_geo()
            model_builder.with_hierarchical_builder(hier_builder)

        model_cfg = model_builder.build()

        # Build trend config
        trend_dict = model_settings_dict.get("trend", {})
        trend_builder = TrendConfigBuilder()

        trend_type = trend_dict.get("type", "linear")
        if trend_type == "none":
            trend_builder.none()
        elif trend_type == "linear":
            trend_builder.linear()
        elif trend_type == "piecewise":
            trend_builder.piecewise()
            if "n_changepoints" in trend_dict:
                trend_builder.with_n_changepoints(trend_dict["n_changepoints"])
            if "changepoint_range" in trend_dict:
                trend_builder.with_changepoint_range(trend_dict["changepoint_range"])
            if "changepoint_prior_scale" in trend_dict:
                trend_builder.with_changepoint_prior_scale(
                    trend_dict["changepoint_prior_scale"]
                )
        elif trend_type == "spline":
            trend_builder.spline()
            if "n_knots" in trend_dict:
                trend_builder.with_n_knots(trend_dict["n_knots"])
        elif trend_type == "gaussian_process":
            trend_builder.gaussian_process()
            if "gp_n_basis" in trend_dict:
                trend_builder.with_gp_n_basis(trend_dict["gp_n_basis"])

        trend_config = trend_builder.build()

        # Create MMM
        await update_status(JobStatus.RUNNING, 30.0, "Initializing Bayesian model...")
        mmm = BayesianMMM(panel, model_cfg, trend_config)

        # Fit model
        await update_status(
            JobStatus.RUNNING, 35.0, "Fitting model (this may take several minutes)..."
        )

        random_seed = model_settings_dict.get("random_seed", 42)
        results = mmm.fit(random_seed=random_seed)

        print("DEBUG: Model fitted successfully")

        await update_status(JobStatus.RUNNING, 85.0, "Processing results...")

        # Extract diagnostics
        print("DEBUG: Extracting diagnostics...")
        try:
            diagnostics = {
                "divergences": int(results.diagnostics.get("divergences", 0)),
                "rhat_max": float(results.diagnostics.get("rhat_max", 1.0)),
                "ess_bulk_min": float(results.diagnostics.get("ess_bulk_min", 0)),
            }
            print(f"DEBUG: Diagnostics extracted: {diagnostics}")
        except Exception as e:
            print(f"DEBUG: Failed at diagnostics: {e}")
            raise

        # Get parameter summary
        print("DEBUG: Getting parameter summary...")
        try:
            summary_df = results.summary()
            print(f"DEBUG: Summary df type: {type(summary_df)}")
        except Exception as e:
            print(f"DEBUG: Failed at results.summary(): {e}")
            raise

        print("DEBUG: Building param_summary list...")
        try:
            param_summary = []
            for idx in summary_df.index:
                row = summary_df.loc[idx]
                param_summary.append(
                    {
                        "parameter": str(idx),
                        "mean": float(row.get("mean", 0)),
                        "sd": float(row.get("sd", 0)),
                        "hdi_3%": float(row.get("hdi_3%", 0)),
                        "hdi_97%": float(row.get("hdi_97%", 0)),
                        "r_hat": (
                            float(row.get("r_hat", 1.0)) if "r_hat" in row else None
                        ),
                    }
                )
            print(f"DEBUG: param_summary built, {len(param_summary)} params")
        except Exception as e:
            print(f"DEBUG: Failed building param_summary: {e}")
            raise

        # Save artifacts
        await update_status(JobStatus.RUNNING, 90.0, "Saving model artifacts...")

        print("DEBUG: Saving mmm artifact...")
        try:
            storage.save_model_artifact(model_id, "mmm", mmm)
            print("DEBUG: mmm saved")
        except Exception as e:
            print(f"DEBUG: Failed saving mmm: {e}")
            raise

        print("DEBUG: Saving results artifact...")
        try:
            storage.save_model_artifact(model_id, "results", results)
            print("DEBUG: results saved")
        except Exception as e:
            print(f"DEBUG: Failed saving results: {e}")
            raise

        print("DEBUG: Saving panel artifact...")
        try:
            storage.save_model_artifact(model_id, "panel", panel)
            print("DEBUG: panel saved")
        except Exception as e:
            print(f"DEBUG: Failed saving panel: {e}")
            raise

        # Save results summary
        print("DEBUG: Saving results summary...")
        try:
            results_summary = {
                "model_id": model_id,
                "diagnostics": diagnostics,
                "parameter_summary": param_summary,
                "n_obs": int(mmm.n_obs),
                "n_channels": int(mmm.n_channels),
                "n_controls": int(mmm.n_controls),
                "channel_names": mmm.channel_names,
                "control_names": mmm.control_names,
                "y_mean": float(mmm.y_mean),
                "y_std": float(mmm.y_std),
            }
            storage.save_results(model_id, "summary", results_summary)
            print("DEBUG: results_summary saved")
        except Exception as e:
            print(f"DEBUG: Failed saving results_summary: {e}")
            raise

        # Update final status
        print("DEBUG: Updating final status...")
        await update_status(
            JobStatus.COMPLETED,
            100.0,
            "Model fitting completed successfully",
        )

        print("DEBUG: Updating model metadata...")
        storage.update_model_metadata(
            model_id,
            {
                "status": JobStatus.COMPLETED.value,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "diagnostics": diagnostics,
            },
        )

        print("DEBUG: All done, returning result")
        return {
            "model_id": model_id,
            "status": "completed",
            "diagnostics": diagnostics,
        }
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)

        error_msg = str(e)
        traceback_str = traceback.format_exc()
        logger.error(f"Model fitting failed for {model_id}: {error_msg}")
        logger.error(traceback_str)

        # Update Redis status to FAILED
        try:
            await update_status(
                JobStatus.FAILED,
                progress=0.0,
                message=f"Model fitting failed: {error_msg}",
            )
        except Exception as redis_err:
            logger.error(f"Failed to update Redis status: {redis_err}")

        # Update model metadata
        try:
            storage.update_model_metadata(
                model_id,
                {
                    "status": JobStatus.FAILED.value,
                    "error_message": error_msg,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as storage_err:
            logger.error(f"Failed to update model metadata: {storage_err}")

        return {
            "model_id": model_id,
            "status": "failed",
            "error": error_msg,
        }


async def compute_contributions_task(
    ctx: dict,
    model_id: str,
    time_period: tuple[int, int] | None = None,
    channels: list[str] | None = None,
    compute_uncertainty: bool = True,
    hdi_prob: float = 0.94,
) -> dict[str, Any]:
    """Compute counterfactual contributions for a fitted model."""
    storage = get_storage()

    try:
        # Load fitted model
        mmm = storage.load_model_artifact(model_id, "mmm")

        # Compute contributions
        contrib_results = mmm.compute_counterfactual_contributions(
            time_period=time_period,
            channels=channels,
            compute_uncertainty=compute_uncertainty,
            hdi_prob=hdi_prob,
            random_seed=42,
        )

        # Convert to serializable format
        results = {
            "model_id": model_id,
            "total_contributions": contrib_results.total_contributions.to_dict(),
            "contribution_pct": contrib_results.contribution_pct.to_dict(),
            "time_period": time_period,
        }

        if contrib_results.contribution_hdi_low is not None:
            results["contribution_hdi_low"] = (
                contrib_results.contribution_hdi_low.to_dict()
            )
            results["contribution_hdi_high"] = (
                contrib_results.contribution_hdi_high.to_dict()
            )

        # Cache results
        storage.save_results(model_id, "contributions", results)

        return results

    except Exception as e:
        return {
            "model_id": model_id,
            "error": str(e),
        }


async def run_scenario_task(
    ctx: dict,
    model_id: str,
    spend_changes: dict[str, float],
    time_period: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Run what-if scenario analysis."""
    storage = get_storage()

    try:
        mmm = storage.load_model_artifact(model_id, "mmm")

        scenario_results = mmm.what_if_scenario(
            spend_changes=spend_changes,
            time_period=time_period,
            random_seed=42,
        )

        # Convert to serializable
        results = {
            "model_id": model_id,
            "baseline_outcome": float(scenario_results["baseline_outcome"]),
            "scenario_outcome": float(scenario_results["scenario_outcome"]),
            "outcome_change": float(scenario_results["outcome_change"]),
            "outcome_change_pct": float(scenario_results["outcome_change_pct"]),
            "spend_changes": scenario_results["spend_changes"],
        }

        return results

    except Exception as e:
        return {
            "model_id": model_id,
            "error": str(e),
        }


async def fit_extended_model_task(
    ctx: dict,
    model_id: str,
    data_id: str,
    config_id: str,
    model_type: str,  # "nested", "multivariate", or "combined"
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Async task to fit an extended MMM model (Nested, Multivariate, or Combined).

    Parameters
    ----------
    ctx : dict
        ARQ context with Redis connection.
    model_id : str
        Model ID for tracking.
    data_id : str
        ID of the dataset to use.
    config_id : str
        ID of the extended configuration to use.
    model_type : str
        Type of extended model: "nested", "multivariate", or "combined".
    overrides : dict, optional
        Parameter overrides (n_chains, n_draws, etc.).

    Returns
    -------
    dict
        Results summary.
    """
    import redis.asyncio as aioredis
    import logging

    logger = logging.getLogger(__name__)

    storage = get_storage()
    redis_client: aioredis.Redis = ctx["redis"]

    async def update_status(
        status: JobStatus, progress: float = 0.0, message: str | None = None, **extra
    ):
        """Update job status in Redis."""
        job_key = f"mmm:job:{model_id}"
        await redis_client.hset(
            job_key,
            mapping={
                "status": status.value,
                "progress": str(progress),
                "progress_message": message or "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **{k: str(v) if v is not None else "" for k, v in extra.items()},
            },
        )

    try:
        # Update status to running
        await update_status(JobStatus.RUNNING, 0.0, "Initializing extended model...")

        # Update model metadata
        storage.update_model_metadata(
            model_id,
            {
                "status": JobStatus.RUNNING.value,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "model_type": model_type,
            },
        )

        # Load data
        await update_status(JobStatus.RUNNING, 5.0, "Loading data...")
        df = storage.load_data(data_id)

        # Load extended config
        await update_status(JobStatus.RUNNING, 10.0, "Loading configuration...")
        config = storage.load_config(config_id)

        # Extract model settings
        model_settings = config.get("model_settings", {})
        if overrides:
            for key in ["n_chains", "n_draws", "n_tune", "random_seed"]:
                if key in overrides and overrides[key] is not None:
                    model_settings[key] = overrides[key]

        # Import extended models
        await update_status(
            JobStatus.RUNNING, 15.0, "Building extended model configuration..."
        )

        from mmm_framework.mmm_extensions import (
            NestedMMM,
            MultivariateMMM,
            CombinedMMM,
            NestedModelConfigBuilder,
            MultivariateModelConfigBuilder,
            CombinedModelConfigBuilder,
            MediatorConfigBuilder,
            OutcomeConfigBuilder,
            CrossEffectConfigBuilder,
        )
        from mmm_framework.mmm_extensions.config import (
            MediatorType,
            CrossEffectType,
            EffectConstraint,
        )

        # Prepare data arrays from DataFrame
        await update_status(JobStatus.RUNNING, 20.0, "Preparing data arrays...")

        # Get media channel columns
        media_columns = config.get("media_channels", [])
        if not media_columns:
            raise ValueError("No media channels specified in config")

        channel_names = [c["name"] if isinstance(c, dict) else c for c in media_columns]
        X_media = df[channel_names].values

        # Get outcome column(s)
        outcome_columns = config.get("outcomes", [])
        if not outcome_columns:
            # Fallback to KPI for nested models
            kpi = config.get("kpi", {})
            y = df[kpi.get("name", "Sales")].values
        else:
            outcome_names = [
                o["name"] if isinstance(o, dict) else o for o in outcome_columns
            ]

        # Build model based on type
        await update_status(JobStatus.RUNNING, 25.0, f"Building {model_type} model...")

        if model_type == "nested":
            # Build nested model config
            nested_config_dict = config.get("nested_config", {})
            mediators_list = nested_config_dict.get("mediators", [])

            builder = NestedModelConfigBuilder()
            for med_dict in mediators_list:
                med_builder = MediatorConfigBuilder(med_dict["name"])

                # Set mediator type
                med_type = med_dict.get("type", "partially_observed")
                if med_type == "fully_latent":
                    med_builder.fully_latent()
                elif med_type == "fully_observed":
                    med_builder.fully_observed(med_dict.get("observation_noise", 0.05))
                else:
                    med_builder.partially_observed(
                        med_dict.get("observation_noise", 0.1)
                    )

                # Set effects
                if med_dict.get("positive_media_effect", True):
                    med_builder.with_positive_media_effect()

                if not med_dict.get("allow_direct_effect", True):
                    med_builder.without_direct_effect()

                builder.add_mediator(med_builder.build())

            # Add channel-mediator mappings
            for med_name, channels in nested_config_dict.get(
                "media_to_mediator_map", {}
            ).items():
                builder.map_channels_to_mediator(med_name, channels)

            nested_config = builder.build()

            # Get mediator data
            mediator_data = {}
            mediator_masks = {}
            for med_dict in mediators_list:
                med_name = med_dict["name"]
                if med_name in df.columns:
                    mediator_data[med_name] = df[med_name].values
                    mediator_masks[med_name] = ~df[med_name].isna().values

            # Create model
            model = NestedMMM(
                X_media=X_media,
                y=y,
                channel_names=channel_names,
                config=nested_config,
                mediator_data=mediator_data,
                mediator_masks=mediator_masks,
                index=df.index,
            )

        elif model_type == "multivariate":
            # Build multivariate model config
            mv_config_dict = config.get("multivariate_config", {})
            outcomes_list = mv_config_dict.get("outcomes", [])

            builder = MultivariateModelConfigBuilder()

            # Add outcomes
            for out_dict in outcomes_list:
                out_builder = OutcomeConfigBuilder(
                    out_dict["name"], out_dict.get("column")
                )
                if out_dict.get("positive_media_effects", False):
                    out_builder.with_positive_media_effects()
                builder.add_outcome(out_builder.build())

            # Add cross-effects
            for ce_dict in mv_config_dict.get("cross_effects", []):
                ce_builder = CrossEffectConfigBuilder(
                    ce_dict["source_outcome"],
                    ce_dict["target_outcome"],
                )

                effect_type = ce_dict.get("effect_type", "cannibalization")
                if effect_type == "halo":
                    ce_builder.halo()
                elif effect_type == "symmetric":
                    ce_builder.symmetric()
                else:
                    ce_builder.cannibalization()

                if ce_dict.get("prior_sigma"):
                    ce_builder.with_prior_sigma(ce_dict["prior_sigma"])

                builder.add_cross_effect(ce_builder.build())

            # Set LKJ eta
            if mv_config_dict.get("lkj_eta"):
                builder.with_lkj_eta(mv_config_dict["lkj_eta"])

            mv_config = builder.build()

            # Get outcome data
            outcome_data = {}
            for out_dict in outcomes_list:
                name = out_dict["name"]
                col = out_dict.get("column", name)
                outcome_data[name] = df[col].values

            # Get promotion data if specified
            promotion_data = {}
            for ce_dict in mv_config_dict.get("cross_effects", []):
                if ce_dict.get("promotion_column"):
                    col = ce_dict["promotion_column"]
                    if col in df.columns:
                        promotion_data[col] = df[col].values

            # Create model
            model = MultivariateMMM(
                X_media=X_media,
                outcome_data=outcome_data,
                channel_names=channel_names,
                config=mv_config,
                promotion_data=promotion_data,
                index=df.index,
            )

        elif model_type == "combined":
            # Build combined model config (nested + multivariate)
            combined_config_dict = config.get("combined_config", {})
            nested_part = combined_config_dict.get("nested", {})
            mv_part = combined_config_dict.get("multivariate", {})

            builder = CombinedModelConfigBuilder()

            # Add mediators
            for med_dict in nested_part.get("mediators", []):
                med_builder = MediatorConfigBuilder(med_dict["name"])
                med_type = med_dict.get("type", "partially_observed")
                if med_type == "fully_latent":
                    med_builder.fully_latent()
                elif med_type == "fully_observed":
                    med_builder.fully_observed(med_dict.get("observation_noise", 0.05))
                else:
                    med_builder.partially_observed(
                        med_dict.get("observation_noise", 0.1)
                    )

                if med_dict.get("positive_media_effect", True):
                    med_builder.with_positive_media_effect()

                builder.add_mediator(med_builder.build())

            # Add outcomes
            for out_dict in mv_part.get("outcomes", []):
                out_builder = OutcomeConfigBuilder(
                    out_dict["name"], out_dict.get("column")
                )
                builder.add_outcome(out_builder.build())

            # Add cross-effects
            for ce_dict in mv_part.get("cross_effects", []):
                if ce_dict.get("effect_type") == "halo":
                    builder.with_halo_effect(
                        ce_dict["source_outcome"], ce_dict["target_outcome"]
                    )
                else:
                    builder.with_cannibalization(
                        ce_dict["source_outcome"],
                        ce_dict["target_outcome"],
                        ce_dict.get("promotion_column"),
                    )

            # Map mediators to outcomes
            for med_name, outcomes in combined_config_dict.get(
                "mediator_to_outcome_map", {}
            ).items():
                builder.map_mediator_to_outcomes(med_name, outcomes)

            combined_config = builder.build()

            # Get data
            mediator_data = {}
            mediator_masks = {}
            for med_dict in nested_part.get("mediators", []):
                med_name = med_dict["name"]
                if med_name in df.columns:
                    mediator_data[med_name] = df[med_name].values
                    mediator_masks[med_name] = ~df[med_name].isna().values

            outcome_data = {}
            for out_dict in mv_part.get("outcomes", []):
                name = out_dict["name"]
                col = out_dict.get("column", name)
                outcome_data[name] = df[col].values

            promotion_data = {}
            for ce_dict in mv_part.get("cross_effects", []):
                if ce_dict.get("promotion_column"):
                    col = ce_dict["promotion_column"]
                    if col in df.columns:
                        promotion_data[col] = df[col].values

            # Create model
            model = CombinedMMM(
                X_media=X_media,
                outcome_data=outcome_data,
                channel_names=channel_names,
                config=combined_config,
                mediator_data=mediator_data,
                mediator_masks=mediator_masks,
                promotion_data=promotion_data,
                index=df.index,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Fit model
        await update_status(
            JobStatus.RUNNING,
            35.0,
            f"Fitting {model_type} model (this may take several minutes)...",
        )

        n_draws = model_settings.get("n_draws", 1000)
        n_tune = model_settings.get("n_tune", 1000)
        n_chains = model_settings.get("n_chains", 4)
        target_accept = model_settings.get("target_accept", 0.9)
        random_seed = model_settings.get("random_seed", 42)

        results = model.fit(
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
        )

        logger.info(f"Extended model {model_id} fitted successfully")

        await update_status(JobStatus.RUNNING, 85.0, "Processing results...")

        # Extract diagnostics
        try:
            import arviz as az

            summary = az.summary(results.trace)
            diagnostics = {
                "divergences": int(
                    results.trace.sample_stats.get("diverging", []).sum()
                    if hasattr(results.trace, "sample_stats")
                    else 0
                ),
                "rhat_max": (
                    float(summary["r_hat"].max()) if "r_hat" in summary.columns else 1.0
                ),
                "ess_bulk_min": (
                    float(summary["ess_bulk"].min())
                    if "ess_bulk" in summary.columns
                    else 0
                ),
            }
        except Exception as e:
            logger.warning(f"Could not extract diagnostics: {e}")
            diagnostics = {"divergences": 0, "rhat_max": 1.0, "ess_bulk_min": 0}

        # Save artifacts
        await update_status(JobStatus.RUNNING, 90.0, "Saving model artifacts...")

        storage.save_model_artifact(model_id, "model", model)
        storage.save_model_artifact(model_id, "results", results)

        # Extract and save specific results based on model type
        results_summary = {
            "model_id": model_id,
            "model_type": model_type,
            "diagnostics": diagnostics,
            "n_obs": model.n_obs,
            "n_channels": model.n_channels,
            "channel_names": model.channel_names,
        }

        if model_type == "nested":
            results_summary["n_mediators"] = model.n_mediators
            results_summary["mediator_names"] = model.mediator_names
            # Get mediation effects
            try:
                mediation_df = model.get_mediation_effects()
                results_summary["mediation_effects"] = mediation_df.to_dict(
                    orient="records"
                )
            except Exception as e:
                logger.warning(f"Could not extract mediation effects: {e}")

        elif model_type == "multivariate":
            results_summary["n_outcomes"] = model.n_outcomes
            results_summary["outcome_names"] = model.outcome_names
            # Get cross-effects
            try:
                cross_effects_df = model.get_cross_effects_summary()
                results_summary["cross_effects"] = cross_effects_df.to_dict(
                    orient="records"
                )
            except Exception as e:
                logger.warning(f"Could not extract cross effects: {e}")
            # Get correlation matrix
            try:
                corr_df = model.get_correlation_matrix()
                results_summary["correlation_matrix"] = corr_df.to_dict()
            except Exception as e:
                logger.warning(f"Could not extract correlation matrix: {e}")

        elif model_type == "combined":
            results_summary["n_mediators"] = model.n_mediators
            results_summary["n_outcomes"] = model.n_outcomes
            results_summary["mediator_names"] = model.mediator_names
            results_summary["outcome_names"] = model.outcome_names
            # Get effect decomposition
            try:
                decomp_df = model.get_effect_decomposition()
                results_summary["effect_decomposition"] = decomp_df.to_dict(
                    orient="records"
                )
            except Exception as e:
                logger.warning(f"Could not extract effect decomposition: {e}")

        storage.save_results(model_id, "summary", results_summary)

        # Update final status
        await update_status(
            JobStatus.COMPLETED,
            100.0,
            f"{model_type.capitalize()} model fitting completed successfully",
        )

        storage.update_model_metadata(
            model_id,
            {
                "status": JobStatus.COMPLETED.value,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "diagnostics": diagnostics,
            },
        )

        return {
            "model_id": model_id,
            "model_type": model_type,
            "status": "completed",
            "diagnostics": diagnostics,
        }

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        logger.error(f"Extended model fitting failed for {model_id}: {error_msg}")
        logger.error(traceback_str)

        try:
            await update_status(
                JobStatus.FAILED,
                progress=0.0,
                message=f"Extended model fitting failed: {error_msg}",
            )
        except Exception as redis_err:
            logger.error(f"Failed to update Redis status: {redis_err}")

        try:
            storage.update_model_metadata(
                model_id,
                {
                    "status": JobStatus.FAILED.value,
                    "error_message": error_msg,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as storage_err:
            logger.error(f"Failed to update model metadata: {storage_err}")

        return {
            "model_id": model_id,
            "model_type": model_type,
            "status": "failed",
            "error": error_msg,
        }


async def generate_report_task(
    ctx: dict,
    model_id: str,
    report_id: str,
    config: dict,
) -> dict:
    """
    Generate HTML report for a fitted model.

    Parameters
    ----------
    ctx : dict
        ARQ context with Redis connection.
    model_id : str
        Model ID to generate report for.
    report_id : str
        Unique report ID for tracking.
    config : dict
        Report configuration options.

    Returns
    -------
    dict
        Report generation results.
    """
    import redis.asyncio as aioredis
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    storage = get_storage()
    settings = get_settings()
    redis_client: aioredis.Redis = ctx["redis"]

    async def update_report_status(status: str, message: str = None, **extra):
        """Update report status in Redis."""
        report_key = f"mmm:report:{report_id}"
        await redis_client.hset(
            report_key,
            mapping={
                "status": status,
                "message": message or "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **{k: str(v) for k, v in extra.items()},
            },
        )
        await redis_client.expire(report_key, 86400)  # 24 hour TTL

    try:
        await update_report_status("generating", "Loading model artifacts...")

        # Load model artifacts
        logger.info(f"Loading model artifacts for report: {model_id}")
        mmm = storage.load_model_artifact(model_id, "mmm")
        panel = storage.load_model_artifact(model_id, "panel")

        # Try to load results
        try:
            results = storage.load_model_artifact(model_id, "results")
        except Exception:
            results = None

        await update_report_status("generating", "Extracting report data...")

        # Import reporting module
        from mmm_framework.reporting import (
            MMMReportGenerator,
            ReportConfig,
            SectionConfig,
        )

        report_config = ReportConfig(
            title=config.get("title") or "Marketing Mix Model Report",
            client=config.get("client"),
            subtitle=config.get("subtitle"),
            analysis_period=config.get("analysis_period"),
            currency_symbol=config.get("currency_symbol", "$"),
            default_credible_interval=config.get("credible_interval", 0.8),
            # Section configs
            executive_summary=SectionConfig(
                enabled=config.get("include_executive_summary", True)
            ),
            model_fit=SectionConfig(enabled=config.get("include_model_fit", True)),
            channel_roi=SectionConfig(enabled=config.get("include_channel_roi", True)),
            decomposition=SectionConfig(
                enabled=config.get("include_decomposition", True)
            ),
            saturation=SectionConfig(enabled=config.get("include_saturation", True)),
            diagnostics=SectionConfig(enabled=config.get("include_diagnostics", True)),
            methodology=SectionConfig(enabled=config.get("include_methodology", True)),
            # Disable sections we don't have data for by default
            geographic=SectionConfig(enabled=False),
            mediators=SectionConfig(enabled=False),
            cannibalization=SectionConfig(enabled=False),
            sensitivity=SectionConfig(enabled=False),
        )

        await update_report_status("generating", "Generating report...")

        # Generate report
        generator = MMMReportGenerator(
            model=mmm,
            config=report_config,
            panel=panel,
            results=results,
        )

        html_content = generator.render()

        # Save report to storage
        reports_dir = Path(settings.storage_path) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_filename = f"{model_id}_{report_id}.html"
        report_path = reports_dir / report_filename

        report_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Report saved to: {report_path}")

        # Update status
        await update_report_status(
            "completed",
            "Report generated successfully",
            filepath=str(report_path),
            filename=report_filename,
        )

        return {
            "report_id": report_id,
            "model_id": model_id,
            "status": "completed",
            "filepath": str(report_path),
            "filename": report_filename,
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback

        tb = traceback.format_exc()
        logger.error(tb)

        await update_report_status("failed", str(e))

        return {
            "report_id": report_id,
            "model_id": model_id,
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# Cron Jobs
# =============================================================================


async def cleanup_old_jobs(ctx: dict):
    """Clean up old completed/failed jobs from Redis."""
    import redis.asyncio as aioredis

    redis_client: aioredis.Redis = ctx["redis"]
    settings = get_settings()

    # Find all job keys
    job_keys = await redis_client.keys("mmm:job:*")

    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.data_retention_days)

    cleaned = 0
    for key in job_keys:
        updated_at = await redis_client.hget(key, "updated_at")
        if updated_at:
            try:
                job_time = datetime.fromisoformat(updated_at)
                if job_time < cutoff:
                    await redis_client.delete(key)
                    cleaned += 1
            except (ValueError, TypeError):
                continue

    return {"cleaned_jobs": cleaned}


# =============================================================================
# Worker Setup
# =============================================================================


async def startup(ctx: dict):
    """Worker startup hook."""
    print("MMM Worker starting up...")


async def shutdown(ctx: dict):
    """Worker shutdown hook."""
    print("MMM Worker shutting down...")


class WorkerSettings:
    """ARQ worker settings."""

    # Use settings from environment variables
    redis_settings = get_settings().redis_settings

    functions = [
        fit_model_task,
        fit_extended_model_task,
        compute_contributions_task,
        run_scenario_task,
        generate_report_task,
    ]

    cron_jobs = [
        cron(cleanup_old_jobs, hour=3, minute=0),  # Run at 3 AM daily
    ]

    on_startup = startup
    on_shutdown = shutdown

    max_jobs = 4
    job_timeout = 10000
    max_tries = 3

    # Health check
    health_check_interval = 30


if __name__ == "__main__":
    # For running worker directly
    import asyncio
    from arq import run_worker

    run_worker(WorkerSettings)
