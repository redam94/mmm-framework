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
from datetime import datetime, timedelta
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
    
    async def update_status(status: JobStatus, progress: float = 0.0, message: str | None = None, **extra):
        """Update job status in Redis."""
        job_key = f"mmm:job:{model_id}"
        await redis_client.hset(job_key, mapping={
            "status": status.value,
            "progress": str(progress),
            "progress_message": message or "",
            "updated_at": datetime.utcnow().isoformat(),
            **{k: str(v) if v is not None else "" for k, v in extra.items()},
        })
    
    try:
        # Update status to running
        await update_status(JobStatus.RUNNING, 0.0, "Initializing...")
        
        # Update model metadata
        storage.update_model_metadata(model_id, {
            "status": JobStatus.RUNNING.value,
            "started_at": datetime.utcnow().isoformat(),
        })
        
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
            
            if media_dict.get("dimensions") == ["Period"]:
                media_builder.national()
            elif "Geography" in media_dict.get("dimensions", []):
                media_builder.by_geo()
            
            adstock_config = media_dict.get("adstock", {})
            l_max = adstock_config.get("l_max", 8)
            media_builder.with_geometric_adstock(l_max)
            media_builder.with_hill_saturation()
            
            if media_dict.get("parent_channel"):
                media_builder.with_parent_channel(media_dict["parent_channel"])
            
            mff_builder.add_media_builder(media_builder)
        
        # Controls
        for control_dict in mff_config_dict.get("controls", []):
            control_builder = ControlVariableConfigBuilder(control_dict["name"]).national()
            if control_dict.get("allow_negative", True):
                control_builder.allow_negative()
            else:
                control_builder.positive_only()
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
                trend_builder.with_changepoint_prior_scale(trend_dict["changepoint_prior_scale"])
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
        await update_status(JobStatus.RUNNING, 35.0, "Fitting model (this may take several minutes)...")
        
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
                param_summary.append({
                    "parameter": str(idx),
                    "mean": float(row.get("mean", 0)),
                    "sd": float(row.get("sd", 0)),
                    "hdi_3%": float(row.get("hdi_3%", 0)),
                    "hdi_97%": float(row.get("hdi_97%", 0)),
                    "r_hat": float(row.get("r_hat", 1.0)) if "r_hat" in row else None,
                })
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
        storage.update_model_metadata(model_id, {
            "status": JobStatus.COMPLETED.value,
            "completed_at": datetime.utcnow().isoformat(),
            "diagnostics": diagnostics,
        })
        
        print("DEBUG: All done, returning result")
        return {
            "model_id": model_id,
            "status": "completed",
            "diagnostics": diagnostics,
        }
    except Exception as e:
        print(f"DEBUG: Failed saving results_summary: {e}")
        raise


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
            results["contribution_hdi_low"] = contrib_results.contribution_hdi_low.to_dict()
            results["contribution_hdi_high"] = contrib_results.contribution_hdi_high.to_dict()
        
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
    
    cutoff = datetime.utcnow() - timedelta(days=settings.data_retention_days)
    
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
    
     
    redis_settings = RedisSettings(
        host="localhost",
        port=6379,
    )
    
    # redis_settings = RedisSettings(
    #     host=settings.redis_url.replace("redis://", "").split(":")[0].split("/")[0],
    #     port=int(settings.redis_url.split(":")[-1].split("/")[0]) if ":" in settings.redis_url else 6379,
    #     database=settings.redis_db,
    #     password=settings.redis_password,
    # )
    
    functions = [
        fit_model_task,
        compute_contributions_task,
        run_scenario_task,
    ]
    
    # cron_jobs = [
    #     cron(cleanup_old_jobs, hour=3, minute=0),  # Run at 3 AM daily
    # ]
    
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