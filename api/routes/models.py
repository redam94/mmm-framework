"""
Model management API routes.

Handles model fitting, tracking, results, and downloads.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from arq import ArqRedis, create_pool
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger

from config import Settings, get_settings
from redis_service import RedisService, get_redis
from schemas import (
    ContributionRequest,
    ContributionResponse,
    ErrorResponse,
    JobStatus,
    ModelFitRequest,
    ModelInfo,
    ModelListResponse,
    ModelResultsResponse,
    PredictionRequest,
    PredictionResponse,
    ScenarioRequest,
    ScenarioResponse,
    SuccessResponse,
)
from storage import StorageError, StorageService, get_storage

router = APIRouter(prefix="/models", tags=["Models"])


# Custom JSON encoder that handles NaN/Inf values
class NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to null."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)
    
    def encode(self, obj):
        return super().encode(self._sanitize(obj))
    
    def _sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return self._sanitize(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return self._sanitize(obj.item())
        return obj


class SafeJSONResponse(JSONResponse):
    """JSONResponse that handles NaN/Inf values."""
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=NaNSafeEncoder,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf values with None for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, float):
        try:
            if math.isnan(obj) or math.isinf(obj):
                return None
        except (TypeError, ValueError):
            pass
        return obj
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Catch-all for other numpy scalars
    elif hasattr(obj, 'item'):
        return _sanitize_for_json(obj.item())
    return obj


async def get_arq_pool(settings: Settings = Depends(get_settings)) -> ArqRedis:
    """Get ARQ Redis connection pool."""
    return await create_pool(settings.redis_settings)


def _metadata_to_model_info(metadata: dict) -> ModelInfo:
    """Convert model metadata to ModelInfo response."""
    return ModelInfo(
        model_id=metadata["model_id"],
        name=metadata.get("name"),
        description=metadata.get("description"),
        data_id=metadata["data_id"],
        config_id=metadata["config_id"],
        status=JobStatus(metadata.get("status", "pending")),
        progress=float(metadata.get("progress", 0)),
        progress_message=metadata.get("progress_message"),
        created_at=datetime.fromisoformat(metadata["created_at"]),
        started_at=datetime.fromisoformat(metadata["started_at"]) if metadata.get("started_at") else None,
        completed_at=datetime.fromisoformat(metadata["completed_at"]) if metadata.get("completed_at") else None,
        error_message=metadata.get("error_message"),
        diagnostics=metadata.get("diagnostics"),
    )


def _check_model_completed(storage: StorageService, model_id: str):
    """Check if model exists and is completed. Raises HTTPException if not."""
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    
    metadata = storage.get_model_metadata(model_id)
    
    if metadata.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model is not completed. Current status: {metadata.get('status')}",
        )
    
    return metadata


@router.post(
    "/fit",
    response_model=ModelInfo,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def start_model_fitting(
    request: ModelFitRequest,
    storage: StorageService = Depends(get_storage),
    arq_pool: ArqRedis = Depends(get_arq_pool),
    redis: RedisService = Depends(get_redis),
):
    """
    Start fitting a Bayesian MMM model.
    
    This is an async operation. The model fitting runs in the background.
    Use the returned model_id to track progress and retrieve results.
    """
    # Validate data exists
    if not storage.data_exists(request.data_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {request.data_id}",
        )
    
    # Validate config exists
    if not storage.config_exists(request.config_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration not found: {request.config_id}",
        )
    
    # Generate model ID
    model_id = storage.generate_id()
    
    # Create model metadata
    metadata = {
        "model_id": model_id,
        "name": request.name or f"Model {model_id[:8]}",
        "description": request.description,
        "data_id": request.data_id,
        "config_id": request.config_id,
        "status": JobStatus.QUEUED.value,
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    storage.save_model_metadata(model_id, metadata)
    
    # Initialize job status in Redis
    await redis.set_job_status(
        model_id,
        status=JobStatus.QUEUED,
        progress=0.0,
        message="Job queued",
    )
    
    # Queue the fitting task
    overrides = {}
    if request.n_chains:
        overrides["n_chains"] = request.n_chains
    if request.n_draws:
        overrides["n_draws"] = request.n_draws
    if request.n_tune:
        overrides["n_tune"] = request.n_tune
    if request.random_seed:
        overrides["random_seed"] = request.random_seed
    
    await arq_pool.enqueue_job(
        "fit_model_task",
        model_id=model_id,
        data_id=request.data_id,
        config_id=request.config_id,
        overrides=overrides if overrides else None,
    )
    
    logger.info(f"Queued model fitting job: {model_id}")
    
    return _metadata_to_model_info(metadata)


@router.get(
    "",
    response_model=ModelListResponse,
)
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status_filter: JobStatus | None = None,
    storage: StorageService = Depends(get_storage),
):
    """List all models with optional status filter."""
    all_models = storage.list_models()
    
    # Apply status filter
    if status_filter:
        all_models = [m for m in all_models if m.get("status") == status_filter.value]
    
    total = len(all_models)
    models = all_models[skip:skip + limit]
    
    return ModelListResponse(
        models=[_metadata_to_model_info(m) for m in models],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_model(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Get model information and status."""
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    
    metadata = storage.get_model_metadata(model_id)
    return _metadata_to_model_info(metadata)


@router.get(
    "/{model_id}/status",
    responses={404: {"model": ErrorResponse}},
)
async def get_model_status(
    model_id: str,
    storage: StorageService = Depends(get_storage),
    redis: RedisService = Depends(get_redis),
):
    """Get real-time model fitting status from Redis."""
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    
    # Get status from Redis (real-time) or fall back to storage
    redis_status = await redis.get_job_status(model_id)
    
    if redis_status:
        return redis_status
    
    # Fall back to stored metadata
    metadata = storage.get_model_metadata(model_id)
    return {
        "model_id": model_id,
        "status": metadata.get("status", "unknown"),
        "progress": metadata.get("progress", 0),
        "message": metadata.get("progress_message"),
    }


@router.get(
    "/{model_id}/results",
    response_model=ModelResultsResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def get_model_results(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """
    Get model results summary (diagnostics, parameter summary).
    
    For detailed results, use the specific endpoints:
    - /models/{model_id}/fit - Model fit data
    - /models/{model_id}/posteriors - Posterior distributions
    - /models/{model_id}/response-curves - Response curves
    - /models/{model_id}/decomposition - Component decomposition
    """
    _check_model_completed(storage, model_id)
    
    try:
        results = storage.load_results(model_id, "summary")
        
        return ModelResultsResponse(
            model_id=model_id,
            status=JobStatus.COMPLETED,
            diagnostics=results.get("diagnostics", {}),
            parameter_summary=results.get("parameter_summary", []),
            channel_contributions=None,
            component_decomposition=None,
        )
    
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results not found. Model may still be processing.",
        )


# =============================================================================
# Detailed Results Endpoints
# =============================================================================

@router.get(
    "/{model_id}/fit",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def get_model_fit(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """
    Get model fit data (observed vs predicted values).
    
    Returns periods, observed values, predicted mean/std, and fit metrics (R², RMSE, MAPE).
    Data is returned both aggregated and by geography (if available).
    """
    _check_model_completed(storage, model_id)
    
    try:
        # Try to load cached fit data
        try:
            fit_data = storage.load_results(model_id, "fit")
            return SafeJSONResponse(content=fit_data)
        except StorageError:
            pass
        
        # Compute fit data from model artifacts
        logger.info(f"Computing fit data for model {model_id}")
        mmm = storage.load_model_artifact(model_id, "mmm")
        panel = storage.load_model_artifact(model_id, "panel")
        
        # Get observed values
        y_obs = panel.y.values.flatten()
        
        # Get predictions
        pred_results = mmm.predict(return_original_scale=True, random_seed=42)
        y_pred_mean = pred_results.y_pred_mean
        y_pred_std = pred_results.y_pred_std
        
        # Compute overall fit metrics
        r2 = 1 - np.sum((y_obs - y_pred_mean) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)
        rmse = np.sqrt(np.mean((y_obs - y_pred_mean) ** 2))
        mape = np.mean(np.abs((y_obs - y_pred_mean) / (y_obs + 1e-8))) * 100
        
        # Get period labels
        periods = panel.coords.periods
        period_labels = [str(p.date()) if hasattr(p, 'date') else str(p) for p in periods]
        n_periods = len(periods)
        
        # Check for geo/product dimensions
        has_geo = panel.coords.has_geo
        has_product = panel.coords.has_product
        geographies = panel.coords.geographies if has_geo else None
        products = panel.coords.products if has_product else None
        
        n_geos = panel.coords.n_geos
        n_products = panel.coords.n_products
        
        # Build aggregated time series (sum over geo and product)
        # Data is assumed to be in order: period, geo, product (innermost to outermost)
        y_obs_reshaped = y_obs.reshape(n_periods, n_geos, n_products) if (has_geo or has_product) else y_obs.reshape(n_periods, 1, 1)
        y_pred_reshaped = y_pred_mean.reshape(n_periods, n_geos, n_products) if (has_geo or has_product) else y_pred_mean.reshape(n_periods, 1, 1)
        
        # Aggregated (sum over geo and product)
        y_obs_agg = y_obs_reshaped.sum(axis=(1, 2))
        y_pred_agg = y_pred_reshaped.sum(axis=(1, 2))
        
        # Compute aggregated metrics
        r2_agg = 1 - np.sum((y_obs_agg - y_pred_agg) ** 2) / np.sum((y_obs_agg - np.mean(y_obs_agg)) ** 2)
        rmse_agg = np.sqrt(np.mean((y_obs_agg - y_pred_agg) ** 2))
        mape_agg = np.mean(np.abs((y_obs_agg - y_pred_agg) / (y_obs_agg + 1e-8))) * 100
        
        # Compute predicted std aggregated if available
        y_pred_std_agg = None
        if y_pred_std is not None:
            y_pred_std_reshaped = y_pred_std.reshape(n_periods, n_geos, n_products) if (has_geo or has_product) else y_pred_std.reshape(n_periods, 1, 1)
            # For summed values, std combines in quadrature
            y_pred_std_agg = np.sqrt((y_pred_std_reshaped ** 2).sum(axis=(1, 2))).tolist()
        
        fit_data = {
            "model_id": model_id,
            "periods": period_labels,
            "has_geo": has_geo,
            "has_product": has_product,
            "geographies": geographies,
            "products": products,
            # Aggregated data (summed over geo/product)
            "aggregated": {
                "observed": y_obs_agg.tolist(),
                "predicted_mean": y_pred_agg.tolist(),
                "predicted_std": y_pred_std_agg,
                "r2": float(r2_agg),
                "rmse": float(rmse_agg),
                "mape": float(mape_agg),
            },
            # Overall metrics (across all observations)
            "r2": float(r2),
            "rmse": float(rmse),
            "mape": float(mape),
            # Legacy fields for backward compatibility
            "observed": y_obs_agg.tolist(),
            "predicted_mean": y_pred_agg.tolist(),
            "predicted_std": y_pred_std_agg,
        }
        
        # Add geo-level breakdowns if available
        if has_geo:
            geo_data = {}
            for g_idx, geo in enumerate(geographies):
                # Sum over products for this geo
                y_obs_geo = y_obs_reshaped[:, g_idx, :].sum(axis=1)
                y_pred_geo = y_pred_reshaped[:, g_idx, :].sum(axis=1)
                
                r2_geo = 1 - np.sum((y_obs_geo - y_pred_geo) ** 2) / np.sum((y_obs_geo - np.mean(y_obs_geo)) ** 2)
                rmse_geo = np.sqrt(np.mean((y_obs_geo - y_pred_geo) ** 2))
                mape_geo = np.mean(np.abs((y_obs_geo - y_pred_geo) / (y_obs_geo + 1e-8))) * 100
                
                # Compute std for this geo (sum over products in quadrature)
                y_pred_std_geo = None
                if y_pred_std is not None:
                    y_pred_std_reshaped = y_pred_std.reshape(n_periods, n_geos, n_products) if (has_geo or has_product) else y_pred_std.reshape(n_periods, 1, 1)
                    # For summed values, std combines in quadrature
                    y_pred_std_geo = np.sqrt((y_pred_std_reshaped[:, g_idx, :] ** 2).sum(axis=1)).tolist()
                
                geo_data[geo] = {
                    "observed": y_obs_geo.tolist(),
                    "predicted_mean": y_pred_geo.tolist(),
                    "predicted_std": y_pred_std_geo,
                    "r2": float(r2_geo),
                    "rmse": float(rmse_geo),
                    "mape": float(mape_geo),
                }
            fit_data["by_geography"] = geo_data
        
        # Sanitize for JSON serialization
        fit_data = _sanitize_for_json(fit_data)
        
        # Cache the results
        storage.save_results(model_id, "fit", fit_data)
        
        return SafeJSONResponse(content=fit_data)
        
    except Exception as e:
        logger.error(f"Error computing fit data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing fit data: {str(e)}",
        )


@router.get(
    "/{model_id}/posteriors",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def get_posteriors(
    model_id: str,
    parameters: list[str] | None = Query(None, description="Filter to specific parameters"),
    n_samples: int = Query(500, ge=100, le=2000, description="Number of samples to return"),
    storage: StorageService = Depends(get_storage),
):
    """
    Get posterior distribution samples for model parameters.
    
    Returns samples for each parameter to enable histogram/KDE visualization.
    """
    _check_model_completed(storage, model_id)
    
    try:
        # Try to load cached posteriors
        try:
            posteriors = storage.load_results(model_id, "posteriors")
            # Filter if parameters specified
            if parameters:
                posteriors = {k: v for k, v in posteriors.items() if k in parameters or k == "model_id"}
            return posteriors
        except StorageError:
            pass
        
        # Compute from model artifacts
        logger.info(f"Computing posteriors for model {model_id}")
        mmm = storage.load_model_artifact(model_id, "mmm")
        
        if mmm._trace is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model trace not available",
            )
        
        trace = mmm._trace
        posterior = trace.posterior
        
        posteriors = {"model_id": model_id}
        
        # Get all variable names
        var_names = list(posterior.data_vars)
        
        for var_name in var_names:
            if parameters and var_name not in parameters:
                continue
            
            try:
                var_data = posterior[var_name]
                
                # Flatten chain and draw dimensions
                samples = var_data.values.flatten()
                
                # Subsample if needed
                if len(samples) > n_samples:
                    idx = np.random.choice(len(samples), n_samples, replace=False)
                    samples = samples[idx]
                
                # Handle multi-dimensional parameters
                if var_data.ndim > 2:
                    # For array parameters, compute summary stats
                    mean_val = float(var_data.mean().values)
                    std_val = float(var_data.std().values)
                    posteriors[var_name] = {
                        "samples": samples.tolist(),
                        "mean": mean_val,
                        "std": std_val,
                        "shape": list(var_data.shape[2:]),  # Exclude chain/draw dims
                    }
                else:
                    posteriors[var_name] = {
                        "samples": samples.tolist(),
                        "mean": float(np.mean(samples)),
                        "std": float(np.std(samples)),
                    }
            except Exception as e:
                logger.warning(f"Could not process parameter {var_name}: {e}")
                continue
        
        # Sanitize for JSON serialization
        posteriors = _sanitize_for_json(posteriors)
        
        # Cache the results
        storage.save_results(model_id, "posteriors", posteriors)
        
        return SafeJSONResponse(content=posteriors)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing posteriors: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing posteriors: {str(e)}",
        )


@router.get(
    "/{model_id}/prior-posterior",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def get_prior_vs_posterior(
    model_id: str,
    n_samples: int = Query(500, ge=100, le=2000, description="Number of samples to return"),
    storage: StorageService = Depends(get_storage),
):
    """
    Get prior vs posterior comparison for model parameters.
    
    Returns prior and posterior samples with shrinkage metrics for each parameter.
    """
    _check_model_completed(storage, model_id)
    
    try:
        # Try to load cached prior-posterior data
        try:
            prior_posterior = storage.load_results(model_id, "prior_posterior")
            return SafeJSONResponse(content=prior_posterior)
        except StorageError:
            pass
        
        # Compute from model artifacts
        logger.info(f"Computing prior vs posterior for model {model_id}")
        mmm = storage.load_model_artifact(model_id, "mmm")
        
        if mmm._trace is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model trace not available",
            )
        
        import pymc as pm
        
        trace = mmm._trace
        posterior = trace.posterior
        
        # Sample from prior
        with mmm._model:
            prior = pm.sample_prior_predictive(samples=n_samples, random_seed=42)
        
        prior_data = prior.prior
        
        result = {
            "model_id": model_id,
            "parameters": {},
            "channel_names": mmm.channel_names,
        }
        
        # Key parameters to compare
        key_params = []
        
        # Add beta parameters
        for ch in mmm.channel_names:
            for prefix in ["beta_", "beta_media_"]:
                param = f"{prefix}{ch}"
                if param in posterior:
                    key_params.append(param)
                    break
        
        # Add adstock parameters
        for ch in mmm.channel_names:
            for prefix in ["adstock_", "adstock_alpha_"]:
                param = f"{prefix}{ch}"
                if param in posterior:
                    key_params.append(param)
                    break
        
        # Add saturation parameters
        for ch in mmm.channel_names:
            for prefix in ["sat_lam_", "saturation_lam_"]:
                param = f"{prefix}{ch}"
                if param in posterior:
                    key_params.append(param)
                    break
        
        # Add other key parameters
        for param in ["intercept", "sigma", "trend_slope", "trend_intercept"]:
            if param in posterior:
                key_params.append(param)
        
        # Also include any trend/seasonality parameters
        for var in posterior.data_vars:
            if any(x in var for x in ["trend", "season", "gp_", "spline"]):
                if var not in key_params:
                    key_params.append(var)
        
        for param in key_params:
            if param not in posterior:
                continue
            
            try:
                # Get posterior samples
                post_samples = posterior[param].values.flatten()
                if len(post_samples) > n_samples:
                    idx = np.random.choice(len(post_samples), n_samples, replace=False)
                    post_samples = post_samples[idx]
                
                # Get prior samples (if available)
                prior_samples = None
                prior_mean = None
                prior_std = None
                shrinkage = None
                
                if param in prior_data:
                    prior_samples = prior_data[param].values.flatten()
                    if len(prior_samples) > n_samples:
                        idx = np.random.choice(len(prior_samples), n_samples, replace=False)
                        prior_samples = prior_samples[idx]
                    
                    prior_mean = float(np.mean(prior_samples))
                    prior_std = float(np.std(prior_samples))
                    
                    # Compute shrinkage (reduction in standard deviation)
                    post_std = float(np.std(post_samples))
                    if prior_std > 1e-8:
                        shrinkage = float((1 - post_std / prior_std) * 100)
                    else:
                        shrinkage = 0.0
                
                post_mean = float(np.mean(post_samples))
                post_std = float(np.std(post_samples))
                
                # Compute HDI
                hdi_low = float(np.percentile(post_samples, 3))
                hdi_high = float(np.percentile(post_samples, 97))
                
                param_data = {
                    "posterior_samples": post_samples.tolist(),
                    "posterior_mean": post_mean,
                    "posterior_std": post_std,
                    "posterior_hdi_3": hdi_low,
                    "posterior_hdi_97": hdi_high,
                }
                
                if prior_samples is not None:
                    param_data.update({
                        "prior_samples": prior_samples.tolist(),
                        "prior_mean": prior_mean,
                        "prior_std": prior_std,
                        "shrinkage_pct": shrinkage,
                    })
                
                result["parameters"][param] = param_data
                
            except Exception as e:
                logger.warning(f"Could not process parameter {param}: {e}")
                continue
        
        # Sanitize for JSON serialization
        result = _sanitize_for_json(result)
        
        # Cache the results
        storage.save_results(model_id, "prior_posterior", result)
        
        return SafeJSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing prior vs posterior: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing prior vs posterior: {str(e)}",
        )


@router.get(
    "/{model_id}/response-curves",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def get_response_curves(
    model_id: str,
    n_points: int = Query(100, ge=50, le=500, description="Number of points per curve"),
    n_samples: int = Query(200, ge=50, le=500, description="Number of posterior samples for uncertainty"),
    storage: StorageService = Depends(get_storage),
):
    """
    Get response curves for each channel.
    
    Returns spend values, response values with uncertainty bands, and current spend markers.
    """
    _check_model_completed(storage, model_id)
    
    try:
        # Try to load cached response curves
        try:
            curves = storage.load_results(model_id, "response_curves")
            return curves
        except StorageError:
            pass
        
        # Compute from model artifacts
        logger.info(f"Computing response curves for model {model_id}")
        mmm = storage.load_model_artifact(model_id, "mmm")
        panel = storage.load_model_artifact(model_id, "panel")
        
        if mmm._trace is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model trace not available",
            )
        
        posterior = mmm._trace.posterior
        channel_names = mmm.channel_names
        X_media_raw = panel.X_media.values
        
        curves = {"model_id": model_id, "channels": {}}
        
        for c, channel in enumerate(channel_names):
            # Look for saturation and beta parameters
            sat_lam_var = f"sat_lam_{channel}"
            beta_var = f"beta_{channel}"
            
            # Also try alternative naming conventions
            if sat_lam_var not in posterior:
                sat_lam_var = f"saturation_lam_{channel}"
            if beta_var not in posterior:
                beta_var = f"beta_media_{channel}"
            
            if sat_lam_var not in posterior or beta_var not in posterior:
                logger.warning(f"Parameters not found for channel {channel}")
                continue
            
            sat_lam_samples = posterior[sat_lam_var].values.flatten()
            beta_samples = posterior[beta_var].values.flatten()
            
            # Get spend range
            spend_raw = X_media_raw[:, c]
            spend_max = float(spend_raw.max())
            current_spend = float(spend_raw.mean())
            
            # Create spend grid
            x_original = np.linspace(0, spend_max * 1.2, n_points)
            x_scaled = x_original / (spend_max + 1e-8)
            
            # Sample posterior for uncertainty
            n_posterior_samples = min(n_samples, len(sat_lam_samples))
            idx = np.random.choice(len(sat_lam_samples), n_posterior_samples, replace=False)
            
            response_curves = np.zeros((n_posterior_samples, n_points))
            for i, j in enumerate(idx):
                saturated = 1 - np.exp(-sat_lam_samples[j] * x_scaled)
                response_curves[i, :] = beta_samples[j] * saturated
            
            # Scale back to original units
            response_curves_original = response_curves * mmm.y_std
            
            curves["channels"][channel] = {
                "spend": x_original.tolist(),
                "response": response_curves_original.mean(axis=0).tolist(),
                "response_hdi_low": np.percentile(response_curves_original, 3, axis=0).tolist(),
                "response_hdi_high": np.percentile(response_curves_original, 97, axis=0).tolist(),
                "current_spend": current_spend,
                "spend_max": spend_max,
            }
        
        # Sanitize for JSON serialization
        curves = _sanitize_for_json(curves)
        
        # Cache the results
        storage.save_results(model_id, "response_curves", curves)
        
        return SafeJSONResponse(content=curves)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing response curves: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing response curves: {str(e)}",
        )


@router.get(
    "/{model_id}/decomposition",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def get_decomposition(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """
    Get component decomposition (trend, seasonality, media contributions, etc.).
    
    Returns time series of each component's contribution to the outcome.
    """
    _check_model_completed(storage, model_id)
    
    try:
        # Try to load cached decomposition
        try:
            decomposition = storage.load_results(model_id, "decomposition")
            return decomposition
        except StorageError:
            pass
        
        # Compute from model artifacts
        logger.info(f"Computing decomposition for model {model_id}")
        mmm = storage.load_model_artifact(model_id, "mmm")
        panel = storage.load_model_artifact(model_id, "panel")
        
        if mmm._trace is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model trace not available",
            )
        
        posterior = mmm._trace.posterior
        
        # Get observed values
        y_obs = panel.y.values.flatten()
        n_obs = len(y_obs)
        
        # Get period labels
        if hasattr(panel, 'periods') and panel.periods is not None:
            periods = [str(p) for p in panel.periods]
        else:
            periods = list(range(n_obs))
        
        decomposition = {
            "model_id": model_id,
            "periods": periods,
            "observed": y_obs.tolist(),
            "components": {},
        }
        
        # Intercept/baseline component
        if "intercept" in posterior:
            intercept = float(posterior["intercept"].mean().values)
            baseline = np.full(n_obs, intercept * mmm.y_std + mmm.y_mean)
            decomposition["components"]["baseline"] = baseline.tolist()
        
        # Trend component
        if "trend" in posterior:
            trend = posterior["trend"].mean(dim=["chain", "draw"]).values
            if len(trend) == n_obs:
                decomposition["components"]["trend"] = (trend * mmm.y_std).tolist()
        
        # Seasonality component
        if "seasonality" in posterior:
            seasonality = posterior["seasonality"].mean(dim=["chain", "draw"]).values
            if len(seasonality) == n_obs:
                decomposition["components"]["seasonality"] = (seasonality * mmm.y_std).tolist()
        
        # Media channel contributions
        channel_names = mmm.channel_names
        for channel in channel_names:
            contrib_var = f"channel_contribution_{channel}"
            if contrib_var not in posterior:
                contrib_var = f"media_contribution_{channel}"
            
            if contrib_var in posterior:
                contrib = posterior[contrib_var].mean(dim=["chain", "draw"]).values
                if len(contrib.flatten()) == n_obs:
                    decomposition["components"][f"media_{channel}"] = (contrib.flatten() * mmm.y_std).tolist()
        
        # Control variable contributions
        if hasattr(mmm, 'control_names') and mmm.control_names:
            for control in mmm.control_names:
                contrib_var = f"control_contribution_{control}"
                if contrib_var in posterior:
                    contrib = posterior[contrib_var].mean(dim=["chain", "draw"]).values
                    if len(contrib.flatten()) == n_obs:
                        decomposition["components"][f"control_{control}"] = (contrib.flatten() * mmm.y_std).tolist()
        
        # Sanitize for JSON serialization
        decomposition = _sanitize_for_json(decomposition)
        
        # Cache the results
        storage.save_results(model_id, "decomposition", decomposition)
        
        return SafeJSONResponse(content=decomposition)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing decomposition: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing decomposition: {str(e)}",
        )


@router.get(
    "/{model_id}/roas",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def get_marginal_roas(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """
    Get marginal ROAS for each channel.
    
    Returns mean ROAS with uncertainty (HDI) for each channel.
    """
    _check_model_completed(storage, model_id)
    
    try:
        # Try to load cached ROAS
        try:
            roas = storage.load_results(model_id, "roas")
            return roas
        except StorageError:
            pass
        
        # Compute from model artifacts
        logger.info(f"Computing ROAS for model {model_id}")
        mmm = storage.load_model_artifact(model_id, "mmm")
        panel = storage.load_model_artifact(model_id, "panel")
        
        if mmm._trace is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model trace not available",
            )
        
        posterior = mmm._trace.posterior
        channel_names = mmm.channel_names
        X_media_raw = panel.X_media.values
        
        roas = {"model_id": model_id, "channels": {}}
        
        for c, channel in enumerate(channel_names):
            beta_var = f"beta_{channel}"
            if beta_var not in posterior:
                beta_var = f"beta_media_{channel}"
            
            if beta_var not in posterior:
                continue
            
            beta_samples = posterior[beta_var].values.flatten()
            
            # Scale beta to original units (contribution per unit spend)
            spend_raw = X_media_raw[:, c]
            total_spend = float(spend_raw.sum())
            
            # ROAS = total contribution / total spend
            # Contribution = beta * saturation response (assume average saturation)
            roas_samples = beta_samples * mmm.y_std
            
            roas["channels"][channel] = {
                "mean": float(np.mean(roas_samples)),
                "std": float(np.std(roas_samples)),
                "hdi_low": float(np.percentile(roas_samples, 3)),
                "hdi_high": float(np.percentile(roas_samples, 97)),
                "total_spend": total_spend,
            }
        
        # Sanitize for JSON serialization
        roas = _sanitize_for_json(roas)
        
        # Cache the results
        storage.save_results(model_id, "roas", roas)
        
        return SafeJSONResponse(content=roas)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing ROAS: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing ROAS: {str(e)}",
        )


# =============================================================================
# Existing Endpoints (updated)
# =============================================================================

@router.delete(
    "/{model_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_model(
    model_id: str,
    storage: StorageService = Depends(get_storage),
    redis: RedisService = Depends(get_redis),
):
    """Delete a model and all its artifacts."""
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    
    # Delete from storage
    storage.delete_model(model_id)
    
    # Clean up Redis
    await redis.delete_job_status(model_id)
    
    return SuccessResponse(
        success=True,
        message=f"Model {model_id} deleted successfully",
    )


@router.get(
    "/{model_id}/download",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def download_model(
    model_id: str,
    artifact: str = Query("mmm", description="Artifact to download: mmm, results, panel"),
    storage: StorageService = Depends(get_storage),
):
    """
    Download a model artifact.
    
    Available artifacts:
    - **mmm**: The fitted BayesianMMM object (pickle)
    - **results**: The MMMResults object (pickle)
    - **panel**: The PanelDataset object (pickle)
    """
    _check_model_completed(storage, model_id)
    
    try:
        artifact_path = storage.get_model_artifact_path(model_id, artifact)
        
        return FileResponse(
            path=artifact_path,
            filename=f"{model_id}_{artifact}.pkl",
            media_type="application/octet-stream",
        )
    
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact not found: {artifact}",
        )


@router.post(
    "/{model_id}/contributions",
    response_model=ContributionResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def compute_contributions(
    model_id: str,
    request: ContributionRequest,
    storage: StorageService = Depends(get_storage),
    arq_pool: ArqRedis = Depends(get_arq_pool),
):
    """
    Compute counterfactual channel contributions.
    
    This computes the incremental contribution of each channel by comparing
    the baseline prediction to counterfactual predictions with each channel
    zeroed out.
    """
    _check_model_completed(storage, model_id)
    
    try:
        mmm = storage.load_model_artifact(model_id, "mmm")
        
        contrib_results = mmm.compute_counterfactual_contributions(
            time_period=tuple(request.time_period) if request.time_period else None,
            channels=request.channels,
            compute_uncertainty=request.compute_uncertainty,
            hdi_prob=request.hdi_prob,
            random_seed=42,
        )
        
        # Build response dict with all fields
        response_data = {
            "model_id": model_id,
            "total_contributions": contrib_results.total_contributions.to_dict(),
            "contribution_pct": contrib_results.contribution_pct.to_dict(),
            "contribution_hdi_low": contrib_results.contribution_hdi_low.to_dict() if contrib_results.contribution_hdi_low is not None else None,
            "contribution_hdi_high": contrib_results.contribution_hdi_high.to_dict() if contrib_results.contribution_hdi_high is not None else None,
            "time_period": request.time_period,
        }
        
        # Cache the results (include baseline for frontend even though not in schema)
        cache_data = response_data.copy()
        if hasattr(contrib_results, 'baseline'):
            cache_data["baseline"] = float(contrib_results.baseline)
        storage.save_results(model_id, "contributions", cache_data)
        
        return ContributionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error computing contributions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing contributions: {str(e)}",
        )


@router.post(
    "/{model_id}/scenario",
    response_model=ScenarioResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def run_scenario(
    model_id: str,
    request: ScenarioRequest,
    storage: StorageService = Depends(get_storage),
):
    """
    Run what-if scenario analysis.
    
    Modify channel spend by percentage and see predicted outcome changes.
    """
    _check_model_completed(storage, model_id)
    
    try:
        mmm = storage.load_model_artifact(model_id, "mmm")
        
        scenario_results = mmm.what_if_scenario(
            spend_changes=request.spend_changes,
            time_period=tuple(request.time_period) if request.time_period else None,
            random_seed=42,
        )
        
        return ScenarioResponse(
            model_id=model_id,
            baseline_outcome=float(scenario_results["baseline_outcome"]),
            scenario_outcome=float(scenario_results["scenario_outcome"]),
            outcome_change=float(scenario_results["outcome_change"]),
            outcome_change_pct=float(scenario_results["outcome_change_pct"]),
            spend_changes=scenario_results["spend_changes"],
        )
        
    except Exception as e:
        logger.error(f"Error running scenario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running scenario: {str(e)}",
        )


@router.post(
    "/{model_id}/predict",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def generate_predictions(
    model_id: str,
    request: PredictionRequest,
    storage: StorageService = Depends(get_storage),
):
    """
    Generate predictions for the model.
    
    Can optionally provide modified media spend for counterfactual predictions.
    """
    _check_model_completed(storage, model_id)
    
    try:
        mmm = storage.load_model_artifact(model_id, "mmm")
        
        # Build media data if provided
        X_media = None
        if request.media_spend:
            channel_names = mmm.channel_names
            n_obs = len(list(request.media_spend.values())[0])
            X_media = np.zeros((n_obs, len(channel_names)))
            for i, ch in enumerate(channel_names):
                if ch in request.media_spend:
                    X_media[:, i] = request.media_spend[ch]
        
        pred_results = mmm.predict(
            X_media=X_media,
            return_original_scale=True,
            random_seed=42,
        )
        
        response = PredictionResponse(
            model_id=model_id,
            y_pred_mean=pred_results.y_pred_mean.tolist(),
            y_pred_std=pred_results.y_pred_std.tolist(),
            y_pred_hdi_low=pred_results.y_pred_hdi_low.tolist(),
            y_pred_hdi_high=pred_results.y_pred_hdi_high.tolist(),
        )
        
        if request.return_samples:
            response.y_pred_samples = pred_results.y_pred_samples[:100].tolist()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating predictions: {str(e)}",
        )


@router.get(
    "/{model_id}/summary",
    responses={404: {"model": ErrorResponse}},
)
async def get_model_summary(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Get a summary of the model including metadata and key results."""
    _check_model_completed(storage, model_id)
    
    try:
        metadata = storage.get_model_metadata(model_id)
        summary_results = storage.load_results(model_id, "summary")
        
        response = {
            "model_id": model_id,
            "name": metadata.get("name"),
            "description": metadata.get("description"),
            "created_at": metadata.get("created_at"),
            "completed_at": metadata.get("completed_at"),
            "data_id": metadata.get("data_id"),
            "config_id": metadata.get("config_id"),
            "n_obs": summary_results.get("n_obs"),
            "n_channels": summary_results.get("n_channels"),
            "n_controls": summary_results.get("n_controls"),
            "channel_names": summary_results.get("channel_names"),
            "control_names": summary_results.get("control_names"),
            "diagnostics": summary_results.get("diagnostics"),
            "parameter_summary": summary_results.get("parameter_summary"),
        }
        
        # Sanitize NaN/Inf values for JSON serialization
        sanitized = _sanitize_for_json(response)
        return SafeJSONResponse(content=sanitized)
        
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Summary not found",
        )