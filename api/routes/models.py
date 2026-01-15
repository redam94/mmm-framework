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
import pandas as pd
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
    ReportRequest,
    ReportResponse,
    ReportStatusResponse,
    ReportListResponse,
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
        elif hasattr(obj, "item"):
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
        elif hasattr(obj, "item"):
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
    elif hasattr(obj, "item"):
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
        started_at=(
            datetime.fromisoformat(metadata["started_at"])
            if metadata.get("started_at")
            else None
        ),
        completed_at=(
            datetime.fromisoformat(metadata["completed_at"])
            if metadata.get("completed_at")
            else None
        ),
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
    "/{model_id}/report",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def generate_report(
    model_id: str,
    request: ReportRequest,
    storage: StorageService = Depends(get_storage),
    arq_pool: ArqRedis = Depends(get_arq_pool),
    redis: RedisService = Depends(get_redis),
):
    """
    Generate an HTML report for a fitted model.

    This is an async operation. Use the returned report_id to track progress
    and download the report when complete.
    """
    _check_model_completed(storage, model_id)

    # Generate report ID
    report_id = storage.generate_id()[:12]

    # Queue the report generation task
    await arq_pool.enqueue_job(
        "generate_report_task",
        model_id=model_id,
        report_id=report_id,
        config={
            "title": request.title,
            "client": request.client,
            "subtitle": request.subtitle,
            "analysis_period": request.analysis_period,
            "include_executive_summary": request.include_executive_summary,
            "include_model_fit": request.include_model_fit,
            "include_channel_roi": request.include_channel_roi,
            "include_decomposition": request.include_decomposition,
            "include_saturation": request.include_saturation,
            "include_diagnostics": request.include_diagnostics,
            "include_methodology": request.include_methodology,
            "credible_interval": request.credible_interval,
            "currency_symbol": request.currency_symbol,
            "currency_scale": request.currency_scale,
            "color_scheme": request.color_scheme,
        },
    )

    logger.info(f"Queued report generation: {report_id} for model {model_id}")

    return ReportResponse(
        model_id=model_id,
        report_id=report_id,
        status="generating",
        message="Report generation started",
        created_at=datetime.utcnow(),
    )


@router.get(
    "/{model_id}/report/{report_id}/status",
    response_model=ReportStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_report_status(
    model_id: str,
    report_id: str,
    redis: RedisService = Depends(get_redis),
):
    """Get report generation status."""
    r = await redis.connect()
    report_key = f"mmm:report:{report_id}"

    report_data = await r.hgetall(report_key)

    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report not found: {report_id}",
        )

    return ReportStatusResponse(
        report_id=report_id,
        model_id=model_id,
        status=report_data.get(b"status", b"unknown").decode(),
        message=report_data.get(b"message", b"").decode() or None,
        filename=report_data.get(b"filename", b"").decode() or None,
    )


@router.get(
    "/{model_id}/report/{report_id}/download",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def download_report(
    model_id: str,
    report_id: str,
    storage: StorageService = Depends(get_storage),
    redis: RedisService = Depends(get_redis),
    settings: Settings = Depends(get_settings),
):
    """Download a generated report."""
    r = await redis.connect()
    report_key = f"mmm:report:{report_id}"

    report_data = await r.hgetall(report_key)

    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report not found: {report_id}",
        )

    # FIX: Use string keys, not bytes
    status_val = report_data.get("status", "")
    if status_val != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report not ready. Status: {status_val}",
        )

    filepath = report_data.get("filepath", "")
    filename = report_data.get("filename", "")

    if not filepath or not Path(filepath).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found",
        )

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="text/html",
    )


@router.get(
    "/{model_id}/reports",
    response_model=ReportListResponse,
    responses={404: {"model": ErrorResponse}},
)
async def list_model_reports(
    model_id: str,
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    """List all reports for a model."""
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )

    reports_dir = Path(settings.storage_path) / "reports"

    reports = []
    if reports_dir.exists():
        for report_file in reports_dir.glob(f"{model_id}_*.html"):
            reports.append(
                {
                    "filename": report_file.name,
                    "report_id": report_file.stem.split("_")[-1],
                    "created_at": datetime.fromtimestamp(
                        report_file.stat().st_mtime
                    ).isoformat(),
                    "size_bytes": report_file.stat().st_size,
                }
            )

    # Sort by created_at descending
    reports.sort(key=lambda x: x["created_at"], reverse=True)

    return ReportListResponse(model_id=model_id, reports=reports)

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
    models = all_models[skip : skip + limit]

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
        r2 = 1 - np.sum((y_obs - y_pred_mean) ** 2) / np.sum(
            (y_obs - np.mean(y_obs)) ** 2
        )
        rmse = np.sqrt(np.mean((y_obs - y_pred_mean) ** 2))
        mape = np.mean(np.abs((y_obs - y_pred_mean) / (y_obs + 1e-8))) * 100

        # Get period labels
        periods = panel.coords.periods
        period_labels = [
            str(p.date()) if hasattr(p, "date") else str(p) for p in periods
        ]
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
        y_obs_reshaped = (
            y_obs.reshape(n_periods, n_geos, n_products)
            if (has_geo or has_product)
            else y_obs.reshape(n_periods, 1, 1)
        )
        y_pred_reshaped = (
            y_pred_mean.reshape(n_periods, n_geos, n_products)
            if (has_geo or has_product)
            else y_pred_mean.reshape(n_periods, 1, 1)
        )

        # Aggregated (sum over geo and product)
        y_obs_agg = y_obs_reshaped.sum(axis=(1, 2))
        y_pred_agg = y_pred_reshaped.sum(axis=(1, 2))

        # Compute aggregated metrics
        r2_agg = 1 - np.sum((y_obs_agg - y_pred_agg) ** 2) / np.sum(
            (y_obs_agg - np.mean(y_obs_agg)) ** 2
        )
        rmse_agg = np.sqrt(np.mean((y_obs_agg - y_pred_agg) ** 2))
        mape_agg = np.mean(np.abs((y_obs_agg - y_pred_agg) / (y_obs_agg + 1e-8))) * 100

        # Compute predicted std aggregated if available
        y_pred_std_agg = None
        if y_pred_std is not None:
            y_pred_std_reshaped = (
                y_pred_std.reshape(n_periods, n_geos, n_products)
                if (has_geo or has_product)
                else y_pred_std.reshape(n_periods, 1, 1)
            )
            # For summed values, std combines in quadrature
            y_pred_std_agg = np.sqrt((y_pred_std_reshaped**2).sum(axis=(1, 2))).tolist()

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

                r2_geo = 1 - np.sum((y_obs_geo - y_pred_geo) ** 2) / np.sum(
                    (y_obs_geo - np.mean(y_obs_geo)) ** 2
                )
                rmse_geo = np.sqrt(np.mean((y_obs_geo - y_pred_geo) ** 2))
                mape_geo = (
                    np.mean(np.abs((y_obs_geo - y_pred_geo) / (y_obs_geo + 1e-8))) * 100
                )

                # Compute std for this geo (sum over products in quadrature)
                y_pred_std_geo = None
                if y_pred_std is not None:
                    y_pred_std_reshaped = (
                        y_pred_std.reshape(n_periods, n_geos, n_products)
                        if (has_geo or has_product)
                        else y_pred_std.reshape(n_periods, 1, 1)
                    )
                    # For summed values, std combines in quadrature
                    y_pred_std_geo = np.sqrt(
                        (y_pred_std_reshaped[:, g_idx, :] ** 2).sum(axis=1)
                    ).tolist()

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
    parameters: list[str] | None = Query(
        None, description="Filter to specific parameters"
    ),
    n_samples: int = Query(
        500, ge=100, le=2000, description="Number of samples to return"
    ),
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
                posteriors = {
                    k: v
                    for k, v in posteriors.items()
                    if k in parameters or k == "model_id"
                }
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
    n_samples: int = Query(
        500, ge=100, le=2000, description="Number of samples to return"
    ),
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
                        idx = np.random.choice(
                            len(prior_samples), n_samples, replace=False
                        )
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
                    param_data.update(
                        {
                            "prior_samples": prior_samples.tolist(),
                            "prior_mean": prior_mean,
                            "prior_std": prior_std,
                            "shrinkage_pct": shrinkage,
                        }
                    )

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
    n_samples: int = Query(
        200, ge=50, le=500, description="Number of posterior samples for uncertainty"
    ),
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
            idx = np.random.choice(
                len(sat_lam_samples), n_posterior_samples, replace=False
            )

            response_curves = np.zeros((n_posterior_samples, n_points))
            for i, j in enumerate(idx):
                saturated = 1 - np.exp(-sat_lam_samples[j] * x_scaled)
                response_curves[i, :] = beta_samples[j] * saturated

            # Scale back to original units
            response_curves_original = response_curves * mmm.y_std

            curves["channels"][channel] = {
                "spend": x_original.tolist(),
                "response": response_curves_original.mean(axis=0).tolist(),
                "response_hdi_low": np.percentile(
                    response_curves_original, 3, axis=0
                ).tolist(),
                "response_hdi_high": np.percentile(
                    response_curves_original, 97, axis=0
                ).tolist(),
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
    force_recompute: bool = Query(
        False, description="Force recomputation of decomposition"
    ),
    storage: StorageService = Depends(get_storage),
):
    """
    Get component decomposition (trend, seasonality, media contributions, etc.).

    Returns time series of each component's contribution to the outcome,
    both aggregated and by geography.
    """
    _check_model_completed(storage, model_id)

    try:
        # Try to load cached decomposition
        if not force_recompute:
            try:
                decomposition = storage.load_results(model_id, "decomposition")
                return SafeJSONResponse(content=decomposition)
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

        # Log available variables for debugging
        posterior_vars = list(posterior.data_vars.keys())
        logger.info(f"Available posterior variables: {posterior_vars}")

        # Get basic dimensions
        n_obs = len(panel.y.values.flatten())
        n_periods = mmm.n_periods if hasattr(mmm, "n_periods") else n_obs

        # Get observed values (in original scale)
        y_obs_standardized = panel.y.values.flatten()
        y_obs = y_obs_standardized * mmm.y_std + mmm.y_mean

        # Get period labels - try multiple sources
        periods = None
        unique_periods = None

        # Method 1: From panel.coords.periods (preferred - contains DatetimeIndex)
        if (
            hasattr(panel, "coords")
            and hasattr(panel.coords, "periods")
            and panel.coords.periods is not None
        ):
            datetime_periods = panel.coords.periods
            # These are unique periods
            unique_periods = [
                p.strftime("%Y-%m-%d") if hasattr(p, "strftime") else str(p)
                for p in datetime_periods
            ]
            logger.info(f"Got {len(unique_periods)} unique periods from coords.periods")

            # Now we need to map each observation to its period
            # If panel has time_idx, use that; otherwise derive from index
            if hasattr(panel, "time_idx") and panel.time_idx is not None:
                # time_idx maps each obs to its period index
                time_idx = np.array(panel.time_idx)
                periods = [
                    unique_periods[min(t, len(unique_periods) - 1)] for t in time_idx
                ]
            elif hasattr(panel, "index") and panel.index is not None:
                # Try to get period from multi-index
                if isinstance(panel.index, pd.MultiIndex):
                    # First level should be date
                    date_level = panel.index.get_level_values(0)
                    periods = [
                        d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                        for d in date_level
                    ]
                    unique_periods = sorted(set(periods))
                elif isinstance(panel.index, pd.DatetimeIndex):
                    periods = [d.strftime("%Y-%m-%d") for d in panel.index]
                    unique_periods = sorted(set(periods))
                else:
                    # Fall back to repeating unique periods
                    n_reps = n_obs // len(unique_periods)
                    periods = unique_periods * n_reps
                    if len(periods) < n_obs:
                        periods = periods + unique_periods[: n_obs - len(periods)]
            else:
                # Assume observations are in period order, possibly with geo/product repeats
                n_geos = (
                    len(mmm.geo_names)
                    if hasattr(mmm, "geo_names") and mmm.geo_names
                    else 1
                )
                n_prods = getattr(mmm, "n_products", 1) or 1
                periods = []
                for p in unique_periods:
                    periods.extend([p] * (n_geos * n_prods))
                periods = periods[:n_obs]

        # Method 2: From panel.periods directly
        elif hasattr(panel, "periods") and panel.periods is not None:
            raw_periods = panel.periods
            if hasattr(raw_periods, "__iter__"):
                periods = [
                    p.strftime("%Y-%m-%d") if hasattr(p, "strftime") else str(p)
                    for p in raw_periods
                ]
            else:
                periods = [str(raw_periods)]
            unique_periods = sorted(set(periods))
            logger.info(f"Got {len(unique_periods)} unique periods from panel.periods")

        # Method 3: Fall back to numeric indices
        if periods is None or unique_periods is None:
            logger.warning("Could not extract date periods, using numeric indices")
            periods = [str(i) for i in range(n_obs)]
            unique_periods = [str(i) for i in range(n_periods)]

        logger.info(
            f"Period info: n_obs={n_obs}, n_unique_periods={len(unique_periods)}, sample periods={unique_periods[:3]}"
        )

        # Get geo and time indices
        has_geo = hasattr(panel, "geo_idx") and panel.geo_idx is not None
        geo_idx = np.array(panel.geo_idx) if has_geo else np.zeros(n_obs, dtype=int)
        geo_names = (
            list(mmm.geo_names)
            if hasattr(mmm, "geo_names") and mmm.geo_names
            else ["National"]
        )
        n_geos = len(geo_names)

        # Get product indices if available
        has_product = hasattr(panel, "product_idx") and panel.product_idx is not None
        product_idx = (
            np.array(panel.product_idx) if has_product else np.zeros(n_obs, dtype=int)
        )
        product_names = (
            list(mmm.product_names)
            if hasattr(mmm, "product_names") and mmm.product_names
            else ["All"]
        )
        n_products = len(product_names)

        # Get time_idx for mapping period-level arrays to obs-level
        if hasattr(panel, "time_idx") and panel.time_idx is not None:
            time_idx = np.array(panel.time_idx)
        else:
            # Create time_idx from periods list
            period_to_idx = {p: i for i, p in enumerate(unique_periods)}
            time_idx = np.array([period_to_idx.get(p, 0) for p in periods])

        # SAFETY: Ensure time_idx is within valid bounds for period-level arrays
        n_unique_periods = len(unique_periods)
        if time_idx is not None and len(time_idx) > 0:
            max_time_idx = int(np.max(time_idx))
            min_time_idx = int(np.min(time_idx))
            logger.info(
                f"time_idx range: [{min_time_idx}, {max_time_idx}], n_unique_periods: {n_unique_periods}"
            )

            # Clip time_idx to valid range
            time_idx = np.clip(time_idx, 0, n_unique_periods - 1)

        logger.info(
            f"Panel structure: n_obs={n_obs}, n_periods={n_unique_periods}, n_geos={n_geos}, n_products={n_products}"
        )

        # Initialize decomposition structure
        decomposition = {
            "model_id": model_id,
            "periods": unique_periods,
            "n_periods": n_unique_periods,
            "n_obs": n_obs,
            "components": {},  # Aggregated (summed over geo/product)
            "by_geography": {},  # Per-geo time series
            "by_product": {},  # Per-product time series
            "by_geo_product": {},  # Geo x Product breakdown
            "observed": [],  # Will be set after aggregation
            "observed_by_geography": {},
            "observed_by_product": {},
            "metadata": {
                "geo_names": geo_names,
                "product_names": product_names,
                "channel_names": mmm.channel_names,
                "control_names": (
                    mmm.control_names if hasattr(mmm, "control_names") else []
                ),
                "has_geo": n_geos > 1,
                "has_product": n_products > 1,
                "has_trend": False,
                "has_seasonality": False,
                "trend_type": (
                    mmm.trend_config.type.value
                    if hasattr(mmm, "trend_config")
                    else "unknown"
                ),
                "posterior_variables": posterior_vars,
            },
        }

        # =================================================================
        # AGGREGATION HELPER FUNCTIONS
        # =================================================================

        def aggregate_to_period(values, agg_func="sum"):
            """Aggregate observation-level values to period level (summing over geo/product)."""
            values = np.array(values)
            if len(values) == n_unique_periods:
                return values  # Already at period level

            df = pd.DataFrame({"period": periods, "value": values})
            if agg_func == "sum":
                agg = df.groupby("period")["value"].sum()
            else:
                agg = df.groupby("period")["value"].mean()
            return agg.reindex(unique_periods).fillna(0).values

        def aggregate_by_geo(values):
            """Aggregate to geo x period level."""
            values = np.array(values)
            result = {}
            df = pd.DataFrame(
                {
                    "period": periods,
                    "geo": [geo_names[int(g)] for g in geo_idx],
                    "value": values,
                }
            )
            for geo in geo_names:
                geo_df = df[df["geo"] == geo]
                if len(geo_df) > 0:
                    agg = geo_df.groupby("period")["value"].sum()
                    result[geo] = agg.reindex(unique_periods).fillna(0).values.tolist()
            return result

        def aggregate_by_product(values):
            """Aggregate to product x period level."""
            values = np.array(values)
            result = {}
            df = pd.DataFrame(
                {
                    "period": periods,
                    "product": [product_names[int(p)] for p in product_idx],
                    "value": values,
                }
            )
            for prod in product_names:
                prod_df = df[df["product"] == prod]
                if len(prod_df) > 0:
                    agg = prod_df.groupby("period")["value"].sum()
                    result[prod] = agg.reindex(unique_periods).fillna(0).values.tolist()
            return result

        def store_component(name, obs_values, scale_factor=1.0):
            """Store a component with all aggregation levels."""
            obs_values = np.array(obs_values) * scale_factor

            # Aggregate to period level (sum over geo/product)
            period_values = aggregate_to_period(obs_values, "sum")
            decomposition["components"][name] = period_values.tolist()

            # By geography
            if n_geos > 1:
                decomposition["by_geography"][name] = aggregate_by_geo(obs_values)

            # By product
            if n_products > 1:
                decomposition["by_product"][name] = aggregate_by_product(obs_values)

        # =================================================================
        # OBSERVED VALUES (in original scale)
        # =================================================================
        # y_obs is already in original scale from earlier
        decomposition["observed"] = aggregate_to_period(y_obs, "sum").tolist()
        if n_geos > 1:
            decomposition["observed_by_geography"] = aggregate_by_geo(y_obs)
        if n_products > 1:
            decomposition["observed_by_product"] = aggregate_by_product(y_obs)

        # =================================================================
        # INTERCEPT / BASELINE
        # The baseline represents the expected value when all effects are zero.
        # In standardized space: y = intercept + effects
        # In original space: Y = (intercept + effects) * y_std + y_mean
        # So baseline contribution per obs = intercept * y_std + y_mean
        # But y_mean is the grand mean, so we attribute it to baseline.
        # =================================================================
        if "intercept" in posterior:
            intercept = float(posterior["intercept"].mean().values)
            # Per-observation baseline in original scale
            baseline_per_obs = intercept * mmm.y_std + mmm.y_mean
            baseline_obs = np.full(n_obs, baseline_per_obs)
            store_component("baseline", baseline_obs)
        else:
            baseline_obs = np.full(n_obs, float(mmm.y_mean))
            store_component("baseline", baseline_obs)

        # =================================================================
        # TREND - Try multiple extraction methods
        # =================================================================
        trend_extracted = False
        trend_obs = np.zeros(n_obs)

        # Method 1: Look for trend_slope with t_scaled
        if "trend_slope" in posterior and hasattr(mmm, "t_scaled"):
            try:
                trend_slope = float(posterior["trend_slope"].mean().values)
                t_scaled = np.array(mmm.t_scaled)
                # t_scaled should have length n_periods (unique periods)
                trend_at_periods = trend_slope * t_scaled
                # Safely index - clip time_idx to valid range
                safe_time_idx = np.clip(time_idx, 0, len(trend_at_periods) - 1)
                trend_obs = trend_at_periods[safe_time_idx] * mmm.y_std
                trend_extracted = True
                logger.info(
                    f"Extracted linear trend with slope={trend_slope:.4f}, t_scaled len={len(t_scaled)}"
                )
            except Exception as e:
                logger.warning(f"Failed to extract linear trend: {e}")

        # Method 2: Look for deterministic trend variables
        if not trend_extracted:
            trend_var_names = [
                "trend",
                "trend_component",
                "trend_contribution",
                "trend_effect",
                "trend_unique",
            ]
            for var in trend_var_names:
                if var in posterior:
                    try:
                        trend_vals = (
                            posterior[var].mean(dim=["chain", "draw"]).values.flatten()
                        )
                        if len(trend_vals) == n_obs:
                            trend_obs = trend_vals * mmm.y_std
                            trend_extracted = True
                            logger.info(f"Extracted trend from {var} (n_obs={n_obs})")
                            break
                        elif len(trend_vals) >= n_unique_periods:
                            # Map from periods to observations with safe indexing
                            safe_time_idx = np.clip(time_idx, 0, len(trend_vals) - 1)
                            trend_obs = (trend_vals * mmm.y_std)[safe_time_idx]
                            trend_extracted = True
                            logger.info(
                                f"Extracted trend from {var} (len={len(trend_vals)})"
                            )
                            break
                    except Exception as e:
                        logger.warning(f"Failed to extract trend from {var}: {e}")

        # Method 3: Look for spline coefficients
        if not trend_extracted and "spline_coef" in posterior:
            if hasattr(mmm, "trend_features") and "spline_basis" in mmm.trend_features:
                try:
                    spline_coef = (
                        posterior["spline_coef"].mean(dim=["chain", "draw"]).values
                    )
                    basis = mmm.trend_features["spline_basis"]
                    trend_at_periods = basis @ spline_coef
                    trend_at_periods = (
                        trend_at_periods - trend_at_periods.mean()
                    )  # Center
                    # Safe indexing
                    safe_time_idx = np.clip(time_idx, 0, len(trend_at_periods) - 1)
                    trend_obs = (trend_at_periods * mmm.y_std)[safe_time_idx]
                    trend_extracted = True
                    logger.info(
                        f"Extracted spline trend from coefficients, basis shape={basis.shape}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract spline trend: {e}")

        # Method 4: Look for piecewise trend components
        if not trend_extracted and "trend_k" in posterior:
            if hasattr(mmm, "trend_features"):
                try:
                    k = float(posterior["trend_k"].mean().values)
                    m = (
                        float(posterior["trend_m"].mean().values)
                        if "trend_m" in posterior
                        else 0
                    )
                    t_unique = np.linspace(0, 1, n_unique_periods)

                    if (
                        "trend_delta" in posterior
                        and "changepoint_matrix" in mmm.trend_features
                    ):
                        delta = (
                            posterior["trend_delta"].mean(dim=["chain", "draw"]).values
                        )
                        A = mmm.trend_features["changepoint_matrix"]
                        s = mmm.trend_features["changepoints"]
                        gamma = -s * delta
                        trend_at_periods = k * t_unique + A @ delta + m + A @ gamma
                    else:
                        trend_at_periods = k * t_unique + m

                    # Safe indexing
                    safe_time_idx = np.clip(time_idx, 0, len(trend_at_periods) - 1)
                    trend_obs = (trend_at_periods * mmm.y_std)[safe_time_idx]
                    trend_extracted = True
                    logger.info(f"Extracted piecewise trend, k={k:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to extract piecewise trend: {e}")

        if trend_extracted:
            store_component("trend", trend_obs)
            decomposition["metadata"]["has_trend"] = True

        # =================================================================
        # SEASONALITY - Try multiple extraction methods
        # =================================================================
        seasonality_extracted = False
        seasonality_obs = np.zeros(n_obs)

        # Method 1: Look for seasonality_component (obs-level)
        seasonality_var_names = [
            "seasonality_component",
            "seasonality",
            "seasonal_effect",
            "seasonality_contribution",
        ]
        for var in seasonality_var_names:
            if var in posterior:
                try:
                    seas_vals = (
                        posterior[var].mean(dim=["chain", "draw"]).values.flatten()
                    )
                    if len(seas_vals) == n_obs:
                        seasonality_obs = seas_vals * mmm.y_std
                        seasonality_extracted = True
                        logger.info(f"Extracted seasonality from {var} (n_obs={n_obs})")
                        break
                    elif len(seas_vals) >= n_unique_periods:
                        # Safe indexing
                        safe_time_idx = np.clip(time_idx, 0, len(seas_vals) - 1)
                        seasonality_obs = (seas_vals * mmm.y_std)[safe_time_idx]
                        seasonality_extracted = True
                        logger.info(
                            f"Extracted seasonality from {var} (len={len(seas_vals)})"
                        )
                        break
                except Exception as e:
                    logger.warning(f"Failed to extract seasonality from {var}: {e}")

        # Method 2: Look for seasonality_by_period
        if not seasonality_extracted and "seasonality_by_period" in posterior:
            try:
                seas_by_period = (
                    posterior["seasonality_by_period"]
                    .mean(dim=["chain", "draw"])
                    .values.flatten()
                )
                if len(seas_by_period) >= n_unique_periods:
                    safe_time_idx = np.clip(time_idx, 0, len(seas_by_period) - 1)
                    seasonality_obs = (seas_by_period * mmm.y_std)[safe_time_idx]
                    seasonality_extracted = True
                    logger.info(
                        f"Extracted seasonality from seasonality_by_period (len={len(seas_by_period)})"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to extract seasonality from seasonality_by_period: {e}"
                )

        # Method 3: Compute from Fourier coefficients
        if not seasonality_extracted and hasattr(mmm, "seasonality_features"):
            for name, features in mmm.seasonality_features.items():
                coef_var = f"season_{name}"
                if coef_var in posterior:
                    try:
                        coef = posterior[coef_var].mean(dim=["chain", "draw"]).values
                        if len(coef) == features.shape[1]:
                            seas_at_periods = features @ coef
                            safe_time_idx = np.clip(
                                time_idx, 0, len(seas_at_periods) - 1
                            )
                            seasonality_obs = (seas_at_periods * mmm.y_std)[
                                safe_time_idx
                            ]
                            seasonality_extracted = True
                            logger.info(
                                f"Computed seasonality from {coef_var}, features shape={features.shape}"
                            )
                            break
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute seasonality from {coef_var}: {e}"
                        )

        if seasonality_extracted:
            store_component("seasonality", seasonality_obs)
            decomposition["metadata"]["has_seasonality"] = True

            # Also store seasonality pattern (one cycle for visualization)
            # Assuming weekly data with 52-week cycle
            period_seasonality = aggregate_to_period(seasonality_obs)
            if len(unique_periods) >= 52:
                decomposition["metadata"]["seasonality_pattern"] = period_seasonality[
                    :52
                ].tolist()

        # =================================================================
        # MEDIA CHANNEL CONTRIBUTIONS
        # All media contributions should be in original scale and summed
        # =================================================================
        channel_names = mmm.channel_names
        media_obs_total = np.zeros(n_obs)

        # Method 1: Look for channel_contributions matrix
        if "channel_contributions" in posterior:
            contrib_matrix = (
                posterior["channel_contributions"].mean(dim=["chain", "draw"]).values
            )
            logger.info(f"channel_contributions shape: {contrib_matrix.shape}")

            # Handle different shapes
            if len(contrib_matrix.shape) == 2:
                if contrib_matrix.shape[0] == n_obs and contrib_matrix.shape[1] == len(
                    channel_names
                ):
                    for i, channel in enumerate(channel_names):
                        ch_obs = contrib_matrix[:, i] * mmm.y_std
                        store_component(f"media_{channel}", ch_obs)
                        media_obs_total += ch_obs
                elif contrib_matrix.shape[0] >= n_unique_periods:
                    # Contributions at period level, need to map to obs with safe indexing
                    for i, channel in enumerate(channel_names):
                        ch_periods = contrib_matrix[:, i] * mmm.y_std
                        safe_time_idx = np.clip(time_idx, 0, len(ch_periods) - 1)
                        ch_obs = ch_periods[safe_time_idx]
                        store_component(f"media_{channel}", ch_obs)
                        media_obs_total += ch_obs

        # Method 2: Look for media_total
        if "media_total" in posterior and not any(
            k.startswith("media_") for k in decomposition["components"]
        ):
            media_total_vals = (
                posterior["media_total"].mean(dim=["chain", "draw"]).values.flatten()
            )
            if len(media_total_vals) == n_obs:
                media_obs_total = media_total_vals * mmm.y_std
            elif len(media_total_vals) >= n_unique_periods:
                safe_time_idx = np.clip(time_idx, 0, len(media_total_vals) - 1)
                media_obs_total = (media_total_vals * mmm.y_std)[safe_time_idx]
            store_component("media_total", media_obs_total)

        # Store total media contribution if we have individual channels
        if np.any(media_obs_total != 0) and any(
            k.startswith("media_") and k != "media_total"
            for k in decomposition["components"]
        ):
            store_component("media_total", media_obs_total)

        # =================================================================
        # CONTROL VARIABLE CONTRIBUTIONS
        # =================================================================
        controls_obs_total = np.zeros(n_obs)
        control_names = (
            mmm.control_names
            if hasattr(mmm, "control_names") and mmm.control_names
            else []
        )

        # Method 1: Compute from beta_controls
        if "beta_controls" in posterior and len(control_names) > 0:
            try:
                beta_controls = (
                    posterior["beta_controls"].mean(dim=["chain", "draw"]).values
                )
                X_controls = (
                    panel.X_controls.values if hasattr(panel, "X_controls") else None
                )

                if X_controls is not None:
                    for i, control in enumerate(control_names):
                        if i < len(beta_controls) and i < X_controls.shape[1]:
                            ctrl_obs = X_controls[:, i] * beta_controls[i] * mmm.y_std
                            store_component(f"control_{control}", ctrl_obs)
                            controls_obs_total += ctrl_obs
            except Exception as e:
                logger.warning(f"Could not compute control contributions: {e}")

        if np.any(controls_obs_total != 0):
            store_component("controls_total", controls_obs_total)

        # =================================================================
        # GEO AND PRODUCT EFFECTS (hierarchical random effects)
        # =================================================================
        if "geo_effects" in posterior or "geo_contribution" in posterior:
            var_name = (
                "geo_effects" if "geo_effects" in posterior else "geo_contribution"
            )
            geo_vals = posterior[var_name].mean(dim=["chain", "draw"]).values.flatten()
            if len(geo_vals) == n_obs:
                geo_obs = geo_vals * mmm.y_std
                store_component("geo_effects", geo_obs)

        if "product_effects" in posterior or "product_contribution" in posterior:
            var_name = (
                "product_effects"
                if "product_effects" in posterior
                else "product_contribution"
            )
            prod_vals = posterior[var_name].mean(dim=["chain", "draw"]).values.flatten()
            if len(prod_vals) == n_obs:
                prod_obs = prod_vals * mmm.y_std
                store_component("product_effects", prod_obs)

        # =================================================================
        # COMPUTE PREDICTED AND RESIDUAL
        # =================================================================
        try:
            observed_period = np.array(decomposition["observed"])
            predicted_period = np.zeros(n_unique_periods)

            for key, values in decomposition["components"].items():
                if isinstance(values, list) and len(values) == n_unique_periods:
                    # Avoid double-counting totals
                    if key == "media_total" and any(
                        k.startswith("media_") and k != "media_total"
                        for k in decomposition["components"]
                    ):
                        continue
                    if key == "controls_total" and any(
                        k.startswith("control_") and k != "controls_total"
                        for k in decomposition["components"]
                    ):
                        continue
                    predicted_period += np.array(values)

            decomposition["predicted"] = predicted_period.tolist()
            decomposition["residual"] = (observed_period - predicted_period).tolist()

            # R-squared
            ss_res = np.sum((observed_period - predicted_period) ** 2)
            ss_tot = np.sum((observed_period - np.mean(observed_period)) ** 2)
            decomposition["metadata"]["r_squared"] = (
                float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            )

        except Exception as e:
            logger.warning(f"Could not compute predicted/residual: {e}")

        # Sanitize for JSON serialization
        decomposition = _sanitize_for_json(decomposition)

        # Log summary
        logger.info(
            f"Decomposition components: {list(decomposition['components'].keys())}"
        )
        logger.info(
            f"Decomposition by_geography keys: {list(decomposition.get('by_geography', {}).keys())}"
        )
        logger.info(
            f"Decomposition by_product keys: {list(decomposition.get('by_product', {}).keys())}"
        )

        # Cache the results
        storage.save_results(model_id, "decomposition", decomposition)

        return SafeJSONResponse(content=decomposition)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing decomposition: {e}")
        import traceback

        logger.error(traceback.format_exc())
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
    artifact: str = Query(
        "mmm", description="Artifact to download: mmm, results, panel"
    ),
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
            "contribution_hdi_low": (
                contrib_results.contribution_hdi_low.to_dict()
                if contrib_results.contribution_hdi_low is not None
                else None
            ),
            "contribution_hdi_high": (
                contrib_results.contribution_hdi_high.to_dict()
                if contrib_results.contribution_hdi_high is not None
                else None
            ),
            "time_period": request.time_period,
        }

        # Cache the results (include baseline for frontend even though not in schema)
        cache_data = response_data.copy()
        if hasattr(contrib_results, "baseline"):
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
