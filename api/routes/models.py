"""
Model management API routes.

Handles model fitting, tracking, results, and downloads.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from arq import ArqRedis, create_pool
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
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
        "name": request.name or f"Model {model_id}",
        "description": request.description,
        "data_id": request.data_id,
        "config_id": request.config_id,
        "status": JobStatus.QUEUED.value,
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    storage.save_model_metadata(model_id, metadata)
    
    # Set initial status in Redis
    await redis.set_job_status(
        model_id,
        JobStatus.QUEUED,
        progress=0.0,
        progress_message="Job queued",
    )
    
    # Build overrides
    overrides = {}
    if request.n_chains is not None:
        overrides["n_chains"] = request.n_chains
    if request.n_draws is not None:
        overrides["n_draws"] = request.n_draws
    if request.n_tune is not None:
        overrides["n_tune"] = request.n_tune
    if request.random_seed is not None:
        overrides["random_seed"] = request.random_seed
    
    # Enqueue the fitting task
    await arq_pool.enqueue_job(
        "fit_model_task",
        model_id=model_id,
        data_id=request.data_id,
        config_id=request.config_id,
        overrides=overrides if overrides else None,
    )
    
    return _metadata_to_model_info(metadata)


@router.get(
    "",
    response_model=ModelListResponse,
)
async def list_models(
    storage: StorageService = Depends(get_storage),
    status_filter: JobStatus | None = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List all models."""
    models = storage.list_models()
    
    # Apply status filter
    if status_filter:
        models = [m for m in models if m.get("status") == status_filter.value]
    
    total = len(models)
    models = models[skip : skip + limit]
    
    return ModelListResponse(
        models=[_metadata_to_model_info(m) for m in models],
        total=total,
    )


@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_model(
    model_id: str,
    storage: StorageService = Depends(get_storage),
    redis: RedisService = Depends(get_redis),
):
    """Get model information and status."""
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    
    metadata = storage.get_model_metadata(model_id)
    
    # Get real-time status from Redis if job is active
    if metadata.get("status") in [JobStatus.QUEUED.value, JobStatus.RUNNING.value]:
        redis_status = await redis.get_job_status(model_id)
        if redis_status:
            metadata["status"] = redis_status.get("status", metadata.get("status"))
            metadata["progress"] = float(redis_status.get("progress", 0))
            metadata["progress_message"] = redis_status.get("progress_message")
    
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
    """Get real-time model status (optimized for polling)."""
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    
    # Try Redis first for real-time status
    redis_status = await redis.get_job_status(model_id)
    
    if redis_status:
        return {
            "model_id": model_id,
            "status": redis_status.get("status"),
            "progress": float(redis_status.get("progress", 0)),
            "progress_message": redis_status.get("progress_message"),
            "error_message": redis_status.get("error_message"),
        }
    
    # Fall back to stored metadata
    metadata = storage.get_model_metadata(model_id)
    return {
        "model_id": model_id,
        "status": metadata.get("status"),
        "progress": float(metadata.get("progress", 0)),
        "progress_message": metadata.get("progress_message"),
        "error_message": metadata.get("error_message"),
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
    """Get model results after fitting is complete."""
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
    
    try:
        results = storage.load_results(model_id, "summary")
        
        return ModelResultsResponse(
            model_id=model_id,
            status=JobStatus.COMPLETED,
            diagnostics=results.get("diagnostics", {}),
            parameter_summary=results.get("parameter_summary", []),
            channel_contributions=None,  # Computed separately
            component_decomposition=None,
        )
    
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results not found. Model may still be processing.",
        )


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
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    logger.info('Fetching data')
    metadata = storage.get_model_metadata(model_id)
    logger.info(metadata)
    if metadata.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model is not completed. Current status: {metadata.get('status')}",
        )
    
    # Run computation (this is relatively fast, so we do it synchronously)
    try:
        mmm = storage.load_model_artifact(model_id, "mmm")
        
        contrib_results = mmm.compute_counterfactual_contributions(
            time_period=request.time_period,
            channels=request.channels,
            compute_uncertainty=request.compute_uncertainty,
            hdi_prob=request.hdi_prob,
            random_seed=42,
        )
        
        response = ContributionResponse(
            model_id=model_id,
            total_contributions=contrib_results.total_contributions.to_dict(),
            contribution_pct=contrib_results.contribution_pct.to_dict(),
            time_period=request.time_period,
        )
        
        if contrib_results.contribution_hdi_low is not None:
            response.contribution_hdi_low = contrib_results.contribution_hdi_low.to_dict()
            response.contribution_hdi_high = contrib_results.contribution_hdi_high.to_dict()
        
        return response
        
    except Exception as e:
        logger.error(f"Error computing contributions: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
    Run a what-if scenario analysis.
    
    Simulates the impact of changing media spend by specified multipliers.
    E.g., {"TV": 1.2, "Digital": 0.8} = +20% TV, -20% Digital.
    """
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
    
    try:
        mmm = storage.load_model_artifact(model_id, "mmm")
        
        scenario_results = mmm.what_if_scenario(
            spend_changes=request.spend_changes,
            time_period=request.time_period,
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
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
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
async def get_predictions(
    model_id: str,
    request: PredictionRequest,
    storage: StorageService = Depends(get_storage),
):
    """
    Get predictions from a fitted model.
    
    Can optionally provide modified media spend for counterfactual predictions.
    """
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
    
    try:
        mmm = storage.load_model_artifact(model_id, "mmm")
        
        # Build media data if provided
        import numpy as np
        X_media = None
        if request.media_spend:
            # Convert dict of lists to numpy array
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
            response.y_pred_samples = pred_results.y_pred_samples[:100].tolist()  # Limit samples
        
        return response
        
    except Exception as e:
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
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    
    metadata = storage.get_model_metadata(model_id)
    
    summary = {
        "model_id": model_id,
        "name": metadata.get("name"),
        "description": metadata.get("description"),
        "status": metadata.get("status"),
        "created_at": metadata.get("created_at"),
        "completed_at": metadata.get("completed_at"),
        "data_id": metadata.get("data_id"),
        "config_id": metadata.get("config_id"),
    }
    
    if metadata.get("status") == JobStatus.COMPLETED.value:
        try:
            results = storage.load_results(model_id, "summary")
            summary["diagnostics"] = results.get("diagnostics")
            summary["n_obs"] = results.get("n_obs")
            summary["n_channels"] = results.get("n_channels")
            summary["channel_names"] = results.get("channel_names")
            summary["y_mean"] = results.get("y_mean")
            summary["y_std"] = results.get("y_std")
        except StorageError:
            pass
    
    return summary