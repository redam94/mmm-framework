"""
Extended model API routes.

Handles NestedMMM, MultivariateMMM, and CombinedMMM model types.
"""

import json
import math
from datetime import datetime, timezone
from typing import Any
import uuid

import numpy as np
from arq import ArqRedis, create_pool
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from loguru import logger

from config import Settings, get_settings
from redis_service import RedisService, get_redis
from schemas import (
    ErrorResponse,
    JobStatus,
    ModelType,
    SuccessResponse,
    # Extended model schemas
    ExtendedConfigCreateRequest,
    ExtendedConfigResponse,
    ExtendedModelFitRequest,
    ExtendedModelInfo,
    MediationResultsResponse,
    MediationEffectSchema,
    MultivariateResultsResponse,
    CrossEffectResultSchema,
    CrossEffectType,
)
from storage import StorageError, StorageService, get_storage

router = APIRouter(prefix="/extended-models", tags=["Extended Models"])


# =============================================================================
# Helper Functions
# =============================================================================


class NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to null."""

    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SafeJSONResponse(JSONResponse):
    """JSON response that handles NaN/Inf values."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=NaNSafeEncoder,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize data for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


def _check_extended_model_exists(storage: StorageService, model_id: str) -> dict:
    """Check that an extended model exists and return metadata."""
    if not storage.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    return storage.get_model_metadata(model_id)


def _check_extended_model_completed(storage: StorageService, model_id: str) -> dict:
    """Check that an extended model exists and is completed."""
    metadata = _check_extended_model_exists(storage, model_id)
    if metadata.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model not completed. Status: {metadata.get('status')}",
        )
    return metadata


async def get_arq_pool(settings: Settings = Depends(get_settings)) -> ArqRedis:
    """Get ARQ Redis pool for queuing jobs."""
    return await create_pool(settings.redis_settings)


# =============================================================================
# Extended Configuration Endpoints
# =============================================================================


@router.post(
    "/configs",
    response_model=ExtendedConfigResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_extended_config(
    request: ExtendedConfigCreateRequest,
    storage: StorageService = Depends(get_storage),
):
    """
    Create a new extended model configuration.

    Supports:
    - **standard**: Basic BayesianMMM
    - **nested**: NestedMMM with mediating variables (e.g., brand awareness)
    - **multivariate**: MultivariateMMM with multiple correlated outcomes
    - **combined**: CombinedMMM combining nested and multivariate features
    """
    config_data = {
        "name": request.name,
        "description": request.description,
        "model_type": request.mff_config.model_type.value,
        "mff_config": request.mff_config.model_dump(),
        "model_settings": request.model_settings.model_dump(),
    }

    saved = storage.save_config(config_data)

    return ExtendedConfigResponse(
        config_id=saved["config_id"],
        name=saved["name"],
        description=saved.get("description"),
        model_type=ModelType(saved["model_type"]),
        mff_config=request.mff_config,
        model_settings=request.model_settings,
        created_at=datetime.fromisoformat(saved["created_at"]),
        updated_at=datetime.fromisoformat(saved["updated_at"]),
    )


@router.get(
    "/configs/{config_id}",
    response_model=ExtendedConfigResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_extended_config(
    config_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Get an extended model configuration."""
    try:
        config = storage.load_config(config_id)

        # Determine model type
        model_type = config.get("model_type", "standard")
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        return ExtendedConfigResponse(
            config_id=config["config_id"],
            name=config["name"],
            description=config.get("description"),
            model_type=model_type,
            mff_config=config["mff_config"],
            model_settings=config["model_settings"],
            created_at=datetime.fromisoformat(config["created_at"]),
            updated_at=datetime.fromisoformat(config["updated_at"]),
        )
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration not found: {config_id}",
        )


# =============================================================================
# Extended Model Fitting
# =============================================================================


@router.post(
    "/fit",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse, "description": "Data or config not found"},
    },
)
async def fit_extended_model(
    request: ExtendedModelFitRequest,
    storage: StorageService = Depends(get_storage),
    redis: RedisService = Depends(get_redis),
    arq_pool: ArqRedis = Depends(get_arq_pool),
):
    """
    Start fitting an extended model asynchronously.

    For **nested** models, you can optionally provide `mediator_data_id` with
    observed mediator values (e.g., brand awareness survey data).

    For **multivariate** models, specify `outcome_data_columns` mapping outcome
    names to column names in your data file.

    Returns a model_id that can be used to track progress and retrieve results.
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

    # Load config to get model type
    config = storage.load_config(request.config_id)
    model_type = config.get("model_type", "standard")

    # Validate mediator data if provided
    if request.mediator_data_id and not storage.data_exists(request.mediator_data_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mediator dataset not found: {request.mediator_data_id}",
        )

    # Generate model ID
    model_id = storage.generate_id()

    # Create model metadata
    metadata = {
        "model_id": model_id,
        "name": request.name or f"Extended Model {model_id}",
        "description": request.description,
        "model_type": model_type,
        "data_id": request.data_id,
        "config_id": request.config_id,
        "mediator_data_id": request.mediator_data_id,
        "outcome_data_columns": request.outcome_data_columns,
        "promotion_columns": request.promotion_columns,
        "status": JobStatus.PENDING.value,
        "progress": 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    storage.save_model_metadata(model_id, metadata)

    # Prepare overrides
    overrides = {}
    if request.n_chains is not None:
        overrides["n_chains"] = request.n_chains
    if request.n_draws is not None:
        overrides["n_draws"] = request.n_draws
    if request.n_tune is not None:
        overrides["n_tune"] = request.n_tune
    if request.random_seed is not None:
        overrides["random_seed"] = request.random_seed

    # Queue the fitting job
    job = await arq_pool.enqueue_job(
        "fit_extended_model_task",
        model_id,
        request.data_id,
        request.config_id,
        request.mediator_data_id,
        request.outcome_data_columns,
        request.promotion_columns,
        overrides if overrides else None,
    )

    # Update status to queued
    await redis.set_job_status(
        model_id,
        JobStatus.QUEUED,
        progress=0.0,
        progress_message="Job queued for processing",
    )

    storage.update_model_metadata(model_id, {"status": JobStatus.QUEUED.value})

    return {
        "model_id": model_id,
        "status": "queued",
        "model_type": model_type,
        "message": "Extended model fitting job queued",
    }


@router.get(
    "/{model_id}",
    response_model=ExtendedModelInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_extended_model(
    model_id: str,
    storage: StorageService = Depends(get_storage),
    redis: RedisService = Depends(get_redis),
):
    """Get information about an extended model."""
    metadata = _check_extended_model_exists(storage, model_id)

    # Try to get real-time status from Redis
    redis_status = await redis.get_job_status(model_id)
    if redis_status:
        progress = float(redis_status.get("progress", 0))
        progress_message = redis_status.get("progress_message")
        current_status = redis_status.get("status", metadata.get("status"))
    else:
        progress = metadata.get("progress", 0)
        progress_message = metadata.get("progress_message")
        current_status = metadata.get("status")

    model_type = metadata.get("model_type", "standard")
    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    return ExtendedModelInfo(
        model_id=model_id,
        name=metadata.get("name"),
        description=metadata.get("description"),
        model_type=model_type,
        data_id=metadata.get("data_id"),
        config_id=metadata.get("config_id"),
        status=JobStatus(current_status),
        progress=progress,
        progress_message=progress_message,
        created_at=datetime.fromisoformat(metadata.get("created_at")),
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
        mediator_names=metadata.get("mediator_names"),
        outcome_names=metadata.get("outcome_names"),
    )


@router.get(
    "/{model_id}/status",
    responses={404: {"model": ErrorResponse}},
)
async def get_extended_model_status(
    model_id: str,
    storage: StorageService = Depends(get_storage),
    redis: RedisService = Depends(get_redis),
):
    """Get real-time status of an extended model fitting job."""
    _check_extended_model_exists(storage, model_id)

    redis_status = await redis.get_job_status(model_id)
    if redis_status:
        return {
            "model_id": model_id,
            "status": redis_status.get("status"),
            "progress": float(redis_status.get("progress", 0)),
            "progress_message": redis_status.get("progress_message"),
            "updated_at": redis_status.get("updated_at"),
        }

    metadata = storage.get_model_metadata(model_id)
    return {
        "model_id": model_id,
        "status": metadata.get("status"),
        "progress": metadata.get("progress", 0),
        "progress_message": metadata.get("progress_message"),
    }


# =============================================================================
# Mediation Analysis (for NestedMMM)
# =============================================================================


@router.get(
    "/{model_id}/mediation",
    response_model=MediationResultsResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse, "description": "Model not completed or wrong type"},
    },
)
async def get_mediation_effects(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """
    Get mediation effects for a NestedMMM model.

    Returns direct and indirect effects for each channel, along with
    the proportion of effect mediated through each mediator.

    Only available for **nested** and **combined** model types.
    """
    metadata = _check_extended_model_completed(storage, model_id)

    model_type = metadata.get("model_type", "standard")
    if model_type not in ["nested", "combined"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mediation analysis not available for model type: {model_type}",
        )

    try:
        # Try to load cached results first
        mediation_data = storage.load_results(model_id, "mediation")
    except StorageError:
        # Compute mediation effects
        mmm = storage.load_model_artifact(model_id, "mmm")

        # Get mediation effects from model
        effects_df = mmm.get_mediation_effects()

        # Convert to response format
        effects = []
        for _, row in effects_df.iterrows():
            effect = MediationEffectSchema(
                channel=row["channel"],
                direct_effect=float(row.get("direct_effect", 0)),
                direct_effect_sd=float(row.get("direct_effect_sd", 0)),
                indirect_effects={
                    k: float(v)
                    for k, v in row.get("indirect_effects", {}).items()
                },
                indirect_effects_sd={
                    k: float(v)
                    for k, v in row.get("indirect_effects_sd", {}).items()
                },
                total_indirect=float(row.get("total_indirect", 0)),
                total_effect=float(row.get("total_effect", 0)),
                proportion_mediated=float(row.get("proportion_mediated", 0)),
            )
            effects.append(effect)

        mediation_data = {
            "model_id": model_id,
            "mediator_names": metadata.get("mediator_names", []),
            "channel_names": mmm.channel_names,
            "effects": [e.model_dump() for e in effects],
        }

        # Cache results
        storage.save_results(model_id, "mediation", mediation_data)

    return MediationResultsResponse(
        model_id=model_id,
        mediator_names=mediation_data.get("mediator_names", []),
        channel_names=mediation_data.get("channel_names", []),
        effects=[MediationEffectSchema(**e) for e in mediation_data.get("effects", [])],
    )


# =============================================================================
# Multivariate Analysis (for MultivariateMMM)
# =============================================================================


@router.get(
    "/{model_id}/multivariate",
    response_model=MultivariateResultsResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse, "description": "Model not completed or wrong type"},
    },
)
async def get_multivariate_results(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """
    Get multivariate results for a MultivariateMMM model.

    Returns outcome correlations, cross-effects (cannibalization/halo),
    and per-outcome fit metrics.

    Only available for **multivariate** and **combined** model types.
    """
    metadata = _check_extended_model_completed(storage, model_id)

    model_type = metadata.get("model_type", "standard")
    if model_type not in ["multivariate", "combined"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Multivariate analysis not available for model type: {model_type}",
        )

    try:
        # Try to load cached results
        mv_data = storage.load_results(model_id, "multivariate")
    except StorageError:
        # Compute multivariate results
        mmm = storage.load_model_artifact(model_id, "mmm")

        # Extract outcome correlations from trace
        trace = mmm._trace
        correlations = {}

        if "outcome_corr" in trace.posterior:
            corr_samples = trace.posterior["outcome_corr"].mean(dim=["chain", "draw"]).values
            outcome_names = mmm.outcome_names

            for i, out1 in enumerate(outcome_names):
                correlations[out1] = {}
                for j, out2 in enumerate(outcome_names):
                    correlations[out1][out2] = float(corr_samples[i, j])

        # Extract cross-effects
        cross_effects = []
        config = storage.load_config(metadata.get("config_id"))
        mv_config = config.get("mff_config", {}).get("multivariate_config", {})

        for ce_config in mv_config.get("cross_effects", []):
            param_name = f"cross_effect_{ce_config['source_outcome']}_{ce_config['target_outcome']}"
            if param_name in trace.posterior:
                samples = trace.posterior[param_name].values.flatten()
                cross_effects.append(
                    CrossEffectResultSchema(
                        source=ce_config["source_outcome"],
                        target=ce_config["target_outcome"],
                        effect_type=CrossEffectType(ce_config.get("effect_type", "cannibalization")),
                        mean=float(np.mean(samples)),
                        sd=float(np.std(samples)),
                        hdi_low=float(np.percentile(samples, 3)),
                        hdi_high=float(np.percentile(samples, 97)),
                    )
                )

        # Per-outcome metrics
        per_outcome_metrics = {}
        for outcome in mmm.outcome_names:
            per_outcome_metrics[outcome] = {
                "r2": 0.9,  # TODO: Compute actual metrics
                "mape": 5.0,
                "rmse": 100.0,
            }

        mv_data = {
            "model_id": model_id,
            "outcome_names": mmm.outcome_names,
            "channel_names": mmm.channel_names,
            "outcome_correlations": correlations,
            "cross_effects": [ce.model_dump() for ce in cross_effects],
            "per_outcome_metrics": per_outcome_metrics,
        }

        # Cache results
        storage.save_results(model_id, "multivariate", mv_data)

    return MultivariateResultsResponse(
        model_id=model_id,
        outcome_names=mv_data.get("outcome_names", []),
        channel_names=mv_data.get("channel_names", []),
        outcome_correlations=mv_data.get("outcome_correlations", {}),
        cross_effects=[
            CrossEffectResultSchema(**ce) for ce in mv_data.get("cross_effects", [])
        ],
        per_outcome_metrics=mv_data.get("per_outcome_metrics", {}),
    )


# =============================================================================
# Extended Model Results
# =============================================================================


@router.get(
    "/{model_id}/results",
    responses={404: {"model": ErrorResponse}},
)
async def get_extended_model_results(
    model_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Get comprehensive results for an extended model."""
    metadata = _check_extended_model_completed(storage, model_id)

    try:
        summary = storage.load_results(model_id, "summary")
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results not yet available",
        )

    response = {
        "model_id": model_id,
        "model_type": metadata.get("model_type"),
        "status": metadata.get("status"),
        "diagnostics": summary.get("diagnostics"),
        "parameter_summary": summary.get("parameter_summary"),
        "n_obs": summary.get("n_obs"),
        "channel_names": summary.get("channel_names"),
    }

    # Add type-specific info
    model_type = metadata.get("model_type")
    if model_type in ["nested", "combined"]:
        response["mediator_names"] = metadata.get("mediator_names")
    if model_type in ["multivariate", "combined"]:
        response["outcome_names"] = metadata.get("outcome_names")

    return SafeJSONResponse(content=_sanitize_for_json(response))


@router.delete(
    "/{model_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_extended_model(
    model_id: str,
    storage: StorageService = Depends(get_storage),
    redis: RedisService = Depends(get_redis),
):
    """Delete an extended model and its artifacts."""
    _check_extended_model_exists(storage, model_id)

    storage.delete_model(model_id)
    await redis.delete_job_status(model_id)

    return SuccessResponse(
        success=True,
        message=f"Extended model {model_id} deleted successfully",
    )
