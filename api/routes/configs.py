"""
Configuration management API routes.

Handles creating, updating, and deleting model configurations.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from schemas import (
    ConfigCreateRequest,
    ConfigListResponse,
    ConfigResponse,
    ConfigUpdateRequest,
    ErrorResponse,
    SuccessResponse,
)
from storage import StorageError, StorageService, get_storage

router = APIRouter(prefix="/configs", tags=["Configurations"])


def _config_to_response(config: dict) -> ConfigResponse:
    """Convert stored config dict to response model."""
    return ConfigResponse(
        config_id=config["config_id"],
        name=config["name"],
        description=config.get("description"),
        mff_config=config["mff_config"],
        model_settings=config["model_settings"],
        created_at=datetime.fromisoformat(config["created_at"]),
        updated_at=datetime.fromisoformat(config["updated_at"]),
    )


@router.post(
    "",
    response_model=ConfigResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_config(
    request: ConfigCreateRequest,
    storage: StorageService = Depends(get_storage),
):
    """
    Create a new model configuration.
    
    The configuration includes:
    - **mff_config**: Data specification (KPI, media channels, controls, alignment)
    - **model_settings**: Model fitting settings (inference method, MCMC params, trend, seasonality)
    """
    config_data = {
        "name": request.name,
        "description": request.description,
        "mff_config": request.mff_config.model_dump(),
        "model_settings": request.model_settings.model_dump(),
    }
    
    saved = storage.save_config(config_data)
    return _config_to_response(saved)


@router.get(
    "",
    response_model=ConfigListResponse,
)
async def list_configs(
    storage: StorageService = Depends(get_storage),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List all configurations."""
    configs = storage.list_configs()
    total = len(configs)
    configs = configs[skip : skip + limit]
    
    return ConfigListResponse(
        configs=[_config_to_response(c) for c in configs],
        total=total,
    )


@router.get(
    "/{config_id}",
    response_model=ConfigResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_config(
    config_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Get a specific configuration."""
    try:
        config = storage.load_config(config_id)
        return _config_to_response(config)
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration not found: {config_id}",
        )


@router.put(
    "/{config_id}",
    response_model=ConfigResponse,
    responses={404: {"model": ErrorResponse}},
)
async def update_config(
    config_id: str,
    request: ConfigUpdateRequest,
    storage: StorageService = Depends(get_storage),
):
    """Update an existing configuration."""
    if not storage.config_exists(config_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration not found: {config_id}",
        )
    
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.description is not None:
        updates["description"] = request.description
    if request.mff_config is not None:
        updates["mff_config"] = request.mff_config.model_dump()
    if request.model_settings is not None:
        updates["model_settings"] = request.model_settings.model_dump()
    
    updated = storage.update_config(config_id, updates)
    return _config_to_response(updated)


@router.delete(
    "/{config_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_config(
    config_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Delete a configuration."""
    if not storage.config_exists(config_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration not found: {config_id}",
        )
    
    storage.delete_config(config_id)
    
    return SuccessResponse(
        success=True,
        message=f"Configuration {config_id} deleted successfully",
    )


@router.post(
    "/{config_id}/duplicate",
    response_model=ConfigResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}},
)
async def duplicate_config(
    config_id: str,
    new_name: str = Query(..., min_length=1, max_length=100),
    storage: StorageService = Depends(get_storage),
):
    """Create a copy of an existing configuration."""
    try:
        original = storage.load_config(config_id)
        
        # Create new config with new name
        new_config = {
            "name": new_name,
            "description": f"Copy of {original.get('name', config_id)}",
            "mff_config": original["mff_config"],
            "model_settings": original["model_settings"],
        }
        
        saved = storage.save_config(new_config)
        return _config_to_response(saved)
        
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration not found: {config_id}",
        )


@router.post(
    "/validate",
    responses={400: {"model": ErrorResponse}},
)
async def validate_config(
    request: ConfigCreateRequest,
):
    """
    Validate a configuration without saving it.
    
    Checks that the configuration is well-formed and compatible.
    """
    # The Pydantic validation already checks most things
    # Here we can add additional semantic validation
    
    errors = []
    
    # Check KPI name is not empty
    if not request.mff_config.kpi.name:
        errors.append("KPI name cannot be empty")
    
    # Check at least one media channel
    if not request.mff_config.media_channels:
        errors.append("At least one media channel is required")
    
    # Check for duplicate channel names
    channel_names = [ch.name for ch in request.mff_config.media_channels]
    if len(channel_names) != len(set(channel_names)):
        errors.append("Duplicate media channel names detected")
    
    # Check control names don't overlap with channels
    control_names = [c.name for c in request.mff_config.controls]
    overlap = set(channel_names) & set(control_names)
    if overlap:
        errors.append(f"Variable names cannot be both media and control: {overlap}")
    
    # Check model settings
    if request.model_settings.n_draws < request.model_settings.n_tune:
        errors.append("n_draws should typically be >= n_tune for reliable inference")
    
    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": errors},
        )
    
    return {
        "valid": True,
        "message": "Configuration is valid",
        "summary": {
            "kpi": request.mff_config.kpi.name,
            "n_media_channels": len(request.mff_config.media_channels),
            "n_controls": len(request.mff_config.controls),
            "inference_method": request.model_settings.inference_method,
            "n_chains": request.model_settings.n_chains,
            "n_draws": request.model_settings.n_draws,
        },
    }