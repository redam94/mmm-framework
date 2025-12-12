"""
Data management API routes.

Handles uploading, listing, and deleting datasets.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from config import Settings, get_settings
from schemas import (
    DataInfo,
    DataListResponse,
    DataUploadResponse,
    ErrorResponse,
    SuccessResponse,
)
from storage import StorageError, StorageService, get_storage

router = APIRouter(prefix="/data", tags=["Data"])


@router.post(
    "/upload",
    response_model=DataUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_data(
    file: UploadFile = File(..., description="MFF data file (CSV, Parquet, or Excel)"),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    """
    Upload a dataset in MFF format.
    
    Supported formats:
    - CSV (.csv)
    - Parquet (.parquet)
    - Excel (.xlsx, .xls)
    
    The file should be in Master Flat File (MFF) format with columns:
    - Period, Geography, Product, Campaign, Outlet, Creative
    - VariableName, VariableValue
    """
    # Check file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb} MB",
        )
    
    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )
    
    valid_extensions = (".csv", ".parquet", ".xlsx", ".xls")
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Supported formats: {valid_extensions}",
        )
    
    try:
        metadata = storage.save_data(content, file.filename)
        
        return DataUploadResponse(
            data_id=metadata["data_id"],
            filename=metadata["filename"],
            rows=metadata["rows"],
            columns=metadata["columns"],
            variables=metadata["variables"],
            dimensions=metadata["dimensions"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            size_bytes=metadata["size_bytes"],
        )
    
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}",
        )


@router.get(
    "",
    response_model=DataListResponse,
)
async def list_data(
    storage: StorageService = Depends(get_storage),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum items to return"),
):
    """List all uploaded datasets."""
    datasets = storage.list_data()
    
    # Apply pagination
    total = len(datasets)
    datasets = datasets[skip : skip + limit]
    
    return DataListResponse(
        datasets=[
            DataInfo(
                data_id=d["data_id"],
                filename=d["filename"],
                rows=d["rows"],
                columns=d["columns"],
                variables=d["variables"],
                dimensions=d["dimensions"],
                created_at=datetime.fromisoformat(d["created_at"]),
                size_bytes=d["size_bytes"],
            )
            for d in datasets
        ],
        total=total,
    )


@router.get(
    "/{data_id}",
    response_model=DataInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_data(
    data_id: str,
    include_preview: bool = Query(False, description="Include data preview"),
    preview_rows: int = Query(10, ge=1, le=100, description="Number of preview rows"),
    storage: StorageService = Depends(get_storage),
):
    """Get information about a specific dataset."""
    try:
        metadata = storage.get_data_info(data_id)
        
        preview = None
        if include_preview:
            df = storage.load_data(data_id)
            preview = df.head(preview_rows).to_dict(orient="records")
        
        return DataInfo(
            data_id=metadata["data_id"],
            filename=metadata["filename"],
            rows=metadata["rows"],
            columns=metadata["columns"],
            variables=metadata["variables"],
            dimensions=metadata["dimensions"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            size_bytes=metadata["size_bytes"],
            preview=preview,
        )
    
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {data_id}",
        )


@router.delete(
    "/{data_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_data(
    data_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Delete a dataset."""
    if not storage.data_exists(data_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {data_id}",
        )
    
    storage.delete_data(data_id)
    
    return SuccessResponse(
        success=True,
        message=f"Dataset {data_id} deleted successfully",
    )


@router.get(
    "/{data_id}/variables",
    responses={404: {"model": ErrorResponse}},
)
async def get_data_variables(
    data_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Get variable names and summary statistics from a dataset."""
    try:
        df = storage.load_data(data_id)
        
        if "VariableName" not in df.columns or "VariableValue" not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset is not in MFF format (missing VariableName/VariableValue columns)",
            )
        
        # Get variable summary
        summary = df.groupby("VariableName")["VariableValue"].agg([
            "count", "mean", "std", "min", "max"
        ]).reset_index()
        
        return {
            "data_id": data_id,
            "variables": summary.to_dict(orient="records"),
        }
    
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {data_id}",
        )