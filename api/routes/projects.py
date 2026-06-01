"""
Project management API routes.

Handles creating, updating, listing, and deleting projects that group
data, configs, models, and sessions.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from schemas import (
    ErrorResponse,
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdateRequest,
    SuccessResponse,
)
from storage import StorageError, StorageService, get_storage

router = APIRouter(prefix="/projects", tags=["Projects"])


def _project_to_response(
    project: dict,
    storage: StorageService,
    include_counts: bool = False,
) -> ProjectResponse:
    counts = {"data_count": 0, "config_count": 0, "model_count": 0, "session_count": 0}
    if include_counts:
        resource_counts = storage.count_by_project(project["project_id"])
        counts.update(resource_counts)
        # session_count populated by sessions route if available; default 0 here
    return ProjectResponse(
        project_id=project["project_id"],
        name=project["name"],
        description=project.get("description"),
        created_at=datetime.fromisoformat(project["created_at"]),
        updated_at=datetime.fromisoformat(project["updated_at"]),
        **counts,
    )


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}},
)
async def create_project(
    request: ProjectCreateRequest,
    storage: StorageService = Depends(get_storage),
):
    """Create a new project."""
    project = storage.save_project(
        name=request.name,
        description=request.description,
    )
    return _project_to_response(project, storage)


@router.get(
    "",
    response_model=ProjectListResponse,
)
async def list_projects(
    storage: StorageService = Depends(get_storage),
):
    """List all projects with resource counts."""
    projects = storage.list_projects()
    items = [_project_to_response(p, storage, include_counts=True) for p in projects]
    return ProjectListResponse(projects=items, total=len(items))


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_project(
    project_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Get a project by ID with resource counts."""
    try:
        project = storage.get_project_metadata(project_id)
    except StorageError:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return _project_to_response(project, storage, include_counts=True)


@router.patch(
    "/{project_id}",
    response_model=ProjectResponse,
    responses={404: {"model": ErrorResponse}},
)
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    storage: StorageService = Depends(get_storage),
):
    """Update project name or description."""
    if not storage.project_exists(project_id):
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    updates = request.model_dump(exclude_none=True)
    project = storage.update_project_metadata(project_id, updates)
    return _project_to_response(project, storage, include_counts=True)


@router.delete(
    "/{project_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_project(
    project_id: str,
    storage: StorageService = Depends(get_storage),
):
    """
    Delete a project record.

    Member resources (data, configs, models) are NOT deleted — they become
    unassigned. To delete them, use the respective resource endpoints.
    """
    if not storage.delete_project(project_id):
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return SuccessResponse(message=f"Project {project_id} deleted")
