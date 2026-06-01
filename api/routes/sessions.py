"""
Session management API routes.

Exposes the SQLite-backed session store from mmm_framework.api.sessions
as a REST API for the frontend session manager.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from auth import verify_api_key
from mmm_framework.api import sessions as session_store
from schemas import ErrorResponse, SuccessResponse

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class SessionInfo(BaseModel):
    thread_id: str
    name: str
    created_at: float
    updated_at: float
    project_id: str | None = None
    artifact_count: int = 0


class SessionCreateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    project_id: str | None = None


class SessionUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    project_id: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]
    total: int


class SessionDetailResponse(BaseModel):
    thread_id: str
    name: str
    created_at: float
    updated_at: float
    project_id: str | None = None
    artifact_count: int = 0
    artifacts: list[dict[str, Any]] = []
    assumptions: list[dict[str, Any]] = []
    workflow_steps: dict[int, dict[str, Any]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_session_info(row: dict[str, Any]) -> SessionInfo:
    return SessionInfo(
        thread_id=row["thread_id"],
        name=row["name"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        project_id=row.get("project_id"),
        artifact_count=row.get("artifact_count", 0),
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.on_event("startup")
async def _init():
    session_store.init_db()


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    project_id: str | None = Query(None, description="Filter sessions by project"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all agent chat sessions, optionally filtered by project."""
    session_store.init_db()
    rows = session_store.list_sessions(project_id=project_id)
    # Enrich with artifact_count for each session
    enriched = []
    for row in rows:
        detail = session_store.get_session(row["thread_id"])
        if detail:
            enriched.append(detail)
        else:
            enriched.append(row)
    total = len(enriched)
    page = enriched[skip: skip + limit]
    return SessionListResponse(
        sessions=[_to_session_info(r) for r in page],
        total=total,
    )


@router.post("", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
async def create_session(request: SessionCreateRequest):
    """Create a new agent session."""
    session_store.init_db()
    row = session_store.create_session(
        name=request.name,
        project_id=request.project_id,
    )
    return _to_session_info(row)


@router.get(
    "/{thread_id}",
    response_model=SessionDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_session(thread_id: str):
    """Get a session with its artifacts, assumptions, and workflow status."""
    session_store.init_db()
    session = session_store.get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {thread_id}")
    artifacts = session_store.list_artifacts(thread_id)
    assumptions = session_store.list_assumptions(thread_id)
    workflow = session_store.get_workflow_overrides(thread_id)
    return SessionDetailResponse(
        thread_id=session["thread_id"],
        name=session["name"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        project_id=session.get("project_id"),
        artifact_count=session.get("artifact_count", 0),
        artifacts=artifacts,
        assumptions=assumptions,
        workflow_steps=workflow,
    )


@router.patch(
    "/{thread_id}",
    response_model=SessionInfo,
    responses={404: {"model": ErrorResponse}},
)
async def update_session(thread_id: str, request: SessionUpdateRequest):
    """Rename a session or reassign it to a project."""
    session_store.init_db()
    updated = session_store.update_session(
        thread_id=thread_id,
        name=request.name,
        project_id=request.project_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Session not found: {thread_id}")
    session = session_store.get_session(thread_id)
    return _to_session_info(session)  # type: ignore[arg-type]


@router.delete(
    "/{thread_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_session(thread_id: str):
    """Delete a session and its artifacts."""
    session_store.init_db()
    deleted = session_store.delete_session(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {thread_id}")
    return SuccessResponse(message=f"Session {thread_id} deleted")


# ── Analysis Plans ─────────────────────────────────────────────────────────────


class AnalysisPlanInfo(BaseModel):
    id: str
    thread_id: str
    name: str
    locked_at: float
    payload: dict[str, Any] = {}


class AnalysisPlanListResponse(BaseModel):
    plans: list[AnalysisPlanInfo]
    total: int


class LockPlanRequest(BaseModel):
    thread_id: str
    name: str = Field(default="Analysis Plan", max_length=200)
    dag: dict[str, Any] | None = None
    research_question: dict[str, Any] | None = None
    assumptions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


analysis_plans_router = APIRouter(prefix="/analysis-plans", tags=["Analysis Plans"])


@analysis_plans_router.post(
    "",
    response_model=AnalysisPlanInfo,
    status_code=status.HTTP_201_CREATED,
)
async def lock_analysis_plan(request: LockPlanRequest):
    """Lock the current DAG + research question + assumptions into a named analysis plan."""
    session_store.init_db()
    payload: dict[str, Any] = {}
    if request.dag is not None:
        payload["dag"] = request.dag
    if request.research_question is not None:
        payload["research_question"] = request.research_question
    if request.assumptions is not None:
        payload["assumptions"] = request.assumptions
    if request.extra:
        payload.update(request.extra)

    plan = session_store.lock_analysis_plan(
        thread_id=request.thread_id,
        name=request.name,
        payload=payload,
    )
    return AnalysisPlanInfo(**plan)


@analysis_plans_router.get("", response_model=AnalysisPlanListResponse)
async def list_analysis_plans(
    thread_id: str | None = Query(None, description="Filter by session thread ID"),
    limit: int = Query(20, ge=1, le=100),
):
    """List locked analysis plans, optionally filtered by session."""
    session_store.init_db()
    plans = session_store.list_analysis_plans(thread_id=thread_id)
    page = plans[:limit]
    return AnalysisPlanListResponse(
        plans=[AnalysisPlanInfo(**p) for p in page],
        total=len(plans),
    )


@analysis_plans_router.get(
    "/{plan_id}",
    response_model=AnalysisPlanInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_analysis_plan(plan_id: str):
    """Get a single analysis plan by ID."""
    session_store.init_db()
    plan = session_store.get_analysis_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Plan not found: {plan_id}")
    return AnalysisPlanInfo(**plan)


@analysis_plans_router.delete(
    "/{plan_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_analysis_plan(plan_id: str):
    """Delete a locked analysis plan."""
    session_store.init_db()
    if not session_store.delete_analysis_plan(plan_id):
        raise HTTPException(status_code=404, detail=f"Plan not found: {plan_id}")
    return SuccessResponse(message=f"Plan {plan_id} deleted")
