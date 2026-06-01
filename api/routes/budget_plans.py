"""
Budget plan API routes.

Handles saving, listing, retrieving, and deleting named budget plans
(scenario runs with spend allocations persisted for later reference).
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from schemas import (
    BudgetPlanCreateRequest,
    BudgetPlanInfo,
    BudgetPlanListResponse,
    ErrorResponse,
    SuccessResponse,
)
from storage import StorageError, StorageService, get_storage

router = APIRouter(prefix="/budget-plans", tags=["Budget Plans"])


def _plan_to_info(plan: dict) -> BudgetPlanInfo:
    return BudgetPlanInfo(
        plan_id=plan["plan_id"],
        name=plan["name"],
        description=plan.get("description"),
        model_id=plan["model_id"],
        spend_changes=plan.get("spend_changes", {}),
        baseline_outcome=float(plan["baseline_outcome"]),
        scenario_outcome=float(plan["scenario_outcome"]),
        outcome_change=float(plan["outcome_change"]),
        outcome_change_pct=float(plan["outcome_change_pct"]),
        channel_details=plan.get("channel_details", {}),
        created_at=datetime.fromisoformat(plan["created_at"]),
        project_id=plan.get("project_id"),
    )


@router.post(
    "",
    response_model=BudgetPlanInfo,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def create_budget_plan(
    request: BudgetPlanCreateRequest,
    storage: StorageService = Depends(get_storage),
):
    """
    Run a scenario and save the result as a named budget plan.

    Loads the specified model, runs what-if scenario analysis with the
    provided spend changes, and persists the result for later retrieval.
    """
    if not storage.model_exists(request.model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {request.model_id}",
        )

    try:
        mmm = storage.load_model_artifact(request.model_id, "mmm")

        scenario_results = mmm.what_if_scenario(
            spend_changes=request.spend_changes,
            time_period=tuple(request.time_period) if request.time_period else None,
            random_seed=42,
        )

        plan = storage.save_budget_plan(
            name=request.name,
            model_id=request.model_id,
            spend_changes=request.spend_changes,
            baseline_outcome=float(scenario_results["baseline_outcome"]),
            scenario_outcome=float(scenario_results["scenario_outcome"]),
            outcome_change=float(scenario_results["outcome_change"]),
            outcome_change_pct=float(scenario_results["outcome_change_pct"]),
            channel_details=scenario_results.get("spend_changes", {}),
            description=request.description,
            project_id=request.project_id,
        )

        return _plan_to_info(plan)

    except StorageError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating budget plan: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running scenario: {str(e)}",
        )


@router.get(
    "",
    response_model=BudgetPlanListResponse,
)
async def list_budget_plans(
    model_id: str | None = Query(None),
    project_id: str | None = Query(None),
    storage: StorageService = Depends(get_storage),
):
    """List saved budget plans, optionally filtered by model or project."""
    plans = storage.list_budget_plans(model_id=model_id, project_id=project_id)
    return BudgetPlanListResponse(
        plans=[_plan_to_info(p) for p in plans],
        total=len(plans),
    )


@router.get(
    "/{plan_id}",
    response_model=BudgetPlanInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_budget_plan(
    plan_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Get a saved budget plan by ID."""
    try:
        plan = storage.get_budget_plan(plan_id)
        return _plan_to_info(plan)
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget plan not found: {plan_id}",
        )


@router.delete(
    "/{plan_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_budget_plan(
    plan_id: str,
    storage: StorageService = Depends(get_storage),
):
    """Delete a saved budget plan."""
    if not storage.budget_plan_exists(plan_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget plan not found: {plan_id}",
        )
    storage.delete_budget_plan(plan_id)
    return SuccessResponse(message=f"Budget plan {plan_id} deleted")
