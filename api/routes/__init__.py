"""
API Routes package.
"""

from .budget_plans import router as budget_plans_router
from .configs import router as configs_router
from .data import router as data_router
from .extended_models import router as extended_models_router
from .models import router as models_router
from .projects import router as projects_router
from .sessions import router as sessions_router
from .sessions import analysis_plans_router
from .templates import router as templates_router

__all__ = [
    "budget_plans_router",
    "data_router",
    "configs_router",
    "models_router",
    "extended_models_router",
    "projects_router",
    "sessions_router",
    "analysis_plans_router",
    "templates_router",
]
