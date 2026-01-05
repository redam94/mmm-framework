"""
API Routes package.
"""

from .configs import router as configs_router
from .data import router as data_router
from .models import router as models_router

__all__ = ["data_router", "configs_router", "models_router"]
