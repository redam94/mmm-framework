"""
MMM Framework API

FastAPI backend for Marketing Mix Model fitting and analysis.
"""

from .main import app, create_app
from .schemas import JobStatus

__all__ = ["app", "create_app"]
