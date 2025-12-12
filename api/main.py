"""
MMM Framework API - Main Application

FastAPI application for the Marketing Mix Model framework.
Provides endpoints for data management, configuration, model fitting, and analysis.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import Settings, get_settings
from redis_service import RedisService, get_redis
from routes import configs_router, data_router, models_router
from schemas import ErrorResponse, HealthResponse
from storage import get_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    settings.ensure_storage_dirs()
    
    # Initialize Redis connection
    redis = await get_redis()
    await redis.connect()
    
    yield
    
    # Shutdown
    await redis.disconnect()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    settings = settings or get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
# MMM Framework API

A comprehensive API for building, fitting, and analyzing Marketing Mix Models.

## Features

- **Data Management**: Upload and manage MFF (Master Flat File) format data
- **Configuration**: Create and manage model configurations
- **Model Fitting**: Async Bayesian model fitting with progress tracking
- **Analysis**: Counterfactual contributions, scenarios, and predictions
- **Export**: Download fitted models and results

## Workflow

1. **Upload Data**: POST to `/data/upload` with your MFF file
2. **Create Config**: POST to `/configs` with model specification
3. **Fit Model**: POST to `/models/fit` to start async fitting
4. **Track Progress**: GET `/models/{id}/status` to monitor fitting
5. **Get Results**: GET `/models/{id}/results` for fitted parameters
6. **Analyze**: Use `/models/{id}/contributions` and `/models/{id}/scenario`
        """,
        openapi_tags=[
            {
                "name": "Health",
                "description": "Health checks and system status",
            },
            {
                "name": "Data",
                "description": "Upload, list, and manage datasets",
            },
            {
                "name": "Configurations",
                "description": "Create and manage model configurations",
            },
            {
                "name": "Models",
                "description": "Fit models, track progress, and analyze results",
            },
        ],
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(data_router)
    app.include_router(configs_router)
    app.include_router(models_router)
    
    # Health endpoints
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
    )
    async def health_check():
        """
        Check API health status.
        
        Returns the status of the API, Redis connection, and worker availability.
        """
        redis = await get_redis()
        
        redis_ok = await redis.ping()
        worker_ok = await redis.check_worker_health()
        
        overall_status = "healthy" if redis_ok else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            version=settings.app_version,
            redis_connected=redis_ok,
            worker_active=worker_ok,
        )
    
    @app.get(
        "/health/detailed",
        tags=["Health"],
    )
    async def detailed_health():
        """Get detailed health information including queue stats."""
        redis = await get_redis()
        storage = get_storage()
        
        redis_ok = await redis.ping()
        queue_stats = await redis.get_queue_stats() if redis_ok else {}
        
        # Count stored items
        n_datasets = len(storage.list_data())
        n_configs = len(storage.list_configs())
        n_models = len(storage.list_models())
        
        return {
            "status": "healthy" if redis_ok else "unhealthy",
            "version": settings.app_version,
            "redis": {
                "connected": redis_ok,
                "url": settings.redis_url,
            },
            "queue": queue_stats,
            "storage": {
                "backend": settings.storage_backend,
                "datasets": n_datasets,
                "configs": n_configs,
                "models": n_models,
            },
        }
    
    @app.get("/", tags=["Health"])
    async def root():
        """API root endpoint."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail if isinstance(exc.detail, str) else "Request failed",
                "detail": exc.detail if isinstance(exc.detail, dict) else None,
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else None,
            },
        )
    
    return app


# Create default application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        # host=settings.host,
        # port=settings.port,
        # reload=settings.debug,
    )