"""
Middleware components for the MMM API.

Includes request logging and performance monitoring.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.

    Logs request method, path, response status code, and duration.
    Adds X-Request-ID header to all responses for tracing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Store request ID in state for access in route handlers
        request.state.request_id = request_id

        # Get client IP (handle proxies)
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        # Log incoming request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} " f"from {client_ip}"
        )

        # Process request and measure time
        start_time = time.time()

        try:
            response = await call_next(request)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"failed after {duration:.3f}s: {str(e)}"
            )
            raise

        duration = time.time() - start_time

        # Log response
        log_level = "info" if response.status_code < 400 else "warning"
        getattr(logger, log_level)(
            f"[{request_id}] {request.method} {request.url.path} "
            f"-> {response.status_code} in {duration:.3f}s"
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for consistent error handling and formatting.

    Catches unhandled exceptions and formats them consistently.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error handling."""
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.exception(f"[{request_id}] Unhandled exception: {str(e)}")
            raise
