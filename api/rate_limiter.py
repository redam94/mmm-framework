"""
Rate limiting module for the MMM API.

Uses slowapi for rate limiting with Redis backend when available.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from fastapi import Request
from fastapi.responses import JSONResponse

from config import get_settings


def get_rate_limit_key(request: Request) -> str:
    """
    Get the key for rate limiting.

    Uses API key if available, otherwise falls back to IP address.

    Parameters
    ----------
    request : Request
        The incoming request.

    Returns
    -------
    str
        The rate limit key.
    """
    # Try to get API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"

    # Fall back to IP address
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(key_func=get_rate_limit_key)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Handle rate limit exceeded errors.

    Parameters
    ----------
    request : Request
        The incoming request.
    exc : RateLimitExceeded
        The rate limit exception.

    Returns
    -------
    JSONResponse
        Error response with 429 status.
    """
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": "Rate limit exceeded",
            "detail": str(exc.detail),
        },
        headers={"Retry-After": str(exc.retry_after) if exc.retry_after else "60"},
    )


def get_rate_limit_string() -> str:
    """
    Get the rate limit string from settings.

    Returns
    -------
    str
        Rate limit string (e.g., "100/minute").
    """
    settings = get_settings()
    return f"{settings.rate_limit_requests}/{settings.rate_limit_period}"
