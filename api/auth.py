"""
Authentication module for the MMM API.

Provides API key authentication that can be enabled/disabled via configuration.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config import get_settings

# API key header - auto_error=False to allow optional authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """
    Verify the API key if authentication is enabled.

    Parameters
    ----------
    api_key : str | None
        The API key from the X-API-Key header.

    Returns
    -------
    str | None
        The validated API key, or None if auth is disabled.

    Raises
    ------
    HTTPException
        If authentication is enabled and the API key is invalid or missing.
    """
    settings = get_settings()

    # If auth is disabled, skip validation
    if not settings.api_keys_enabled:
        return None

    # Auth is enabled - require valid key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key not in settings.valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


async def optional_api_key(
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """
    Optional API key verification - doesn't fail if auth is enabled but no key provided.
    Useful for endpoints that should work both with and without authentication.

    Parameters
    ----------
    api_key : str | None
        The API key from the X-API-Key header.

    Returns
    -------
    str | None
        The API key if valid, or None.
    """
    settings = get_settings()

    if not settings.api_keys_enabled:
        return None

    if api_key and api_key in settings.valid_api_keys:
        return api_key

    return None
