"""FastAPI dependencies for resolving the principal and enforcing access.

Importing this module pulls in ``fastapi`` — only the API apps should import it,
not the modeling core (so ``import mmm_framework`` stays fastapi-free).

Usage::

    from mmm_framework.auth.deps import get_current_principal, require_project_access

    @router.get("/projects/{project_id}/experiments")
    async def list_experiments(
        project_id: str,
        principal = Depends(require_project_access(Role.VIEWER)),
    ): ...

When ``MMM_AUTH_ENABLED`` is false, ``get_current_principal`` returns a
single-tenant dev principal so existing flows keep working without a token.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from fastapi import Depends, Header, HTTPException, status

from . import store
from .config import AuthSettings, get_auth_settings
from .models import AuthContext, Role
from .tokens import AuthError, TokenVerifier, build_verifier

_verifier_cache: dict[str, TokenVerifier] = {}


def _get_verifier(settings: AuthSettings) -> TokenVerifier:
    import hashlib

    # Include a secret fingerprint so a rotated MMM_AUTH_SECRET (or a different
    # secret per test) rebuilds the verifier instead of reusing a stale one.
    fp = hashlib.sha256(
        f"{settings.secret}|{settings.oidc_jwks_url}".encode()
    ).hexdigest()[:12]
    key = f"{settings.provider}:{settings.issuer}:{settings.audience}:{fp}"
    v = _verifier_cache.get(key)
    if v is None:
        v = build_verifier(settings)
        _verifier_cache[key] = v
    return v


def _dev_principal(settings: AuthSettings) -> AuthContext:
    return AuthContext(
        user_id=settings.dev_user_id,
        org_id=settings.dev_org_id,
        email=settings.dev_email,
        org_role=Role.OWNER,
        is_dev=True,
    )


async def get_current_principal(
    authorization: str | None = Header(default=None),
    settings: AuthSettings = Depends(get_auth_settings),
) -> AuthContext:
    """Resolve the verified principal from the ``Authorization: Bearer`` header."""
    if not settings.enabled:
        return _dev_principal(settings)

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = _get_verifier(settings).verify(token)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if claims.get("typ") not in (None, "access"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not an access token",
        )
    # Instant revocation: reject any token minted before the user's current
    # token version (bumped on deactivate / password reset / sign-out-everywhere).
    # The version read is a blocking sqlite call on the request hot path, so it's
    # offloaded to a worker thread to avoid stalling the event loop.
    sub = claims.get("sub")
    current_tv = (
        await asyncio.to_thread(store.get_token_version, sub)
        if sub is not None
        else None
    )
    if sub is None or int(claims.get("tv", 0)) != current_tv:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token superseded",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return AuthContext(
            user_id=claims["sub"],
            org_id=claims["org"],
            email=claims.get("email", ""),
            org_role=claims.get("role", Role.VIEWER.value),
            token_id=claims.get("jti"),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token claims",
        ) from exc


def require_org_role(min_role: Role) -> Callable:
    """Dependency factory: principal must hold at least ``min_role`` in its org."""

    async def _dep(
        principal: AuthContext = Depends(get_current_principal),
    ) -> AuthContext:
        if not principal.has_role(min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {min_role.value} role",
            )
        return principal

    return _dep


def ensure_project_access(
    principal: AuthContext, project_id: str | None, min_role: Role = Role.VIEWER
) -> None:
    """Raise unless ``principal``'s org owns ``project_id`` at ``min_role``.

    Cross-tenant / unknown / unattributable projects all raise **404** (not 403)
    so one org cannot probe another's project existence. Dev principal (auth
    disabled) is a no-op. Use this inline for routes keyed by a non-project id
    (e.g. an experiment) after resolving the owning ``project_id``.
    """
    if principal.is_dev:
        return
    owner_org = store.get_project_org(project_id) if project_id else None
    if not owner_org or owner_org != principal.org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not principal.has_role(min_role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires {min_role.value} role",
        )


def require_project_access(min_role: Role = Role.VIEWER) -> Callable:
    """Dependency factory: principal's org must own ``project_id`` (path param).

    Mount via the route decorator's ``dependencies=[...]`` so the handler
    signature stays untouched::

        @app.get("/projects/{project_id}/x", dependencies=[Depends(require_project_access())])
    """

    async def _dep(
        project_id: str,
        principal: AuthContext = Depends(get_current_principal),
    ) -> AuthContext:
        ensure_project_access(principal, project_id, min_role)
        return principal

    return _dep


def require_plan_feature(feature: str) -> Callable:
    """Dependency factory: 402 unless the principal's org plan includes ``feature``.

    Dev principal (auth off) is a no-op. See ``auth.plans.FEATURES``.
    """

    async def _dep(
        principal: AuthContext = Depends(get_current_principal),
    ) -> AuthContext:
        if not principal.is_dev:
            from .plans import entitlements_for_org

            if not entitlements_for_org(principal.org_id).has(feature):
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=f"Your plan does not include '{feature}'. Upgrade to enable it.",
                )
        return principal

    return _dep
