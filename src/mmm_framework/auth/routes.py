"""Auth API router: signup, login, refresh, me.

Mount in any FastAPI app::

    from mmm_framework.auth.routes import create_auth_router
    app.include_router(create_auth_router())
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from . import service, store
from .config import AuthSettings, get_auth_settings
from .deps import get_current_principal, require_org_role
from .ratelimit import require_ip_rate_limit
from .models import (
    AcceptInviteRequest,
    AuthContext,
    InviteOut,
    InviteRequest,
    LoginRequest,
    LogoutRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    RefreshRequest,
    Role,
    SignupRequest,
    TokenResponse,
    UserOut,
)


def create_auth_router() -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["Auth"])

    # IP-based throttle for the UNAUTHENTICATED routes (brute-force /
    # credential-stuffing / token-grinding guard the per-org limiter can't cover).
    _ip_auth = Depends(require_ip_rate_limit("auth"))

    @router.post("/signup", response_model=TokenResponse, dependencies=[_ip_auth])
    async def signup(
        body: SignupRequest,
        settings: AuthSettings = Depends(get_auth_settings),
    ) -> TokenResponse:
        store.init_auth_schema()
        try:
            _user, tokens = service.signup_organization(
                organization=body.organization,
                email=body.email,
                password=body.password,
                name=body.name,
                settings=settings,
            )
        except service.AuthServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        return tokens

    @router.post("/login", response_model=TokenResponse, dependencies=[_ip_auth])
    async def login(
        body: LoginRequest,
        settings: AuthSettings = Depends(get_auth_settings),
    ) -> TokenResponse:
        try:
            user = service.authenticate(body.email, body.password)
        except service.AuthServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
            )
        role = store.get_org_role(user.get("org_id", ""), user["user_id"]) or user.get(
            "role", "viewer"
        )
        return service.issue_tokens(
            user_id=user["user_id"],
            org_id=user.get("org_id", ""),
            email=user["email"],
            org_role=role,
            token_version=int(user.get("token_version") or 0),
            settings=settings,
        )

    @router.post("/refresh", response_model=TokenResponse, dependencies=[_ip_auth])
    async def refresh(
        body: RefreshRequest,
        settings: AuthSettings = Depends(get_auth_settings),
    ) -> TokenResponse:
        try:
            return service.refresh_tokens(body.refresh_token, settings=settings)
        except service.AuthServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
            )

    @router.get("/me", response_model=UserOut)
    async def me(
        principal: AuthContext = Depends(get_current_principal),
    ) -> UserOut:
        return UserOut(
            user_id=principal.user_id,
            email=principal.email,
            org_id=principal.org_id,
            org_role=principal.org_role,
        )

    @router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
    async def logout(
        body: LogoutRequest,
        settings: AuthSettings = Depends(get_auth_settings),
    ) -> None:
        # Idempotent; revokes the refresh token so it can't be replayed.
        service.logout(body.refresh_token, settings=settings)

    @router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
    async def logout_all(
        principal: AuthContext = Depends(get_current_principal),
    ) -> None:
        # "Sign out everywhere": bumps the token version so every outstanding
        # access AND refresh token for this user is rejected on next use.
        service.revoke_all_sessions(principal.user_id)

    @router.post("/invite", response_model=InviteOut)
    async def invite(
        body: InviteRequest,
        settings: AuthSettings = Depends(get_auth_settings),
        principal: AuthContext = Depends(require_org_role(Role.ADMIN)),
    ) -> InviteOut:
        try:
            inv = service.create_invite(
                org_id=principal.org_id,
                email=body.email,
                role=body.role.value,
                invited_by=principal.user_id,
                settings=settings,
            )
        except service.AuthServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        return InviteOut(**inv)

    @router.post("/accept-invite", response_model=TokenResponse)
    async def accept_invite(
        body: AcceptInviteRequest,
        settings: AuthSettings = Depends(get_auth_settings),
    ) -> TokenResponse:
        try:
            _user, tokens = service.accept_invite(
                token=body.token,
                password=body.password,
                name=body.name,
                settings=settings,
            )
        except service.AuthServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        return tokens

    @router.post(
        "/password-reset/request",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[_ip_auth],
    )
    async def password_reset_request(
        body: PasswordResetRequest,
        settings: AuthSettings = Depends(get_auth_settings),
    ) -> dict:
        # Always 202 (never reveal whether the email exists). The token is
        # delivered out-of-band (email); it is intentionally NOT returned here.
        service.request_password_reset(body.email, settings=settings)
        return {"status": "if the account exists, a reset link has been sent"}

    @router.post("/password-reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
    async def password_reset_confirm(
        body: PasswordResetConfirm,
        settings: AuthSettings = Depends(get_auth_settings),
    ) -> None:
        try:
            service.confirm_password_reset(
                token=body.token, new_password=body.new_password, settings=settings
            )
        except service.AuthServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )

    @router.post("/users/{user_id}/deactivate", status_code=status.HTTP_204_NO_CONTENT)
    async def deactivate_user(
        user_id: str,
        principal: AuthContext = Depends(require_org_role(Role.ADMIN)),
    ) -> None:
        # Admins may only deactivate users within their own org.
        target = store.get_user(user_id)
        if target is None or target.get("org_id") != principal.org_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
            )
        service.deactivate_user(user_id, actor=principal.user_id)

    return router
