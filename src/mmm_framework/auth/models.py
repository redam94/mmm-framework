"""Auth domain models: roles, the resolved principal, and request/response DTOs."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    """Lower-case + strip an email; raise ``ValueError`` if it isn't one."""
    e = (email or "").strip().lower()
    if not _EMAIL_RE.match(e):
        raise ValueError("invalid email address")
    return e


class Role(str, Enum):
    """Org/project roles, ordered least → most privileged."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    OWNER = "owner"


_ROLE_ORDER = {Role.VIEWER: 0, Role.ANALYST: 1, Role.ADMIN: 2, Role.OWNER: 3}


def role_rank(role: str | Role) -> int:
    """Numeric rank for a role string/enum; unknown roles rank lowest."""
    try:
        return _ROLE_ORDER[Role(role)]
    except (ValueError, KeyError):
        return -1


def role_satisfies(have: str | Role, need: str | Role) -> bool:
    """True if ``have`` is at least as privileged as ``need``."""
    return role_rank(have) >= role_rank(need)


class AuthContext(BaseModel):
    """The verified principal attached to a request."""

    user_id: str
    org_id: str
    email: str
    org_role: Role = Role.VIEWER
    token_id: str | None = None
    is_dev: bool = False

    def has_role(self, need: Role) -> bool:
        return role_satisfies(self.org_role, need)


# ----- request / response DTOs ------------------------------------------------


class SignupRequest(BaseModel):
    organization: str = Field(min_length=1, max_length=120)
    email: str
    password: str
    name: str | None = None

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return normalize_email(v)


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return normalize_email(v)


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    user_id: str
    email: str
    name: str | None = None
    org_id: str
    org_role: Role


class OrgOut(BaseModel):
    org_id: str
    name: str
    slug: str
    plan: str


# ----- lifecycle DTOs (Phase 1.4) ---------------------------------------------


class LogoutRequest(BaseModel):
    refresh_token: str


class InviteRequest(BaseModel):
    email: str
    role: Role = Role.ANALYST

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return normalize_email(v)


class InviteOut(BaseModel):
    token: str
    email: str
    role: Role
    org_id: str
    expires_at: float


class RoleUpdate(BaseModel):
    role: Role


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class AcceptInviteRequest(BaseModel):
    token: str
    password: str
    name: str | None = None


class PasswordResetRequest(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return normalize_email(v)


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str
