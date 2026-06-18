"""Per-org rate limiting (abuse protection) for principal-authenticated routes.

An in-memory fixed-window limiter keyed by the caller's ``org_id``. It is
**off by default** (``MMM_RATELIMIT_ENABLED``) so dev/single-tenant behavior is
unchanged, and the dev principal is never limited. Scope is per-process — for a
multi-worker deployment, back this with Redis (documented as the production
upgrade). Billing-grade *quotas* (fits/month per org) live with Track 3 metering;
this module is purely abuse/DoS protection.

Usage (mount via the route decorator's ``dependencies=[...]``)::

    _rl_chat = Depends(require_org_rate_limit("chat"))
    @app.post("/chat", dependencies=[_rl_chat])
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from pydantic_settings import BaseSettings, SettingsConfigDict

from .deps import get_current_principal
from .models import AuthContext


class RateLimitSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MMM_RATELIMIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = False
    window_seconds: int = 60
    default_per_window: int = 120
    chat_per_window: int = 30
    heavy_per_window: int = 10  # model fits / job spawns are expensive


@lru_cache
def get_ratelimit_settings() -> RateLimitSettings:
    return RateLimitSettings()


class _FixedWindow:
    """Thread-safe fixed-window counter: key -> (window_start, count)."""

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        """Record a hit; return (allowed, retry_after_seconds)."""
        now = time.time()
        with self._lock:
            start, count = self._buckets.get(key, (now, 0))
            if now - start >= window:
                start, count = now, 0
            count += 1
            self._buckets[key] = (start, count)
        return (count <= limit, max(int(start + window - now), 0))

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


_BUCKETS = _FixedWindow()


def _limit_for(category: str, s: RateLimitSettings) -> int:
    return {"chat": s.chat_per_window, "heavy": s.heavy_per_window}.get(
        category, s.default_per_window
    )


def require_org_rate_limit(category: str = "default"):
    """Dependency factory: 429 when the caller's org exceeds the ``category`` rate."""

    async def _dep(
        principal: AuthContext = Depends(get_current_principal),
    ) -> AuthContext:
        s = get_ratelimit_settings()
        if not s.enabled or principal.is_dev:
            return principal
        allowed, retry = _BUCKETS.hit(
            f"{category}:{principal.org_id}", _limit_for(category, s), s.window_seconds
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for your organization",
                headers={"Retry-After": str(retry)},
            )
        return principal

    return _dep
