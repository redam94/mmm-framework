"""Rate limiting (abuse protection) for the API surface.

Two layers, both off by default (``MMM_RATELIMIT_ENABLED``) so dev/single-tenant
behavior is unchanged:

- :func:`require_org_rate_limit` — per-**org** (keyed on the verified principal's
  ``org_id``) for authenticated, expensive routes. The dev principal is never
  limited.
- :func:`require_ip_rate_limit` — per-**client-IP** for the *unauthenticated*
  auth routes (login / signup / refresh / password-reset), the
  credential-stuffing / brute-force surface a per-org limiter structurally cannot
  protect (there is no principal yet).

Implementation is an in-memory **fixed-window** counter. Caveats (by design,
documented rather than hidden):

- **Per-process scope.** ``_BUCKETS`` is a module global, so a multi-worker
  server (``uvicorn --workers N``) or the separate ARQ worker each hold their own
  counts and the effective limit becomes ``N × limit``. Back this with Redis for
  multi-process deployments. The dev/`--reload` server is single-worker.
- **~2× burst tolerance** at the window-reset instant (a fixed window is not a
  smooth limiter). Fine for DoS/abuse protection; use a token bucket if you need
  tight shaping.
- The heavy routes enqueue ARQ jobs — this throttles the HTTP *enqueue*, not the
  worker; size the worker/queue independently.
- Behind a reverse proxy, ``request.client.host`` is the proxy. Set
  ``MMM_RATELIMIT_TRUST_FORWARDED=1`` (only behind a trusted proxy that sets it)
  to key on the leftmost ``X-Forwarded-For`` hop instead.

Billing-grade *quotas* (fits/month per org) live with Track 3 metering.
"""

from __future__ import annotations

import math
import threading
import time
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status
from pydantic_settings import BaseSettings, SettingsConfigDict

from .deps import get_current_principal
from .models import AuthContext

# Periodically evict expired windows so the bucket dict can't grow unbounded.
_SWEEP_EVERY = 1024


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
    heavy_per_window: int = 10  # model fits / job spawns / embedding ingest
    auth_per_window: int = 10  # unauthenticated login/signup/refresh per IP
    trust_forwarded: bool = False


@lru_cache
def get_ratelimit_settings() -> RateLimitSettings:
    return RateLimitSettings()


class _FixedWindow:
    """Thread-safe fixed-window counter: key -> (window_start, count)."""

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()
        self._hits = 0

    def hit(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        """Record a hit; return (allowed, retry_after_seconds)."""
        now = time.time()
        with self._lock:
            self._hits += 1
            if self._hits % _SWEEP_EVERY == 0:
                self._sweep(now, window)
            start, count = self._buckets.get(key, (now, 0))
            if now - start >= window:
                start, count = now, 0
            count += 1
            self._buckets[key] = (start, count)
            allowed = count <= limit
        # Retry-After: round UP and floor at 1s when blocked (a "0" would invite
        # an immediate retry that just gets rejected again).
        retry = max(int(math.ceil(start + window - now)), 1) if not allowed else 0
        return (allowed, retry)

    def _sweep(self, now: float, window: float) -> None:
        # caller holds the lock
        expired = [k for k, (s, _) in self._buckets.items() if now - s >= window]
        for k in expired:
            del self._buckets[k]

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._hits = 0


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


def _client_ip(request: Request, trust_forwarded: bool) -> str:
    if trust_forwarded:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_ip_rate_limit(category: str = "auth"):
    """Dependency factory: throttle the *unauthenticated* auth routes per client IP.

    This is the brute-force / credential-stuffing guard the per-org limiter can't
    provide (no principal exists yet). Off until ``MMM_RATELIMIT_ENABLED=1``.
    """

    async def _dep(request: Request) -> None:
        s = get_ratelimit_settings()
        if not s.enabled:
            return
        ip = _client_ip(request, s.trust_forwarded)
        allowed, retry = _BUCKETS.hit(
            f"{category}:ip:{ip}", s.auth_per_window, s.window_seconds
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many attempts — please slow down",
                headers={"Retry-After": str(retry)},
            )

    return _dep
