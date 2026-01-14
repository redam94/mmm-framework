"""
Redis service for job tracking, caching, and pub/sub.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import redis.asyncio as redis

from config import Settings, get_settings
from schemas import JobStatus


class RedisService:
    """
    Redis service for async operations.

    Handles:
    - Job status tracking
    - Progress updates
    - Caching
    - Pub/sub for real-time updates
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._redis: redis.Redis | None = None

    async def connect(self) -> redis.Redis:
        """Connect to Redis."""
        if self._redis is None:
            self._redis = await redis.from_url(
                self.settings.redis_url,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=True,
            )
        return self._redis

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def ping(self) -> bool:
        """Check Redis connection."""
        try:
            r = await self.connect()
            return await r.ping()
        except Exception:
            return False

    # =========================================================================
    # Job Status
    # =========================================================================

    def _job_key(self, model_id: str) -> str:
        """Get Redis key for job status."""
        return f"mmm:job:{model_id}"

    async def set_job_status(
        self,
        model_id: str,
        status: JobStatus,
        progress: float = 0.0,
        progress_message: str | None = None,
        error_message: str | None = None,
        **extra,
    ) -> None:
        """Set job status in Redis."""
        r = await self.connect()

        job_data = {
            "model_id": model_id,
            "status": status.value,
            "progress": progress,
            "progress_message": progress_message,
            "error_message": error_message,
            "updated_at": datetime.utcnow().isoformat(),
            **extra,
        }

        # Store status
        await r.hset(
            self._job_key(model_id),
            mapping={
                k: (
                    json.dumps(v)
                    if isinstance(v, (dict, list))
                    else str(v) if v is not None else ""
                )
                for k, v in job_data.items()
            },
        )

        # Set expiry (7 days)
        await r.expire(self._job_key(model_id), 7 * 24 * 3600)

        # Publish update for real-time subscribers
        await r.publish(f"mmm:updates:{model_id}", json.dumps(job_data, default=str))

    async def get_job_status(self, model_id: str) -> dict[str, Any] | None:
        """Get job status from Redis."""
        r = await self.connect()

        data = await r.hgetall(self._job_key(model_id))
        if not data:
            return None

        # Parse values
        result = {}
        for k, v in data.items():
            if v == "":
                result[k] = None
            elif v in ("True", "False"):
                result[k] = v == "True"
            else:
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = v

        return result

    async def update_job_progress(
        self,
        model_id: str,
        progress: float,
        message: str | None = None,
    ) -> None:
        """Update job progress."""
        r = await self.connect()

        updates = {
            "progress": str(progress),
            "updated_at": datetime.utcnow().isoformat(),
        }
        if message:
            updates["progress_message"] = message

        await r.hset(self._job_key(model_id), mapping=updates)

        # Publish update
        await r.publish(
            f"mmm:updates:{model_id}",
            json.dumps(
                {
                    "model_id": model_id,
                    "progress": progress,
                    "progress_message": message,
                }
            ),
        )

    async def delete_job_status(self, model_id: str) -> bool:
        """Delete job status."""
        r = await self.connect()
        return bool(await r.delete(self._job_key(model_id)))

    # =========================================================================
    # Caching
    # =========================================================================

    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
    ) -> None:
        """Set a cached value."""
        r = await self.connect()
        cache_key = f"mmm:cache:{key}"
        await r.setex(cache_key, ttl_seconds, json.dumps(value, default=str))

    async def cache_get(self, key: str) -> Any | None:
        """Get a cached value."""
        r = await self.connect()
        cache_key = f"mmm:cache:{key}"
        data = await r.get(cache_key)
        if data:
            return json.loads(data)
        return None

    async def cache_delete(self, key: str) -> bool:
        """Delete a cached value."""
        r = await self.connect()
        cache_key = f"mmm:cache:{key}"
        return bool(await r.delete(cache_key))

    # =========================================================================
    # Worker Health
    # =========================================================================

    async def check_worker_health(self) -> bool:
        """Check if any workers are active."""
        r = await self.connect()
        # ARQ stores worker info in redis
        workers = await r.keys("arq:worker:*")
        return len(workers) > 0

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        r = await self.connect()

        # Get queue lengths
        pending = await r.zcard("arq:queue")

        # Get active jobs (approximate)
        active_keys = await r.keys("mmm:job:*")

        active_count = 0
        for key in active_keys:
            status = await r.hget(key, "status")
            if status == JobStatus.RUNNING.value:
                active_count += 1

        return {
            "pending": pending,
            "active": active_count,
            "total_tracked": len(active_keys),
        }


# Global redis service
_redis_service: RedisService | None = None


async def get_redis() -> RedisService:
    """Get global Redis service instance."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service
