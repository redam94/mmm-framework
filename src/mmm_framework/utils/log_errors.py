"""Observable error suppression.

The codebase has many ``try: ... except Exception: pass`` blocks that swallow
failures silently — the dangerous failure mode where a report section renders
empty or a tool no-ops while the caller (and the user) see a clean result with a
hole in it. This is the sanctioned replacement: suppress the exception (so the
best-effort path still degrades gracefully) but LOG it, so the failure is
observable instead of invisible.

Use it for genuinely-optional best-effort work::

    from mmm_framework.utils.log_errors import logged_suppress

    with logged_suppress("extracting saturation curves"):
        bundle.saturation = self._extract_saturation()

For code that should NOT swallow (anything producing a business number), let the
exception propagate — do not reach for this.
"""

from __future__ import annotations

from contextlib import contextmanager

from loguru import logger


@contextmanager
def logged_suppress(
    context: str,
    *exceptions: type[BaseException],
    level: str = "DEBUG",
):
    """Suppress ``exceptions`` (default: ``Exception``) but log them with ``context``.

    Parameters
    ----------
    context : str
        Short description of the optional work being attempted, included in the
        log line so a swallowed failure can be traced.
    *exceptions : exception types
        Which exceptions to suppress. Defaults to ``Exception`` (not
        ``BaseException`` — KeyboardInterrupt/SystemExit always propagate).
    level : str
        loguru level for the log line (default ``DEBUG``).
    """
    exc_types = exceptions or (Exception,)
    try:
        yield
    except exc_types as e:  # noqa: BLE001 - this is the sanctioned suppression point
        logger.opt(exception=True).log(
            level, "Suppressed error while {}: {}", context, e
        )
