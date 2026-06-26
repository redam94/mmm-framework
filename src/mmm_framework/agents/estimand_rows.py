"""Shared estimand-row builder.

One place that turns a fitted model's declarative estimands into flat, JSON-safe
rows (mean + HDI + units + tail probabilities), with wildcard-channel estimands
split into ``(estimand, channel)``. Used by the ``compute_estimands`` agent op
(human-facing markdown/table), by fit-time persistence (``agents.fitting`` stores
the rows on the ``model_run`` artifact), by the backfill, and by the Performance
estimands endpoint (``api.estimands``) — so every surface shows the same numbers.

Kept dependency-light (only ``mmm.evaluate_estimands``) so the API/backfill can
import it without dragging in the whole agent toolchain.
"""

from __future__ import annotations

from typing import Any


def evaluate_estimand_rows(
    mmm: Any,
    *,
    estimands: Any = None,
    random_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Realize ``mmm``'s estimands and return one row per realized estimand.

    Rows mirror the ``compute_estimands`` op shape exactly::

        {estimand, channel, kind, status, mean, hdi_low, hdi_high,
         units, prob_positive, prob_profitable}

    ``channel`` is the wildcard expansion (``"contribution_roi:TV"`` -> channel
    ``"TV"``) or ``"—"`` for a scalar estimand. Unsupported estimands are
    returned (``status="unsupported"``, ``mean=None``), not dropped, so the UI
    can show *why* a model lacks a number rather than silently omitting it.
    """
    out = mmm.evaluate_estimands(estimands=estimands, random_seed=random_seed)
    rows: list[dict[str, Any]] = []
    for key, r in out.items():
        name, _, channel = key.partition(":")
        extra = r.extra or {}
        rows.append(
            {
                "estimand": name,
                "channel": channel or "—",
                "kind": r.kind,
                "status": r.status,
                "mean": None if r.mean is None else float(r.mean),
                "hdi_low": None if r.hdi_low is None else float(r.hdi_low),
                "hdi_high": None if r.hdi_high is None else float(r.hdi_high),
                "units": r.units,
                "hdi_prob": getattr(r, "hdi_prob", None),
                "prob_positive": extra.get("prob_positive"),
                "prob_profitable": extra.get("prob_profitable"),
            }
        )
    return rows
