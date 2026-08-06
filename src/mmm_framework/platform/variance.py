"""Assemble the inputs for a variance-to-plan bridge from the project stores
(issue #227).

The engine (:mod:`mmm_framework.planning.variance`) is pure compute; this
module is the store-reading half: it finds the committed plan of record, aligns
the delivery ledger to the committed periods, fetches the latest-as-of realized
KPI, resolves the valuation, and detects a refit. Everything it returns is
JSON-safe, so the same dict crosses the REST job boundary and the agent-kernel
boundary unchanged.

Refusals here are ``VarianceInputError`` with a stated reason — the conditions
a job should never be started under (no committed plan, no actuals, a delivery
ledger that does not cover the committed window, a dataset that changed since
the commit). The engine's own refusals (window coverage, refit split,
non-monetary supplied lines) stay in the engine, where they are enforced for
every caller including tests that bypass the store.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["VarianceInputError", "collect_variance_inputs"]


class VarianceInputError(ValueError):
    """A variance bridge cannot be assembled, with the reason stated."""


def collect_variance_inputs(
    project_id: str,
    *,
    supplied: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Everything ``variance_to_plan`` (the model op) needs, from the stores.

    Returns a JSON-safe dict::

        {
          "committed_version": {...},      # full plan_versions row, payload included
          "actual_media": {ch: [spend]},   # delivery aligned to committed periods
          "actuals": [{period, kpi_value}],# latest-as-of realized KPI
          "valuation": {...},              # kpi_to_dollars(...).to_dict()
          "supplied": [...],               # validated pass-through
          "refit_run_id": str | None,      # latest run when it differs from committed
          "run_diff": dict | None,         # compare_runs(committed, latest) when it does
          "committed_run_name": str,       # model dir basename — what the job loads
          "dataset_path": str | None,
        }

    Raises :class:`VarianceInputError` when the bridge should not start:
    no committed plan of record, a commitment without a per-period plan, no
    realized KPI uploaded, delivery gaps over the committed window, or a
    dataset whose fingerprint no longer matches the commitment.
    """
    from ..finance.valuation import kpi_to_dollars
    from . import sessions as sessions_store
    from .history import latest_model_run_payload
    from .runs import compare_runs, data_fingerprint

    # ── supplied lines: shape-checked here so a bad one fails the POST ─────
    clean_supplied: list[dict[str, Any]] = []
    for item in supplied or []:
        name = str(item.get("name") or "").strip()
        note = str(item.get("source_note") or "").strip()
        if not name or not note:
            raise VarianceInputError(
                "Every supplied line needs a name and a source_note — an "
                "unattributed manual adjustment is exactly what the SUPPLIED "
                "provenance exists to prevent."
            )
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            raise VarianceInputError(
                f"Supplied line {name!r} has a non-numeric value."
            ) from None
        clean_supplied.append({"name": name, "value": value, "source_note": note})

    committed = sessions_store.latest_committed_plan(project_id)
    if committed is None:
        raise VarianceInputError(
            "No committed plan of record: a variance to plan needs a committed "
            "version to be a variance FROM. Commit a plan first — drafts do "
            "not qualify, because a draft can be edited after the fact."
        )
    payload = committed.get("payload") or {}
    snapshot = payload.get("forecast") or {}
    periods = [str(p) for p in (snapshot.get("periods") or [])]
    if not periods:
        raise VarianceInputError(
            "The committed version carries no forecast periods; there is "
            "nothing to bridge against."
        )
    plan_media = payload.get("plan_media") or snapshot.get("plan_media") or {}
    if not plan_media:
        raise VarianceInputError(
            "The committed version records no per-period spend plan "
            "(plan_media), so delivery variance cannot be computed. "
            "Commitments made before v1.4 predate the recorded plan; "
            "re-commit from a fresh forecast."
        )

    # ── realized KPI ────────────────────────────────────────────────────────
    actuals = sessions_store.latest_actuals_for_project(project_id)
    if not actuals:
        raise VarianceInputError(
            "No realized KPI has been uploaded for this project. The bridge "
            "compares the committed forecast against an independent actuals "
            "record — upload one via POST /projects/{id}/actuals."
        )

    # ── delivery, aligned to the committed periods per channel ─────────────
    delivery = sessions_store.list_delivery(project_id)
    spend_at: dict[str, dict[str, float]] = {}
    for row in delivery:
        ch = str(row.get("channel") or "")
        per = str(row.get("period") or "")
        try:
            spend_at.setdefault(ch, {})[per] = float(row.get("spend"))
        except (TypeError, ValueError):
            continue
    actual_media: dict[str, list[float]] = {}
    gaps: list[str] = []
    for ch in plan_media:
        series: list[float] = []
        have = spend_at.get(ch, {})
        for per in periods:
            if per in have:
                series.append(have[per])
            else:
                gaps.append(f"{ch} @ {per}")
        actual_media[ch] = series
    if gaps:
        shown = ", ".join(gaps[:6]) + (", …" if len(gaps) > 6 else "")
        raise VarianceInputError(
            f"The delivery ledger does not cover the committed window: "
            f"{len(gaps)} channel-period cell(s) have no actual spend "
            f"({shown}). Assuming plan-as-delivered would fabricate a zero "
            "delivery variance as if it were measured — upload the missing "
            "delivery rows instead."
        )

    # ── the dataset must still be the committed one ────────────────────────
    prov = payload.get("provenance") or {}
    recorded = prov.get("data_fingerprint")
    recorded_fp = recorded.get("md5") if isinstance(recorded, dict) else recorded
    dataset_path = prov.get("dataset_path") or (
        recorded.get("path") if isinstance(recorded, dict) else None
    )
    if dataset_path and recorded_fp:
        current = data_fingerprint(dataset_path) or {}
        if current.get("md5") != recorded_fp:
            raise VarianceInputError(
                "The dataset behind the committed plan has changed "
                f"(committed {recorded_fp}, now {current.get('md5')}). The "
                "committed forecast cannot be regenerated against different "
                "data, so the delivery bucket would not be the committed "
                "model's statement."
            )

    model_path = prov.get("model_path")
    committed_run_name = os.path.basename(str(model_path)) if model_path else ""
    if not committed_run_name:
        raise VarianceInputError(
            "The committed version records no model path, so the committed "
            "model cannot be reloaded. It should not have been committable."
        )

    # ── refit detection: stated, never silently absorbed ───────────────────
    refit_run_id: str | None = None
    run_diff: dict[str, Any] | None = None
    latest = latest_model_run_payload(project_id) or {}
    committed_run_id = prov.get("run_id")
    latest_run_id = latest.get("run_id")
    if latest_run_id and committed_run_id and latest_run_id != committed_run_id:
        refit_run_id = str(latest_run_id)
        try:
            run_diff = compare_runs(str(committed_run_id), str(latest_run_id))
        except Exception:  # noqa: BLE001 — the diff is best-effort context
            run_diff = None

    # ── valuation (project preferences / branding economics) ───────────────
    prefs = branding = None
    try:
        prefs = {"economics": sessions_store.get_preference(project_id, "economics")}
    except Exception:  # noqa: BLE001
        pass
    try:
        branding = {"economics": sessions_store.get_preference(project_id, "branding")}
    except Exception:  # noqa: BLE001
        pass
    valuation = kpi_to_dollars(preferences=prefs, branding=branding).to_dict()

    return {
        "committed_version": committed,
        "actual_media": actual_media,
        "actuals": [
            {"period": str(r.get("period")), "kpi_value": r.get("kpi_value")}
            for r in actuals
        ],
        "valuation": valuation,
        "supplied": clean_supplied,
        "refit_run_id": refit_run_id,
        "run_diff": run_diff,
        "committed_run_name": committed_run_name,
        "dataset_path": str(dataset_path) if dataset_path else None,
    }
