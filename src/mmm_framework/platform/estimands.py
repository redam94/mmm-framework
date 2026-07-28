"""Project estimand aggregation for the Performance page.

Turns the estimand rows persisted on each ``model_run`` artifact (see
``agents.fitting`` / ``agents.estimand_rows``) into **comparability clusters**:
one cluster per ``(estimand, KPI)`` so that the same estimand measured on the
same KPI by different models sits side-by-side, while a model on a *different*
KPI lands in its own cluster and is never silently compared. Two ROI estimands
of the same statistical *kind* but different methodology (``contribution_roi``
vs ``counterfactual_roi``) stay distinct clusters — they are different numbers.

The labeling / reference-value / evidence logic is *shared* with the report's
``EstimandsSection`` (``reporting/sections.py``) rather than mirrored, so the
dashboard and the generated report cannot tell different stories: both call
:mod:`mmm_framework.finance.evidence`, which owns the grading bar, the
direction (a cost per outcome is graded LOWER-is-better) and the verdict. It
lives in ``finance`` because ``platform`` must not import the reporting stack
and ``reporting`` must not import the services layer.

``group_estimands`` is pure (no DB) and unit-tested; ``build_project_estimands``
reads the sessions store.
"""

from __future__ import annotations

from typing import Any

from mmm_framework.finance.evidence import (
    UNRESOLVED_COST_REASON,
    EvidenceReference,
)
from mmm_framework.finance.evidence import classify_evidence as _classify_evidence
from mmm_framework.finance.evidence import is_ratio_kind as _is_ratio_kind
from mmm_framework.finance.evidence import resolve_reference

# Display labels mirror reporting/sections.py::EstimandsSection._KIND_LABELS.
_ESTIMAND_LABELS: dict[str, str] = {
    "contribution_roi": "Contribution ROI",
    "counterfactual_roi": "Counterfactual ROI",
    "marginal_roas": "Marginal ROAS",
    "contribution": "Incremental contribution",
    "awareness_lift": "Awareness lift",
    "cost_per_conversion": "Cost per conversion",
}

# Canonical sort order for estimands within a KPI section; unknowns sort after,
# alphabetically.
_ESTIMAND_ORDER = [
    "contribution_roi",
    "counterfactual_roi",
    "marginal_roas",
    "contribution",
    "awareness_lift",
    "cost_per_conversion",
]


def estimand_label(name: str) -> str:
    """Human label for an estimand name (title-cased fallback for unknowns)."""
    return _ESTIMAND_LABELS.get(name, name.replace("_", " ").title())


# The grading rule lives in finance.evidence (one mint, two consumers). These
# names stay importable from here because callers and tests already use them.
is_ratio_kind = _is_ratio_kind
classify_evidence = _classify_evidence


def _estimand_sort_key(name: str) -> tuple[int, str]:
    try:
        return (_ESTIMAND_ORDER.index(name), "")
    except ValueError:
        return (len(_ESTIMAND_ORDER), name)


def group_estimands(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Cluster per-run estimand rows into ``(estimand, KPI)`` comparability groups.

    Parameters
    ----------
    runs
        One dict per fitted run with keys ``run_id``, ``label``, ``model_kind``,
        ``model_key``, ``kpi``, ``created_at`` and ``estimands`` (a list of rows
        as produced by :func:`mmm_framework.agents.estimand_rows.evaluate_estimand_rows`).

    Returns
    -------
    dict with ``runs`` (incl. ``is_latest_for_model`` for the default selection),
    ``kpis``, and ``groups`` (each a comparability cluster). Pure; no I/O.
    """
    # Latest run per structural model identity (for the FE default selection).
    # Ties on created_at break deterministically on run_id so the flagged "latest"
    # run is stable regardless of input order.
    latest_for_key: dict[str, tuple[float, str]] = {}
    for r in runs:
        mk = r.get("model_key") or r.get("run_id")
        rid = r.get("run_id") or ""
        ca = float(r.get("created_at") or 0)
        if mk not in latest_for_key or (ca, rid) > latest_for_key[mk]:
            latest_for_key[mk] = (ca, rid)

    run_summaries: list[dict[str, Any]] = []
    # group key -> accumulator
    groups: dict[str, dict[str, Any]] = {}

    for r in runs:
        run_id = r.get("run_id")
        kpi = r.get("kpi") or ""
        model_key = r.get("model_key") or run_id
        est_rows = r.get("estimands") or []
        # Per-channel evidence tier map (issue #124); {} for runs without it.
        ch_evidence = r.get("channel_evidence") or {}
        run_summaries.append(
            {
                "run_id": run_id,
                "label": r.get("label") or run_id,
                "model_kind": r.get("model_kind") or "mmm",
                "model_key": model_key,
                "kpi": kpi,
                "created_at": r.get("created_at"),
                "n_estimands": len(est_rows),
                "is_latest_for_model": latest_for_key.get(model_key, (0.0, ""))[1]
                == run_id,
            }
        )

        for row in est_rows:
            name = row.get("estimand") or ""
            if not name:
                continue
            gkey = f"{name}|||{kpi}"
            grp = groups.get(gkey)
            if grp is None:
                kind = row.get("kind") or ""
                units = row.get("units") or ""
                # The row's explicit measurement reference wins (impression-
                # level ROI: efficiency metrics carry reference 0 even though
                # their kind is still "roi"; a profit basis carries 1/margin);
                # else the kind/units rule decides, and a cost per outcome
                # resolves to *no* reference rather than a fabricated 0.
                ref_obj = resolve_reference(
                    kind,
                    units,
                    explicit=(
                        float(row["reference"])
                        if row.get("reference") is not None
                        else None
                    ),
                )
                grp = {
                    "key": gkey,
                    "estimand": name,
                    "label": estimand_label(name),
                    "kpi": kpi,
                    "kind": kind,
                    "units": units,
                    "is_ratio": ref_obj.is_ratio,
                    "reference": ref_obj.value,
                    # Which way is better, and the sentence naming the bar. The
                    # UI renders `reference_hint` rather than deriving one: it
                    # used to print "vs 0 (no effect)" beside a profit
                    # break-even of 2.5, and "Strong" for every cost.
                    "direction": ref_obj.direction,
                    "reference_hint": ref_obj.hint,
                    "reference_basis": ref_obj.basis,
                    "reference_note": (
                        UNRESOLVED_COST_REASON
                        if ref_obj.basis == "unresolved"
                        else None
                    ),
                    "channels": [],
                    "_models": {},  # run_id -> model entry (collapsed below)
                }
                groups[gkey] = grp
            # Fill kind/units if the first row was blank.
            if not grp["kind"] and row.get("kind"):
                grp["kind"] = row.get("kind")
            if not grp["units"] and row.get("units"):
                grp["units"] = row.get("units")

            channel = row.get("channel") or "—"
            if channel not in grp["channels"]:
                grp["channels"].append(channel)

            model = grp["_models"].get(run_id)
            if model is None:
                model = {
                    "run_id": run_id,
                    "label": r.get("label") or run_id,
                    "model_kind": r.get("model_kind") or "mmm",
                    "model_key": model_key,
                    "created_at": r.get("created_at"),
                    "rows": {},  # channel -> cell
                }
                grp["_models"][run_id] = model

            # Per-row reference: a row may carry its own (a mixed-measurement
            # portfolio puts an efficiency channel beside a spend one), else the
            # group's. Direction always comes from the metric, not the row.
            ref = (
                EvidenceReference(
                    value=float(row["reference"]),
                    direction=grp["direction"],
                    hint=grp["reference_hint"],
                    basis=grp["reference_basis"],
                )
                if row.get("reference") is not None
                else EvidenceReference(
                    value=grp["reference"],
                    direction=grp["direction"],
                    hint=grp["reference_hint"],
                    basis=grp["reference_basis"],
                )
            )
            mean = row.get("mean")
            lower = row.get("hdi_low")
            upper = row.get("hdi_high")
            cell = {
                "channel": channel,
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "units": row.get("units") or grp["units"],
                "status": row.get("status") or "ok",
                # CI-vs-reference verdict (strong/below/uncertain) — NOT the tier.
                "evidence": classify_evidence(
                    status=row.get("status"),
                    mean=mean,
                    lower=lower,
                    upper=upper,
                    reference=ref,
                ),
                "prob_positive": row.get("prob_positive"),
                "prob_profitable": row.get("prob_profitable"),
            }
            # Evidence tier + identifiability flag (issue #124) — same chip the
            # report renders. Absent for runs fitted before evidence persistence.
            tier = ch_evidence.get(channel)
            if tier:
                cell["tier"] = tier
            model["rows"][channel] = cell

    # Materialize groups: order channels, collapse model dicts -> lists, count
    # models that actually carry a usable number.
    out_groups: list[dict[str, Any]] = []
    for grp in groups.values():
        channels = grp.pop("channels")
        models_map = grp.pop("_models")
        models = []
        n_with_data = 0
        for m in models_map.values():
            row_map = m.pop("rows")
            m["rows"] = [row_map[ch] for ch in channels if ch in row_map]
            if any(
                c.get("status") == "ok" and c.get("mean") is not None for c in m["rows"]
            ):
                n_with_data += 1
            models.append(m)
        models.sort(key=lambda m: -(float(m.get("created_at") or 0)))
        grp["channels"] = channels
        grp["models"] = models
        grp["n_models"] = len(models)
        grp["n_models_with_data"] = n_with_data
        out_groups.append(grp)

    out_groups.sort(key=lambda g: (g["kpi"], _estimand_sort_key(g["estimand"])))

    kpis = sorted({g["kpi"] for g in out_groups})
    run_summaries.sort(key=lambda r: -(float(r.get("created_at") or 0)))
    return {"runs": run_summaries, "kpis": kpis, "groups": out_groups}


def _model_key(model_kind: str, kpi: str, channels: list[str]) -> str:
    """Structural model identity for grouping latest-run-per-model on the FE."""
    return "|".join([model_kind or "mmm", kpi or "", ",".join(sorted(channels or []))])


def build_project_estimands(project_id: str | None) -> dict[str, Any]:
    """Read the project's ``model_run`` artifacts and group their persisted
    estimand rows. Runs fitted before estimand persistence (or with no estimands)
    are skipped; run the backfill (``python -m mmm_framework.platform.backfill
    --what estimands``) to populate them."""
    from . import sessions as sessions_store

    runs: list[dict[str, Any]] = []
    for s in sessions_store.list_sessions(project_id=project_id):
        for art in sessions_store.list_artifacts(s["thread_id"]):
            if art.get("kind") != "model_run":
                continue
            p = art.get("payload") or {}
            est = p.get("estimands")
            if not est:
                continue
            kpi = p.get("kpi") or ""
            channels = p.get("channels") or []
            model_kind = p.get("model_kind") or "mmm"
            runs.append(
                {
                    "run_id": p.get("run_id") or art["id"],
                    "label": p.get("run_name") or p.get("run_id") or art["id"],
                    "model_kind": model_kind,
                    "model_key": _model_key(model_kind, kpi, channels),
                    "kpi": kpi,
                    "created_at": art.get("created_at"),
                    "estimands": est,
                    # Per-channel evidence tier + identifiability flag persisted at
                    # fit time (issue #124); None for runs fitted before it / after
                    # the evidence backfill. Grouped onto each cell below.
                    "channel_evidence": p.get("channel_evidence"),
                }
            )
    # Deterministic input order (newest first, run_id tiebreak) so the grouped
    # output — incl. the channel union, which follows first-seen order — is stable
    # across calls regardless of how the store enumerates sessions/artifacts.
    runs.sort(key=lambda r: (-(float(r.get("created_at") or 0)), r.get("run_id") or ""))
    return group_estimands(runs)
