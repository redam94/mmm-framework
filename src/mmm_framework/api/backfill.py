"""Backfill run snapshots for model runs fitted before a feature existed.

Two backfills over the same ``model_run`` artifacts:

* ``run_metrics`` — for every artifact without a run_metrics row: rebuild the
  panel from the recorded spec + dataset, deserialize the saved model, run
  ``compute_run_metrics``, and persist with the artifact's original timestamp so
  trajectory series stay ordered.
* ``estimands`` — for every artifact whose payload has no ``estimands``: load the
  model, realize its declared/default estimands, and write the rows (+
  ``model_kind``) back onto the artifact payload so the Performance estimands
  view can show pre-existing models.

Skip-and-report anything unrecoverable (empty model dirs, moved datasets) — old
runs saved on other machines often are.

Usage:
    python -m mmm_framework.api.backfill [--project ID] [--dry-run]
                                         [--max-draws 200]
                                         [--what {metrics,estimands,all}]
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from mmm_framework.api import sessions as sessions_store


def _candidate_runs(project_id: str | None) -> list[dict[str, Any]]:
    """(session, artifact) pairs for model_run artifacts, oldest first."""
    out: list[dict[str, Any]] = []
    for s in sessions_store.list_sessions(project_id=project_id):
        for art in sessions_store.list_artifacts(s["thread_id"]):
            if art.get("kind") != "model_run":
                continue
            p = art.get("payload") or {}
            if not p.get("run_id"):
                continue
            out.append(
                {
                    "thread_id": s["thread_id"],
                    "project_id": s.get("project_id"),
                    "artifact_id": art.get("id"),
                    "created_at": art.get("created_at"),
                    "payload": p,
                }
            )
    out.sort(key=lambda r: r["created_at"] or 0)
    return out


def backfill_run_metrics(
    project_id: str | None = None,
    *,
    dry_run: bool = False,
    max_draws: int = 200,
) -> list[dict[str, str]]:
    """Returns a per-run status report: ``{run_id, status, detail}`` with
    status ∈ {done, exists, skipped, error, would-run}."""
    report: list[dict[str, str]] = []
    for cand in _candidate_runs(project_id):
        p = cand["payload"]
        run_id = p["run_id"]
        if sessions_store.get_run_metrics(run_id) is not None:
            report.append({"run_id": run_id, "status": "exists", "detail": ""})
            continue

        model_path = p.get("model_path")
        dataset_path = p.get("dataset_path")
        spec = p.get("spec")
        problems = []
        if (
            not model_path
            or not os.path.isdir(model_path)
            or not os.listdir(model_path)
        ):
            problems.append(f"model dir missing/empty: {model_path}")
        if not dataset_path or not os.path.exists(dataset_path):
            problems.append(f"dataset missing: {dataset_path}")
        if not spec or not spec.get("kpi"):
            problems.append("no spec recorded")
        if problems:
            report.append(
                {"run_id": run_id, "status": "skipped", "detail": "; ".join(problems)}
            )
            continue
        if dry_run:
            report.append({"run_id": run_id, "status": "would-run", "detail": ""})
            continue

        try:
            from mmm_framework.agents.fitting import _mff_config_from_spec
            from mmm_framework.api.history import persist_run_metrics
            from mmm_framework.planning.history import compute_run_metrics
            from mmm_framework import load_mff
            from mmm_framework.serialization import MMMSerializer

            panel = load_mff(dataset_path, _mff_config_from_spec(spec))
            mmm = MMMSerializer.load(model_path, panel)
            p["metrics"] = compute_run_metrics(mmm, max_draws=max_draws)
            persist_run_metrics(
                p,
                cand["thread_id"],
                artifact_id=cand["artifact_id"],
                created_at=cand["created_at"],
            )
            report.append({"run_id": run_id, "status": "done", "detail": ""})
        except Exception as exc:  # noqa: BLE001
            report.append({"run_id": run_id, "status": "error", "detail": str(exc)})
    return report


def _load_saved_model(p: dict[str, Any]) -> Any:
    """Rebuild the panel + deserialize the saved model from a run payload.

    Raises on any unrecoverable problem (missing model dir / dataset / spec) so
    callers can record a 'skipped'/'error' status.
    """
    from mmm_framework.agents.fitting import _mff_config_from_spec
    from mmm_framework import load_mff
    from mmm_framework.serialization import MMMSerializer

    model_path = p.get("model_path")
    dataset_path = p.get("dataset_path")
    spec = p.get("spec")
    problems = []
    if not model_path or not os.path.isdir(model_path) or not os.listdir(model_path):
        problems.append(f"model dir missing/empty: {model_path}")
    if not dataset_path or not os.path.exists(dataset_path):
        problems.append(f"dataset missing: {dataset_path}")
    if not spec or not spec.get("kpi"):
        problems.append("no spec recorded")
    if problems:
        raise FileNotFoundError("; ".join(problems))

    panel = load_mff(dataset_path, _mff_config_from_spec(spec))
    return MMMSerializer.load(model_path, panel)


def backfill_estimands(
    project_id: str | None = None,
    *,
    dry_run: bool = False,
    random_seed: int | None = 42,
) -> list[dict[str, str]]:
    """Populate ``payload['estimands']`` (+ ``model_kind``) on every model_run
    artifact that lacks it, so the Performance estimands view covers models
    fitted before estimand persistence existed. Returns a per-run status report
    with status ∈ {done, exists, skipped, error, would-run}."""
    report: list[dict[str, str]] = []
    for cand in _candidate_runs(project_id):
        p = cand["payload"]
        run_id = p["run_id"]
        if p.get("estimands"):
            report.append({"run_id": run_id, "status": "exists", "detail": ""})
            continue
        try:
            if dry_run:
                # Cheap path check without deserializing the model.
                mp = p.get("model_path")
                if not mp or not os.path.isdir(mp) or not os.listdir(mp):
                    raise FileNotFoundError(f"model dir missing/empty: {mp}")
                report.append({"run_id": run_id, "status": "would-run", "detail": ""})
                continue
            from mmm_framework.agents.estimand_rows import evaluate_estimand_rows

            mmm = _load_saved_model(p)
            p["estimands"] = evaluate_estimand_rows(mmm, random_seed=random_seed)
            p["model_kind"] = getattr(mmm, "__garden_model_kind__", "mmm")
            sessions_store.update_artifact_payload(cand["artifact_id"], p)
            report.append({"run_id": run_id, "status": "done", "detail": ""})
        except FileNotFoundError as exc:
            report.append({"run_id": run_id, "status": "skipped", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            report.append({"run_id": run_id, "status": "error", "detail": str(exc)})
    return report


def _print_report(title: str, report: list[dict[str, str]]) -> None:
    print(f"\n== {title} ==")
    if not report:
        print("  (no model_run artifacts)")
        return
    width = max(len(r["run_id"]) for r in report)
    for r in report:
        print(f"  {r['run_id']:<{width}}  {r['status']:<9}  {r['detail']}".rstrip())
    counts: dict[str, int] = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="restrict to one project id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-draws", type=int, default=200)
    parser.add_argument(
        "--what",
        choices=("metrics", "estimands", "all"),
        default="all",
        help="which backfill(s) to run (default: all)",
    )
    args = parser.parse_args()

    sessions_store.init_db()
    if args.what in ("metrics", "all"):
        _print_report(
            "run_metrics",
            backfill_run_metrics(
                args.project, dry_run=args.dry_run, max_draws=args.max_draws
            ),
        )
    if args.what in ("estimands", "all"):
        _print_report(
            "estimands",
            backfill_estimands(args.project, dry_run=args.dry_run),
        )


if __name__ == "__main__":
    main()
