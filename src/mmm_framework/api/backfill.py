"""Backfill run_metrics rows for model runs fitted before metrics existed.

For every ``model_run`` artifact without a run_metrics row: rebuild the panel
from the recorded spec + dataset, deserialize the saved model, run
``compute_run_metrics``, and persist with the artifact's original timestamp so
trajectory series stay ordered. Skip-and-report anything unrecoverable (empty
model dirs, moved datasets) — old runs saved on other machines often are.

Usage:
    python -m mmm_framework.api.backfill [--project ID] [--dry-run]
                                         [--max-draws 200]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="restrict to one project id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-draws", type=int, default=200)
    args = parser.parse_args()

    sessions_store.init_db()
    report = backfill_run_metrics(
        args.project, dry_run=args.dry_run, max_draws=args.max_draws
    )
    if not report:
        print("No model_run artifacts found.")
        return
    width = max(len(r["run_id"]) for r in report)
    for r in report:
        line = f"{r['run_id']:<{width}}  {r['status']:<9}  {r['detail']}"
        print(line.rstrip())
    counts: dict[str, int] = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
