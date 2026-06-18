"""Self-serve onboarding readiness for a project.

Computes the path-to-first-model as an ordered checklist, derived entirely from
real project state (brief, uploaded data, fitted models, reports, experiments).
A self-serve UI renders this to guide a brand-new user — and to validate their
progress — without an analyst on the call. Pure read over ``sessions.db``.
"""

from __future__ import annotations

from typing import Any

from mmm_framework.api import sessions as store


def _step(key: str, title: str, done: bool, hint: str) -> dict[str, Any]:
    return {"key": key, "title": title, "done": bool(done), "hint": hint}


def summarize_eda_issues(issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact data-quality summary from the agent's ``dashboard_data['eda']``
    issues (each ``{severity, check, variable, message}``). ``fit_ready`` is true
    when there are no blocking errors. Surfaced inline at the onboarding "add
    data" step so a new user fixes problems before fitting."""
    errs = [i for i in issues if i.get("severity") == "error"]
    warns = [i for i in issues if i.get("severity") == "warning"]
    infos = [i for i in issues if i.get("severity") == "info"]
    return {
        "n_errors": len(errs),
        "n_warnings": len(warns),
        "n_info": len(infos),
        "fit_ready": len(errs) == 0,
        "top_issues": [
            {
                "severity": i.get("severity"),
                "check": i.get("check"),
                "variable": i.get("variable") or None,
                "message": i.get("message"),
            }
            for i in (errs + warns)[:5]
        ],
    }


def project_onboarding_status(project_id: str) -> dict[str, Any] | None:
    """The onboarding checklist + next action for ``project_id`` (None if absent)."""
    proj = store.get_project(project_id)
    if proj is None:
        return None

    sessions = store.list_sessions(project_id=project_id)
    thread_ids = [s["thread_id"] for s in sessions]
    artifacts = [a for tid in thread_ids for a in store.list_artifacts(tid)]
    n_files = sum(len(store.list_files(tid)) for tid in thread_ids)
    model_runs = [a for a in artifacts if a.get("kind") == "model_run"]
    has_report = any(a.get("kind") == "report" for a in artifacts) or any(
        (mr.get("payload") or {}).get("report_path") for mr in model_runs
    )
    n_experiments = len(store.list_experiments(project_id=project_id))
    meta = proj.get("meta") or {}

    steps = [
        _step(
            "create_project",
            "Create your project",
            True,
            "Done — your workspace is ready.",
        ),
        _step(
            "add_brief",
            "Add a project brief",
            meta.get("onboarded") is True,
            "Tell the assistant your client, goals, KPI, and channels so it "
            "grounds every answer in your context.",
        ),
        _step(
            "add_data",
            "Add your data",
            n_files > 0,
            "Upload an MFF dataset — or ask the assistant to generate synthetic "
            "data to explore the workflow first.",
        ),
        _step(
            "fit_model",
            "Fit your first model",
            len(model_runs) > 0,
            "Ask the assistant to configure and fit a Bayesian MMM "
            "(a national fit takes ~17 s).",
        ),
        _step(
            "review_results",
            "Review the results",
            has_report,
            "Generate a report to see channel ROI, decomposition, and "
            "saturation curves.",
        ),
        _step(
            "plan_experiment",
            "Plan an experiment",
            n_experiments > 0,
            "Use the experiment priorities (EIG/EVOI) to design your "
            "highest-value lift test.",
        ),
    ]

    completed = [s for s in steps if s["done"]]
    nxt = next((s for s in steps if not s["done"]), None)
    return {
        "project_id": project_id,
        "project_name": proj.get("name"),
        "steps": steps,
        "completed": len(completed),
        "total": len(steps),
        "percent": round(100 * len(completed) / len(steps)),
        "complete": nxt is None,
        "next_step": nxt["key"] if nxt else None,
        "next_hint": (
            nxt["hint"]
            if nxt
            else "You're set up — keep iterating the measurement loop."
        ),
        "counts": {
            "data_files": n_files,
            "model_runs": len(model_runs),
            "experiments": n_experiments,
        },
    }
