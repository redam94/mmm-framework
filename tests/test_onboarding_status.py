"""P1 self-serve onboarding — the path-to-first-model checklist."""

from __future__ import annotations

import pytest

from mmm_framework.api import sessions as ss
from mmm_framework.api.onboarding import project_onboarding_status


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "sessions.db"
    monkeypatch.setattr(ss, "DB_PATH", p)
    ss.init_db()
    return p


def _next(pid):
    return project_onboarding_status(pid)["next_step"]


def test_onboarding_progresses_step_by_step(db):
    pid = ss.create_project("Acme")["project_id"]

    st = project_onboarding_status(pid)
    assert st["steps"][0] == {
        "key": "create_project",
        "title": "Create your project",
        "done": True,
        "hint": "Done — your workspace is ready.",
    }
    assert st["next_step"] == "add_brief"
    assert st["percent"] == 17  # 1 / 6
    assert st["complete"] is False

    # brief (the onboarding endpoint sets meta.onboarded = True)
    ss.set_project_meta(pid, {"onboarded": True, "client_name": "Acme"})
    assert _next(pid) == "add_data"

    # data
    tid = ss.create_session("S1", project_id=pid)["thread_id"]
    ss.register_file(tid, "/tmp/d.csv", "d.csv", "upload")
    assert _next(pid) == "fit_model"

    # first fit
    ss.add_artifact(tid, "model_run", {"run_name": "m1"})
    assert _next(pid) == "review_results"

    # report
    ss.add_artifact(tid, "report", {"path": "/tmp/r.html"})
    assert _next(pid) == "plan_experiment"

    # experiment planned -> fully onboarded
    ss.upsert_experiment(project_id=pid, thread_id=tid, channel="TV")
    final = project_onboarding_status(pid)
    assert final["complete"] is True
    assert final["next_step"] is None
    assert final["percent"] == 100
    assert final["counts"] == {"data_files": 1, "model_runs": 1, "experiments": 1}


def test_report_path_on_model_run_counts_as_reviewed(db):
    pid = ss.create_project("Acme")["project_id"]
    ss.set_project_meta(pid, {"onboarded": True})
    tid = ss.create_session("S1", project_id=pid)["thread_id"]
    ss.register_file(tid, "/tmp/d.csv", "d.csv", "upload")
    # a model_run carrying a report_path satisfies "review results" with no
    # separate report artifact
    ss.add_artifact(tid, "model_run", {"run_name": "m1", "report_path": "/tmp/r.html"})
    assert _next(pid) == "plan_experiment"


def test_unknown_project_is_none(db):
    assert project_onboarding_status("ghost") is None


def test_summarize_eda_issues():
    from mmm_framework.api.onboarding import summarize_eda_issues

    assert summarize_eda_issues([]) == {
        "n_errors": 0,
        "n_warnings": 0,
        "n_info": 0,
        "fit_ready": True,
        "top_issues": [],
    }
    issues = [
        {
            "severity": "warning",
            "check": "date_gaps",
            "variable": "Sales",
            "message": "gaps",
        },
        {
            "severity": "error",
            "check": "negative_spend",
            "variable": "TV",
            "message": "neg",
        },
        {"severity": "info", "check": "x", "variable": "", "message": "fyi"},
    ]
    s = summarize_eda_issues(issues)
    assert s["n_errors"] == 1 and s["n_warnings"] == 1 and s["n_info"] == 1
    assert s["fit_ready"] is False  # an error blocks
    # errors sort ahead of warnings in top_issues
    assert s["top_issues"][0]["severity"] == "error"
    assert s["top_issues"][1]["variable"] == "Sales"
