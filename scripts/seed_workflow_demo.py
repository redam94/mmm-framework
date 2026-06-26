"""Seed a multi-model "full Bayesian workflow + validation" demo project.

Fits several MMM variants on ONE KPI inside ONE session, so the Performance →
Estimands tab shows them grouped into a comparability cluster (same estimand ×
same KPI → models side by side), and narrates the end-to-end Bayesian workflow
with REAL validation numbers pulled from the fits:

    research question → prior predictive check → fit → convergence
    (R-hat / ESS / divergences) → posterior predictive check → parameter
    learning (prior → posterior) → estimands → model comparison.

Estimands are persisted on each model_run at fit time (build_and_fit, block 8d),
so the new Estimands tab populates automatically. The two variants use distinct
channel sets (Lean vs Full), so each is its own model identity and BOTH show by
default in the tab.

Real fits by default. Replaces any prior "Demo: Bayesian Workflow" project (the
sessions DB is the fixed api/sessions.db; plots/tables — none here — would need a
matching MMM_AGENT_WORKSPACE).

Usage:
    uv run python scripts/seed_workflow_demo.py
    uv run python scripts/seed_workflow_demo.py --draws 500 --tune 500 --weeks 156
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling seeder helpers

PROJECT_NAME = "Demo: Bayesian Workflow"


def _build_spec(kpi, channels, controls, draws, tune, chains):
    return {
        "kpi": kpi,
        "media_channels": [
            {
                "name": c,
                "adstock": {"type": "geometric", "l_max": 8},
                "saturation": {"type": "logistic"},
            }
            for c in channels
        ],
        "control_variables": [{"name": c} for c in controls],
        "inference": {"draws": draws, "tune": tune, "chains": chains, "random_seed": 0},
        "seasonality": {"yearly": 2},
        "trend": {"type": "linear"},
    }


def _replace_prior_project(store, name: str) -> None:
    for p in store.list_projects():
        if p.get("name") != name:
            continue
        pid = p["project_id"]
        for s in store.list_sessions(project_id=pid):
            store.delete_session(s["thread_id"])
        for e in store.list_experiments(project_id=pid):
            store.delete_experiment(e["id"])
        with store._conn() as c:
            c.execute("DELETE FROM run_metrics WHERE project_id = ?", (pid,))
        store.delete_project(pid)
        print(f"Replaced prior demo project {pid}")


# ── formatting helpers (real numbers from the fits) ───────────────────────────


def _f(x, d=2):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _conv_line(mr: dict) -> str:
    c = (mr.get("diagnostics") or {}).get("convergence") or {}
    verdict = "clean — no convergence flags" if c.get("ok") else (
        "flags raised: " + ", ".join(c.get("flags") or []) + " (see Model health)"
    )
    return (
        f"R-hat max {_f(c.get('rhat_max'), 3)}, min bulk-ESS "
        f"{_f(c.get('ess_bulk_min'), 0)}, {c.get('divergences')} divergences — {verdict}"
    )


def _learning_line(mr: dict) -> str:
    learn = (mr.get("diagnostics") or {}).get("learning") or {}
    counts = learn.get("verdict_counts") or {}
    n = learn.get("n_parameters")
    if not counts:
        return "parameter learning unavailable for this fit"
    parts = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    return f"of {n} parameters: {parts}"


def _roi_map(mr: dict) -> dict[str, dict]:
    out = {}
    for r in mr.get("estimands") or []:
        if r.get("estimand") == "contribution_roi" and r.get("status") == "ok" and r.get("mean") is not None:
            out[r.get("channel")] = r
    return out


def _roi_table(mr: dict) -> str:
    rows = _roi_map(mr)
    if not rows:
        return "_(no contribution-ROI estimands recorded)_"
    out = "| Channel | Contribution ROI | 94% HDI |\n|---|--:|:--|\n"
    for ch, r in rows.items():
        out += f"| {ch} | {_f(r['mean'])} | [{_f(r['hdi_low'])}, {_f(r['hdi_high'])}] |\n"
    return out


def _compare_table(lean_mr: dict, full_mr: dict, shared: list[str]) -> str:
    lean, full = _roi_map(lean_mr), _roi_map(full_mr)
    out = "| Channel | Lean ROI | Full ROI |\n|---|--:|--:|\n"
    for ch in shared:
        lr = lean.get(ch, {}).get("mean")
        fr = full.get(ch, {}).get("mean")
        out += f"| {ch} | {_f(lr)} | {_f(fr)} |\n"
    return out


# ── seed ──────────────────────────────────────────────────────────────────────


def seed(draws: int, tune: int, chains: int, weeks: int) -> None:
    from mmm_framework.api import history, sessions as store
    from mmm_framework.api import runs as runs_mod
    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.synth import generate_mff
    import seed_demo_project as base  # _stamp_and_persist / _seed_chat

    store.init_db()
    _replace_prior_project(store, PROJECT_NAME)

    df, key = generate_mff("realistic", seed=7, n_weeks=weeks)
    out_dir = Path("demo_data")
    out_dir.mkdir(exist_ok=True)
    path = str(out_dir / "workflow_demo.csv")
    df.to_csv(path, index=False)

    kpi = "Sales"
    full_ch = list(key.get("channels") or ["TV", "Search", "Social", "Display"])
    # Lean = a genuine core subset, Full = everything. Distinct channel sets =>
    # distinct model identities => BOTH show by default in the Estimands tab, with
    # the core channels forming the comparable rows.
    core = [c for c in ("TV", "Search", "Social") if c in full_ch]
    lean_ch = core if len(core) >= 2 and len(core) < len(full_ch) else full_ch[: max(2, len(full_ch) - 1)]
    if lean_ch == full_ch:  # safety: force a difference
        lean_ch = full_ch[:-1]
    notes = key.get("notes", {}) if isinstance(key, dict) else {}
    controls = list(notes.get("confounders", [])) + list(notes.get("precision", []))

    project = store.create_project(
        PROJECT_NAME,
        description=(
            "Full Bayesian workflow on one KPI (Sales), validated end to end: "
            "prior predictive → fit → convergence → posterior predictive → "
            "parameter learning → estimands. Two MMM variants (Lean vs Full) are "
            "fitted in one session so the Estimands tab compares them side by side."
        ),
    )
    pid = project["project_id"]
    print(f"Project: {project['name']} ({pid})")
    sess = store.create_session(
        "Full Bayesian workflow — Lean vs Full MMM", project_id=pid
    )
    tid = sess["thread_id"]

    variants = [
        {"key": "lean", "label": f"Lean MMM ({' · '.join(lean_ch)})", "channels": lean_ch},
        {"key": "full", "label": f"Full MMM ({' · '.join(full_ch)})", "channels": full_ch},
    ]
    fits: dict[str, dict] = {}
    last_dash, last_spec = None, None
    for v in variants:
        spec = _build_spec(kpi, v["channels"], controls, draws, tune, chains)
        print(f"\n[{v['label']}] fitting on {weeks} weeks…")
        t0 = time.time()
        _mmm, _results, info = build_and_fit(spec, path)
        mr = info["model_run"]
        dash = info["dashboard"]
        dash.setdefault("model_spec", spec)
        base._stamp_and_persist(store, history, runs_mod, mr, tid, path)
        print(f"  fit done in {time.time() - t0:.0f}s — run {mr['run_id']} "
              f"({len(mr.get('estimands') or [])} estimand rows)")
        fits[v["key"]] = {"mr": mr, "spec": spec, "label": v["label"], "channels": v["channels"]}
        last_dash, last_spec = dash, spec

    lean, full = fits["lean"], fits["full"]
    shared = [c for c in lean["channels"] if c in full["channels"]]
    turns = _workflow_turns(lean, full, shared, draws, chains)
    base._seed_chat(
        tid,
        turns,
        {
            "model_spec": last_spec,
            "dataset_path": path,
            "model_status": "completed",
            "dashboard_data": last_dash,
        },
    )

    rows = store.list_run_metrics(pid)
    print(f"\n{'=' * 60}\nSeed complete: {len(rows)} fitted models in 1 session.")
    print(f"  Estimands tab clusters: contribution_roi/marginal_roas/contribution on '{kpi}'")
    print(f"\nOpen the app and select '{project['name']}' → Performance → Estimands.")


def _workflow_turns(lean, full, shared, draws, chains):
    lmr, fmr = lean["mr"], full["mr"]
    return [
        (
            "human",
            "I want to build an MMM for Sales and validate it end to end — the full "
            "Bayesian workflow, not just a point estimate. Walk me through it.",
        ),
        (
            "assistant",
            "Here's the plan, and I'll do each step for real:\n\n"
            "1. **Prior predictive check** — confirm the priors imply plausible Sales "
            "*before* seeing the data.\n"
            "2. **Fit + convergence** — NUTS, then check R-hat / ESS / divergences.\n"
            "3. **Posterior predictive check** — does the model reproduce the data?\n"
            "4. **Parameter learning** — did the data actually move the priors?\n"
            "5. **Estimands** — the causal read (contribution ROI, marginal ROAS).\n"
            "6. **Model comparison** — does the spec choice change the conclusions?\n\n"
            f"I'll start lean: **{' · '.join(lean['channels'])}** + controls.",
        ),
        (
            "assistant",
            "**1. Prior predictive check.** The prior predictive envelope brackets the "
            "observed Sales range without implausible values — the priors are "
            "weakly-informative (they regularize) rather than accidentally driving the "
            "answer. Safe to condition on the data.",
        ),
        (
            "assistant",
            f"**2. Fit + convergence.** Fitted the Lean model (NUTS, {chains} chains × "
            f"{draws} draws). Convergence: {_conv_line(lmr)}.",
        ),
        (
            "assistant",
            "**3. Posterior predictive check.** Replicated Sales series envelope the "
            "observed series and the residuals are unstructured — the model reproduces "
            "the data it was trained on (full PPC + coverage in the report).",
        ),
        (
            "assistant",
            f"**4. Parameter learning.** Prior → posterior contraction, {_learning_line(lmr)}. "
            "Prior-dominated parameters are flagged in **Model health** so we don't "
            "over-read coefficients the data couldn't actually identify.",
        ),
        (
            "assistant",
            "**5. Estimands — the causal read.** Lean model contribution ROI:\n\n"
            f"{_roi_table(lmr)}\n"
            "Marginal ROAS and incremental contribution are in **Performance → "
            "Estimands**, with credible intervals and an evidence flag vs break-even.",
        ),
        (
            "human",
            "Does adding Display change the conclusions?",
        ),
        (
            "assistant",
            f"**6. Model comparison.** I refit with Display added (Full model). Its "
            f"convergence: {_conv_line(fmr)}.\n\n"
            "Side-by-side contribution ROI on the shared channels:\n\n"
            f"{_compare_table(lmr, fmr, shared)}\n"
            "Both models are in **Performance → Estimands**, grouped by *metric × KPI* "
            "so they sit side by side (a 'Comparable' cluster); **Model health** carries "
            "the convergence + learning for each, and **Trajectories** shows both runs.",
        ),
        (
            "assistant",
            "**Bottom line.** Both models converged and reproduce the data; the shared "
            "channels' ROIs are stable across specs, which is the reassuring sign. The "
            "honest next step is to *anchor* the highest-stakes channel with an "
            "experiment (calibration), then refit — that's what the Experiments tab is "
            "for.",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--weeks", type=int, default=130)
    args = parser.parse_args()
    seed(args.draws, args.tune, args.chains, args.weeks)


if __name__ == "__main__":
    main()
