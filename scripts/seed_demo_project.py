"""Seed a demo project with a multi-cycle measurement program.

Builds the full T₀–T₅ story from a synthetic world with known causal truth:
each quarterly cycle fits the MMM on the data observed so far, computes
EIG/EVOI experiment priorities, pre-registers and "runs" the top experiments
(readouts drawn from the DGP answer key + noise), and calibrates them into the
next fit. The result is a populated experiment registry + run-metrics history
demonstrating ROI CI contraction, budget-share migration, falling
misallocation cost, and rising portfolio mROI — the showcase behind the
Program / Experiments / Performance pages.

Run from the repo root (models auto-save to ./mmm_models):

    uv run python scripts/seed_demo_project.py                  # national demo (~5-15 min)
    uv run python scripts/seed_demo_project.py --geo            # 8-DMA geo-lift demo
    uv run python scripts/seed_demo_project.py --synthetic-records  # no MCMC (seconds)

The geo demo ("Demo: Geo Lift Program") fits geo-level models on a
heterogeneous 8-DMA panel and registers experiments with REAL design-studio
payloads: randomized matched-pair assignment, DiD power curves, placebo bars —
so the Experiments drawer and design studio render fully.

Re-running REPLACES the existing demo project (its sessions, experiments, and
run metrics are removed first) so the project switcher never stacks duplicates.
Each session also gets a scripted copilot conversation + live workspace state
(spec, ROI, decomposition) written into the LangGraph checkpointer, so opening
a demo session shows the full chat-aided workflow.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

BASE_WEEKS = 52
WEEKS_PER_CYCLE = 13
EXPERIMENTS_PER_CYCLE = 2

# Two demo programs. Hero channels: rescale the recorded spend UNITS down by
# these factors — the channel's baked-in contribution to Sales is untouched,
# so its true ROAS rises by the same factor (a self-consistent world where
# those media clear 2.4x; the model only ever sees the rescaled series).
MODES: dict[str, dict] = {
    "national": {
        "project_name": "Demo: Adaptive Measurement Loop",
        "scenario": "realistic",
        "geographies": None,
        "kpi_level": "national",
        "world_seed": 7,
        "roas_boost": {"Video": 1.7, "TV": 3.9, "Social": 4.1},
        "default_cycles": 4,
        "description": (
            "Seeded showcase (realistic national synthetic world, {cycles} "
            "quarterly cycles): baseline fit → EIG/EVOI priorities → "
            "pre-registered experiments → calibrated refits. Ground truth in "
            "demo_data/synthetic_truth_national.json."
        ),
    },
    "geo": {
        "project_name": "Demo: Geo Lift Program",
        "scenario": "geo_heterogeneous",
        "geographies": [
            "NYC",
            "LA",
            "Chicago",
            "Dallas",
            "Houston",
            "Atlanta",
            "Phoenix",
            "Seattle",
        ],
        "kpi_level": "geo",
        "world_seed": 11,
        "roas_boost": {"Social": 9.0, "Search": 10.5},
        "default_cycles": 3,
        "description": (
            "Seeded showcase (heterogeneous 8-DMA synthetic world, {cycles} "
            "quarterly cycles): geo-level fits → EIG/EVOI priorities → "
            "RANDOMIZED matched-pair geo-lift experiments (designs from the "
            "design studio, with DiD power analysis) → calibrated refits. "
            "Ground truth incl. per-DMA ROAS in demo_data/"
            "synthetic_truth_geo.json."
        ),
    },
}


def _mode_controls(mode: str, key: dict) -> list[str]:
    if mode == "geo":
        return ["Price"]
    return list(key["notes"]["confounders"]) + list(key["notes"]["precision"])


def _build_spec(
    channels: list[str],
    controls: list[str],
    draws: int,
    tune: int,
    chains: int,
    kpi_level: str = "national",
) -> dict:
    return {
        "kpi": "Sales",
        "kpi_level": kpi_level,
        "time_granularity": "weekly",
        "media_channels": [{"name": c} for c in channels],
        "control_variables": [{"name": c} for c in controls],
        "seasonality": {"yearly": 2},
        "trend": {"type": "linear"},
        "inference": {
            "chains": chains,
            "draws": draws,
            "tune": tune,
            "target_accept": 0.9,
            "random_seed": 42,
        },
    }


def _write_prefix_csv(
    df, periods: list, n: int, out_dir: Path, cycle: int, prefix: str
) -> str:
    """Write the MFF rows for the first ``n`` periods (the data observed by
    cycle ``cycle``)."""
    keep = set(periods[:n])
    path = out_dir / f"demo_{prefix}_mff_cycle{cycle}.csv"
    df[df["Period"].isin(keep)].to_csv(path, index=False)
    return str(path)


def _studio_design(
    mode: str,
    dataset_path: str,
    channel: str,
    fallback_se: float,
    true_roas: dict,
    seed: int,
) -> tuple[dict, float, str]:
    """(design payload, readout se, method) for a planned experiment.

    Geo mode computes the REAL design-studio output (matched pairs, randomized
    assignment, DiD power curve) on the data observed at planning time, so the
    demo registry entries render fully in the studio/drawer. The readout se is
    the design's achievable precision, floored at a useful relative precision
    (the simulated team extends the test until calibration genuinely bites).
    """
    if mode == "geo":
        from mmm_framework.planning.design import geo_lift_design

        design = geo_lift_design(
            dataset_path,
            "Sales",
            channel,
            design="scaling",
            intensity_pct=50.0,
            duration=8,
            seed=seed,
        )
        design["hypothesis"] = (
            f"{channel} ROAS is mis-estimated by the correlational fit"
        )
        se = float(
            min(design["se_roas"], 0.2 * abs(true_roas.get(channel, 1.0)) + 0.02)
        )
        pairs = ", ".join(
            f"{p['treatment']}→T/{p['control']}→C" for p in design["assignment"]
        )
        return design, se, f"randomized geo lift DiD (simulated; pairs {pairs})"
    design = {
        "design_type": "geo holdout / geo lift test",
        "design_key": "geo_holdout",
        "hypothesis": f"{channel} ROAS is mis-estimated by the correlational fit",
        "min_duration_periods": 8,
        "target_se": fallback_se,
    }
    return design, fallback_se, "geo holdout DiD (simulated)"


def _stage_current_portfolio(
    store,
    pid: str,
    tid: str,
    mode: str,
    path: str,
    channels: list[str],
    true_roas: dict,
    periods: list,
    run_id: str,
    world_seed: int,
    rng,
) -> list[dict]:
    """Leave the program mid-flight after the final fit: one experiment in
    each pre-calibration state (completed → running → planned → draft), on the
    channels whose calibrated evidence is OLDEST, with windows at the live
    edge of the calendar. This is what a real program looks like between
    cycles — and the 'completed' readout is the next second-calibration
    waiting to happen (it drives the calibrate next-action on the Program
    page)."""
    iso = lambda p: str(p)[:10]  # noqa: E731
    evidence = store.latest_calibrated_evidence(pid)
    by_age = sorted(
        (c for c in channels if c in evidence),
        key=lambda c: evidence[c].get("end_date") or "",
    )
    states = ["completed", "running", "planned", "draft"]
    staged: list[dict] = []
    for state, name in zip(states, by_age):
        old_end = evidence[name].get("end_date") or "the prior test"
        sigma_fallback = 0.15 * abs(true_roas.get(name, 1.0)) + 0.02
        design_payload, sigma_exp, method = _studio_design(
            mode, path, name, sigma_fallback, true_roas, seed=world_seed + 97
        )
        # honest framing: only claim decay when the evidence is actually old
        try:
            age_weeks = (
                np.datetime64(str(periods[-1])[:10]) - np.datetime64(old_end)
            ) / np.timedelta64(1, "W")
        except Exception:  # noqa: BLE001
            age_weeks = 999.0
        design_payload["hypothesis"] = (
            f"Re-test: {name} evidence from {old_end} has decayed — verify "
            "current effectiveness before the next allocation."
            if age_weeks >= 26
            else f"Rolling roadmap: scheduled re-validation of {name} "
            f"(prior evidence: {old_end})."
        )
        exp = store.upsert_experiment(
            project_id=pid,
            thread_id=tid,
            channel=name,
            design_type=design_payload.get("design_key", "geo_holdout"),
            status="draft",
            recommending_run_id=run_id,
            design=design_payload,
        )
        entry = {"channel": name, "state": state, "prior_evidence_end": old_end}
        if state == "draft":
            staged.append(entry)
            continue
        store.transition_experiment(exp["id"], "planned", note="pre-registered")
        if state == "planned":
            # launches next quarter: window starts after the data edge
            entry["window"] = "next quarter"
            staged.append(entry)
            continue
        store.transition_experiment(exp["id"], "running", note="launched")
        w_start = periods[-8]
        w_end = periods[-1]
        if state == "running":
            store.upsert_experiment(
                experiment_id=exp["id"], start_date=iso(periods[-3])
            )
            entry["window"] = f"started {iso(periods[-3])}"
            staged.append(entry)
            continue
        # completed: fresh readout over the last 8 observed weeks — the next
        # calibration (a second one for this channel) is ready to apply
        value = float(true_roas[name] * (1.0 + rng.normal(0.0, 0.05)))
        store.transition_experiment(
            exp["id"],
            "completed",
            value=value,
            se=sigma_exp,
            estimand="roas",
            start_date=iso(w_start),
            end_date=iso(w_end),
            readout={"method": method, "value": value, "se": sigma_exp},
            note=method,
        )
        entry.update(
            {
                "window": f"{iso(w_start)} → {iso(w_end)}",
                "value": value,
                "se": sigma_exp,
            }
        )
        staged.append(entry)
    return staged


def _endgame_turns(staged: list[dict]) -> list[tuple[str, str]]:
    if not staged:
        return []
    label = {
        "completed": "readout in — awaiting its SECOND calibration",
        "running": "in market now",
        "planned": "pre-registered, launches next quarter",
        "draft": "designed in the studio, pending sign-off",
    }
    lines = []
    for s in staged:
        extra = f" ({s['window']})" if s.get("window") else ""
        readout = (
            f" — measured **{s['value']:.2f} ± {s['se']:.2f}**"
            if s.get("value") is not None
            else ""
        )
        lines.append(
            f"- `{s['channel']}`: {label[s['state']]}{extra}{readout}; prior "
            f"evidence from {s['prior_evidence_end']}"
        )
    return [
        ("human", "Where does the experiment portfolio stand going into next quarter?"),
        (
            "ai",
            "Every channel is experiment-backed, so the portfolio has rotated to "
            "re-tests — oldest evidence first, as the information decay schedule "
            "dictates:\n\n" + "\n".join(lines) + "\n\n"
            "When you're ready, say the word and I'll apply the completed readout "
            "as a calibration likelihood (apply_experiment_calibration) and refit — "
            "that channel's second calibration. The lifecycle board on the "
            "Experiments page shows all of these in their columns.",
        ),
    ]


def _stamp_and_persist(
    store, history, runs_mod, model_run: dict, tid: str, path: str
) -> None:
    """Replicate fit_mmm_model's host-side close-out: lineage + artifact +
    run-metrics persistence (the seed calls build_and_fit directly)."""
    model_run["data_fingerprint"] = runs_mod.data_fingerprint(path)
    model_run["spec_hash"] = runs_mod.spec_hash(model_run.get("spec"))
    prior_runs = [a for a in store.list_artifacts(tid) if a.get("kind") == "model_run"]
    model_run["parent_run_id"] = (
        prior_runs[-1]["payload"].get("run_id") if prior_runs else None
    )
    store.add_artifact(tid, "model_run", model_run)
    history.persist_run_metrics(model_run, tid)


# ── Demo chat seeding ─────────────────────────────────────────────────────────
# Each demo session gets a believable agent conversation + the live workspace
# state (model_spec / dashboard_data / model_status), written straight into the
# LangGraph checkpointer (same sessions.db the server uses), so opening a demo
# session shows the chat AND populated Model/Results/Experiments tabs.


class _StubLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, *_a, **_kw):  # pragma: no cover - never called
        raise RuntimeError("stub")


def _seed_chat(thread_id: str, turns: list[tuple[str, str]], extras: dict) -> None:
    import asyncio

    from langchain_core.messages import AIMessage, HumanMessage

    async def _run() -> None:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from mmm_framework.agents.graph import create_agent_graph
        from mmm_framework.agents.serde import MsgpackSafeSerializer
        from mmm_framework.api import sessions as store
        from mmm_framework.api.main import safe_json_dumps_load

        conn = await aiosqlite.connect(str(store.DB_PATH))
        try:
            memory = AsyncSqliteSaver(conn, serde=MsgpackSafeSerializer())
            await memory.setup()
            graph = create_agent_graph(_StubLLM(), checkpointer=memory)
            messages = [
                (
                    HumanMessage(content=text)
                    if role == "human"
                    else AIMessage(content=text)
                )
                for role, text in turns
            ]
            # numpy scalars in the dashboard payload would break msgpack
            await graph.aupdate_state(
                {"configurable": {"thread_id": thread_id}},
                {"messages": messages, **safe_json_dumps_load(extras)},
                as_node="agent",
            )
        finally:
            await conn.close()

    asyncio.run(_run())


def _roi_lines(metrics: dict, top: int = 4) -> str:
    chs = sorted(
        (metrics.get("channels") or {}).items(),
        key=lambda kv: kv[1].get("roi_mean") or 0,
        reverse=True,
    )[:top]
    return "\n".join(
        f"- `{name}`: ROI **{m['roi_mean']:.2f}** "
        f"(90% CI [{m['roi_hdi_low']:.2f}, {m['roi_hdi_high']:.2f}])"
        for name, m in chs
    )


def _baseline_turns(
    model_run: dict,
    planned: list[tuple[str, dict]],
    designs: dict[str, dict] | None = None,
    mode: str = "national",
) -> list[tuple[str, str]]:
    metrics = model_run.get("metrics") or {}
    port = metrics.get("portfolio") or {}
    if mode == "geo":
        first_human = (
            "I've loaded this quarter's DMA-level MFF panel (8 markets, weekly). "
            "Configure a geo-level MMM for Sales with the four media channels "
            "and the price control, run the prior predictive check, then fit "
            "the baseline."
        )
    else:
        first_human = (
            "I've loaded this quarter's MFF dataset. Configure a national weekly "
            "MMM for Sales — all seven media channels, the category-demand and "
            "distribution confounders, and the price/competitor/weather/holiday "
            "precision controls — run the prior predictive check, then fit the "
            "baseline."
        )
    turns = [
        ("human", first_human),
        (
            "ai",
            "Baseline fitted ✅ — convergence clean (0 divergences, R-hat < 1.01).\n\n"
            f"**Posterior ROI, top channels**\n{_roi_lines(metrics)}\n\n"
            f"Portfolio marginal ROI is **{port.get('portfolio_marginal_roi', 0):.2f}**, "
            f"and the optimizer leaves ~**{port.get('expected_uplift', 0):,.0f}** KPI units "
            "on the table vs the current allocation — but several of these ROI "
            "posteriors are wide, so that reallocation is not yet decision-grade. "
            "This is a correlational fit: the demand confounder is controlled, but "
            "I'd treat the headline ROIs as provisional until we anchor the big "
            "channels with experiments.\n\n"
            "Next step (T₁): compute EIG/EVOI experiment priorities.",
        ),
        ("human", "Which experiments should we run next quarter?"),
    ]
    if planned:
        body = "\n".join(
            _planned_line(i, name, m, (designs or {}).get(name))
            for i, (name, m) in enumerate(planned, 1)
        )
        turns.append(
            (
                "ai",
                "I scored every channel on what an experiment would teach (EIG) "
                "times what that learning is worth to the budget decision (EVOI). "
                f"The portfolio for next quarter:\n\n{body}\n\n"
                "Both are pre-registered in the experiment registry (design locked "
                "before launch). When the readouts land, I'll calibrate them into "
                "the next fit as likelihood terms — check the **Experiments** tab "
                "or the Experiments page to track them.",
            )
        )
    return turns


def _planned_line(i: int, name: str, m: dict, design: dict | None) -> str:
    base = (
        f"**{i}. `{name}`** — {m.get('quadrant', 'test_now')}; "
        f"EIG {m.get('eig', 0):.2f} nats, EVOI {m.get('evoi', 0):,.0f} KPI units."
    )
    if design and design.get("assignment"):
        pairs = "; ".join(
            f"**{p['treatment']}** (T) vs {p['control']} (C)"
            for p in design["assignment"]
        )
        return (
            f"{base} Randomized matched-pair geo lift, +"
            f"{design.get('intensity_pct', 50):.0f}% spend in treated DMAs for "
            f"{design.get('duration', 8)} weeks — {pairs}. "
            f"MDE ≈ {design.get('mde_roas', 0):.2f} ROAS at 80% power."
        )
    return f"{base} Geo holdout, ≥ 8 periods, target SE ≤ {m.get('sigma_exp', 0):.2f}."


def _cycle_turns(
    t: int,
    model_run: dict,
    calibrated: list[dict],
    planned: list[tuple[str, dict]],
    prev_metrics: dict | None,
    designs: dict[str, dict] | None = None,
) -> list[tuple[str, str]]:
    metrics = model_run.get("metrics") or {}
    port = metrics.get("portfolio") or {}
    names = ", ".join(
        f"`{r['measurement']['channel']}`" + (" (re-test)" if r.get("retest") else "")
        for r in calibrated
    )
    readouts = "\n".join(
        f"- `{r['measurement']['channel']}`: measured ROAS "
        f"**{r['measurement']['value']:.2f} ± {r['measurement']['se']:.2f}**"
        + (
            f" — re-test; supersedes the {r['prior_evidence_end']} evidence"
            if r.get("retest") and r.get("prior_evidence_end")
            else ""
        )
        for r in calibrated
    )
    moves = ""
    if prev_metrics:
        lines = []
        for r in calibrated:
            ch = r["measurement"]["channel"]
            before = (prev_metrics.get("channels") or {}).get(ch) or {}
            after = (metrics.get("channels") or {}).get(ch) or {}
            if before and after:
                lines.append(
                    f"- `{ch}`: ROI {before.get('roi_mean', 0):.2f} → "
                    f"**{after.get('roi_mean', 0):.2f}**; CI width "
                    f"{before.get('ci_width', 0):.2f} → **{after.get('ci_width', 0):.2f}**"
                )
        if lines:
            moves = (
                "\n\n**Belief revision (prior fit → calibrated fit)**\n"
                + "\n".join(lines)
            )
    turns = [
        (
            "human",
            f"Quarter {t} data is in, and the {names} experiment readouts landed:\n"
            f"{readouts}\n\nCalibrate them into the model and refit.",
        ),
        (
            "ai",
            f"Recorded the readouts, staged them as in-graph likelihood terms, and "
            f"refit on the extended panel. The registry entries are now **calibrated** "
            f"and linked to this run.{moves}"
            + (
                "\n\nNote the **second calibration**: the re-tested channel now has "
                "two experiment anchors — the fresh one supersedes the decayed "
                "evidence in the coverage map, and its re-test clock restarts. "
                "This is the loop working as designed: calibrate, decay, re-test, "
                "re-calibrate."
                if any(r.get("retest") for r in calibrated)
                else ""
            )
            + f"\n\nMean ROI CI width across channels is now "
            f"**{port.get('mean_ci_width', 0):.2f}**, portfolio marginal ROI "
            f"**{port.get('portfolio_marginal_roi', 0):.2f}**. "
            "The Performance page has the full cycle-over-cycle trajectory.",
        ),
    ]
    if planned:
        body = "\n".join(
            _planned_line(i, name, m, (designs or {}).get(name))
            for i, (name, m) in enumerate(planned, 1)
        )
        turns.append(("human", "What should we test next?"))
        turns.append(
            (
                "ai",
                "Re-scored the grid with the tightened posteriors — calibrated "
                f"channels' EIG dropped, so the next portfolio is:\n\n{body}\n\n"
                "Designs are pre-registered in the registry; launch when the "
                "quarter starts.",
            )
        )
    return turns


def seed(
    cycles: int,
    draws: int,
    tune: int,
    chains: int,
    synthetic: bool,
    mode: str = "national",
) -> None:
    from mmm_framework.api import history, sessions as store
    from mmm_framework.api import runs as runs_mod
    from mmm_framework.synth import generate_mff

    store.init_db()
    rng = np.random.default_rng(MODES[mode]["world_seed"])

    cfg = MODES[mode]
    total_weeks = BASE_WEEKS + WEEKS_PER_CYCLE * cycles
    df, key = generate_mff(
        cfg["scenario"],
        seed=cfg["world_seed"],
        n_weeks=total_weeks,
        geographies=cfg["geographies"],
    )
    channels = list(key["channels"])
    controls = _mode_controls(mode, key)

    # Rescale hero channels' spend units so their true ROAS clears 2.4x.
    for ch, boost in cfg["roas_boost"].items():
        if ch in channels:
            mask = df["VariableName"] == ch
            df.loc[mask, "VariableValue"] = df.loc[mask, "VariableValue"] / boost
            key["true_roas"][ch] = key["true_roas"][ch] * boost
            if "true_roas_by_geo" in key:  # uniform unit rescale: per-geo ROAS ×k too
                for cell, row in key["true_roas_by_geo"].items():
                    if ch in row:
                        row[ch] = row[ch] * boost
    true_roas = key["true_roas"]
    print(
        "True ROAS (after hero rescale): "
        + ", ".join(f"{c}={v:.2f}" for c, v in sorted(true_roas.items()))
    )

    # Shift the synthetic calendar so the panel ENDS this week: experiment
    # windows then live in the recent past, early calibrations are genuinely
    # 1–2 years old (information decay is real, not cosmetic), and the staged
    # in-flight experiments below carry current dates.
    import pandas as pd

    dates = pd.to_datetime(df["Period"])
    today = pd.Timestamp.now().normalize()
    target_last = today - pd.Timedelta(days=int(today.weekday()))  # this Monday
    offset = target_last - dates.max()
    df["Period"] = (dates + offset).dt.strftime("%Y-%m-%d")
    periods = sorted(df["Period"].unique())

    out_dir = REPO_ROOT / "demo_data"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"synthetic_truth_{mode}.json").write_text(
        __import__("json").dumps(key, indent=2)
    )

    # Replace any prior demo project (sessions, experiments, run metrics) so
    # re-seeding doesn't stack duplicates in the project switcher.
    demo_name = cfg["project_name"]
    for old in store.list_projects():
        if old.get("name") != demo_name:
            continue
        old_pid = old["project_id"]
        for s in store.list_sessions(project_id=old_pid):
            store.delete_session(s["thread_id"])
        for e in store.list_experiments(project_id=old_pid):
            store.delete_experiment(e["id"])
        with store._conn() as c:  # run_metrics has no public delete; demo-only cleanup
            c.execute("DELETE FROM run_metrics WHERE project_id = ?", (old_pid,))
        store.delete_project(old_pid)
        print(f"Replaced prior demo project {old_pid}")

    project = store.create_project(
        demo_name, description=cfg["description"].format(cycles=cycles)
    )
    pid = project["project_id"]
    print(f"Project: {project['name']} ({pid})")

    pending_readouts: list[dict] = []  # experiments completed, awaiting calibration
    prev_metrics: dict | None = None  # last cycle's metrics, for belief-revision chat
    for t in range(cycles + 1):  # cycle 0 = baseline, then `cycles` refreshes
        n_weeks = BASE_WEEKS + WEEKS_PER_CYCLE * t
        if n_weeks > total_weeks:
            break
        label = "Baseline (T₀)" if t == 0 else f"Cycle {t} refresh"
        sess = store.create_session(label, project_id=pid)
        tid = sess["thread_id"]
        path = _write_prefix_csv(df, periods, n_weeks, out_dir, t, mode)
        spec = _build_spec(
            channels, controls, draws, tune, chains, kpi_level=cfg["kpi_level"]
        )

        # ── record readouts that "landed" since the last cycle, stage calibration
        calibrated_ids: list[str] = []
        consumed_readouts: list[dict] = []
        if pending_readouts:
            spec["experiments"] = [r["measurement"] for r in pending_readouts]
            spec["experiment_ids"] = [r["id"] for r in pending_readouts]
            calibrated_ids = [r["id"] for r in pending_readouts]
            consumed_readouts = pending_readouts
            pending_readouts = []

        print(
            f"\n[{label}] fitting on {n_weeks} weeks "
            f"({len(spec.get('experiments', []))} calibration likelihood(s))…"
        )
        t0 = time.time()
        if synthetic:
            model_run = _synthetic_model_run(t, spec, path, channels, true_roas, rng)
            dashboard = {
                "model_status": "completed",
                "summary": model_run["summary"],
                "model_run": model_run,
            }
        else:
            from mmm_framework.agents.fitting import build_and_fit

            _mmm, _results, info = build_and_fit(spec, path)
            model_run = info["model_run"]
            dashboard = info["dashboard"]
        print(f"  fit done in {time.time() - t0:.0f}s — run {model_run['run_id']}")
        # the workspace Model tab reads the spec from dashboard_data.model_spec
        dashboard.setdefault("model_spec", spec)

        _stamp_and_persist(store, history, runs_mod, model_run, tid, path)

        # close the loop: registry entries consumed by THIS fit become calibrated
        for eid in calibrated_ids:
            store.transition_experiment(
                eid,
                "calibrated",
                calibrated_run_id=model_run["run_id"],
                note=f"folded into {model_run['run_name']}",
            )
        if calibrated_ids:
            print(f"  calibrated {len(calibrated_ids)} experiment(s) into this run")

        if t == cycles:
            # Final fit: leave the program MID-FLIGHT — one experiment in each
            # pre-calibration state with current dates (the completed readout
            # is the next second-calibration waiting to happen).
            staged = _stage_current_portfolio(
                store,
                pid,
                tid,
                mode,
                path,
                channels,
                true_roas,
                periods[:n_weeks],
                model_run["run_id"],
                cfg["world_seed"],
                rng,
            )
            for s in staged:
                print(f"  staged {s['channel']}: {s['state']}")
            _seed_chat(
                tid,
                _cycle_turns(t, model_run, consumed_readouts, [], prev_metrics)
                + _endgame_turns(staged),
                {
                    "model_spec": spec,
                    "dataset_path": path,
                    "model_status": "completed",
                    "dashboard_data": dashboard,
                },
            )
            break

        # ── T₁/T₂: plan + run the top-priority channels this quarter. Untested
        # channels go first; once every channel has calibrated evidence, the
        # loop moves to RE-TESTS (oldest evidence first) — information decay
        # sending a channel back into the portfolio is the T₅ → T₁ edge, and a
        # re-tested channel's refit is its SECOND calibration.
        metrics = model_run.get("metrics") or {}
        ch_metrics = metrics.get("channels") or {}
        all_exps = store.list_experiments(project_id=pid)
        active = {
            e["channel"]
            for e in all_exps
            if e["status"] in ("planned", "running", "completed")
        }
        evidence = store.latest_calibrated_evidence(pid)
        untested = [c for c in ch_metrics if c not in active and c not in evidence]
        fresh = sorted(
            ((c, ch_metrics[c]) for c in untested),
            key=lambda kv: kv[1].get("priority") or 0,
            reverse=True,
        )[:EXPERIMENTS_PER_CYCLE]
        # fill any remaining slots with re-tests, oldest evidence first
        retest_pool = sorted(
            (c for c in ch_metrics if c not in active and c in evidence),
            key=lambda c: evidence.get(c, {}).get("end_date") or "",
        )
        retests = [
            (c, ch_metrics[c])
            for c in retest_pool[: max(0, EXPERIMENTS_PER_CYCLE - len(fresh))]
        ]
        ranked = fresh + retests
        retest_channels = {c for c, _ in retests}

        # the experiment runs over the first 8 weeks of the NEXT quarter, so
        # its window is inside the next fit's observed data
        w_start = periods[n_weeks]
        w_end = periods[min(n_weeks + 7, total_weeks - 1)]
        iso = lambda p: str(p)[:10]  # noqa: E731
        cycle_designs: dict[str, dict] = {}
        for name, m in ranked:
            sigma_exp = float(m.get("sigma_exp") or 0.1 * abs(true_roas.get(name, 1.0)))
            design_payload, sigma_exp, method = _studio_design(
                mode, path, name, sigma_exp, true_roas, seed=cfg["world_seed"] + t
            )
            is_retest = name in retest_channels
            if is_retest:
                old_end = evidence.get(name, {}).get("end_date") or "the prior test"
                design_payload["hypothesis"] = (
                    f"Re-test: {name} evidence from {old_end} has decayed past "
                    "the information threshold — verify current effectiveness "
                    "before the next allocation."
                )
            cycle_designs[name] = design_payload
            exp = store.upsert_experiment(
                project_id=pid,
                thread_id=tid,
                channel=name,
                design_type=design_payload.get("design_key", "geo_holdout"),
                status="draft",
                recommending_run_id=model_run["run_id"],
                design=design_payload,
                priority={
                    "eig": m.get("eig"),
                    "evoi": m.get("evoi"),
                    "quadrant": m.get("quadrant"),
                    "priority": m.get("priority"),
                },
            )
            store.transition_experiment(exp["id"], "planned", note="pre-registered")
            store.transition_experiment(exp["id"], "running", note="launched")
            # ground-truth readout: true ROAS + design-level measurement noise
            value = float(true_roas[name] * (1.0 + rng.normal(0.0, 0.05)))
            store.transition_experiment(
                exp["id"],
                "completed",
                value=value,
                se=sigma_exp,
                estimand="roas",
                start_date=iso(w_start),
                end_date=iso(w_end),
                readout={"method": method, "value": value, "se": sigma_exp},
                note=method,
            )
            pending_readouts.append(
                {
                    "id": exp["id"],
                    "retest": is_retest,
                    "prior_evidence_end": (
                        evidence.get(name, {}).get("end_date") if is_retest else None
                    ),
                    "measurement": {
                        "channel": name,
                        "test_period": [iso(w_start), iso(w_end)],
                        "value": value,
                        "se": sigma_exp,
                        "estimand": "roas",
                    },
                }
            )
            print(
                f"  experiment {name}: priority {m.get('priority', 0):.2f} "
                f"({m.get('quadrant')}), readout {value:.2f} ± {sigma_exp:.2f}"
            )

        # ── seed the session's chat + workspace state ─────────────────────────
        turns = (
            _baseline_turns(model_run, ranked, cycle_designs, mode)
            if t == 0
            else _cycle_turns(
                t, model_run, consumed_readouts, ranked, prev_metrics, cycle_designs
            )
        )
        _seed_chat(
            tid,
            turns,
            {
                "model_spec": spec,
                "dataset_path": path,
                "model_status": "completed",
                "dashboard_data": dashboard,
            },
        )
        prev_metrics = metrics

    # ── closing summary ─────────────────────────────────────────────────────
    rows = store.list_run_metrics(pid)
    print(f"\n{'='*60}\nSeed complete: {len(rows)} runs with metrics.")
    if rows:
        first, last = rows[0]["metrics"]["portfolio"], rows[-1]["metrics"]["portfolio"]
        print(
            f"  mean ROI CI width : {first['mean_ci_width']:.2f} → {last['mean_ci_width']:.2f}"
        )
        print(
            f"  misallocation     : {first['expected_uplift']:,.0f} → {last['expected_uplift']:,.0f}"
        )
        print(
            f"  portfolio mROI    : {first['portfolio_marginal_roi']:.2f} → {last['portfolio_marginal_roi']:.2f}"
        )
    exps = store.list_experiments(project_id=pid)
    by_status: dict[str, int] = {}
    for e in exps:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    print(f"  experiments       : {by_status}")
    print(f"\nOpen the app and select '{project['name']}' in the header.")


def _synthetic_model_run(t, spec, path, channels, true_roas, rng):
    """--synthetic-records: fabricate a plausible run + metrics without MCMC.
    Tested channels' ROI converges toward truth with contracting CIs."""
    from datetime import datetime, timezone

    run_id = f"demo{t}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    calibrated = {e["channel"] for e in spec.get("experiments", [])}
    ch_payload = {}
    n = len(channels)
    for i, c in enumerate(channels):
        truth = float(true_roas.get(c, 1.0))
        # bias shrinks and CI contracts once the channel is calibrated (and
        # everything tightens slowly as data accumulates)
        tested_bonus = 0.35 if c in calibrated else 1.0
        sd = max(0.05, (0.45 - 0.04 * t) * tested_bonus)
        mean = truth + float(rng.normal(0, 0.3)) * sd
        spend = 1000.0 * (1.0 + 0.5 * np.sin(i))
        eig = 0.5 * np.log1p((sd / max(0.1 * abs(truth), 0.02)) ** 2)
        ch_payload[c] = {
            "spend": spend,
            "spend_share": 1.0 / n,
            "roi_mean": mean,
            "roi_sd": sd,
            "roi_hdi_low": mean - 1.64 * sd,
            "roi_hdi_high": mean + 1.64 * sd,
            "ci_width": 2 * 1.64 * sd,
            "marginal_roi": 0.6 * mean,
            "current_share": 1.0 / n,
            "optimal_share": min(
                0.5, max(0.02, truth / (sum(true_roas.values()) or 1))
            ),
            "share_gap": 0.0,
            "allocation_instability": sd / 2,
            "eig": float(eig),
            "evoi": float(200 * sd * spend / 1000.0),
            "sigma_exp": max(0.1 * abs(truth), 0.02),
            "priority": float(eig * sd),
            "quadrant": "test_now" if sd > 0.25 else "monitor",
        }
    mean_ci = float(np.mean([c["ci_width"] for c in ch_payload.values()]))
    return {
        "run_id": run_id,
        "run_name": f"run_{run_id}",
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "dataset_path": path,
        "kpi": "Sales",
        "channels": channels,
        "spec": spec,
        "summary": f"Synthetic demo record (cycle {t}).",
        "metrics": {
            "schema_version": 1,
            "n_draws": 0,
            "channels": ch_payload,
            "portfolio": {
                "total_spend": float(sum(c["spend"] for c in ch_payload.values())),
                "portfolio_marginal_roi": 1.6 + 0.18 * t,
                "expected_uplift": max(2000.0 * (1 - 0.18 * t), 150.0),
                "uplift_hdi": [0.0, 1.0],
                "prob_positive_uplift": 0.9,
                "v_current": 10000.0,
                "evpi": max(900.0 * (1 - 0.15 * t), 80.0),
                "mean_ci_width": mean_ci,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geo",
        action="store_true",
        help="seed the geo demo: 8-DMA panel, geo-level fits, RANDOMIZED "
        "matched-pair geo-lift experiments with real design-studio payloads",
    )
    parser.add_argument(
        "--cycles", type=int, default=None, help="default: 4 national / 3 geo"
    )
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--synthetic-records",
        action="store_true",
        help="skip MCMC; fabricate plausible run metrics (seconds, for demos/CI)",
    )
    args = parser.parse_args()
    mode = "geo" if args.geo else "national"
    cycles = args.cycles if args.cycles is not None else MODES[mode]["default_cycles"]
    seed(cycles, args.draws, args.tune, args.chains, args.synthetic_records, mode=mode)


if __name__ == "__main__":
    main()
