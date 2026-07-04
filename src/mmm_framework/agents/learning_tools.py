"""Agent tools for continuous-learning programs (the model-free geo bandit).

Thin wrappers around :mod:`mmm_framework.continuous_learning.service` following
the same contract as ``agents/tools.py``: ``@tool`` + injected state/config,
returning a ``Command`` whose update carries a ToolMessage plus
``dashboard_data`` (content-addressed plot/table refs and a
``learning_program`` payload for the UI).

These are **spine** tools — they need NO fitted MMM and no media-channel spec:
a learning program fits the spend→KPI response surface directly from designed
geo experiments and/or past lift-test readouts. Helper imports from
``agents.tools`` happen inside functions (the ``eda_tools`` direction) because
``tools.py`` imports ``LEARNING_TOOLS`` from here at module load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

InjectedConfig = Annotated[RunnableConfig, InjectedToolArg]

_HONESTY_NOTE = (
    "\n\n*Reading the results honestly: trust the FUNDED SET and the ranking; "
    "channel-by-channel magnitudes stay prior-sensitive until a channel has "
    "seen ≥3 distinct spend levels. Keep the geo set stable across waves.*"
)


# ── shared helpers ────────────────────────────────────────────────────────────


def _tid(config) -> str | None:
    from mmm_framework.agents.runtime import get_current_thread
    from mmm_framework.agents.tools import _activate_thread

    return get_current_thread() if config is None else _activate_thread(config)


def _project_id(tid: str | None) -> str | None:
    from mmm_framework.api import sessions as sessions_store

    try:
        sess = sessions_store.get_session(tid) if tid else None
        return (sess or {}).get("project_id")
    except Exception:
        return None


def _msg(text: str, tool_call_id) -> Command:
    return Command(
        update={"messages": [ToolMessage(content=text, tool_call_id=tool_call_id)]}
    )


def _resolve_program(
    project_id: str | None, program_id: str | None
) -> tuple[dict | None, str | None]:
    """(program, error): explicit id, else the project's single ACTIVE program,
    else a clear error listing what exists. A project-less session must pass
    program_id explicitly — never adopt some other project's program from a
    global listing."""
    from mmm_framework.api import sessions as sessions_store

    if program_id:
        prog = sessions_store.get_learning_program(program_id)
        if prog is None or (
            project_id is not None and prog.get("project_id") not in (None, project_id)
        ):
            return None, (
                f"Unknown learning program id '{program_id}' in this project. "
                "Call get_learning_program_status without an id to see what "
                "exists, or start_learning_program to create one."
            )
        return prog, None
    if project_id is None:
        return None, (
            "This session is not attached to a project, so learning programs "
            "cannot be resolved by project — pass program_id explicitly."
        )
    progs = sessions_store.list_learning_programs(
        project_id=project_id, status="active"
    )
    if len(progs) == 1:
        return progs[0], None
    if not progs:
        every = sessions_store.list_learning_programs(project_id=project_id)
        if every:
            names = ", ".join(
                f"`{p['id'][:8]}` ({p.get('name') or 'unnamed'}, {p['status']})"
                for p in every
            )
            return None, (
                f"No ACTIVE learning program in this project (existing: {names}). "
                "Pass program_id explicitly or start a new one."
            )
        return None, (
            "No learning programs in this project yet — call "
            "start_learning_program first."
        )
    names = ", ".join(f"`{p['id']}` ({p.get('name') or 'unnamed'})" for p in progs)
    return None, (
        f"Multiple active learning programs — pass program_id. Available: {names}."
    )


def _economics(
    cl_state, config: dict[str, Any], overrides: dict[str, Any] | None = None
):
    """(margin, population, wave_cost, population_warning) at fit time.

    ``population`` is GEO-PERIODS: an explicit override is final; otherwise
    ``config['horizon_periods']`` (legacy ``population`` read as horizon) is
    scaled by the program's geo count via ``service.resolve_population``.
    """
    from mmm_framework.continuous_learning import service as cl_service

    econ = dict(overrides or {})
    margin = float(econ.get("margin") or config.get("margin") or 1.0)
    population, pop_warning = cl_service.resolve_population(
        cl_state, config, econ.get("population")
    )
    wave_cost = float(
        econ.get("wave_cost")
        if econ.get("wave_cost") is not None
        else (config.get("wave_cost") or 0.0)
    )
    return margin, population, wave_cost, pop_warning


def _fit_save_and_record(
    project_id: str,
    prog: dict[str, Any],
    state,
    *,
    source: str,
    observations: dict | None = None,
    experiment_ids: list[str] | None = None,
    economics: dict | None = None,
) -> dict[str, Any]:
    """Refit on all accumulated evidence, snapshot, persist state + rows.

    Callers must hold ``service.program_lock(prog['id'])`` across their
    load → ingest → this-call span (concurrent fits race on state.npz).
    ``experiment_ids`` must be the SUCCESSFULLY IMPORTED ids only (provenance);
    skipped ids belong in ``observations``.
    """
    from mmm_framework.api import sessions as sessions_store
    from mmm_framework.continuous_learning import service as cl_service

    config = prog.get("config") or {}
    margin, population, wave_cost, pop_warning = _economics(state, config, economics)
    plan_kwargs: dict[str, Any] = {}
    group_budgets = cl_service.group_budgets_for(state, config)
    if group_budgets:
        plan_kwargs["group_budgets"] = group_budgets
    # n_waves counts INGESTED evidence batches (this call adds one) — never
    # fits, so refits/status calls can't inflate the wave count.
    waves = sessions_store.list_learning_waves(prog["id"])
    n_waves = sum(1 for w in waves if w.get("status") == "ingested") + 1
    snapshot = cl_service.fit_and_plan(
        state,
        fit_kwargs=config.get("fit_kwargs"),
        plan_kwargs=plan_kwargs,
        margin=margin,
        population=population,
        wave_cost=wave_cost,
        n_waves=n_waves,
        extra_warnings=[pop_warning] if pop_warning else None,
    )
    path = cl_service.save_program_state(project_id, prog["id"], state)
    sessions_store.record_ingested_wave(
        prog["id"],
        project_id=project_id,
        source=source,
        observations=observations,
        snapshot=snapshot,
        experiment_ids=experiment_ids,
    )
    sessions_store.update_learning_program(
        prog["id"], state_path=path, summary=snapshot
    )
    return snapshot


def _dashboard_payload(prog: dict[str, Any], snapshot: dict | None) -> dict:
    return {
        "program_id": prog["id"],
        "name": prog.get("name"),
        "status": prog.get("status"),
        "channels": prog.get("channels") or [],
        "config": prog.get("config") or {},
        "snapshot": snapshot,
    }


def _funding_records(snapshot: dict) -> list[dict]:
    return [
        {
            "channel": row["channel"],
            "mroas_mean": round(float(row["mroas_mean"]), 3),
            "prob_above_line": round(float(row["prob_above_line"]), 3),
            "verdict": row["verdict"],
            "funded": bool(row["funded"]),
        }
        for row in snapshot.get("funding") or []
    ]


def _allocation_records(snapshot: dict) -> list[dict]:
    rec = snapshot.get("recommendation") or {}
    sd = snapshot.get("allocation_sd") or {}
    curves = snapshot.get("response_curves") or {}
    return [
        {
            "channel": ch,
            "current_dollars": (curves.get(ch) or {}).get("current"),
            "recommended_dollars": round(float(val), 2),
            "sd_dollars": round(float(sd.get(ch, 0.0)), 2),
        }
        for ch, val in rec.items()
    ]


def _snapshot_figures(snapshot: dict) -> list[tuple[str, Any]]:
    """Funding-line bars + response-curve bands, built from the snapshot."""
    import plotly.graph_objects as go

    figs: list[tuple[str, Any]] = []
    funding = snapshot.get("funding") or []
    if funding:
        channels = [r["channel"] for r in funding]
        probs = [float(r["prob_above_line"]) for r in funding]
        fig = go.Figure(
            go.Bar(
                x=channels,
                y=probs,
                text=[r["verdict"] for r in funding],
                textposition="outside",
            )
        )
        fig.add_hline(
            y=0.65, line_dash="dash", annotation_text="FUND above", line_width=1
        )
        fig.add_hline(
            y=0.35, line_dash="dot", annotation_text="CUT below", line_width=1
        )
        fig.update_layout(
            title="Funding line — P(marginal ROAS > break-even) at the recommendation",
            yaxis={"title": "P(value · dR/d$ > 1)", "range": [0, 1.15]},
        )
        figs.append(("Funding line", fig))

    curves = snapshot.get("response_curves") or {}
    if curves:
        fig = go.Figure()
        for ch, curve in curves.items():
            x = curve.get("spend_dollars") or []
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=curve.get("hi") or [],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    legendgroup=ch,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=curve.get("lo") or [],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    opacity=0.3,
                    showlegend=False,
                    legendgroup=ch,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=curve.get("mean") or [],
                    mode="lines",
                    name=ch,
                    legendgroup=ch,
                )
            )
        fig.update_layout(
            title="Incremental response curves (posterior mean, 90% band)",
            xaxis={"title": "Spend per geo-period ($)"},
            yaxis={"title": "Incremental KPI per geo-period"},
        )
        figs.append(("Response curves", fig))
    return figs


def _snapshot_markdown(prog: dict[str, Any], snapshot: dict) -> str:
    ev = snapshot.get("evidence") or {}
    diag = snapshot.get("diagnostics") or {}
    regret = snapshot.get("regret") or {}
    lines = [
        f"### Learning program **{prog.get('name') or prog['id'][:8]}** — wave "
        f"{ev.get('n_waves', '?')} readout",
        "",
        f"Evidence: {ev.get('n_rows', 0)} geo-period rows + "
        f"{ev.get('n_summaries', 0)} past-experiment summaries. "
        f"Diagnostics: R̂={diag.get('max_rhat')}, ESS={diag.get('min_ess')}, "
        f"{diag.get('n_draws')} draws.",
        "",
        "| Channel | Recommended $ per geo-period | ± sd | P(mROAS>1) | Verdict |",
        "|---|---|---|---|---|",
    ]
    rec = snapshot.get("recommendation") or {}
    sd = snapshot.get("allocation_sd") or {}
    verdicts = {r["channel"]: r for r in snapshot.get("funding") or []}
    for ch, val in rec.items():
        v = verdicts.get(ch, {})
        lines.append(
            f"| {ch} | {val:,.0f} | {sd.get(ch, 0.0):,.0f} | "
            f"{v.get('prob_above_line', float('nan')):.2f} | "
            f"**{v.get('verdict', '—')}** |"
        )
    lines.append("")
    stop = bool(regret.get("stop"))
    lines.append(
        f"**Stopping (ENBS):** E[regret] ≈ ${regret.get('e_regret_dollars', 0):,.0f} "
        f"vs wave cost ${regret.get('wave_cost', 0):,.0f} → ENBS "
        f"${regret.get('enbs', 0):,.0f} — "
        + (
            "**stop testing** (another wave no longer pays for itself)."
            if stop
            else "**keep testing** (another wave is still worth its cost)."
        )
    )
    for flag in diag.get("flags") or []:
        lines.append(f"\n⚠️ {flag}")
    for warn in snapshot.get("warnings") or []:
        lines.append(f"\n⚠️ {warn}")
    return "\n".join(lines)


def _publish_snapshot(
    prog: dict[str, Any],
    snapshot: dict,
    state_dict: dict,
    tid: str | None,
    md: str,
) -> tuple[str, dict]:
    """Publish tables + figures + the learning_program dashboard payload."""
    from mmm_framework.agents.eda_tools import _plots_note, _publish_figures
    from mmm_framework.agents.tables import (
        publish_tables,
        records_to_table_json,
        tables_note,
    )

    dashboard_data = dict(state_dict.get("dashboard_data") or {})
    tables = [
        records_to_table_json(
            _allocation_records(snapshot),
            title="Recommended allocation ($ per geo-period)",
            source="learning_program",
        ),
        records_to_table_json(
            _funding_records(snapshot),
            title="Funding line (FUND / HOLD / CUT)",
            source="learning_program",
        ),
    ]
    t_refs, t_dropped = publish_tables(tables, dashboard_data, tid)
    p_refs, p_dropped = _publish_figures(
        _snapshot_figures(snapshot), dashboard_data, tid
    )
    dashboard_data["learning_program"] = _dashboard_payload(prog, snapshot)
    md += tables_note(t_refs, t_dropped) + _plots_note(p_refs, p_dropped)
    return md, dashboard_data


# ── tools ─────────────────────────────────────────────────────────────────────


@tool
def start_learning_program(
    name: str,
    channels: list[str],
    budget_per_period: float,
    value_per_unit: float,
    state: Annotated[dict, InjectedState],
    center: dict = None,
    arms: dict = None,
    activation: str = "hill",
    kpi: str = None,
    wave_cost: float = None,
    horizon_periods: int = 13,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Start a continuous-learning program — a model-free geo response-surface
    bandit that learns how spend drives the KPI DIRECTLY from designed geo
    experiments and/or past lift tests. Use it when there is no usable modeling
    history for an MMM (short panel, new client, evidence = a pile of old lift
    tests) or when the client runs an always-on geo-testing loop.

    **UNITS — read this before setting any dollar figure**: every dollar in a
    program is PER GEO, PER PERIOD. `budget_per_period` and `center` are the $
    ONE test geo spends per period, NOT a national/program total — divide a
    national budget by the number of test geos first. Example: a $2,000,000/
    week national budget over 50 geos is `budget_per_period=40000` ($40k per
    geo-week). Wave designs hand EACH geo a spend cell of this magnitude, and
    recorded wave rows are one geo-period each in the same units.

    Args (all money in dollars PER GEO-PERIOD): `channels` = the spend lines to
    learn; `budget_per_period` = the $ per period one geo's allocation governs;
    `value_per_unit` = $ of value per KPI unit (sets the mROAS funding line);
    `center` = optional {channel: $} current per-geo allocation (default:
    budget split equally); `arms` = optional {channel: [sub-channel names]} to
    split a channel into creative/keyword arms (siblings get substitution
    priors and the parent's budget stays fixed while the mix is free);
    `wave_cost` = $ cost of running one test wave (drives the ENBS stopping
    rule); `horizon_periods` = how many periods the allocation decision governs
    — the ENBS population is horizon_periods × the program's geo count,
    resolved at fit time.

    Follow with: import_past_experiments (leverage completed registry readouts
    — no model needed) and/or design_learning_wave → run it →
    record_learning_wave.
    """
    from mmm_framework.api import sessions as sessions_store
    from mmm_framework.continuous_learning import service as cl_service

    tid = _tid(config)
    project_id = _project_id(tid)
    program_config = {
        "channels": list(channels or []),
        "arms": arms or None,
        "center": center or None,
        "budget": float(budget_per_period),
        "value_per_unit": float(value_per_unit),
        "activation": activation,
        "kpi": kpi,
        "mode": "fixed",
        "margin": 1.0,
        # ENBS population = horizon_periods × geo count, resolved at fit time
        # (service.resolve_population); never pre-bake a population here.
        "horizon_periods": int(horizon_periods),
        "wave_cost": float(wave_cost) if wave_cost is not None else 0.0,
    }
    try:
        cl_state = cl_service.new_program_state(program_config)
    except ValueError as exc:
        return _msg(f"Could not start the learning program: {exc}", tool_call_id)
    prog = sessions_store.create_learning_program(
        project_id=project_id,
        thread_id=tid,
        name=name,
        channels=cl_state.channels,
        config=program_config,
    )
    path = cl_service.save_program_state(project_id or "default", prog["id"], cl_state)
    prog = sessions_store.update_learning_program(prog["id"], state_path=path)

    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["learning_program"] = _dashboard_payload(prog, None)
    arm_note = ""
    if arms:
        arm_note = (
            f" Sub-channel arms: {json.dumps(arms)} — flattened surface "
            f"dimensions: {cl_state.channels}."
        )
    md = (
        f"Learning program **{name}** started (id `{prog['id']}`) over "
        f"{len(cl_state.channels)} channel(s) at ${budget_per_period:,.0f} per "
        f"GEO-period (per-geo dollars, not a national total), "
        f"funding line at ${value_per_unit:g} per KPI unit.{arm_note}\n\n"
        "Next: `import_past_experiments` to fold in completed lift tests as "
        "evidence (no MMM required), or `design_learning_wave` to design the "
        "first geo wave."
    )
    return Command(
        update={
            "messages": [ToolMessage(content=md, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def import_past_experiments(
    state: Annotated[dict, InjectedState],
    program_id: str = None,
    experiment_ids: list[str] = None,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Leverage PAST experiments as learning-program evidence — no MMM needed.
    Pulls the project registry's completed/calibrated lift-test readouts,
    converts each into a summary observation on the response surface
    (`lift ~ Normal(scale · (R(test) − R(base)), se)` — the geo intercept
    cancels, so no pre-period is required), refits the surface on ALL
    accumulated evidence, and reports the funding line + allocation.

    `experiment_ids` limits the import; default = every completed/calibrated
    registry row. Idempotent: experiments already imported into this program
    are skipped ("already imported"), so calling again after new readouts only
    folds in the NEW evidence. Experiments belonging to a different project are
    skipped, never imported. The registry is NOT modified. Sign convention: a
    readout's `spend_per_period` (from record_experiment_readout) is a SIGNED
    per-period spend delta — holdout/dark tests must record it NEGATIVE; when
    it is absent, the design snapshot's spend level is used with the sign
    restored from its holdout marker. Skipped rows are reported with reasons
    (mroas readouts are slopes, not lifts; rows missing value/se or a spend
    level can't convert).
    """
    from mmm_framework.api import sessions as sessions_store
    from mmm_framework.continuous_learning import service as cl_service

    tid = _tid(config)
    project_id = _project_id(tid)
    prog, err = _resolve_program(project_id, program_id)
    if err:
        return _msg(err, tool_call_id)
    proj = prog.get("project_id") or project_id or "default"

    pre_skipped: list[dict] = []
    if experiment_ids:
        # Project scoping: a stale/copy-pasted id from another project must
        # never become surface evidence here.
        prog_project = prog.get("project_id") or project_id
        exps = []
        for x in experiment_ids:
            e = sessions_store.get_experiment(x)
            if e is None:
                pre_skipped.append({"id": x, "reason": "not found in this project"})
            elif (
                prog_project is not None
                and e.get("project_id")
                and e.get("project_id") != prog_project
            ):
                pre_skipped.append(
                    {"id": x, "reason": "experiment belongs to a different project"}
                )
            else:
                exps.append(e)
    else:
        exps = sessions_store.list_experiments(
            project_id=project_id, status="completed"
        ) + sessions_store.list_experiments(project_id=project_id, status="calibrated")
    if not exps and not pre_skipped:
        return _msg(
            "No completed/calibrated experiments found in the registry to import. "
            "Record readouts with record_experiment_readout first.",
            tool_call_id,
        )

    cfg = prog.get("config") or {}
    period_days = 7.0 * float(cfg.get("cadence_weeks") or 1)
    # Hold the program lock across load → ingest → fit → save: a concurrent
    # fit would otherwise clobber state.npz (last-writer-wins evidence loss).
    with cl_service.program_lock(prog["id"]):
        try:
            cl_state = cl_service.load_program_state(proj, prog["id"])
        except cl_service.ProgramStateError as exc:
            return _msg(
                f"Cannot load the learning program: {exc} (use "
                "start_learning_program to recreate it).",
                tool_call_id,
            )
        try:
            report = cl_service.import_experiment_summaries(
                cl_state, exps, period_days=period_days
            )
        except ValueError as exc:
            return _msg(f"Could not import experiments: {exc}", tool_call_id)
        report["skipped"] = pre_skipped + report["skipped"]
        skipped_md = ""
        if report["skipped"]:
            skipped_md = "\n\nSkipped:\n" + "\n".join(
                f"- `{str(s.get('id'))[:8]}`: {s.get('reason')}"
                for s in report["skipped"]
            )
        if not report["imported"]:
            return _msg(
                "None of the experiments could be converted to evidence." + skipped_md,
                tool_call_id,
            )

        snapshot = _fit_save_and_record(
            proj,
            prog,
            cl_state,
            source="experiment_import",
            observations={
                "imported": report["imported"],
                "skipped": report["skipped"],
            },
            # Provenance: only the ids that actually became evidence.
            experiment_ids=list(report.get("imported_ids") or []),
        )
    md = (
        f"Imported **{report['imported']}** past experiment readout(s) as "
        "surface evidence and refit.\n\n"
        + _snapshot_markdown(prog, snapshot)
        + skipped_md
        + _HONESTY_NOTE
    )
    md, dashboard_data = _publish_snapshot(prog, snapshot, state, tid, md)
    return Command(
        update={
            "messages": [ToolMessage(content=md, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def design_learning_wave(
    state: Annotated[dict, InjectedState],
    program_id: str = None,
    delta: float = 0.6,
    probe_pairs: str = "auto",
    n_geo: int = None,
    n_holdout: int = 0,
    optimize: bool = False,
    candidate_deltas: list[float] = None,
    stratify: bool = True,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Design the next learning wave: the central-composite set of geo spend
    cells around the program's current allocation (1 center + 2 axial per
    channel ± delta + 2 off-axis per probed pair + 1 shutoff per channel).
    Axial cells identify each channel's effect, off-axis cells the synergies,
    shutoffs break the beta/gamma collinearity. `delta` is a multiplicative
    spend-variation fraction (0.6 = ±60%). `probe_pairs`: 'auto' (all program
    pairs), 'none', or a JSON list like '[[0,1]]' to probe only the
    decision-pivotal synergies (each pair costs 2 extra cells). `n_geo` adds a
    geo→cell assignment — stratified on the accumulated per-geo KPI when the
    program has ingested data (blocked randomization), else round-robin; set
    `stratify=false` to force round-robin (`n_holdout` geos stay at the
    status-quo center as a counterfactual). `optimize=true` scores
    `candidate_deltas` (default 0.3/0.6/0.9) with the Laplace
    knowledge-gradient (decision-aware EVSI, no refit) and designs the best
    one — needs a fitted panel posterior with its observation sites (any
    registered activation/likelihood family);
    otherwise it falls back to `delta` with a warning. Stores the design on
    the program's wave timeline; after the wave runs, record results with
    record_learning_wave.
    """
    from mmm_framework.api import sessions as sessions_store
    from mmm_framework.continuous_learning import service as cl_service

    tid = _tid(config)
    project_id = _project_id(tid)
    prog, err = _resolve_program(project_id, program_id)
    if err:
        return _msg(err, tool_call_id)
    try:
        cl_state = cl_service.load_program_state(
            prog.get("project_id") or project_id or "default", prog["id"]
        )
    except cl_service.ProgramStateError as exc:
        return _msg(
            f"Cannot load the learning program: {exc} (use "
            "start_learning_program to recreate it).",
            tool_call_id,
        )

    pairs = None
    if probe_pairs and probe_pairs.strip().lower() == "none":
        pairs = []
    elif probe_pairs and probe_pairs.strip().lower() != "auto":
        try:
            pairs = [(int(i), int(j)) for i, j in json.loads(probe_pairs)]
        except (ValueError, TypeError) as exc:
            return _msg(
                f"Could not parse probe_pairs {probe_pairs!r} (use 'auto', 'none', "
                f"or JSON like '[[0,1]]'): {exc}",
                tool_call_id,
            )
    try:
        design = cl_service.design_wave(
            cl_state,
            delta=float(delta),
            probe_pairs=pairs,
            n_geo=n_geo,
            n_holdout=int(n_holdout or 0),
            stratify=bool(stratify),
            optimize=bool(optimize),
            candidate_deltas=(
                [float(d) for d in candidate_deltas] if candidate_deltas else None
            ),
        )
    except ValueError as exc:
        return _msg(f"Could not design the wave: {exc}", tool_call_id)
    wave = sessions_store.add_learning_wave(
        prog["id"],
        project_id=prog.get("project_id") or project_id,
        status="designed",
        source="wave",
        design=design,
    )

    from mmm_framework.agents.tables import (
        publish_tables,
        records_to_table_json,
        tables_note,
    )

    channels = prog.get("channels") or cl_state.channels
    chosen_delta = float(design["delta"])
    records = [
        {
            "cell": design["cell_labels"][i],
            **{
                ch: round(float(design["cells_dollars"][i][c]), 2)
                for c, ch in enumerate(channels)
            },
        }
        for i in range(design["n_cells"])
    ]
    dashboard_data = dict(state.get("dashboard_data") or {})
    refs, dropped = publish_tables(
        [
            records_to_table_json(
                records,
                title=(
                    f"Wave design — {design['n_cells']} cells "
                    f"(±{int(chosen_delta * 100)}%)"
                ),
                source="learning_program",
            )
        ],
        dashboard_data,
        tid,
    )
    md = (
        f"Wave designed for **{prog.get('name') or prog['id'][:8]}** "
        f"(wave row `{wave['id'][:8]}`): {design['n_cells']} cells "
        f"(1 center + {2 * len(channels)} axial + "
        f"{2 * len(design['probe_pairs'])} off-axis + {len(channels)} shutoff), "
        f"delta ±{int(chosen_delta * 100)}%."
    )
    if (design.get("kg") or {}).get("used"):
        scores_md = ", ".join(
            f"±{int(float(s['delta']) * 100)}%"
            f"{' (no probes)' if not s['probe_pairs'] else ''}"
            f" → {float(s['score']):.4g}"
            for s in design["kg"].get("scores") or []
        )
        md += (
            f"\n\nDelta ±{int(chosen_delta * 100)}% chosen by the Laplace "
            f"knowledge-gradient (EVSI per candidate: {scores_md})."
        )
    if design.get("assignment"):
        how = (
            "stratified on the accumulated per-geo KPI (blocked randomization)"
            if design["assignment"].get("stratified_on") == "accumulated_kpi"
            else "round-robin"
        )
        md += (
            f" Assigned {how} to {len(design['assignment']['cell_idx'])} "
            f"geos ({design['assignment']['n_holdout']} holdout)."
        )
    for w in design.get("warnings") or []:
        md += f"\n\n⚠️ {w}"
    md += (
        "\n\nRun the wave (same geos as previous waves!), then call "
        "`record_learning_wave` with the observed geo,spend,y rows."
    )
    md += tables_note(refs, dropped)
    return Command(
        update={
            "messages": [ToolMessage(content=md, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def record_learning_wave(
    state: Annotated[dict, InjectedState],
    program_id: str = None,
    csv_path: str = None,
    rows: list[dict] = None,
    period_col: str = None,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Record a completed learning wave's observed panel and refit the response
    surface on ALL accumulated evidence. Provide either `rows` (a list of
    {geo, <one $ spend column per channel>, y} dicts — one row per geo-period)
    or `csv_path` (a workspace CSV with columns geo,<channel $ columns>,y; an
    optional week column is fine). Spend is in DOLLARS; y in natural KPI units
    (never normalized). For a program with a national time effect
    (time_effect='national'), pass `period_col` naming the week/date/period
    column so each row's period can be indexed (omitted: a period/week/date
    column is auto-detected when present). The geo set must be the SAME as
    previous waves — a changed geo set makes the loop diverge and is rejected.
    Publishes the refreshed funding line (FUND/HOLD/CUT), the recommended
    allocation, and the response curves.
    """
    from mmm_framework.continuous_learning import service as cl_service

    tid = _tid(config)
    project_id = _project_id(tid)
    prog, err = _resolve_program(project_id, program_id)
    if err:
        return _msg(err, tool_call_id)
    proj = prog.get("project_id") or project_id or "default"

    if rows is None and not csv_path:
        return _msg(
            "Provide the wave's observations: `rows` (list of "
            "{geo, <channel $ spends>, y}) or `csv_path` to a workspace CSV.",
            tool_call_id,
        )
    if rows is None:
        from mmm_framework.agents import workspace as _ws

        p = Path(csv_path)
        if not p.is_absolute():
            try:
                p = _ws.safe_join(_ws.thread_dir(tid), csv_path)
            except ValueError as exc:
                return _msg(f"Bad csv_path: {exc}", tool_call_id)
        if not p.exists() or not _ws.is_within(p):
            return _msg(f"CSV not found in the workspace: {csv_path}", tool_call_id)
        try:
            rows = cl_service.rows_from_csv(p.read_text(), period_col=period_col)
        except ValueError as exc:
            return _msg(f"Could not parse the CSV: {exc}", tool_call_id)

    # Hold the program lock across load → ingest → fit → save: a concurrent
    # fit would otherwise clobber state.npz (last-writer-wins evidence loss).
    with cl_service.program_lock(prog["id"]):
        try:
            cl_state = cl_service.load_program_state(proj, prog["id"])
        except cl_service.ProgramStateError as exc:
            return _msg(
                f"Cannot load the learning program: {exc} (use "
                "start_learning_program to recreate it).",
                tool_call_id,
            )
        try:
            ing = cl_service.ingest_wave_rows(cl_state, rows, period_col=period_col)
        except ValueError as exc:
            return _msg(f"Could not ingest the wave: {exc}", tool_call_id)

        snapshot = _fit_save_and_record(
            proj,
            prog,
            cl_state,
            source="wave",
            observations={"n_rows": ing["n_rows"], "n_geo": ing["n_geo"]},
        )
    md = (
        f"Wave recorded: {ing['n_rows']} rows over {ing['n_geo']} geos. "
        "Refit on all accumulated evidence.\n\n"
    )
    for w in ing.get("warnings") or []:
        md += f"⚠️ {w}\n\n"
    md += _snapshot_markdown(prog, snapshot) + _HONESTY_NOTE
    md, dashboard_data = _publish_snapshot(prog, snapshot, state, tid, md)
    return Command(
        update={
            "messages": [ToolMessage(content=md, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def get_learning_program_status(
    state: Annotated[dict, InjectedState],
    program_id: str = None,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Show a learning program's latest readout: the funding line
    (FUND/HOLD/CUT per channel), the recommended allocation with uncertainty,
    the response curves, synergy estimates, and the ENBS stop/continue verdict.
    Reads the stored snapshot — no refit. Call without program_id when the
    project has a single active program."""
    from mmm_framework.api import sessions as sessions_store

    tid = _tid(config)
    project_id = _project_id(tid)
    prog, err = _resolve_program(project_id, program_id)
    if err:
        return _msg(err, tool_call_id)
    snapshot = prog.get("summary")
    dashboard_data = dict(state.get("dashboard_data") or {})
    if not snapshot:
        dashboard_data["learning_program"] = _dashboard_payload(prog, None)
        cfg = prog.get("config") or {}
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Program **{prog.get('name') or prog['id'][:8]}** "
                            f"({prog['status']}) has no fits yet. Channels: "
                            f"{', '.join(prog.get('channels') or [])}; budget "
                            f"${float(cfg.get('budget') or 0):,.0f}/period. Add "
                            "evidence with import_past_experiments or "
                            "design_learning_wave → record_learning_wave."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
                "dashboard_data": dashboard_data,
            }
        )
    waves = sessions_store.list_learning_waves(prog["id"])
    md = _snapshot_markdown(prog, snapshot)
    gamma = [g for g in snapshot.get("gamma") or [] if not g.get("prior_dominated")]
    if gamma:
        md += "\n\nData-informed synergies: " + "; ".join(
            f"{g['pair'][0]}×{g['pair'][1]} = {g['mean']:+.2f} "
            f"[{g['p5']:+.2f}, {g['p95']:+.2f}]"
            for g in gamma
        )
    md += f"\n\nWave timeline: {len(waves)} recorded wave row(s)." + _HONESTY_NOTE
    md, dashboard_data2 = _publish_snapshot(prog, snapshot, state, tid, md)
    dashboard_data.update(dashboard_data2)
    return Command(
        update={
            "messages": [ToolMessage(content=md, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def check_learning_stopping(
    program_id: str = None,
    margin: float = None,
    population: float = None,
    wave_cost: float = None,
    confirm_stop: bool = False,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Re-evaluate the ENBS stopping rule for a learning program, optionally
    with overridden economics: expected net benefit of one more wave =
    E[regret]$ × margin × population − wave_cost (`population` is GEO-PERIODS
    — geos × horizon periods; omitted overrides fall back to the latest fit's
    stored economics). Uses the latest fit's expected regret — no refit. This
    only RECOMMENDS stopping; to confirm, call again with confirm_stop=True
    and the SAME margin/population/wave_cost arguments (an omitted override
    silently falls back to the fit-time value, which can flip the verdict).
    Only pass confirm_stop=true after the user agrees."""
    from mmm_framework.api import sessions as sessions_store
    from mmm_framework.continuous_learning.planner import should_stop

    tid = _tid(config)
    project_id = _project_id(tid)
    prog, err = _resolve_program(project_id, program_id)
    if err:
        return _msg(err, tool_call_id)
    snapshot = prog.get("summary") or {}
    regret = snapshot.get("regret") or {}
    if regret.get("e_regret_kpi") is None:
        return _msg(
            "No fit yet — record a wave or import past experiments first; the "
            "stopping rule needs a posterior to price the remaining uncertainty.",
            tool_call_id,
        )
    e_regret = float(regret["e_regret_kpi"])
    m = float(margin) if margin is not None else float(regret.get("margin") or 1.0)
    p = (
        float(population)
        if population is not None
        else float(regret.get("population") or 1.0)
    )
    w = (
        float(wave_cost)
        if wave_cost is not None
        else float(regret.get("wave_cost") or 0.0)
    )
    stop, enbs_val = should_stop(e_regret, margin=m, population=p, wave_cost=w)
    md = (
        f"**ENBS check** for {prog.get('name') or prog['id'][:8]}: "
        f"E[regret] ${e_regret * m * p:,.0f} (per geo-period regret "
        f"${e_regret:,.2f} × margin {m:g} × population {p:g} geo-periods) vs "
        f"wave cost ${w:,.0f} → **ENBS ${enbs_val:,.0f}**.\n\n"
        f"Economics used: margin={m:g}, population={p:g}, wave_cost={w:g} — a "
        "follow-up call must repeat these EXACT arguments (omitted overrides "
        "fall back to the latest fit's stored economics, not this call's).\n\n"
    )
    if stop:
        if confirm_stop:
            sessions_store.update_learning_program(prog["id"], status="stopped")
            md += (
                "Another wave no longer pays for itself — program marked "
                "**stopped**. Re-open testing later if information decays "
                "(spend drift, creative refresh, seasonality)."
            )
        else:
            md += (
                "Recommendation: **stop testing** — the expected value of "
                "another wave does not cover its cost. Confirm with the user, "
                "then call check_learning_stopping again with "
                "confirm_stop=true AND the same margin/population/wave_cost "
                "arguments to mark the program stopped."
            )
    else:
        md += (
            "Recommendation: **keep testing** — resolving the remaining "
            "allocation uncertainty is still worth more than the wave costs."
        )
    return _msg(md, tool_call_id)


LEARNING_TOOLS = [
    start_learning_program,
    import_past_experiments,
    design_learning_wave,
    record_learning_wave,
    get_learning_program_status,
    check_learning_stopping,
]
