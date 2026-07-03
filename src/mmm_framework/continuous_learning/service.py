"""Host-side orchestration for continuous-learning programs (wiring §3.1).

One shared service layer used by BOTH the agent tools
(:mod:`mmm_framework.agents.learning_tools`) and the REST endpoints
(``api/main.py``'s ``/projects/{pid}/learning-programs`` family). No sqlite in
here — callers own the sessions store; this module owns config validation,
dollar↔scaled conversion, state-file IO, wave design/ingest, past-experiment
import, and the ONE-Thompson-pass ``fit_and_plan`` snapshot.

Units & conversions (the load-bearing decisions, documented once)
-----------------------------------------------------------------
* **Every dollar at this boundary is PER GEO, PER PERIOD.** The surface is fit
  on per-geo-period rows, design cells are per-geo spend vectors, and the
  recommendation is the allocation ONE geo runs each period — so ``budget``
  and ``center`` in the program config are $ per period **per geo**, never a
  program/national total. A $2,000,000/week national budget spread over 50
  test geos is ``budget = 40000`` ($40k per geo-week). ``fit_and_plan`` warns
  when the ingested panel's spend levels sit far from the configured center
  (the tell that totals and per-geo units were conflated).
* **All money at this boundary is dollars.** Internally everything is scaled
  units: ``scaled = dollars / spend_ref`` per channel
  (:func:`~mmm_framework.continuous_learning.scaling.to_scaled`). ``spend_ref``
  defaults to ONE GLOBAL constant for every channel —
  ``max(mean(center dollars), $1)`` — so the fixed-budget simplex, group
  budgets, and the per-channel cap are all EXACT in dollars (a scaled unit
  costs the same $ in every channel). Channel centers then scale to ≠ 1 but
  stay O(1), which the κ ~ O(1) priors tolerate. Per-channel ``spend_ref``
  overrides remain possible via the config; heterogeneous references make the
  budget/groups/cap reference-weighted and draw snapshot warnings.
* **Budget constraint.** The planner enforces ``Σ s_scaled = B_scaled`` with
  ``B_scaled = budget · Σ(center_scaled) / Σ(center_dollars)``. With a uniform
  ``spend_ref`` (the default) this is exactly the dollar simplex
  ``Σ s = budget``; with heterogeneous references it is a reference-weighted
  budget, and a snapshot warning says so.
* **Funding line (per DOLLAR).** ``plan_from_posterior`` evaluates marginal
  ROAS per *scaled* unit; the snapshot converts each channel's draws to
  per-dollar terms (``mroas_scaled / spend_ref[c]``) before applying the
  break-even test ``P(value · dR/d$ > 1)``, so FUND/HOLD/CUT verdicts compare
  dollars to dollars across channels. When the config ``margin`` is < 1 the
  break-even test is applied to MARGIN-adjusted draws
  (``P(margin · value · dR/d$ > 1)`` — marginal *profit* break-even, matching
  the ENBS rule's economics); funding rows keep ``mroas_mean`` per REVENUE
  dollar and add ``mroas_margin_adjusted``, and a warning names any channel
  whose verdict the margin adjustment flipped.
* **ENBS dollar conversion (pinned).** In ``fixed`` mode the per-draw regret is
  a *difference* of profits at the same budget, so the spend term cancels:
  ``regret = value · (R_best − R_consensus)`` — already in VALUE DOLLARS per
  geo-period when ``value`` is $-per-KPI-unit. The snapshot therefore reports
  ``e_regret_kpi`` (per-geo-period profit, value dollars) and converts with
  ``e_regret_dollars = e_regret_kpi × margin × population`` where ``margin``
  maps value dollars to margin dollars and ``population`` is the number of
  GEO-PERIODS the allocation decision governs. Use
  :func:`resolve_population`: it computes ``population = n_geos ×
  horizon_periods`` at fit/stop time (config key ``horizon_periods``
  preferred; a legacy ``population`` config entry is read as horizon periods).
  ``enbs = e_regret_dollars − wave_cost`` — identical to
  :func:`~mmm_framework.continuous_learning.planner.enbs`.
* **``mode="free"`` requires unit spend_ref.** Free-mode profit prices spend
  in SCALED units (``value·R(s) − Σ s_scaled``), so unless a scaled unit is
  exactly $1 the self-funding threshold is off by the reference;
  :func:`new_program_state` rejects ``mode='free'`` with ``spend_ref != 1``.
* **Prior-domination flag (gamma).** A synergy row is flagged
  ``prior_dominated`` when ``|posterior mean| < 0.1 × gamma_scale`` (the
  posterior never escaped a tenth of its prior scale — indistinguishable from
  the prior's shrinkage toward 0) OR the pair's sign prior is ``"zero"``
  (deliberately prior-pinned, e.g. a demoted walled-garden channel).
* **Shape identification.** A channel's response-curve *shape* (κ/α) is only
  identified with ≥ 3 distinct spend levels in the evidence (panel rows +
  summary test/base vectors). Below that the funded set is still trustworthy
  but the curve is prior-dominated — flagged per channel.

Snapshot schema (``schema_version`` 1) is pinned in
``technical-docs/continuous-learning-wiring.md`` §3.1 — the React Sextant page
builds against it byte-for-byte.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from . import planner as _planner
from .arms import default_arm_pair_signs, expand_arms
from .design import assign_geos
from .evidence import experiments_to_summaries
from .loop import LearningState, WaveRecord
from .model import VALID_LIKELIHOODS, VALID_SIGNS, VALID_TIME_EFFECTS, pair_name
from .scaling import to_dollars, to_scaled
from .serialize import state_from_npz, state_to_npz
from .surface import ACTIVATIONS

PROGRAM_STATE_FILENAME = "state.npz"
SNAPSHOT_SCHEMA_VERSION = 1

#: Funding verdicts (story-notebook convention): FUND above, CUT below, HOLD between.
FUND_THRESHOLD = 0.65
CUT_THRESHOLD = 0.35

#: Response-curve grid: 25 points over [0, 2×center] per channel, ≤200 draws.
_CURVE_POINTS = 25
_CURVE_MAX_DRAWS = 200

#: Minimum distinct spend levels for a channel's curve shape to be identified.
_SHAPE_MIN_LEVELS = 3


# ── paths + state-file IO ──────────────────────────────────────────────────────


class ProgramStateError(RuntimeError):
    """A program's ``state.npz`` is missing or unreadable (corrupt/partial).

    Raised by :func:`load_program_state` for BOTH the missing-file and the
    corrupt-file case so every caller degrades to one clean "recreate the
    program" message (endpoints map it to 409; agent tools report it as text)
    instead of a raw ``zipfile.BadZipFile`` 500.
    """


#: Per-program mutexes serializing state IO (load → ingest → fit → save).
#: Single-process posture (in-process job tasks + agent tools both run in
#: threads of one server); a filelock is the upgrade path for multi-worker
#: deployments.
_PROGRAM_LOCKS: dict[str, threading.Lock] = {}
_PROGRAM_LOCKS_GUARD = threading.Lock()


def program_lock(program_id: str) -> threading.Lock:
    """The mutex guarding one program's load→ingest→fit→save critical section.

    Hold it across the WHOLE span in every writer (the REST fit worker and the
    agent tools' record/import paths) — two concurrent fits otherwise race on
    ``state.npz`` and the last writer silently drops the other's evidence.
    """
    with _PROGRAM_LOCKS_GUARD:
        return _PROGRAM_LOCKS.setdefault(str(program_id), threading.Lock())


def program_dir(project_id: str, program_id: str) -> Path:
    """``<workspace_root>/projects/<pid>/learning/<prog>`` (created on demand)."""
    from mmm_framework.agents import workspace as _ws

    d = (
        _ws.workspace_root()
        / "projects"
        / _ws._safe_segment(project_id)
        / "learning"
        / _ws._safe_segment(program_id)
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def delete_program_dir(project_id: str, program_id: str) -> None:
    """Remove a deleted program's on-disk directory (state.npz: the client's
    accumulated geo panel + posterior draws — a retention gap if orphaned).

    Guarded: only removes paths strictly under
    ``<workspace_root>/projects/<pid>/learning/``.
    """
    from mmm_framework.agents import workspace as _ws

    root = (
        _ws.workspace_root() / "projects" / _ws._safe_segment(project_id) / "learning"
    ).resolve()
    d = (root / _ws._safe_segment(program_id)).resolve()
    if d == root or root not in d.parents:
        return
    shutil.rmtree(d, ignore_errors=True)


def load_program_state(project_id: str, program_id: str) -> LearningState:
    """Reload a program's :class:`LearningState` from its ``state.npz``.

    Raises :class:`ProgramStateError` when the file is missing OR corrupt
    (partial write, bad zip, schema mismatch) — never a raw parser exception.
    """
    path = program_dir(project_id, program_id) / PROGRAM_STATE_FILENAME
    if not path.exists():
        raise ProgramStateError(
            f"the program's state file is missing on disk ({path}) — recreate "
            "the program"
        )
    try:
        return state_from_npz(path)
    except Exception as exc:  # noqa: BLE001 — BadZipFile/ValueError/KeyError/OSError
        raise ProgramStateError(
            f"the program's state file is corrupt or unreadable ({path}: {exc}) "
            "— recreate the program"
        ) from exc


def save_program_state(project_id: str, program_id: str, state: LearningState) -> str:
    """Persist ``state`` to the program's ``state.npz`` ATOMICALLY; returns the path.

    Writes to a unique ``state.tmp-<hex>.npz`` sibling then ``os.replace``s it
    onto ``state.npz``, so a concurrent reader (or a crash mid-write) never
    observes a partial zip. (The tmp name already ends in ``.npz`` because
    ``np.savez`` appends the suffix otherwise.)
    """
    d = program_dir(project_id, program_id)
    path = d / PROGRAM_STATE_FILENAME
    tmp = d / f"state.tmp-{uuid.uuid4().hex[:8]}.npz"
    try:
        state_to_npz(state, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # only on a failed write/replace
            try:
                tmp.unlink()
            except OSError:
                pass
    return str(path)


# ── program config -> LearningState ────────────────────────────────────────────


def _channel_dollar_map(
    raw: dict[str, float] | None,
    flat_channels: list[str],
    parents: list[str],
    groups: dict[str, list[int]],
    default: float,
    what: str,
) -> np.ndarray:
    """Resolve a per-channel dollar dict onto the flattened arm list.

    Lookup order per arm: the flat arm name → the parent name (a parent-level
    dollar figure is split equally across its arms) → ``default``. A NaN
    ``default`` marks "no entry" for the caller to fill; explicit entries must
    be finite and non-negative.
    """
    raw = dict(raw or {})
    by_lower = {str(k).lower(): float(v) for k, v in raw.items()}
    for key, val in by_lower.items():
        if not np.isfinite(val) or val < 0:
            raise ValueError(
                f"{what}[{key!r}] must be finite and non-negative, got {val}"
            )
    out = np.empty(len(flat_channels), dtype=float)
    for i, ch in enumerate(flat_channels):
        if ch.lower() in by_lower:
            out[i] = by_lower[ch.lower()]
        elif parents[i].lower() in by_lower:
            out[i] = by_lower[parents[i].lower()] / max(
                len(groups.get(parents[i], [i])), 1
            )
        else:
            out[i] = default
    return out


def _parse_pair_signs(raw: dict[str, str] | None) -> dict[tuple[int, int], str]:
    """Parse the config's ``{"i,j": sign}`` keys into planner pair tuples."""
    out: dict[tuple[int, int], str] = {}
    for key, sign in (raw or {}).items():
        try:
            i_s, j_s = str(key).split(",")
            pair = (int(i_s), int(j_s))
        except ValueError as exc:
            raise ValueError(
                f"pair_signs keys must look like 'i,j' (channel indices), got {key!r}"
            ) from exc
        sign = str(sign)
        if sign not in VALID_SIGNS:
            raise ValueError(
                f"pair sign must be one of {VALID_SIGNS}, got {sign!r} for {key!r}"
            )
        out[pair] = sign
    return out


def new_program_state(config: dict[str, Any]) -> LearningState:
    """Validate a program config (dollars) and build its :class:`LearningState`.

    **Units**: ``budget`` and ``center`` are $ per period PER GEO (see the
    module header) — never a program/national total. ``spend_ref`` defaults to
    ONE GLOBAL constant, ``max(mean(center dollars), $1)``, identical for every
    channel: a uniform reference makes the fixed-budget simplex, group budgets,
    and the cap EXACT in dollars, while the κ ~ O(1) priors tolerate the O(1)
    per-channel spend ratios it produces. Per-channel ``spend_ref`` overrides
    remain possible (heterogeneous refs draw snapshot warnings).

    ``mode='free'`` is rejected unless ``spend_ref == 1`` for every channel:
    free-mode profit prices spend in scaled units, so the self-funding
    threshold is only correct when one scaled unit is exactly one dollar.

    Raises ``ValueError`` on an invalid config (callers map this to HTTP 400).
    """
    channels = [str(c) for c in (config.get("channels") or [])]
    if not channels:
        raise ValueError("config.channels must be a non-empty list of channel names")
    if len(set(channels)) != len(channels):
        raise ValueError(f"config.channels contains duplicates: {channels}")

    spec = expand_arms(channels, config.get("arms") or {})
    flat = spec.channels
    k = len(flat)

    budget = float(config.get("budget") or 0.0)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError(
            f"config.budget must be a positive dollar amount, got {budget}"
        )
    value = float(config.get("value_per_unit") or 0.0)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"config.value_per_unit must be positive ($ per KPI unit), got {value}"
        )

    activation = str(config.get("activation") or "hill")
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {activation!r}; known: {tuple(ACTIVATIONS)}"
        )
    likelihood = str(config.get("likelihood") or "normal")
    if likelihood not in VALID_LIKELIHOODS:
        raise ValueError(
            f"unknown likelihood {likelihood!r}; known: {VALID_LIKELIHOODS}"
        )
    time_effect = str(config.get("time_effect") or "none")
    if time_effect not in VALID_TIME_EFFECTS:
        raise ValueError(
            f"config.time_effect must be one of {VALID_TIME_EFFECTS}, "
            f"got {time_effect!r}"
        )
    mode = str(config.get("mode") or "fixed")
    if mode not in ("fixed", "free"):
        raise ValueError(f"config.mode must be 'fixed' or 'free', got {mode!r}")

    center_dollars = _channel_dollar_map(
        config.get("center"), flat, spec.parents, spec.groups, budget / k, "center"
    )
    if not np.all(np.isfinite(center_dollars)):
        raise ValueError("config.center must be finite per channel")
    # spend_ref defaults to ONE GLOBAL constant (mean center, floored at $1):
    # a uniform reference keeps the budget simplex / group budgets / cap exact
    # in dollars (a scaled unit costs the same $ everywhere).
    ref_default = np.full(k, max(float(center_dollars.mean()), 1.0))
    ref = _channel_dollar_map(
        config.get("spend_ref"), flat, spec.parents, spec.groups, np.nan, "spend_ref"
    )
    ref = np.where(np.isfinite(ref), ref, ref_default)
    if np.any(ref <= 0):
        raise ValueError("config.spend_ref must be strictly positive per channel")
    if mode == "free" and not np.allclose(ref, 1.0):
        raise ValueError(
            "mode='free' prices spend in SCALED units (profit = value·R(s) − "
            "Σ s_scaled), so the self-funding threshold is only correct when a "
            "scaled unit is exactly $1 — set spend_ref: 1 per channel (spend "
            "in raw dollars) to use free mode, or use mode='fixed'"
        )

    center_scaled = to_scaled(center_dollars, ref)
    total_center = float(center_dollars.sum())
    if total_center <= 0:
        raise ValueError("config.center must have positive total spend")
    # Reference-weighted budget conversion (== budget / ref for a uniform ref).
    budget_scaled = budget * float(center_scaled.sum()) / total_center
    cap = config.get("cap")
    cap_scaled = (
        None if cap is None else float(cap) * float(center_scaled.sum()) / total_center
    )

    pair_signs = _parse_pair_signs(config.get("pair_signs"))
    if spec.split_parents():
        # Arms: within-parent siblings default to "neg" (substitution), cross-
        # parent to "weak"; explicit config entries override.
        pair_signs = default_arm_pair_signs(spec, base=pair_signs)
    # Validate pair indices against the flattened arm count.
    for i, j in pair_signs:
        if not (0 <= i < k and 0 <= j < k) or i == j:
            raise ValueError(
                f"pair_signs pair ({i}, {j}) is invalid for {k} channels/arms"
            )

    return LearningState(
        channels=flat,
        center=center_scaled,
        B=budget_scaled,
        value=value,
        pair_signs=pair_signs or None,
        activation=activation,
        likelihood=likelihood,
        time_effect=time_effect,
        mode=mode,
        cap=cap_scaled,
        beta_scale=float(config.get("beta_scale") or 1.0),
        gamma_scale=float(config.get("gamma_scale") or 0.8),
        spend_ref=ref,
    )


def group_budgets_for(
    state: LearningState, config: dict[str, Any]
) -> list[tuple[list[int], float]] | None:
    """Grouped budget constraints for a program with arms (parent budget fixed).

    Each split parent's budget is the sum of its arms' center allocation,
    rescaled onto the program budget (so the groups stay jointly feasible when
    ``budget != Σ center``; with the default ``center = budget/K`` the factor
    is 1 and this is exactly "the sum of the arms' center dollars, scaled").
    """
    arms = config.get("arms") or {}
    channels = [str(c) for c in (config.get("channels") or [])]
    if not arms or not channels:
        return None
    spec = expand_arms(channels, arms)
    if spec.channels != state.channels:
        raise ValueError(
            "config channels/arms no longer match the program state's channels: "
            f"{spec.channels} vs {state.channels}"
        )
    total = float(np.sum(state.center))
    factor = float(state.B) / total if total > 0 else 1.0
    out: list[tuple[list[int], float]] = []
    for parent in spec.split_parents():
        idx = spec.groups[parent]
        out.append((list(idx), float(np.sum(state.center[idx])) * factor))
    return out or None


# ── wave design ────────────────────────────────────────────────────────────────


def _cell_labels(
    channels: list[str], delta: float, probe_pairs: list[tuple[int, int]]
) -> list[str]:
    """Human labels in :func:`central_composite`'s exact row order."""
    pct = int(round(delta * 100))
    labels = ["center"]
    for c in channels:
        labels.append(f"{c} +{pct}%")
        labels.append(f"{c} -{pct}%")
    for i, j in probe_pairs:
        labels.append(f"{channels[i]} & {channels[j]} +{pct}%")
        labels.append(f"{channels[i]} & {channels[j]} -{pct}%")
    for c in channels:
        labels.append(f"{c} shutoff")
    return labels


def design_wave(
    state: LearningState,
    *,
    delta: float = 0.6,
    probe_pairs: list[tuple[int, int]] | None = None,
    n_geo: int | None = None,
    n_holdout: int = 0,
    seed: int = 0,
    stratify: bool = True,
    optimize: bool = False,
    candidate_deltas: list[float] | None = None,
    kg_n_outcomes: int = 32,
    t_test: int = 10,
) -> dict[str, Any]:
    """Central-composite wave design around the program's current center.

    Returns ``{cells_scaled, cells_dollars, cell_labels, assignment?, n_cells,
    delta, probe_pairs, warnings}``. ``assignment`` (geo → cell) is included
    when ``n_geo`` is given or the program already knows its geo set; it is
    **stratified on the accumulated per-geo KPI** (blocked randomization, see
    :func:`~mmm_framework.continuous_learning.design.assign_geos`) when
    ``stratify`` is on and the program has ingested data for the same geo set
    (``assignment["stratified_on"] = "accumulated_kpi"``), else shuffled
    round-robin (``stratified_on = None``).

    ``optimize=True`` scores ``candidate_deltas`` (default ``0.3/0.6/0.9``)
    with the Laplace knowledge-gradient
    (:func:`~mmm_framework.continuous_learning.loop.select_next_design`) and
    designs the EVSI-best candidate; the returned ``delta``/``probe_pairs``/
    cells/labels reflect the CHOSEN candidate and a ``"kg"`` key carries the
    per-candidate scores. Requires a fitted panel posterior carrying its
    observation sites (any registered activation; Gaussian/Student-t need
    ``sigma``, NegBinomial needs ``phi`` + baseline) — otherwise the
    fixed-``delta`` design is kept and a warning explains why.
    """
    from .loop import select_next_design

    pairs = (
        list(state.pairs or [])
        if probe_pairs is None
        else [(int(i), int(j)) for i, j in probe_pairs]
    )
    ref = (
        state.spend_ref if state.spend_ref is not None else np.ones(len(state.channels))
    )
    warnings: list[str] = []

    resolved_n_geo = n_geo
    if resolved_n_geo is None and state.geo_ids:
        resolved_n_geo = len(state.geo_ids)

    chosen_delta = float(delta)
    kg_info: dict[str, Any] | None = None
    if optimize:
        if state.posterior is None:
            warnings.append(
                "knowledge-gradient optimization unavailable: the program has "
                "no fitted posterior yet — record a wave or import experiments "
                "and fit first (using the fixed delta)"
            )
        else:
            cells_kg, meta = select_next_design(
                state.posterior,
                state.center,
                pairs,
                state.B,
                state.value,
                mode=state.mode,
                cap=state.cap,
                candidate_deltas=(
                    tuple(float(d) for d in candidate_deltas)
                    if candidate_deltas
                    else (0.3, 0.6, 0.9)
                ),
                n_geo=int(resolved_n_geo) if resolved_n_geo else 80,
                t_test=int(t_test),
                n_outcomes=int(kg_n_outcomes),
                seed=int(seed),
                fallback_delta=float(delta),
            )
            if meta.get("kg_used"):
                chosen_delta = float(meta["chosen_delta"])
                pairs = [(int(i), int(j)) for i, j in meta["chosen_probe_pairs"]]
                kg_info = {
                    "used": True,
                    "chosen_delta": chosen_delta,
                    "chosen_probe_pairs": [[int(i), int(j)] for i, j in pairs],
                    "scores": meta.get("kg_scores") or [],
                    "sigma": meta.get("sigma"),
                }
            else:
                warnings.append(
                    "knowledge-gradient optimization unavailable: "
                    f"{meta.get('reason')} (using the fixed delta)"
                )

    cells = state.next_design(chosen_delta, probe_pairs=pairs)

    out: dict[str, Any] = {
        "cells_scaled": [[float(x) for x in row] for row in cells],
        "cells_dollars": [[float(x) for x in row] for row in to_dollars(cells, ref)],
        "cell_labels": _cell_labels(state.channels, chosen_delta, pairs),
        "n_cells": int(cells.shape[0]),
        "delta": chosen_delta,
        "probe_pairs": [[int(i), int(j)] for i, j in pairs],
        "warnings": warnings,
    }
    if kg_info is not None:
        out["kg"] = kg_info
    if resolved_n_geo is not None:
        resolved_n_geo = int(resolved_n_geo)
        if resolved_n_geo < 1:
            raise ValueError(f"n_geo must be >= 1, got {resolved_n_geo}")
        if cells.shape[0] > resolved_n_geo:
            warnings.append(
                f"the design has {cells.shape[0]} cells but only "
                f"{resolved_n_geo} geos — every cell needs at least one geo; "
                "drop probe pairs or add geos"
            )
        # Stratification baseline: the accumulated per-geo mean KPI (only when
        # the panel's geo set matches — never raise on a first-wave design).
        baseline = None
        stratified_on = None
        if (
            stratify
            and state.data is not None
            and int(state.data["n_geo"]) == resolved_n_geo
        ):
            g_idx = np.asarray(state.data["geo_idx"], dtype=int)
            y = np.asarray(state.data["y"], dtype=float)
            baseline = np.bincount(
                g_idx, weights=y, minlength=resolved_n_geo
            ) / np.maximum(np.bincount(g_idx, minlength=resolved_n_geo), 1)
            stratified_on = "accumulated_kpi"
        rng = np.random.default_rng(int(seed))
        geo_alloc, cell_idx = assign_geos(
            cells,
            resolved_n_geo,
            rng,
            n_holdout=int(n_holdout),
            center=state.center if n_holdout else None,
            baseline=baseline,
        )
        geo_ids = (
            list(state.geo_ids)
            if state.geo_ids and len(state.geo_ids) == resolved_n_geo
            else [f"geo_{i}" for i in range(resolved_n_geo)]
        )
        out["assignment"] = {
            "geo_ids": geo_ids,
            "cell_idx": [int(i) for i in cell_idx],
            "n_holdout": int(n_holdout),
            "stratified_on": stratified_on,
            "spend_dollars": [
                [float(x) for x in row] for row in to_dollars(geo_alloc, ref)
            ],
        }
    return out


# ── wave ingestion (rows / CSV) ────────────────────────────────────────────────


def rows_from_csv(
    csv_text: str,
    *,
    geo_col: str = "geo",
    y_col: str = "y",
    period_col: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a ``geo,<channel $ columns>,y`` CSV into ingestable row dicts.

    An optional week/date/period column is fine (kept as a string; ignored by
    :func:`ingest_wave_rows` beyond row ordering — unless the program models a
    national time effect and ``period_col`` names it, in which case pass the
    same ``period_col`` to :func:`ingest_wave_rows`). Naming it here keeps its
    labels as strings (a numeric week id would otherwise coerce to float).
    Numeric columns are coerced.
    """
    reader = csv.DictReader(io.StringIO(csv_text or ""))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        row: dict[str, Any] = {}
        for key, val in raw.items():
            if key is None:
                continue
            key = str(key).strip()
            if val is None:
                row[key] = None
                continue
            sval = str(val).strip()
            if key == geo_col or (period_col is not None and key == period_col):
                row[key] = sval
                continue
            try:
                row[key] = float(sval)
            except ValueError:
                row[key] = sval
        if row:
            rows.append(row)
    if not rows:
        raise ValueError("the CSV has no data rows")
    return rows


def ingest_wave_rows(
    state: LearningState,
    rows: list[dict[str, Any]],
    *,
    geo_col: str = "geo",
    y_col: str = "y",
    period_col: str | None = None,
) -> dict[str, Any]:
    """Append observed wave rows (dollars) to the program's accumulated panel.

    Each row is ``{geo, <one $ column per channel/arm>, y}`` (case-insensitive
    column match; extra columns like ``week`` are ignored). The first wave pins
    the program's geo identities; later waves must use the same geo set.

    When the program models a national time effect (``state.time_effect !=
    "none"``), each row's period must be identifiable: pass ``period_col``
    naming the column; its labels are mapped to wave-local 0-based indices in
    **sorted label order** (ISO dates sort correctly) and
    :meth:`LearningState.ingest` applies the cross-wave offset. When
    ``period_col`` is omitted, a recognizable period column (``period`` /
    ``week`` / ``date``, case-insensitive — the same names
    :func:`rows_from_csv` tolerates) is auto-detected with a warning note in
    the returned ``warnings``; if none exists the ingest fails loudly. With
    ``time_effect="none"`` any ``period_col`` is ignored.
    """
    if not rows:
        raise ValueError("no rows to ingest")
    channels = state.channels
    ref = state.spend_ref if state.spend_ref is not None else np.ones(len(channels))

    keys = {str(k).strip().lower(): str(k).strip() for k in rows[0].keys()}
    if geo_col.lower() not in keys:
        raise ValueError(f"rows are missing the geo column {geo_col!r}")
    if y_col.lower() not in keys:
        raise ValueError(f"rows are missing the outcome column {y_col!r}")
    col_geo, col_y = keys[geo_col.lower()], keys[y_col.lower()]
    warnings_out: list[str] = []
    col_period: str | None = None
    if state.time_effect != "none":
        if period_col is None:
            # Auto-detect a period column so nationally-time-effected programs
            # can ingest through callers that never learned to pass period_col.
            reserved = {geo_col.lower(), y_col.lower()} | {
                ch.lower() for ch in channels
            }
            for cand in ("period", "week", "date"):
                if cand in keys and cand not in reserved:
                    period_col = cand
                    warnings_out.append(
                        f"this program models a national time effect "
                        f"(time_effect={state.time_effect!r}) and no "
                        f"period_col was given — auto-detected the period "
                        f"column {keys[cand]!r} (pass period_col explicitly "
                        "to override)"
                    )
                    break
        if period_col is None:
            raise ValueError(
                f"this program models a national time effect (time_effect="
                f"{state.time_effect!r}): pass period_col naming the "
                "week/date/period column so each row's period can be indexed "
                "(no 'period'/'week'/'date' column was found to auto-detect)"
            )
        if period_col.lower() not in keys:
            raise ValueError(f"rows are missing the period column {period_col!r}")
        col_period = keys[period_col.lower()]
    col_by_channel: dict[str, str] = {}
    missing = []
    for ch in channels:
        col = keys.get(ch.lower())
        if col is None:
            missing.append(ch)
        else:
            col_by_channel[ch] = col
    if missing:
        raise ValueError(
            f"rows are missing spend columns for channels {missing}; expected one "
            f"dollar column per program channel: {channels}"
        )

    labels = [str(r.get(col_geo)) for r in rows]
    if state.geo_ids:
        geo_ids = list(state.geo_ids)
        unknown = sorted({g for g in labels if g not in set(geo_ids)})
        if unknown:
            raise ValueError(
                f"rows reference geos outside the program's stable geo set: "
                f"{unknown} (program geos: {geo_ids}) — the loop requires the "
                "SAME geos every wave"
            )
    else:
        geo_ids = sorted(set(labels))
    index = {g: i for i, g in enumerate(geo_ids)}

    n = len(rows)
    spend_dollars = np.empty((n, len(channels)), dtype=float)
    y = np.empty(n, dtype=float)
    geo_idx = np.empty(n, dtype=int)
    for r_i, row in enumerate(rows):
        geo_idx[r_i] = index[str(row.get(col_geo))]
        try:
            y[r_i] = float(row.get(col_y))
            for c_i, ch in enumerate(channels):
                spend_dollars[r_i, c_i] = float(row.get(col_by_channel[ch]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"row {r_i} has a non-numeric spend/outcome value: {row}"
            ) from exc

    wave = {
        "spend": to_scaled(spend_dollars, ref),
        "geo_idx": geo_idx,
        "y": y,
        "n_geo": len(geo_ids),
        "geo_ids": geo_ids,
    }
    if col_period is not None:
        labels_p = [str(r.get(col_period)) for r in rows]
        period_labels = sorted(set(labels_p))
        p_index = {p: i for i, p in enumerate(period_labels)}
        wave["period_idx"] = np.array([p_index[p] for p in labels_p], dtype=int)
    state.ingest(wave)
    return {"n_rows": n, "n_geo": len(geo_ids), "warnings": warnings_out}


# ── past-experiment import ─────────────────────────────────────────────────────


def import_experiment_summaries(
    state: LearningState,
    experiments: list[dict[str, Any]],
    *,
    period_days: float = 7.0,
) -> dict[str, Any]:
    """Convert registry experiment dicts into summaries and ingest them.

    Thin wrapper over
    :func:`~mmm_framework.continuous_learning.evidence.experiments_to_summaries`
    bound to this program's channels/spend_ref/center. **Idempotent**: any
    experiment whose id already appears in ``state.summaries``' provenance (or
    repeats within one call) is skipped with reason ``"already imported"`` —
    re-importing would duplicate its likelihood term and shrink the posterior
    SEs by √2 per repeat. Returns ``{"imported": n, "imported_ids": [...],
    "skipped": [{"id", "reason"}, ...]}`` — ``imported_ids`` is the provenance
    list callers should persist on the wave row (never the requested ids).
    """
    already = {
        str(s.get("experiment_id"))
        for s in state.summaries
        if s.get("experiment_id") is not None
    }
    fresh: list[dict[str, Any]] = []
    dup_skipped: list[dict[str, Any]] = []
    seen_this_call: set[str] = set()
    for exp in experiments or []:
        eid = exp.get("id")
        key = str(eid) if eid is not None else None
        if key is not None and (key in already or key in seen_this_call):
            dup_skipped.append({"id": eid, "reason": "already imported"})
            continue
        if key is not None:
            seen_this_call.add(key)
        fresh.append(exp)

    ref = (
        state.spend_ref if state.spend_ref is not None else np.ones(len(state.channels))
    )
    summaries, skipped = experiments_to_summaries(
        fresh,
        channels=state.channels,
        spend_ref=np.asarray(ref, dtype=float),
        center_scaled=np.asarray(state.center, dtype=float),
        period_days=float(period_days),
    )
    if summaries:
        state.ingest_summaries(summaries)
    imported_ids = [
        str(s.get("experiment_id"))
        for s in summaries
        if s.get("experiment_id") is not None
    ]
    return {
        "imported": len(summaries),
        "imported_ids": imported_ids,
        "skipped": skipped + dup_skipped,
    }


# ── fit + plan -> SNAPSHOT ─────────────────────────────────────────────────────


def resolve_population(
    state: LearningState,
    config: dict[str, Any],
    override: float | None = None,
) -> tuple[float, str | None]:
    """The ENBS population in GEO-PERIODS (the units ``e_regret_kpi`` is priced
    in — the recommendation applies to EVERY geo, every period of the horizon).

    ``override`` (an explicit economics override from a tool arg / request
    body) is taken as final geo-periods, untouched. Otherwise
    ``config["horizon_periods"]`` (preferred; the legacy ``population`` config
    key is accepted as horizon periods for back-compat — the Sextant wizard
    historically sent horizon there) is multiplied by the program's geo count:
    ``len(state.geo_ids)`` once the geo set is pinned, else the panel's
    ``n_geo``, else 1 — the summaries-only fallback returns a warning that the
    value of information is per-geo-understated.
    """
    if override is not None:
        return float(override), None
    horizon = config.get("horizon_periods")
    if horizon is None:
        horizon = config.get("population")
    horizon = float(horizon) if horizon else 1.0
    if horizon <= 0:
        horizon = 1.0
    n_geos = 0
    if state.geo_ids:
        n_geos = len(state.geo_ids)
    elif state.data is not None:
        n_geos = int(state.data.get("n_geo") or 0)
    warning = None
    if n_geos < 1:
        n_geos = 1
        warning = (
            "no geo set is pinned yet (summaries-only program): the ENBS "
            "population assumes 1 geo, so the value of information is "
            "per-geo-understated and the stopping rule may fire early"
        )
    return float(n_geos) * horizon, warning


def _distinct_levels(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size == 0:
        return 0
    return int(np.unique(np.round(vals, 6)).size)


def _shape_identified(state: LearningState) -> dict[str, bool]:
    """Per channel: does the evidence hold >= 3 distinct spend levels?"""
    k = len(state.channels)
    columns: list[np.ndarray] = []
    if state.data is not None:
        columns.append(np.asarray(state.data["spend"], dtype=float))
    for s in state.summaries:
        columns.append(np.asarray(s["spend_test"], dtype=float)[None, :])
        columns.append(np.asarray(s["spend_base"], dtype=float)[None, :])
    if columns:
        stacked = np.vstack(columns)
    else:
        stacked = np.zeros((0, k))
    return {
        ch: _distinct_levels(stacked[:, c]) >= _SHAPE_MIN_LEVELS
        for c, ch in enumerate(state.channels)
    }


def _curve_draws(n_draws: int) -> np.ndarray:
    if n_draws <= _CURVE_MAX_DRAWS:
        return np.arange(n_draws)
    return np.unique(np.linspace(0, n_draws - 1, _CURVE_MAX_DRAWS).astype(int))


def _response_curves(
    state: LearningState, rec_scaled: np.ndarray, ref: np.ndarray
) -> dict[str, dict[str, Any]]:
    """25-point per-channel response curves (90% band), other channels at the
    recommendation, spend converted to dollars."""
    post = state.posterior
    draws = _curve_draws(post.n_draws)
    out: dict[str, dict[str, Any]] = {}
    for c, ch in enumerate(state.channels):
        grid = np.linspace(0.0, 2.0 * float(state.center[c]), _CURVE_POINTS)
        spend_matrix = np.repeat(rec_scaled[None, :], _CURVE_POINTS, axis=0)
        spend_matrix[:, c] = grid
        resp = _planner.response_grid(post, spend_matrix, draws)  # (G, D)
        out[ch] = {
            "spend_dollars": [float(x) for x in grid * float(ref[c])],
            "mean": [float(x) for x in resp.mean(axis=1)],
            "lo": [float(x) for x in np.percentile(resp, 5, axis=1)],
            "hi": [float(x) for x in np.percentile(resp, 95, axis=1)],
            "current": float(state.center[c] * ref[c]),
        }
    return out


def _verdict(prob: float) -> str:
    if prob > FUND_THRESHOLD:
        return "FUND"
    if prob < CUT_THRESHOLD:
        return "CUT"
    return "HOLD"


def fit_and_plan(
    state: LearningState,
    *,
    fit_kwargs: dict[str, Any] | None = None,
    plan_kwargs: dict[str, Any] | None = None,
    margin: float,
    population: float,
    wave_cost: float,
    n_waves: int | None = None,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Refit on ALL accumulated evidence, run ONE Thompson plan, snapshot.

    Uses :meth:`LearningState.plan` (one ``plan_from_posterior`` pass) so the
    recommendation, funding line, and expected regret all describe one
    consensus allocation — never the separate ``recommend``/``funding``/
    ``regret`` calls (3× the SLSQP cost and inconsistent seeds). Appends a
    :class:`WaveRecord` to ``state.history`` and returns the pinned SNAPSHOT
    dict (see the module header for every unit decision).

    ``population`` is GEO-PERIODS (use :func:`resolve_population`). ``n_waves``
    is the count of INGESTED evidence batches this snapshot reflects (the
    caller counts its ingested wave rows) — pure refits must pass the existing
    count so ``evidence.n_waves`` never inflates with fit clicks; when omitted
    the legacy fit-count fallback (``len(state.history) + 1``) is used.
    ``extra_warnings`` (e.g. the summaries-only population caveat) are appended
    to the snapshot's warnings.
    """
    fit_kwargs = dict(fit_kwargs or {})
    plan_kwargs = dict(plan_kwargs or {})
    k = len(state.channels)
    ref = np.asarray(
        state.spend_ref if state.spend_ref is not None else np.ones(k), dtype=float
    )

    post = state.fit(**fit_kwargs)
    plan = state.plan(**plan_kwargs)

    rec_scaled = np.asarray(plan.recommendation, dtype=float)
    rec_dollars = to_dollars(rec_scaled, ref)
    alloc_sd_dollars = to_dollars(np.asarray(plan.alloc_sd, dtype=float), ref)

    # Funding line in per-DOLLAR terms (see module header). When margin < 1
    # the break-even test runs on MARGIN-adjusted draws (marginal PROFIT
    # break-even, consistent with the ENBS rule); mroas_mean stays per revenue
    # dollar and mroas_margin_adjusted carries the adjusted mean.
    margin_f = float(margin)
    mroas_per_dollar = np.asarray(plan.mroas_draws, dtype=float) / ref[None, :]
    mroas_mean = mroas_per_dollar.mean(axis=0)
    prob_above_revenue = (mroas_per_dollar > 1.0).mean(axis=0)
    if margin_f < 1.0:
        mroas_margin = mroas_per_dollar * margin_f
    else:
        mroas_margin = mroas_per_dollar
    mroas_margin_mean = mroas_margin.mean(axis=0)
    prob_above = (mroas_margin > 1.0).mean(axis=0)
    funding = [
        {
            "channel": ch,
            "mroas_mean": float(mroas_mean[c]),
            "mroas_margin_adjusted": float(mroas_margin_mean[c]),
            "prob_above_line": float(prob_above[c]),
            "funded": bool(prob_above[c] > 0.5),
            "verdict": _verdict(float(prob_above[c])),
        }
        for c, ch in enumerate(state.channels)
    ]

    # ENBS in dollars (pinned conversion; == planner.enbs on the same inputs).
    e_regret = float(plan.e_regret)
    stop, enbs_val = _planner.should_stop(
        e_regret,
        margin=float(margin),
        population=float(population),
        wave_cost=float(wave_cost),
    )
    e_regret_dollars = e_regret * float(margin) * float(population)

    # Synergies + prior-domination flags.
    gamma_rows: list[dict[str, Any]] = []
    summary = post.gamma_summary()
    for i, j in post.pairs:
        row = summary[pair_name(post.channels, (i, j))]
        prior_dominated = bool(
            abs(row["mean"]) < 0.1 * float(state.gamma_scale) or row["sign"] == "zero"
        )
        gamma_rows.append(
            {
                "pair": [state.channels[i], state.channels[j]],
                "mean": float(row["mean"]),
                "p5": float(row["p5"]),
                "p95": float(row["p95"]),
                "sign": row["sign"],
                "prior_dominated": prior_dominated,
            }
        )

    shape_ok = _shape_identified(state)
    evidence = post.diagnostics.get("evidence") or {}
    n_rows = int(evidence.get("n_rows", 0))
    n_summaries = int(evidence.get("n_summaries", len(state.summaries)))

    max_rhat = post.diagnostics.get("max_rhat")
    min_ess = post.diagnostics.get("min_ess")
    flags: list[str] = []
    if max_rhat is not None and float(max_rhat) > 1.1:
        flags.append(
            f"max R-hat {float(max_rhat):.2f} > 1.1 — the sampler did not "
            "converge; widen the activation family or add data before acting"
        )
    unidentified = [ch for ch, ok in shape_ok.items() if not ok]
    if unidentified:
        flags.append(
            "curve shape is prior-dominated (fewer than 3 distinct spend "
            f"levels) for: {', '.join(unidentified)} — trust the funded set, "
            "not the curve"
        )

    warnings: list[str] = list(extra_warnings or [])
    uniform_ref = bool(np.allclose(ref, ref[0]))
    if not uniform_ref:
        warnings.append(
            "spend_ref differs across channels: the budget constraint is "
            "enforced in scaled units (a reference-weighted budget), so the "
            "recommendation's dollar total can drift from the stated budget"
        )
        if plan_kwargs.get("group_budgets"):
            warnings.append(
                "spend_ref differs across channels while group budgets are "
                "active: a scaled unit costs different dollars per arm, so a "
                "split parent's DOLLAR total can drift with the arm mix — "
                "prefer a uniform spend_ref"
            )
        if state.cap is not None:
            warnings.append(
                "spend_ref differs across channels while a per-channel cap is "
                "set: the dollar cap converts through an average ratio, so "
                "each channel's effective dollar cap is scaled by its own "
                "spend_ref — prefer a uniform spend_ref"
            )
    if state.mode == "free" and not uniform_ref:
        warnings.append(
            "mode='free' with a non-uniform spend_ref makes the self-funding "
            "threshold channel-inconsistent — prefer a uniform spend_ref"
        )
    if margin_f < 1.0:
        flipped = [
            ch
            for c, ch in enumerate(state.channels)
            if _verdict(float(prob_above_revenue[c])) != _verdict(float(prob_above[c]))
        ]
        if flipped:
            warnings.append(
                f"margin {margin_f:g} < 1 changes the funding verdict for: "
                f"{', '.join(flipped)} — break-even is tested on "
                "margin-adjusted mROAS (P(margin · value · dR/d$ > 1)); "
                "mroas_mean stays per revenue dollar"
            )
    # Per-geo vs total units sanity check: the surface/center are per-geo-period
    # dollars, so a panel whose per-channel mean spend sits far from the
    # configured center is the tell that program totals were entered.
    if state.data is not None and int(np.asarray(state.data["spend"]).shape[0]) > 0:
        mean_spend = np.asarray(state.data["spend"], dtype=float).mean(axis=0)
        far = []
        for c, ch in enumerate(state.channels):
            center_c = float(state.center[c])
            if center_c <= 0:
                continue
            ratio = float(mean_spend[c]) / center_c
            if ratio < 1.0 / 3.0 or ratio > 3.0:
                far.append(ch)
        if far:
            warnings.append(
                "panel spend far from the program center for: "
                + ", ".join(far)
                + " — recommendations extrapolate; check per-geo vs total "
                "units (budget/center are $ per period PER GEO, not program "
                "totals)"
            )

    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "fitted_at": time.time(),
        "evidence": {
            "n_rows": n_rows,
            "n_summaries": n_summaries,
            "n_waves": (
                int(n_waves) if n_waves is not None else len(state.history) + 1
            ),
            "shape_identified": shape_ok,
        },
        "diagnostics": {
            "max_rhat": None if max_rhat is None else float(max_rhat),
            "min_ess": None if min_ess is None else float(min_ess),
            "n_draws": int(post.n_draws),
            "flags": flags,
        },
        "recommendation": {
            ch: float(rec_dollars[c]) for c, ch in enumerate(state.channels)
        },
        "recommendation_scaled": {
            ch: float(rec_scaled[c]) for c, ch in enumerate(state.channels)
        },
        "allocation_sd": {
            ch: float(alloc_sd_dollars[c]) for c, ch in enumerate(state.channels)
        },
        "funding": funding,
        "regret": {
            "e_regret_kpi": e_regret,
            "e_regret_dollars": float(e_regret_dollars),
            "enbs": float(enbs_val),
            "stop": bool(stop),
            "margin": float(margin),
            "population": float(population),
            "wave_cost": float(wave_cost),
        },
        "gamma": gamma_rows,
        "response_curves": _response_curves(state, rec_scaled, ref),
        "warnings": warnings,
    }

    state.history.append(
        WaveRecord(
            wave=len(state.history),
            n_rows=n_rows,
            e_regret=e_regret,
            enbs=float(enbs_val),
            stop=bool(stop),
            recommendation=[float(x) for x in rec_scaled],
            funded=[bool(p > 0.5) for p in prob_above],
            mroas_mean=[float(x) for x in mroas_mean],
            prob_above_line=[float(x) for x in prob_above],
            profit_gap=0.0,  # truth unknown outside the synthetic closed loop
            profit_gap_rel=0.0,
            max_rhat=None if max_rhat is None else float(max_rhat),
            n_summaries=n_summaries,
        )
    )
    return snapshot
