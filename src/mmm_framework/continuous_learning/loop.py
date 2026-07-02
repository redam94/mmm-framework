"""The outer control loop: carry the posterior across waves and know when to stop.

:class:`LearningState` accumulates every wave's data and refits the response
surface on **all** of it each pass — that is how the posterior is "carried"
across waves (each wave borrows strength from all prior data). It exposes the
per-wave decision readouts (recommend / funding line / regret / stop) over the
current posterior.

:func:`run_closed_loop` drives a :class:`~mmm_framework.continuous_learning.dgp.TrueWorld`
through the full loop — fit -> score -> run designed wave -> update -> ENBS stop
— and is both the runnable demo and the closure/stopping test backbone. In a
real deployment you would replace the synthetic ``simulate_wave`` collector with
the actual geo holdout results; nothing else changes.

Re-test scheduling reuses the framework's existing information-decay model
(:func:`mmm_framework.planning.eig.reexperiment_due`) so a continuous-learning
program and a model-anchored program agree on when evidence has gone stale.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import planner as _planner
from .design import central_composite
from .dgp import TrueWorld, simulate_panel, simulate_wave
from .model import Pair, Posterior, _validate_summaries, default_pairs, fit


def _profit_under(
    response: float, alloc: np.ndarray, value: float, mode: str, B: float
) -> float:
    """Net profit for an allocation given its incremental ``response``."""
    if mode == "fixed":
        return float(value) * float(response) - float(B)
    return float(value) * float(response) - float(np.sum(alloc))


def world_optimal_allocation(
    world: TrueWorld,
    B: float,
    value: float,
    *,
    mode: str = "fixed",
    cap: float | None = None,
    n_starts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """The truth-optimal allocation + profit (the recovery/closure target)."""
    params = {
        "beta": world.beta,
        "gamma": world.gamma_matrix(),
        "act_fn": world.act_fn(),
        "shape": world.shape_tuple(),
    }
    return _planner.allocate_under_sample(
        params, B, value, mode=mode, cap=cap, n_starts=n_starts, seed=seed
    )


@dataclass
class WaveRecord:
    """A JSON-safe snapshot of one wave's decision state."""

    wave: int
    n_rows: int
    e_regret: float
    enbs: float
    stop: bool
    recommendation: list[float]
    funded: list[bool]
    mroas_mean: list[float]
    prob_above_line: list[float]
    profit_gap: float
    profit_gap_rel: float
    max_rhat: float | None
    n_summaries: int = 0


@dataclass
class LearningState:
    """Accumulated experiment data + the current posterior + decision readouts.

    ``spend_ref`` (dollars per scaled unit, per channel) is carried for the
    dollar boundary — convert with
    :func:`~mmm_framework.continuous_learning.scaling.to_dollars` /
    :func:`~mmm_framework.continuous_learning.scaling.to_scaled`; everything
    inside the state (``center``, panel spend, summaries) is scaled units.
    ``geo_ids`` (optional, from the data dict's ``"geo_ids"`` key) pins the geo
    *identity* across waves — the misspecification study showed the loop
    diverges if geo baselines are silently re-drawn under the same ``geo_idx``.
    """

    channels: list[str]
    center: np.ndarray
    B: float
    value: float
    pairs: list[Pair] | None = None
    pair_signs: dict[Pair, str] | None = None
    activation: str = "hill"
    mode: str = "fixed"
    cap: float | None = None
    beta_scale: float = 1.0
    gamma_scale: float = 0.8
    spend_ref: np.ndarray | None = None

    data: dict[str, Any] | None = None
    posterior: Posterior | None = None
    history: list[WaveRecord] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    geo_ids: list[str] | None = None

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        if self.pairs is None:
            self.pairs = default_pairs(len(self.channels))

    # -- data accumulation -----------------------------------------------------

    def _check_geo_ids(self, wave_data: dict[str, Any], n_geo: int) -> None:
        """Geo-identity guard (review fix F6): same ids, same order, every wave."""
        geo_ids = wave_data.get("geo_ids")
        if geo_ids is None:
            return  # count-only check retained (the caller passed no identities)
        geo_ids = [str(g) for g in geo_ids]
        if len(geo_ids) != n_geo:
            raise ValueError(
                f"geo_ids has {len(geo_ids)} entries but the wave has {n_geo} geos"
            )
        if self.geo_ids is None:
            self.geo_ids = geo_ids
            return
        if geo_ids != self.geo_ids:
            dropped = sorted(set(self.geo_ids) - set(geo_ids))
            added = sorted(set(geo_ids) - set(self.geo_ids))
            detail = (
                f"dropped={dropped}, added={added}"
                if (dropped or added)
                else "same geo set but reordered (geo_idx would point at "
                "different geos)"
            )
            raise ValueError(
                "wave geo_ids do not match the program's geo set — the "
                "continuous-learning loop requires a STABLE geo set (re-drawn "
                f"geo baselines make it diverge): {detail}"
            )

    def ingest(self, wave_data: dict[str, Any]) -> None:
        """Append a wave's rows to the accumulated panel (same geos throughout).

        ``wave_data`` may carry an optional ``"geo_ids"`` key (``list[str]`` of
        length ``n_geo``): the first wave pins the program's geo identities and
        later waves must match them exactly (order included).
        """
        spend = np.asarray(wave_data["spend"], dtype=float)
        geo_idx = np.asarray(wave_data["geo_idx"], dtype=int)
        y = np.asarray(wave_data["y"], dtype=float)
        n_geo = int(wave_data["n_geo"])
        if self.data is not None and n_geo != self.data["n_geo"]:
            raise ValueError(
                f"wave has {n_geo} geos but the program has {self.data['n_geo']}; "
                "the continuous-learning loop assumes a stable geo set"
            )
        self._check_geo_ids(wave_data, n_geo)
        if self.data is None:
            self.data = {"spend": spend, "geo_idx": geo_idx, "y": y, "n_geo": n_geo}
            return
        self.data = {
            "spend": np.vstack([self.data["spend"], spend]),
            "geo_idx": np.concatenate([self.data["geo_idx"], geo_idx]),
            "y": np.concatenate([self.data["y"], y]),
            "n_geo": n_geo,
        }

    def ingest_summaries(self, items: list[dict[str, Any]]) -> None:
        """Append summary observations (historical lift readouts, no panel).

        Each item follows the summary schema in
        :mod:`mmm_framework.continuous_learning.model` (``spend_test``/
        ``spend_base`` in SCALED units, total ``lift`` ± ``se`` in natural KPI
        units, ``scale`` = geo-periods aggregated). Validated eagerly.
        """
        self.summaries.extend(_validate_summaries(items, len(self.channels)))

    # -- inference + decisions -------------------------------------------------

    def fit(self, **fit_kwargs: Any) -> Posterior:
        """Refit the surface on ALL accumulated evidence (carries the posterior)."""
        if self.data is None and not self.summaries:
            raise RuntimeError(
                "ingest at least one wave or some summaries before fitting"
            )
        data = dict(self.data) if self.data is not None else {"n_geo": 0}
        if self.summaries:
            data["summaries"] = list(self.summaries)
        self.posterior = fit(
            data,
            channels=self.channels,
            pairs=self.pairs,
            pair_signs=self.pair_signs,
            activation=self.activation,
            beta_scale=self.beta_scale,
            gamma_scale=self.gamma_scale,
            spend_ref=self.spend_ref,
            **fit_kwargs,
        )
        return self.posterior

    def _require_posterior(self) -> Posterior:
        if self.posterior is None:
            raise RuntimeError("fit the model before requesting a decision readout")
        return self.posterior

    def recommend(self, **kwargs: Any) -> np.ndarray:
        return _planner.recommend_allocation(
            self._require_posterior(),
            self.B,
            self.value,
            mode=self.mode,
            cap=self.cap,
            **kwargs,
        )

    def funding(self, alloc: np.ndarray | None = None, **kwargs):
        post = self._require_posterior()
        if alloc is None:
            alloc = self.recommend()
        return _planner.marginal_roas(post, alloc, self.value, **kwargs)

    def regret(self, **kwargs):
        return _planner.expected_regret(
            self._require_posterior(),
            self.B,
            self.value,
            mode=self.mode,
            cap=self.cap,
            **kwargs,
        )

    def plan(self, **kwargs: Any) -> _planner.PlanResult:
        """All per-wave readouts from ONE Thompson pass (see fix F4).

        Delegates to :func:`~mmm_framework.continuous_learning.planner.plan_from_posterior`
        with this program's budget/value/mode/cap; pass ``q``/``seed``/
        ``group_budgets`` etc. through as keyword arguments.
        """
        return _planner.plan_from_posterior(
            self._require_posterior(),
            self.B,
            self.value,
            mode=self.mode,
            cap=self.cap,
            **kwargs,
        )

    def next_design(
        self,
        delta: float,
        *,
        center: np.ndarray | None = None,
        probe_pairs: list[Pair] | None = None,
    ) -> np.ndarray:
        center = self.center if center is None else np.asarray(center, dtype=float)
        probe_pairs = probe_pairs if probe_pairs is not None else self.pairs
        return central_composite(center, delta, probe_pairs)

    def recenter(self, alloc: np.ndarray, *, min_frac: float = 0.05) -> None:
        """Move the trust-region center to ``alloc``, floored away from zero.

        The CCD is *multiplicative* (review fix F5): recentering a channel onto
        ~0 zeroes every axial/off-axis/shutoff cell for it — no variation, so
        beta is unidentified and the spend floor's zero gradient means the
        channel can never be resurrected by the loop. Each channel is therefore
        clamped to ``>= min_frac * (B / K)`` in the center's scaled units
        (``min_frac=0`` restores the old unfloored behavior). Warns whenever a
        clamp fires.
        """
        alloc = np.asarray(alloc, dtype=float)
        floor = float(min_frac) * float(self.B) / max(len(self.channels), 1)
        if min_frac > 0 and np.any(alloc < floor):
            clamped = [self.channels[i] for i in np.nonzero(alloc < floor)[0]]
            warnings.warn(
                f"recenter: clamping {clamped} up to the CCD floor {floor:.4g} "
                "(a multiplicative design at a ~zero center has no variation, "
                "so the channel could never be identified or resurrected)",
                stacklevel=2,
            )
            alloc = np.maximum(alloc, floor)
        self.center = alloc


def run_closed_loop(
    world: TrueWorld,
    *,
    center: np.ndarray,
    B: float,
    value: float,
    n_geo: int = 80,
    t_pre: int = 6,
    t_test: int = 10,
    delta: float = 0.6,
    noise: float = 0.6,
    n_holdout: int = 0,
    mode: str = "fixed",
    cap: float | None = None,
    gamma_scale: float = 0.8,
    pair_signs: dict[Pair, str] | None = None,
    margin: float = 1.0,
    population: float = 1.0,
    wave_cost: float = 0.05,
    max_waves: int = 4,
    recenter: bool = True,
    planner_q: int = 120,
    fit_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full loop against a known world and return the decision trace.

    Each pass: refit on all data, compute the recommendation + funding line +
    expected regret, evaluate the ENBS stopping rule, then (unless stopping)
    recenter on the recommendation, run a fresh designed wave over the SAME geos,
    and ingest it. Returns the per-wave :class:`WaveRecord` history plus the
    final recommendation and the truth-optimal target for comparison.
    """
    fit_kwargs = fit_kwargs or {}
    center = np.asarray(center, dtype=float)
    state = LearningState(
        channels=world.channels,
        center=center,
        B=B,
        value=value,
        pairs=world.pairs,
        pair_signs=pair_signs,
        activation=world.activation,
        mode=mode,
        cap=cap,
        gamma_scale=gamma_scale,
    )

    # wave 0: the initial CCD panel with the pre-period that pins the baselines
    wave0 = simulate_panel(
        world,
        center,
        n_geo=n_geo,
        t_pre=t_pre,
        t_test=t_test,
        delta=delta,
        noise=noise,
        n_holdout=n_holdout,
        seed=seed,
    )
    a_geo = np.asarray(wave0["a_geo"], dtype=float)  # the closure driver owns truth
    state.ingest(wave0)
    state.fit(**fit_kwargs)

    true_alloc, true_profit = world_optimal_allocation(
        world, B, value, mode=mode, cap=cap
    )

    for wave in range(max_waves):
        e_regret, _consensus, _alloc_sd, _profit_sd = state.regret(q=planner_q)
        rec = state.recommend(q=planner_q)
        mroas_mean, prob_above, _ = state.funding(rec, q=planner_q)
        stop, enbs_val = _planner.should_stop(
            e_regret, margin=margin, population=population, wave_cost=wave_cost
        )

        rec_response = float(world.response_mean(rec[None, :])[0])
        rec_profit = _profit_under(rec_response, rec, value, mode, B)
        gap = float(true_profit - rec_profit)
        gap_rel = gap / max(abs(true_profit), 1e-9)

        state.history.append(
            WaveRecord(
                wave=wave,
                n_rows=int(state.data["spend"].shape[0]),
                e_regret=float(e_regret),
                enbs=float(enbs_val),
                stop=bool(stop),
                recommendation=[float(x) for x in rec],
                funded=[bool(p > 0.5) for p in prob_above],
                mroas_mean=[float(x) for x in mroas_mean],
                prob_above_line=[float(x) for x in prob_above],
                profit_gap=gap,
                profit_gap_rel=gap_rel,
                max_rhat=state.posterior.diagnostics.get("max_rhat"),
                n_summaries=len(state.summaries),
            )
        )

        if stop or wave == max_waves - 1:
            break

        new_center = rec if recenter else center
        design = central_composite(new_center, delta, world.pairs)
        wave_next = simulate_wave(
            world,
            design,
            a_geo,
            t_test=t_test,
            center=new_center,
            n_holdout=n_holdout,
            noise=noise,
            seed=seed + wave + 1,
        )
        state.recenter(new_center)
        state.ingest(wave_next)
        state.fit(**fit_kwargs)

    return {
        "channels": world.channels,
        "history": [asdict(r) for r in state.history],
        "final_recommendation": state.history[-1].recommendation,
        "true_allocation": [float(x) for x in true_alloc],
        "true_profit": float(true_profit),
        "answer_key": world.answer_key(),
        "state": state,
    }


def due_for_retest(
    sigma_post: float,
    weeks_elapsed: float,
    channel: str,
    sigma_exp: float,
    *,
    half_life_overrides: dict[str, float] | None = None,
) -> tuple[bool, float]:
    """Whether a channel's evidence has decayed enough to warrant a re-test.

    Thin reuse of :func:`mmm_framework.planning.eig.reexperiment_due` so the
    continuous-learning loop's re-test trigger uses the framework's existing
    information-decay model (``sigma_eff^2(t) = sigma_post^2 exp(lambda t)``,
    ``lambda = ln2 / half_life``) rather than a private one. Returns
    ``(due, current_eig_nats)``.
    """
    from mmm_framework.planning.eig import channel_half_life, reexperiment_due

    half_life = channel_half_life(channel, half_life_overrides)
    return reexperiment_due(sigma_post, weeks_elapsed, half_life, sigma_exp)
