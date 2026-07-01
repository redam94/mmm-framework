"""Synthetic geo-RSM world with causal ground truth (recovery + fantasies).

A :class:`TrueWorld` is a known response surface (``beta``, ``kappa``, ``alpha``,
``gamma``) plus geo-intercept hyperparameters. :func:`simulate_panel` runs a
central-composite wave against it and returns the data contract of guide §3 plus
the answer key — this is BOTH the recovery harness (fit and compare to truth) and
the engine that fantasizes wave outcomes inside the knowledge-gradient
acquisition.

The DGP evaluates the **same** :func:`mmm_framework.continuous_learning.surface`
functions the model fits, so a successful recovery test certifies the exact map
the planner later exploits — there is no second, drifting implementation of the
surface.

Identification (guide §3.2) is built in: a pre-period where every geo shares the
``center`` allocation (pins each geo intercept), then a test-period of designed
cross-sectional variation (the CCD cells). Without the pre-period the geo
intercept and the incremental response are collinear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .design import assign_geos, central_composite
from .model import Pair, default_pairs
from .surface import ACTIVATIONS, surface_over_rows


@dataclass
class TrueWorld:
    """A known response surface for recovery / closure testing.

    ``gamma_pairs`` is aligned to ``pairs`` (one synergy value per pair); the
    matrix form is assembled on demand. ``A``/``sigma_a`` set the geo-intercept
    distribution. The activation is pluggable: pass Hill ``kappa``/``alpha`` (the
    default and back-compatible path), or set ``activation`` + ``shape`` for
    another family (e.g. ``activation="logistic"``, ``shape={"lam": …}``).
    """

    beta: np.ndarray  # (K,)
    gamma_pairs: np.ndarray  # (len(pairs),) aligned to `pairs`
    channels: list[str]
    kappa: np.ndarray | None = None  # Hill half-saturation (back-compat kwarg)
    alpha: np.ndarray | None = None  # Hill shape (back-compat kwarg)
    activation: str = "hill"
    shape: dict[str, np.ndarray] = field(default_factory=dict)  # {param: (K,)}
    pairs: list[Pair] = field(default_factory=list)
    a_level: float = 4.0
    sigma_a: float = 1.0

    def __post_init__(self) -> None:
        self.beta = np.asarray(self.beta, dtype=float)
        self.gamma_pairs = np.asarray(self.gamma_pairs, dtype=float)
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.activation!r}; known: {tuple(ACTIVATIONS)}"
            )
        names = ACTIVATIONS[self.activation][0]
        # Hill back-compat: kappa/alpha kwargs populate the generic shape dict.
        if self.activation == "hill" and not self.shape and self.kappa is not None:
            self.shape = {"kappa": self.kappa, "alpha": self.alpha}
        self.shape = {k: np.asarray(v, dtype=float) for k, v in self.shape.items()}
        missing = [n for n in names if n not in self.shape]
        if missing:
            raise ValueError(
                f"activation {self.activation!r} needs shape params {names}; "
                f"missing {missing}"
            )
        if self.activation == "hill":  # convenience attributes
            self.kappa, self.alpha = self.shape["kappa"], self.shape["alpha"]
        if not self.pairs:
            self.pairs = default_pairs(len(self.channels))
        if self.gamma_pairs.shape[0] != len(self.pairs):
            raise ValueError(
                f"gamma_pairs has {self.gamma_pairs.shape[0]} values but there "
                f"are {len(self.pairs)} pairs"
            )

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def act_fn(self):
        """The JAX activation for this world's family."""
        return ACTIVATIONS[self.activation][1]

    def shape_tuple(self) -> tuple:
        """Shape-parameter arrays in the activation's parameter order."""
        return tuple(self.shape[n] for n in ACTIVATIONS[self.activation][0])

    def gamma_matrix(self) -> np.ndarray:
        k = self.n_channels
        g = np.zeros((k, k))
        for (i, j), val in zip(self.pairs, self.gamma_pairs):
            g[i, j] = float(val)
        return g

    def response_mean(self, spend_matrix: np.ndarray) -> np.ndarray:
        """Mean incremental response for a ``(N, K)`` spend panel (no noise)."""
        return np.asarray(
            surface_over_rows(
                spend_matrix,
                self.beta,
                self.gamma_matrix(),
                self.act_fn(),
                self.shape_tuple(),
            ),
            dtype=float,
        )

    def answer_key(self) -> dict[str, object]:
        key: dict[str, object] = {
            "channels": list(self.channels),
            "beta": self.beta.tolist(),
            "activation": self.activation,
            "shape": {k: v.tolist() for k, v in self.shape.items()},
            "pairs": [list(p) for p in self.pairs],
            "gamma_pairs": self.gamma_pairs.tolist(),
            "a_level": self.a_level,
            "sigma_a": self.sigma_a,
        }
        if self.activation == "hill":  # keep flat kappa/alpha for older readers
            key["kappa"], key["alpha"] = self.kappa.tolist(), self.alpha.tolist()
        return key


def make_world(seed: int = 0, channels: list[str] | None = None) -> TrueWorld:
    """A reproducible, non-trivial 4-channel world for tests and the demo.

    Channels ``["Chatter", "Pulse", "Orbit", "Vibe"]`` (generic social-network
    surfaces) with a mix of strong and weak main effects, distinct half-
    saturations and Hill shapes, and a synergy structure that includes a genuine
    **cannibalization** (negative gamma) and a **complementarity** (positive
    gamma) so recovery has to get signs right.
    """
    channels = channels or ["Chatter", "Pulse", "Orbit", "Vibe"]
    rng = np.random.default_rng(seed)
    k = len(channels)
    pairs = default_pairs(k)
    # Deterministic-ish core with a little seed jitter so multiple worlds differ.
    beta = np.array([2.4, 1.8, 1.1, 0.9])[:k] * (1 + 0.05 * rng.standard_normal(k))
    kappa = np.array([0.8, 0.6, 1.0, 0.7])[:k] * (1 + 0.05 * rng.standard_normal(k))
    alpha = np.array([2.2, 1.6, 2.6, 1.3])[:k]
    # gamma per pair aligned to default_pairs order:
    # (0,1) Chatter x Pulse -> negative (audience-overlap cannibalization)
    # (0,2),(0,3) ~0 ; (1,2) Pulse x Orbit -> positive (complementarity)
    # (1,3) ~0 ; (2,3) Orbit x Vibe -> positive
    gamma_lookup = {
        (0, 1): -0.6,
        (0, 2): 0.0,
        (0, 3): 0.0,
        (1, 2): 0.5,
        (1, 3): 0.0,
        (2, 3): 0.4,
    }
    gamma_pairs = np.array([gamma_lookup.get(p, 0.0) for p in pairs])
    return TrueWorld(
        beta=beta,
        kappa=kappa,
        alpha=alpha,
        gamma_pairs=gamma_pairs,
        channels=channels,
        pairs=pairs,
        a_level=4.0,
        sigma_a=1.0,
    )


def make_world_logistic(seed: int = 0, channels: list[str] | None = None) -> TrueWorld:
    """A known **logistic** (exponential-saturation) world, sibling of
    :func:`make_world`.

    Same channels and synergy structure, but each channel saturates as
    ``f(s) = 1 - exp(-lam * s)`` — a concave curve with no S-shape — rather than a
    Hill. Demonstrates that the whole loop (fit, plan, stop, animate) is
    activation-agnostic. The saturation rates ``lam`` are chosen so half-
    saturation ``ln(2)/lam`` lands in the same O(1) scaled range as the Hill
    ``kappa`` above, keeping the two worlds comparable in scale.
    """
    channels = channels or ["Chatter", "Pulse", "Orbit", "Vibe"]
    rng = np.random.default_rng(seed)
    k = len(channels)
    pairs = default_pairs(k)
    beta = np.array([2.4, 1.8, 1.1, 0.9])[:k] * (1 + 0.05 * rng.standard_normal(k))
    lam = np.array([1.1, 1.4, 0.8, 1.2])[:k] * (1 + 0.05 * rng.standard_normal(k))
    gamma_lookup = {(0, 1): -0.6, (1, 2): 0.5, (2, 3): 0.4}
    gamma_pairs = np.array([gamma_lookup.get(p, 0.0) for p in pairs])
    return TrueWorld(
        beta=beta,
        gamma_pairs=gamma_pairs,
        channels=channels,
        activation="logistic",
        shape={"lam": lam},
        pairs=pairs,
        a_level=4.0,
        sigma_a=1.0,
    )


def make_world_hill_mixture(
    seed: int = 0, channels: list[str] | None = None
) -> TrueWorld:
    """A known **weighted-sum-of-two-Hills** world (a misspecification stress test).

    Each channel's true response is ``w·Hill(κ1,α1) + (1-w)·Hill(κ2,α2)`` with a
    low-κ, high-α early component (a soft activation threshold) plus a high-κ
    later component — a two-phase / shoulder shape a **single** Hill can only
    average over and a **logistic** (concave, no inflection) cannot represent at
    all. Fit it with ``activation="hill"`` (mild misspecification) or
    ``"logistic"`` (severe) to see what a wrong response family does to the
    recovered curve, the funding line, and the recommended allocation.
    """
    channels = channels or ["Chatter", "Pulse", "Orbit", "Vibe"]
    rng = np.random.default_rng(seed)
    k = len(channels)
    pairs = default_pairs(k)
    beta = np.array([2.4, 1.8, 1.1, 0.9])[:k] * (1 + 0.05 * rng.standard_normal(k))
    # per-channel two-Hill mixtures with a genuine early-threshold + late shoulder
    kappa1 = np.array([0.40, 0.32, 0.50, 0.55])[:k]
    alpha1 = np.array([4.0, 5.0, 3.5, 3.0])[:k]  # steep early activation
    kappa2 = np.array([1.60, 1.35, 1.80, 1.30])[:k]
    alpha2 = np.array([2.0, 2.2, 2.4, 1.8])[:k]  # gentler late rise
    w = np.array([0.50, 0.45, 0.55, 0.50])[:k]
    gamma_lookup = {(0, 1): -0.6, (1, 2): 0.5, (2, 3): 0.4}
    gamma_pairs = np.array([gamma_lookup.get(p, 0.0) for p in pairs])
    return TrueWorld(
        beta=beta,
        gamma_pairs=gamma_pairs,
        channels=channels,
        activation="hill_mixture",
        shape={
            "kappa1": kappa1,
            "alpha1": alpha1,
            "kappa2": kappa2,
            "alpha2": alpha2,
            "w": w,
        },
        pairs=pairs,
        a_level=4.0,
        sigma_a=1.0,
    )


def draw_geo_intercepts(
    world: TrueWorld, n_geo: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample the per-geo baseline intercepts ``a_geo ~ Normal(A, sigma_a)``."""
    return rng.normal(world.a_level, world.sigma_a, size=n_geo)


def _stack_window(
    geo_alloc_per_geo: np.ndarray, n_geo: int, n_weeks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Tile a per-geo allocation over ``n_weeks`` -> (rows_spend, rows_geo)."""
    spend = np.repeat(geo_alloc_per_geo[None, :, :], n_weeks, axis=0).reshape(
        n_weeks * n_geo, -1
    )
    geo = np.tile(np.arange(n_geo), n_weeks)
    return spend, geo


def simulate_panel(
    world: TrueWorld,
    center: np.ndarray,
    *,
    n_geo: int = 80,
    t_pre: int = 6,
    t_test: int = 10,
    delta: float = 0.6,
    probe_pairs: list[Pair] | None = None,
    noise: float = 0.6,
    n_holdout: int = 0,
    a_geo: np.ndarray | None = None,
    adstock_alpha: float | None = None,
    adstock_l_max: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    """Simulate a CCD wave against a known world.

    Returns the data contract (``spend``, ``geo_idx``, ``n_geo``, ``y``) plus
    ``design``, ``geo_alloc``, ``cell_idx``, ``a_geo`` and ``answer_key``. Pass an
    existing ``a_geo`` (and the same ``seed`` offset) to simulate a *later* wave
    over the same geos with their baselines intact.

    ``adstock_alpha`` adds **carryover**: the response is driven by the
    geometric-adstocked spend series (within each geo over weeks), while the
    returned ``spend`` stays raw. Fitting on the raw panel is then biased; the
    ``preprocess.adstock_prepass`` recovers it (guide §9.4).
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=float)
    probe_pairs = probe_pairs if probe_pairs is not None else world.pairs

    design = central_composite(center, delta, probe_pairs)
    geo_per_geo, cell_idx = assign_geos(
        design, n_geo, rng, n_holdout=n_holdout, center=center
    )
    if a_geo is None:
        a_geo = draw_geo_intercepts(world, n_geo, rng)
    a_geo = np.asarray(a_geo, dtype=float)

    # pre-period: every geo at center (pins the intercept)
    pre_spend, pre_geo = _stack_window(
        np.repeat(center[None, :], n_geo, axis=0), n_geo, t_pre
    )
    # test-period: each geo at its assigned cell
    test_spend, test_geo = _stack_window(geo_per_geo, n_geo, t_test)

    spend = np.vstack([pre_spend, test_spend])
    geo_idx = np.concatenate([pre_geo, test_geo])
    # carryover: the response sees the adstocked series; the observed spend is raw
    response_spend = spend
    if adstock_alpha is not None:
        from .preprocess import adstock_panel

        response_spend = adstock_panel(
            spend, n_geo, t_pre, t_test, alpha=adstock_alpha, l_max=adstock_l_max
        )
    mu = a_geo[geo_idx] + world.response_mean(response_spend)
    y = mu + rng.normal(0.0, noise, size=mu.shape[0])

    return {
        "spend": spend,
        "geo_idx": geo_idx,
        "n_geo": n_geo,
        "y": y,
        "design": design,
        "geo_alloc": geo_per_geo,
        "cell_idx": cell_idx,
        "a_geo": a_geo,
        "answer_key": world.answer_key(),
    }


def simulate_wave(
    world: TrueWorld,
    design: np.ndarray,
    a_geo: np.ndarray,
    *,
    t_test: int = 10,
    center: np.ndarray | None = None,
    n_holdout: int = 0,
    noise: float = 0.6,
    seed: int = 0,
) -> dict[str, object]:
    """Simulate a single test window for a recentered design over the SAME geos.

    Used by the closed loop to generate wave ``t > 0`` outcomes: the geos and
    their baselines (``a_geo``) persist, only the test allocations change.
    """
    rng = np.random.default_rng(seed)
    n_geo = a_geo.shape[0]
    geo_per_geo, cell_idx = assign_geos(
        design, n_geo, rng, n_holdout=n_holdout, center=center
    )
    test_spend, test_geo = _stack_window(geo_per_geo, n_geo, t_test)
    mu = a_geo[test_geo] + world.response_mean(test_spend)
    y = mu + rng.normal(0.0, noise, size=mu.shape[0])
    return {
        "spend": test_spend,
        "geo_idx": test_geo,
        "n_geo": n_geo,
        "y": y,
        "design": design,
        "geo_alloc": geo_per_geo,
        "cell_idx": cell_idx,
        "a_geo": a_geo,
    }
