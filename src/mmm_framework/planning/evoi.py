"""Expected value of information (EVOI) for the budget decision.

EIG says how much an experiment teaches; EVOI says how much that learning is
WORTH — the expected improvement in the budget allocation's outcome from
deciding with the experiment's information instead of without it
(preposterior analysis / EVSI for the allocation decision):

    EVOI_k = E_y[ max_a E_{theta|y} f(a, theta) ] - max_a E_theta f(a, theta)

with the outer expectation over the prior predictive of the experiment outcome
``y ~ N(roi_k, sigma_exp^2)``. EVPI (perfect information about everything) is
the upper bound: ``EVPI = E_d[f_d(a_d*)] - E_d[f_d(a*)]``.

Units: KPI contribution over the response-curve window — the same units as
``optimize_budget``'s ``expected_uplift``.

Reuses the greedy allocator and per-draw machinery from ``planning.budget``.
Import-light (numpy only) so it can run inside the session kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .budget import ResponseCurves, _eval_allocation, _greedy_allocate


def _eval_allocation_all_draws(
    alloc: np.ndarray,  # (C,)
    contributions: np.ndarray,  # (D, C, G)
    spend_grid: np.ndarray,  # (C, G)
) -> np.ndarray:
    """Total contribution of one allocation under EVERY draw's curves, (D,).
    Vectorized: the interpolation weights depend only on (alloc, spend_grid),
    so they're computed once per channel and applied across draws."""
    D, C, G = contributions.shape
    out = np.zeros(D)
    for c in range(C):
        grid = spend_grid[c]
        v = float(np.clip(alloc[c], grid[0], grid[-1]))
        i = int(np.clip(np.searchsorted(grid, v), 1, G - 1))
        denom = grid[i] - grid[i - 1]
        frac = 0.0 if denom <= 0 else (v - grid[i - 1]) / denom
        out += contributions[:, c, i - 1] * (1 - frac) + contributions[:, c, i] * frac
    return out


def _default_bounds(
    curves: ResponseCurves,
    total_budget: float | None,
    lo_spend: np.ndarray | None,
    hi_spend: np.ndarray | None,
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Same defaults as optimize_budget: pure reallocation of current total
    spend, per-channel bounds as multiples of current spend clamped to the
    sampled curve range."""
    base = curves.base_spend.astype(float)
    if total_budget is None:
        total_budget = float(base.sum())
    grid_max = float(curves.multipliers.max())
    if lo_spend is None:
        lo_spend = base * min_multiplier
    if hi_spend is None:
        hi_spend = base * min(max_multiplier, grid_max)
    return float(total_budget), np.asarray(lo_spend), np.asarray(hi_spend)


@dataclass
class EvoiResult:
    """Portfolio-level value-of-information summary."""

    v_current: float  # E_d[f_d(a*)] — value of acting on today's posterior
    evpi: float  # value of perfect information (upper bound on every EVOI)
    n_draws: int


def compute_evpi(
    curves: ResponseCurves,
    *,
    total_budget: float | None = None,
    lo_spend: np.ndarray | None = None,
    hi_spend: np.ndarray | None = None,
    n_steps: int = 400,
    per_draw_alloc: np.ndarray | None = None,
    optimal_alloc: np.ndarray | None = None,
) -> EvoiResult:
    """V_current and EVPI from the response curves.

    Pass ``per_draw_alloc``/``optimal_alloc`` from a ``BudgetOptimizationResult``
    to reuse its D greedy optimizations instead of recomputing them.
    """
    total_budget, lo_spend, hi_spend = _default_bounds(
        curves, total_budget, lo_spend, hi_spend
    )
    spend_grid = curves.spend_grid
    D = curves.contributions.shape[0]

    if optimal_alloc is None:
        optimal_alloc = _greedy_allocate(
            curves.mean_curves(), spend_grid, total_budget, lo_spend, hi_spend, n_steps
        )
    v_current = float(
        _eval_allocation_all_draws(
            optimal_alloc, curves.contributions, spend_grid
        ).mean()
    )
    if per_draw_alloc is None:
        per_draw_alloc = np.stack(
            [
                _greedy_allocate(
                    curves.contributions[d],
                    spend_grid,
                    total_budget,
                    lo_spend,
                    hi_spend,
                    n_steps,
                )
                for d in range(D)
            ]
        )
    v_perfect = float(
        np.mean(
            [
                _eval_allocation(per_draw_alloc[d], curves.contributions[d], spend_grid)
                for d in range(D)
            ]
        )
    )
    return EvoiResult(
        v_current=v_current, evpi=max(0.0, v_perfect - v_current), n_draws=D
    )


def compute_evoi_for_channel(
    curves: ResponseCurves,
    channel_idx: int,
    roi_draws: np.ndarray,
    sigma_exp: float,
    *,
    optimal_alloc: np.ndarray | None = None,
    n_outcomes: int = 48,
    n_steps: int = 400,
    total_budget: float | None = None,
    lo_spend: np.ndarray | None = None,
    hi_spend: np.ndarray | None = None,
    outcome_draws: tuple[np.ndarray, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """EVOI of an experiment on one channel, by paired preposterior Monte Carlo.

    For each simulated outcome ``y_j = roi[d_j] + sigma_exp * z_j``: reweight
    the posterior draws by ``w_d ∝ N(y_j | roi_d, sigma_exp)``, re-optimize the
    allocation on the REWEIGHTED mean curves, and score the per-outcome gain

        gain_j = V_j(a_j*) - V_j(a*)      with V_j(a) = E_{theta|y_j}[f(a, theta)]

    against the no-experiment allocation ``a*`` under the SAME reweighted
    posterior (the paired estimator: identical estimand by the tower property,
    but the common-mode outcome noise cancels, so EVOI cannot go negative from
    sampling luck — each gain_j >= 0 up to allocator discretization, enforced
    by a per-term floor).

    ``outcome_draws`` = (d_idx, z) lets the caller share common random numbers
    across channels so near-ties don't rank-flip on MC noise. The caller should
    cap the result at EVPI.
    """
    if sigma_exp <= 0:
        raise ValueError(f"sigma_exp must be positive, got {sigma_exp}")
    total_budget, lo_spend, hi_spend = _default_bounds(
        curves, total_budget, lo_spend, hi_spend
    )
    spend_grid = curves.spend_grid
    contributions = curves.contributions  # (D, C, G)
    D = contributions.shape[0]
    x = np.asarray(roi_draws, dtype=float)
    if x.shape[0] != D:
        raise ValueError(f"roi_draws has {x.shape[0]} draws, curves have {D}")

    if optimal_alloc is None:
        optimal_alloc = _greedy_allocate(
            curves.mean_curves(), spend_grid, total_budget, lo_spend, hi_spend, n_steps
        )

    if outcome_draws is not None:
        d_idx, z = outcome_draws
    else:
        rng = rng or np.random.default_rng(0)
        d_idx = rng.integers(0, D, size=n_outcomes)
        z = rng.standard_normal(n_outcomes)
    ys = x[d_idx] + sigma_exp * z

    static_vals = _eval_allocation_all_draws(optimal_alloc, contributions, spend_grid)
    gains = np.empty(len(ys))
    for j, y in enumerate(ys):
        logw = -0.5 * ((y - x) / sigma_exp) ** 2
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        # posterior-mean curves after the experiment: (C, G)
        weighted_curves = np.tensordot(w, contributions, axes=(0, 0))
        a_j = _greedy_allocate(
            weighted_curves, spend_grid, total_budget, lo_spend, hi_spend, n_steps
        )
        v_post = float(w @ _eval_allocation_all_draws(a_j, contributions, spend_grid))
        v_static = float(w @ static_vals)
        gains[j] = max(0.0, v_post - v_static)
    return float(gains.mean())


# ── Gaussian EVOI surrogate ────────────────────────────────────────────────────
#
# The preposterior MC above is exact but costs one re-optimization per simulated
# outcome — prohibitive across a design GRID (dozens of candidate designs per
# channel). The surrogate below prices every candidate from at most TWO anchored
# MC EVOIs, using the Gaussian preposterior geometry (Raiffa–Schlaifer):
#
# With prior roi ~ N(mu, tau^2) and an experiment observing y ~ N(roi,
# sigma_exp^2), the posterior mean is itself random BEFORE the experiment runs,
# with sd  s(sigma_exp) = tau * sqrt(tau^2 / (tau^2 + sigma_exp^2)).
# In the two-action linear-loss problem the experiment's value is
#
#     EVSI(sigma) = k * s(sigma) * Psi(delta / s(sigma))
#
# where Psi(u) = phi(u) − u * (1 − PHI(u)) is the unit-normal loss integral,
# delta = |mu − b| the distance from the prior mean to the decision boundary,
# and k the $-value per unit of decision-relevant mean shift. The budget
# decision is not literally two-action, but empirically the MC EVOI curve has
# exactly this shape: fitting (k, delta) to two MC anchors reproduces the MC
# EVOI to ~±15% ANYWHERE BETWEEN the anchors (even for a bimodal ROI
# posterior); extrapolation far outside the bracket under-estimates, so anchor
# at the extremes of the sigma range you need to price.


def _phi(u: float) -> float:
    return math.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)


def _Phi(u: float) -> float:
    return 0.5 * (1.0 + math.erf(u / math.sqrt(2.0)))


def _psi(u: float) -> float:
    """Unit-normal loss integral ``Psi(u) = phi(u) − u (1 − PHI(u))`` —
    monotone decreasing from Psi(0) = 0.3989 toward 0. Floored at a tiny
    positive value: for large ``u`` the float expression underflows (or goes
    negative by cancellation) and would zero-divide the anchor-ratio solve."""
    return max(_phi(u) - u * (1.0 - _Phi(u)), 1e-300)


def preposterior_sd_ratio(tau: float, sigma_exp: float) -> float:
    """sd of the preposterior mean as a FRACTION of the prior sd ``tau``:
    ``sqrt(tau^2 / (tau^2 + sigma_exp^2))`` — 1 for a perfect experiment
    (sigma_exp → 0), → 0 for an uninformative one."""
    tau = float(tau)
    sigma_exp = float(sigma_exp)
    if tau <= 0:
        return 0.0
    if sigma_exp <= 0:
        return 1.0
    return float(np.sqrt(tau**2 / (tau**2 + sigma_exp**2)))


def surrogate_evoi(
    evoi_ref: float,
    sigma_ref: float,
    sigma_new: float,
    tau: float,
    *,
    evpi: float | None = None,
) -> float:
    """Single-anchor fallback: EVOI scaled by the preposterior-sd ratio
    ``s(sigma_new)/s(sigma_ref)`` (the ``delta = 0`` special case of the fitted
    surrogate — decision boundary at the prior mean). Prefer
    :func:`fit_evoi_surrogate` with two anchors when you can afford the second
    MC evaluation; the single-anchor form decays too slowly for weak designs.
    Capped at ``evpi`` when given; 0 when the anchor or prior is degenerate."""
    ref_scale = preposterior_sd_ratio(tau, sigma_ref)
    if not np.isfinite(evoi_ref) or evoi_ref <= 0 or ref_scale <= 0:
        return 0.0
    v = float(evoi_ref) * preposterior_sd_ratio(tau, sigma_new) / ref_scale
    if evpi is not None and np.isfinite(evpi):
        v = min(v, max(float(evpi), 0.0))
    return max(0.0, v)


@dataclass(frozen=True)
class EvoiSurrogate:
    """Calibrated Gaussian EVOI surrogate ``sigma → k·s(sigma)·Psi(delta/s)``.

    ``tau`` is the channel's prior (pre-experiment) ROI sd; ``k`` and ``delta``
    are fitted to the MC anchors. Call it with any design precision to price
    that design's EVOI without another preposterior MC. Accurate BETWEEN the
    anchor sigmas; extrapolation beyond the weak anchor under-estimates.
    """

    k: float
    delta: float
    tau: float
    sigma_anchors: tuple[float, ...] = ()

    def __call__(self, sigma_exp: float, *, evpi: float | None = None) -> float:
        s = self.tau * preposterior_sd_ratio(self.tau, sigma_exp)
        if s <= 0 or self.k <= 0:
            return 0.0
        v = self.k * s * _psi(self.delta / s)
        if evpi is not None and np.isfinite(evpi):
            v = min(v, max(float(evpi), 0.0))
        return max(0.0, v)


def fit_evoi_surrogate(
    tau: float,
    anchors: list[tuple[float, float]],
) -> EvoiSurrogate | None:
    """Fit ``(k, delta)`` of :class:`EvoiSurrogate` to MC anchors
    ``[(sigma_exp, evoi_mc), ...]``.

    Two usable anchors (distinct sigmas, both positive EVOI) pin both
    parameters by bisection on the anchor ratio; one usable anchor degrades to
    ``delta = 0`` (the :func:`surrogate_evoi` scaling). Returns ``None`` when
    ``tau`` is degenerate or no anchor has positive EVOI."""
    tau = float(tau)
    if tau <= 0 or not np.isfinite(tau):
        return None
    usable = sorted(
        {
            (float(s), float(v))
            for s, v in anchors
            if np.isfinite(s) and s > 0 and np.isfinite(v) and v > 0
        }
    )
    if not usable:
        return None

    def s_of(sig: float) -> float:
        return tau * preposterior_sd_ratio(tau, sig)

    if len(usable) == 1:
        sig1, v1 = usable[0]
        return EvoiSurrogate(
            k=v1 / (s_of(sig1) * _psi(0.0)),
            delta=0.0,
            tau=tau,
            sigma_anchors=(sig1,),
        )

    # two anchors: the sharpest and the weakest usable design
    (sig1, v1), (sig2, v2) = usable[0], usable[-1]
    s1, s2 = s_of(sig1), s_of(sig2)
    if s1 <= s2 or v2 <= 0:  # indistinguishable precisions → single-anchor
        return EvoiSurrogate(
            k=v1 / (s1 * _psi(0.0)), delta=0.0, tau=tau, sigma_anchors=(sig1,)
        )

    def ratio(delta: float) -> float:
        return (s1 * _psi(delta / s1)) / (s2 * _psi(delta / s2))

    target = v1 / v2
    if ratio(0.0) >= target:
        delta = 0.0  # anchors flatter than pure s-scaling → boundary at mean
    else:
        lo, hi = 0.0, 20.0 * tau
        if ratio(hi) < target:
            delta = hi  # extreme separation — clamp (surrogate stays monotone)
        else:
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if ratio(mid) < target:
                    lo = mid
                else:
                    hi = mid
            delta = 0.5 * (lo + hi)
    k = v1 / (s1 * _psi(delta / s1))
    return EvoiSurrogate(k=k, delta=delta, tau=tau, sigma_anchors=(sig1, sig2))
