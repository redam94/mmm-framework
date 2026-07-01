"""Layer 2 — experimental design (central-composite geo cells).

A *wave* is a designed batch of geo cells run for a fixed window. The
central-composite design (CCD) is the minimal set of scaled allocations that
identifies the local response surface around an operating point:

* **1 center cell** — the current operating allocation (the trust-region
  center).
* **2K axial cells** — each channel scaled to ``(1 +/- delta)`` while the others
  hold at center; these give the gradient / main effects.
* **2 off-axis cells per probed pair** — two channels moved *jointly*
  ``(1 +/- delta)``; the only way to recover ``d^2 R / ds_c ds_c'`` (the synergy
  gamma). Probe the decision-pivotal pairs only.
* **K shutoff cells** — one channel set to 0. These isolate the remaining terms
  and **break the beta/gamma collinearity**; without them beta attenuates.

``delta`` is a *multiplicative* trust-region step (a spend-variation fraction):
``delta = 0.6`` moves a channel +/-60% off its center. A shutoff is the
``-100%`` corner. Working multiplicatively keeps the design scale-free, so it
behaves the same whether spend is scaled to O(1) or left in dollars.
"""

from __future__ import annotations

import numpy as np

Pair = tuple[int, int]


def central_composite(
    center: np.ndarray,
    delta: float,
    probe_pairs: list[Pair],
) -> np.ndarray:
    """Build the CCD cells (rows = cells, columns = channels).

    Args:
        center: the operating allocation, shape ``(K,)`` (scaled spend).
        delta: multiplicative trust-region step in ``(0, 1]`` (e.g. ``0.6``).
        probe_pairs: channel pairs to identify the cross-partial for.

    Returns:
        A ``(n_cells, K)`` array of non-negative scaled allocations, with
        ``n_cells = 1 + 2K + 2 * len(probe_pairs) + K``.
    """
    center = np.asarray(center, dtype=float)
    if center.ndim != 1:
        raise ValueError(f"center must be 1-D (K,), got shape {center.shape}")
    if not 0.0 < delta <= 1.0:
        raise ValueError(f"delta must be in (0, 1], got {delta}")
    k = center.shape[0]
    cells = [center.copy()]  # center

    for c in range(k):  # axial +/- delta
        hi = center.copy()
        hi[c] = center[c] * (1.0 + delta)
        cells.append(hi)
        lo = center.copy()
        lo[c] = center[c] * (1.0 - delta)
        cells.append(lo)

    for i, j in probe_pairs:  # off-axis joint moves
        if not (0 <= i < k and 0 <= j < k) or i == j:
            raise ValueError(f"probe pair {(i, j)} invalid for {k} channels")
        plus = center.copy()
        plus[i] *= 1.0 + delta
        plus[j] *= 1.0 + delta
        cells.append(plus)
        minus = center.copy()
        minus[i] *= 1.0 - delta
        minus[j] *= 1.0 - delta
        cells.append(minus)

    for c in range(k):  # shutoff
        z = center.copy()
        z[c] = 0.0
        cells.append(z)

    return np.clip(np.stack(cells), 0.0, None)


def assign_geos(
    design: np.ndarray,
    n_geo: int,
    rng: np.random.Generator,
    *,
    n_holdout: int = 0,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Round-robin (shuffled) assignment of CCD cells to geos.

    Round-robin keeps cells balanced across geos. In production, stratify on
    pre-period KPI level/variance (matched-market style) rather than pure
    round-robin to minimize baseline imbalance — see
    :func:`mmm_framework.planning.design.matched_pairs`.

    Args:
        design: CCD cells, shape ``(n_cells, K)``.
        n_geo: number of geos.
        rng: a numpy random generator (caller owns the seed).
        n_holdout: hold the first ``n_holdout`` geos at ``center`` for the test
            window (a status-quo counterfactual). Requires ``center``.
        center: the status-quo allocation for holdout geos, shape ``(K,)``.

    Returns:
        ``(geo_alloc, cell_idx)`` where ``geo_alloc`` is ``(n_geo, K)`` (each
        geo's test allocation) and ``cell_idx`` is ``(n_geo,)`` (the design-row
        index, or ``-1`` for a holdout geo).
    """
    n_cells = design.shape[0]
    reps = int(np.ceil(n_geo / n_cells))
    cell_idx = np.tile(np.arange(n_cells), reps)[:n_geo].copy()
    rng.shuffle(cell_idx)
    geo_alloc = design[cell_idx].copy()

    if n_holdout > 0:
        if center is None:
            raise ValueError("n_holdout > 0 requires center for the status-quo cell")
        n_holdout = min(n_holdout, n_geo)
        geo_alloc[:n_holdout] = np.asarray(center, dtype=float)
        cell_idx[:n_holdout] = -1
    return geo_alloc, cell_idx
