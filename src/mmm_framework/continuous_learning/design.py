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

from typing import Sequence

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
    baseline: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign CCD cells to geos — shuffled round-robin, or stratified/blocked.

    Without a ``baseline`` (the default, byte-identical to the historical
    behavior) cells are tiled round-robin and shuffled: balanced cell counts,
    zero covariate awareness, and the *first* ``n_holdout`` geos (positional)
    become holdouts.

    With a ``baseline`` (a per-geo covariate such as the pre-period KPI level,
    matched-market style) the assignment is a classic **blocked randomization**:
    geos are sorted by baseline, walked in blocks of ``n_cells``, and each block
    receives a random permutation of the cell indices (the ragged tail gets a
    random subset without replacement). Every cell's geos are then spread evenly
    across the baseline distribution, so between-cell baseline means are nearly
    equal. Holdouts are carved FIRST and are stratum-aware too: exactly
    ``n_holdout`` evenly spaced positions in baseline-sorted order, so the
    status-quo counterfactual spans the baseline range AND honors the requested
    count (a strided pick would silently under-deliver whenever ``n_holdout``
    does not divide ``n_geo`` evenly).

    Args:
        design: CCD cells, shape ``(n_cells, K)``.
        n_geo: number of geos.
        rng: a numpy random generator (caller owns the seed).
        n_holdout: hold ``n_holdout`` geos at ``center`` for the test window (a
            status-quo counterfactual). Requires ``center``.
        center: the status-quo allocation for holdout geos, shape ``(K,)``.
        baseline: optional per-geo covariate, length ``n_geo``, positionally
            aligned with the geo indices (the caller resolves geo ids). ``None``
            keeps the legacy shuffled round-robin path.

    Returns:
        ``(geo_alloc, cell_idx)`` where ``geo_alloc`` is ``(n_geo, K)`` (each
        geo's test allocation) and ``cell_idx`` is ``(n_geo,)`` (the design-row
        index, or ``-1`` for a holdout geo).
    """
    n_cells = design.shape[0]

    if baseline is None:
        # Legacy shuffled round-robin (byte-identical rng stream).
        reps = int(np.ceil(n_geo / n_cells))
        cell_idx = np.tile(np.arange(n_cells), reps)[:n_geo].copy()
        rng.shuffle(cell_idx)
        geo_alloc = design[cell_idx].copy()

        if n_holdout > 0:
            if center is None:
                raise ValueError(
                    "n_holdout > 0 requires center for the status-quo cell"
                )
            n_holdout = min(n_holdout, n_geo)
            geo_alloc[:n_holdout] = np.asarray(center, dtype=float)
            cell_idx[:n_holdout] = -1
        return geo_alloc, cell_idx

    baseline = np.asarray(baseline, dtype=float)
    if baseline.shape != (n_geo,):
        raise ValueError(
            f"baseline must have shape ({n_geo},) — one value per geo, "
            f"positionally aligned; got {baseline.shape}"
        )
    order = np.argsort(baseline, kind="stable")  # geo positions, baseline-sorted

    # Holdouts first: EXACTLY n_holdout evenly spaced positions in
    # baseline-sorted order — the status-quo counterfactual covers the baseline
    # range. (A strided ``order[::step][:n_holdout]`` pick returns only
    # ceil(n_geo/step) geos — fewer than requested whenever n_holdout does not
    # divide n_geo evenly — so the pre-registered design would contradict its
    # own assignment.)
    holdout_geos = np.empty(0, dtype=int)
    if n_holdout > 0:
        if center is None:
            raise ValueError("n_holdout > 0 requires center for the status-quo cell")
        n_holdout = min(n_holdout, n_geo)
        pos = np.unique(np.round(np.linspace(0.0, n_geo - 1.0, n_holdout)).astype(int))
        if pos.size < n_holdout:  # defensive: rounding collisions (spacing >= 1
            # makes them impossible for n_holdout <= n_geo, but never silently
            # under-deliver) — top up with the nearest unused sorted positions.
            unused = np.setdiff1d(np.arange(n_geo), pos)
            dist = np.abs(unused[:, None] - pos[None, :]).min(axis=1)
            take = unused[np.argsort(dist, kind="stable")][: n_holdout - pos.size]
            pos = np.sort(np.concatenate([pos, take]))
        holdout_geos = order[pos]

    # Blocked randomization over the remaining geos (still baseline-sorted):
    # each block of n_cells gets a random permutation of the cell indices, so
    # per-cell counts match the round-robin tiling and each cell's geos spread
    # evenly over the baseline distribution.
    remaining = order[~np.isin(order, holdout_geos)]
    cell_idx = np.zeros(n_geo, dtype=int)
    for start in range(0, remaining.size, n_cells):
        block = remaining[start : start + n_cells]
        if block.size == n_cells:
            cell_idx[block] = rng.permutation(n_cells)
        else:  # ragged tail: a random subset without replacement
            cell_idx[block] = rng.choice(n_cells, size=block.size, replace=False)
    geo_alloc = design[cell_idx].copy()

    if holdout_geos.size:
        geo_alloc[holdout_geos] = np.asarray(center, dtype=float)
        cell_idx[holdout_geos] = -1
    return geo_alloc, cell_idx
