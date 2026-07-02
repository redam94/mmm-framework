"""Sub-channel arms — creatives / keywords / campaigns as surface dimensions.

A *split* channel (e.g. ``Search`` into ``Brand`` / ``NonBrand``) becomes
several **arms**, each a full surface dimension with its own ``beta``/shape/
interactions. Everything downstream is K-generic, so mechanically this is just
"grow K" — but three things need care, and this module owns them:

* **Naming.** Arms are flattened to ``f"{parent}{ARM_SEP}{sub}"`` (the same
  ``" │ "`` separator ``planning/budget.py`` uses for geo arms), so an arm name
  round-trips to its parent unambiguously.
* **Priors.** Within-parent siblings share an audience, so they *substitute* —
  their pairwise interaction defaults to ``"neg"`` (cannibalization), while
  cross-parent pairs default to ``"weak"``. :func:`default_arm_pair_signs`
  builds the full sign map.
* **Budgets.** The natural constraint is "the PARENT's budget is fixed, the mix
  within it is free": :class:`ArmSpec.groups` maps each parent to its arm
  indices, ready to feed the planner's ``group_budgets`` (one SLSQP equality
  constraint ``sum(s[group]) == B_parent`` per group).

Design-cost warning (why you should not split everything): the CCD cell count
is ``1 + 2K + 2|pairs| + K``, so every extra arm costs about 3 cells (2 axial +
1 shutoff) plus any probed pairs — and each cell needs at least one geo.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import Pair, PairSign, Posterior

# Same separator as planning/budget.py's geo arms — one house convention for
# "a thing inside a thing" names.
ARM_SEP = " │ "


@dataclass
class ArmSpec:
    """The flattened arm layout of a (possibly partially) split channel list.

    ``channels`` are the flattened arm names (surface dimensions, in order);
    ``parents[i]`` is arm ``i``'s parent channel (== the name itself when the
    channel is unsplit); ``groups`` maps every parent to its arm indices.
    """

    channels: list[str]
    parents: list[str]
    groups: dict[str, list[int]]

    @property
    def n_arms(self) -> int:
        return len(self.channels)

    def split_parents(self) -> list[str]:
        """Parents that were actually split into more than one arm."""
        return [p for p, idx in self.groups.items() if len(idx) > 1]


def expand_arms(channels: list[str], arms: dict[str, list[str]]) -> ArmSpec:
    """Flatten ``channels`` with ``arms`` (parent -> sub-names) into an ArmSpec.

    Channels absent from ``arms`` stay single arms named after themselves;
    split channels expand in place, preserving the channel order. Raises on an
    unknown parent, an empty/duplicated sub-list, or a flattened-name clash.
    """
    arms = arms or {}
    unknown = [p for p in arms if p not in channels]
    if unknown:
        raise ValueError(f"arms reference unknown channels {unknown} (of {channels})")
    flat: list[str] = []
    parents: list[str] = []
    for c in channels:
        subs = arms.get(c)
        if subs:
            if len(set(subs)) != len(subs):
                raise ValueError(f"duplicate arm names for channel {c!r}: {subs}")
            for s in subs:
                flat.append(f"{c}{ARM_SEP}{s}")
                parents.append(c)
        else:
            flat.append(c)
            parents.append(c)
    if len(set(flat)) != len(flat):
        raise ValueError(f"flattened arm names collide: {flat}")
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(parents):
        groups.setdefault(p, []).append(i)
    return ArmSpec(channels=flat, parents=parents, groups=groups)


def within_parent_pairs(spec: ArmSpec) -> list[Pair]:
    """All sibling pairs ``(i, j)``, ``i < j``, sharing a parent (substitutes)."""
    out: list[Pair] = []
    for idx in spec.groups.values():
        out.extend(
            (idx[a], idx[b]) for a in range(len(idx)) for b in range(a + 1, len(idx))
        )
    return sorted(out)


def cross_parent_pairs(
    spec: ArmSpec, pairs_of_parents: list[tuple[str, str]] | None = None
) -> list[Pair]:
    """Arm pairs spanning two different parents.

    ``pairs_of_parents`` restricts to specific (unordered) parent pairs — probe
    only the decision-pivotal cross-channel synergies rather than paying the
    full off-axis cell cost for every arm combination.
    """
    wanted: set[frozenset[str]] | None = None
    if pairs_of_parents is not None:
        for a, b in pairs_of_parents:
            if a not in spec.groups or b not in spec.groups:
                raise ValueError(
                    f"parent pair ({a!r}, {b!r}) not in parents "
                    f"{sorted(spec.groups)}"
                )
        wanted = {frozenset((a, b)) for a, b in pairs_of_parents}
    out: list[Pair] = []
    k = spec.n_arms
    for i in range(k):
        for j in range(i + 1, k):
            pi, pj = spec.parents[i], spec.parents[j]
            if pi == pj:
                continue
            if wanted is not None and frozenset((pi, pj)) not in wanted:
                continue
            out.append((i, j))
    return out


def arm_shares(
    post: Posterior,
    spec: ArmSpec,
    parent: str,
    spend_ref: np.ndarray,
    *,
    breakout_name_map: dict[str, str],
    mode: str = "zero_out",
    draws: int = 500,
    rng: Any = None,
) -> dict[str, Any]:
    """Export a parent's within-channel share composition from a CL posterior.

    The outbound bridge to the breakout-weighted MMM's share calibration
    (``ShareMeasurement`` in :mod:`mmm_framework.calibration.likelihood`): per
    posterior draw, compute each of the parent's arms' incremental response at
    the reference spend ``spend_ref``, normalize to a within-parent share
    simplex, and summarize the draws as shares + the empirical covariance of
    the ``K-1`` additive log-ratios (ALR, last arm as reference).

    Location/covariance consistency: the exported ``shares`` are the
    **inverse-ALR (softmax) of the mean log-ratios** over the surviving draws
    -- NOT the arithmetic mean of the share draws -- so the consumer's
    ``MvNormal`` location and its observed ``z_hat = ALR(shares)`` are exactly
    ``mean(z)``, on the same draws as the exported covariance.

    Non-positive responses (``"zero_out"`` mode): a draw where any arm's
    zero-out response is non-positive (strong cannibalization at ``spend_ref``)
    has NO well-defined share composition, so such draws are **excluded** from
    both the shares and the ALR covariance rather than floored into the
    statistics (a floored draw would inject ``log(eps)`` outliers that inflate
    the covariance by orders of magnitude and silently down-weight the share
    evidence). Any exclusion warns; more than 20% excluded raises (the
    zero-out shares are ill-defined -- use ``mode="main_effect"`` or a
    different ``spend_ref``); fewer than 10 surviving draws raises (too few to
    estimate the covariance).

    Estimand caveat (read before use): the MMM's ``breakout_share_<C>`` is an
    *effectiveness share through one shared response curve at the panel's spend
    mix*, while an arm share here is a ratio of PER-ARM response curves at
    ``spend_ref`` (with interactions in ``"zero_out"`` mode). The two coincide
    only near the panel's observed sub-stream spend mix and under the breakout
    model's own shared-curve assumption -- so ``spend_ref`` should approximate
    the panel's operating point, NOT a reallocated/optimal spend.

    Args:
        post: A fitted continuous-learning :class:`Posterior` over the arms
            surface (each arm a full dimension).
        spec: The :class:`ArmSpec` describing the flattened arm layout.
        parent: The split parent channel to export (must have >= 2 arms).
        spend_ref: Scaled reference spend vector over ALL arms, shape
            ``(spec.n_arms,)`` -- the operating point the shares are read at.
        breakout_name_map: REQUIRED mapping from each arm's *sub-name* (the
            part after ``ARM_SEP``, e.g. ``"Brand"`` from ``"Search │ Brand"``)
            to the MFF breakout column name (e.g. ``"Search_Brand"``). No fuzzy
            fallback: a missing key raises. The returned ``breakouts`` are the
            mapped names in the ARM (group) order; the MMM consumer requires
            them to match its model order exactly, so build the map (and the
            arm order) against ``BreakoutWeightedParams.breakout_groups``.
        mode: ``"zero_out"`` (default) -- arm ``i``'s response is
            ``R(s_ref) - R(s_ref with arm i zeroed)``, capturing interactions
            ``gamma``; ``"main_effect"`` -- ``beta_i * act_fn(s_ref)_i``,
            main-effect only (strictly positive since ``beta`` is HalfNormal).
        draws: Maximum posterior draws to use; the posterior is subsampled
            without replacement when it has more.
        rng: Seed or :class:`numpy.random.Generator` for the subsample.

    Returns:
        A ``ShareMeasurement``-shaped payload: ``{channel, breakouts, shares,
        log_ratio_cov, distribution, source}``. ``channel`` is set to
        ``parent``; overwrite it if the MMM's virtual channel name differs.
        ``source`` records ``{"mode", "spend_ref", "n_draws", "n_excluded"}``
        as provenance for the double-counting guard (``n_draws`` counts the
        SURVIVING draws the summary is computed on).

    Raises:
        ValueError: If more than 20% of the used draws have a non-positive
            arm response in ``"zero_out"`` mode, or if fewer than 10 draws
            survive the exclusion.
    """
    from .surface import surface_value

    if mode not in ("zero_out", "main_effect"):
        raise ValueError(f"mode must be 'zero_out' or 'main_effect', got {mode!r}")
    if parent not in spec.groups:
        raise ValueError(f"parent {parent!r} not in parents {sorted(spec.groups)}")
    idx = list(spec.groups[parent])
    if len(idx) < 2:
        raise ValueError(f"parent {parent!r} has {len(idx)} arm(s); shares need >= 2.")

    spend_ref = np.asarray(spend_ref, dtype=float)
    if spend_ref.shape != (spec.n_arms,):
        raise ValueError(
            f"spend_ref must have shape ({spec.n_arms},) (one entry per arm), "
            f"got {spend_ref.shape}."
        )

    # Map arm sub-names -> MFF breakout column names, explicitly. No fallback.
    breakouts: list[str] = []
    for i in idx:
        arm = spec.channels[i]
        sub = arm.split(ARM_SEP, 1)[1] if ARM_SEP in arm else arm
        if sub not in breakout_name_map:
            raise ValueError(
                f"breakout_name_map is missing arm sub-name {sub!r} (arm "
                f"{arm!r}); available keys: {sorted(breakout_name_map)}."
            )
        breakouts.append(str(breakout_name_map[sub]))

    n_total = post.n_draws
    if n_total > draws:
        gen = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        use = np.sort(gen.choice(n_total, size=draws, replace=False))
    else:
        use = np.arange(n_total)

    n_use = len(use)
    surviving: list[np.ndarray] = []
    n_excluded = 0
    for d in use:
        p = post.draw_params(int(d))
        beta, gamma = p["beta"], p["gamma"]
        act_fn, shape = p["act_fn"], p["shape"]
        if mode == "zero_out":
            base = float(surface_value(spend_ref, beta, gamma, act_fn, shape))
            r = np.empty(len(idx))
            for a, i in enumerate(idx):
                s0 = spend_ref.copy()
                s0[i] = 0.0
                r[a] = base - float(surface_value(s0, beta, gamma, act_fn, shape))
        else:  # main_effect
            f = np.asarray(act_fn(spend_ref, *shape), dtype=float)
            r = np.array([float(beta[i]) * f[i] for i in idx])
        if np.any(r <= 0):
            # A draw with a non-positive arm response has no well-defined
            # share composition -- exclude it from the statistics entirely
            # (flooring it would inject log(eps) outliers into the ALR cov).
            n_excluded += 1
            continue
        surviving.append(r / r.sum())

    if n_excluded > 0:
        if n_excluded > 0.20 * n_use:
            raise ValueError(
                f"arm_shares({parent!r}, mode={mode!r}): {n_excluded}/{n_use} "
                "draws (> 20%) had a non-positive arm response at spend_ref -- "
                "the zero-out shares are ill-defined under strong "
                "cannibalization. Use mode='main_effect' or a different "
                "spend_ref."
            )
        warnings.warn(
            f"arm_shares({parent!r}, mode={mode!r}): excluded {n_excluded}/"
            f"{n_use} draws with a non-positive arm response at spend_ref "
            "(strong cannibalization); the exported shares and ALR covariance "
            "summarize only the surviving draws. Heavy exclusion means the "
            "zero-out shares are ill-defined -- consider mode='main_effect' "
            "or a different spend_ref.",
            stacklevel=2,
        )
    if len(surviving) < 10:
        raise ValueError(
            f"arm_shares({parent!r}, mode={mode!r}): only {len(surviving)} "
            f"draw(s) survived the non-positive-response exclusion (of "
            f"{n_use}); need at least 10 to estimate the ALR covariance."
        )

    # Surviving draws are strictly positive by construction; the eps floor is
    # a pure numerical guard before the log, not a statistics device.
    share_draws = np.maximum(np.asarray(surviving), 1e-9)

    # ALR log-ratios w.r.t. the LAST breakout; empirical covariance + ridge.
    z = np.log(share_draws[:, :-1] / share_draws[:, -1:])
    cov = np.atleast_2d(np.cov(z, rowvar=False)) + 1e-9 * np.eye(len(idx) - 1)

    # Export the inverse-ALR (softmax) of mean(z) -- NOT the mean share draw --
    # so the consumer's observed z_hat = ALR(shares) is exactly mean(z), i.e.
    # the MvNormal location and covariance describe the same distribution.
    z_bar = z.mean(axis=0)
    expz = np.exp(np.concatenate([z_bar, [0.0]]))
    mean_shares = expz / expz.sum()

    return {
        "channel": parent,
        "breakouts": breakouts,
        "shares": [float(s) for s in mean_shares],
        "log_ratio_cov": [[float(v) for v in row] for row in cov],
        "distribution": "logistic_normal",
        "source": {
            "mode": mode,
            "spend_ref": [float(s) for s in spend_ref],
            "n_draws": int(len(surviving)),
            "n_excluded": int(n_excluded),
        },
    }


def default_arm_pair_signs(
    spec: ArmSpec,
    *,
    within: str = "neg",
    base: dict[Pair, PairSign] | None = None,
) -> dict[Pair, PairSign]:
    """The default sign map for an arms surface.

    Within-parent siblings get ``within`` (default ``"neg"`` — shared-audience
    substitution: creatives/keywords cannibalize each other), cross-parent
    pairs get ``"weak"``, and ``base`` entries override everything.
    """
    signs: dict[Pair, PairSign] = {}
    for p in within_parent_pairs(spec):
        signs[p] = within
    for p in cross_parent_pairs(spec):
        signs[p] = "weak"
    if base:
        signs.update(base)
    return signs
