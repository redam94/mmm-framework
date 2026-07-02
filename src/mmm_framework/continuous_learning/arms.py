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

from dataclasses import dataclass

from .model import Pair, PairSign

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
