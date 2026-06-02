"""Pre-specification lock + diff (P2-3).

Pre-registration is only meaningful if divergences from the locked spec are
*detected*, not just logged. :func:`diff_spec` is the pure, testable core: it
compares two serialized specs (a DAG, an MFFConfig, a ModelConfig -- any nested
dict) and reports what changed. List items are matched by identity (``name`` /
``id`` / ``source→target``) rather than position, so reordering is not a false
divergence and a genuinely added/removed channel or control is caught.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpecChange:
    """A single difference between a frozen spec and the current one."""

    path: str
    kind: str  # "added" | "removed" | "changed"
    old: Any
    new: Any

    def describe(self) -> str:
        if self.kind == "added":
            return f"+ {self.path} = {self.new!r} (added)"
        if self.kind == "removed":
            return f"- {self.path} = {self.old!r} (removed)"
        return f"~ {self.path}: {self.old!r} → {self.new!r}"


def _list_item_key(x: Any) -> str | None:
    """Stable identity key for a list item, or None to fall back to index."""
    if isinstance(x, dict):
        for k in ("name", "id", "variable_name"):
            if x.get(k):
                return f"{k}={x[k]}"
        if "source" in x and "target" in x:
            return f"{x['source']}->{x['target']}"
    return None


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested structure to ``dotted.path -> leaf`` entries.

    Lists of identifiable dicts are keyed by identity so order does not matter.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            # Record an empty *nested* container as a leaf so it can change to/
            # from {}, but never emit a bare top-level key.
            if prefix:
                out[prefix] = {}
            return out
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(obj, list):
        if not obj:
            if prefix:
                out[prefix] = []
            return out
        keys = [_list_item_key(x) for x in obj]
        if all(k is not None for k in keys):
            # Identity-keyed; duplicate names are disambiguated by occurrence so
            # both specs key consistently (a genuinely duplicated channel shows
            # as one clean diff, not a re-indexing of the whole list).
            counts = Counter(keys)
            occ: dict[str, int] = {}
            for k, x in zip(keys, obj):
                if counts[k] > 1:
                    i = occ.get(k, 0)
                    occ[k] = i + 1
                    item_key = f"{k}#{i}"
                else:
                    item_key = k
                out.update(_flatten(x, f"{prefix}[{item_key}]"))
        else:
            # Some items have no stable identity -> positional keying.
            for i, x in enumerate(obj):
                out.update(_flatten(x, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def diff_spec(frozen: dict[str, Any], current: dict[str, Any]) -> list[SpecChange]:
    """Return the structural differences from ``frozen`` to ``current``.

    Each :class:`SpecChange` is an added / removed / changed leaf, identified by
    a dotted path. An empty list means the current spec matches what was locked.
    """
    f = _flatten(frozen)
    c = _flatten(current)
    changes: list[SpecChange] = []
    for key in sorted(set(f) | set(c)):
        if key not in c:
            changes.append(SpecChange(key, "removed", f[key], None))
        elif key not in f:
            changes.append(SpecChange(key, "added", None, c[key]))
        elif f[key] != c[key]:
            changes.append(SpecChange(key, "changed", f[key], c[key]))
    return changes


def summarize_spec_diff(changes: list[SpecChange]) -> str:
    """Human-readable summary; empty changes => an explicit 'no divergence'."""
    if not changes:
        return "No divergence from the pre-registered specification."
    lines = [f"{len(changes)} divergence(s) from the pre-registered specification:"]
    lines += [f"  {ch.describe()}" for ch in changes]
    return "\n".join(lines)


__all__ = ["SpecChange", "diff_spec", "summarize_spec_diff"]
