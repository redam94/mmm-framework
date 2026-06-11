"""Field-level locking for the agent's evolving ``model_spec``.

The user can manually edit the model configuration in the UI. Anything they
touch becomes *locked*: the LLM may no longer silently overwrite it. Instead, a
proposed change to a locked field is deferred into ``pending_spec_changes`` and
surfaced to the user for confirmation (see ``api/main.py`` ``/spec`` endpoints
and the frontend modal).

Lock paths use the SAME dot-notation as ``update_model_setting`` in
``tools.py``: ``media_channels`` and ``control_variables`` are lists whose
items are addressed by their ``name`` (e.g. ``media_channels.TV.adstock.l_max``),
everything else is plain nested-dict access (e.g. ``inference.draws``).

This module is intentionally free of LangChain / LangGraph imports so it can be
unit-tested and reused from both the tools and the API layer.
"""

from __future__ import annotations

import copy
from typing import Any

_MISSING = object()


def _is_named_list(value: Any) -> bool:
    """A list whose entries are all ``{"name": ...}`` dicts (channels/controls)."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, dict) and "name" in x for x in value)
    )


def flatten_leaves(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a spec into ``{dot_path: leaf_value}``.

    Named lists recurse by item name; other lists and scalars are treated as
    opaque leaf values (so e.g. a list of changepoint positions is compared as a
    whole rather than by fragile positional index).
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj and prefix:
            out[prefix] = obj
            return out
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_leaves(value, path))
    elif _is_named_list(obj):
        for item in obj:
            name = item["name"]
            path = f"{prefix}.{name}" if prefix else str(name)
            for key, value in item.items():
                out.update(flatten_leaves(value, f"{path}.{key}"))
    else:
        out[prefix] = obj
    return out


def get_at(spec: Any, path: str, default: Any = None) -> Any:
    """Read the value at a dot-path; return ``default`` if any segment is absent."""
    obj = spec
    for key in path.split("."):
        if isinstance(obj, list):
            obj = next(
                (x for x in obj if isinstance(x, dict) and x.get("name") == key),
                _MISSING,
            )
        elif isinstance(obj, dict):
            obj = obj.get(key, _MISSING)
        else:
            return default
        if obj is _MISSING:
            return default
    return obj


def set_at(spec: dict, path: str, value: Any) -> None:
    """Set the value at a dot-path, auto-creating intermediate dicts and named
    list items as needed. Mutates ``spec`` in place."""
    keys = path.split(".")
    _walk_set(spec, keys, value)


def _walk_set(obj: Any, keys: list[str], value: Any) -> None:
    key, rest = keys[0], keys[1:]

    if isinstance(obj, list):
        item = next(
            (x for x in obj if isinstance(x, dict) and x.get("name") == key), None
        )
        if item is None:
            item = {"name": key}
            obj.append(item)
        if rest:
            _walk_set(item, rest, value)
        return

    # dict
    if not rest:
        obj[key] = value
        return
    if key not in obj or not isinstance(obj[key], (dict, list)):
        obj[key] = {}
    _walk_set(obj[key], rest, value)


# ── Spec patches ──────────────────────────────────────────────────────────────
# LangGraph's ToolNode runs all tool_calls from one AIMessage concurrently
# against the SAME state snapshot. If each update_model_setting call wrote a
# full spec, the last writer would silently erase the others' changes (each
# full spec carries stale values for every field it didn't touch). Instead,
# single-field updates are wrapped in a patch envelope; the model_spec reducer
# applies each patch against the LATEST folded value, so concurrent updates
# compose. Full spec dicts (configure_model / load_config / UI edits) keep
# replace semantics. The reducer always stores a concrete spec — a patch
# envelope never survives into a checkpoint.

SPEC_PATCH_KEY = "__spec_patch__"


def make_spec_patch(changes: list[dict]) -> dict:
    """Wrap ``[{"path": ..., "value": ...}, ...]`` in a patch envelope."""
    return {SPEC_PATCH_KEY: changes}


def is_spec_patch(value: Any) -> bool:
    return isinstance(value, dict) and SPEC_PATCH_KEY in value


def apply_spec_patch(base: Any, patch: dict) -> dict:
    """Apply a patch envelope to ``base``, returning a new spec dict."""
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    for change in patch.get(SPEC_PATCH_KEY) or []:
        set_at(merged, change["path"], change["value"])
    return merged


def diff_locked(old_spec: Any, new_spec: Any) -> list[str]:
    """Leaf paths that were added or changed going from ``old_spec`` to
    ``new_spec`` — i.e. the fields a manual edit "touched". Removals are ignored.
    """
    old_leaves = flatten_leaves(old_spec or {})
    new_leaves = flatten_leaves(new_spec or {})
    changed: list[str] = []
    for path, val in new_leaves.items():
        if path not in old_leaves or old_leaves[path] != val:
            changed.append(path)
    return changed


def reconcile_with_locks(
    candidate: dict,
    current: dict | None,
    locked_fields: list[str] | None,
    reason: str | None = None,
    tool_call_id: str | None = None,
) -> tuple[dict, list[dict]]:
    """Apply locked-field protection to an LLM-proposed ``candidate`` spec.

    For each locked path:
      * if the candidate diverges from the locked (current) value, the candidate
        is reverted to the locked value and a pending proposal is recorded;
      * if the candidate dropped the locked path entirely, the locked value is
        re-asserted (no proposal — the user's value simply survives).

    Returns ``(merged_spec, pending_changes)``. ``pending_changes`` entries:
    ``{path, current, proposed, reason, tool_call_id}``.
    """
    merged = copy.deepcopy(candidate)
    pending: list[dict] = []

    for path in locked_fields or []:
        locked_val = get_at(current or {}, path, _MISSING)
        if locked_val is _MISSING:
            # The locked field no longer exists in the current spec; nothing to
            # protect (e.g. the channel it referenced was removed).
            continue

        cand_val = get_at(merged, path, _MISSING)
        if cand_val is _MISSING:
            # Candidate omitted a locked field — keep the user's value.
            set_at(merged, path, locked_val)
            continue

        if cand_val != locked_val:
            # Conflict: revert to the user's value and surface for confirmation.
            set_at(merged, path, locked_val)
            pending.append(
                {
                    "path": path,
                    "current": locked_val,
                    "proposed": cand_val,
                    "reason": reason,
                    "tool_call_id": tool_call_id,
                }
            )

    return merged, pending


def merge_pending(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """Merge pending proposals, de-duplicated by ``path`` (newest wins). Keeps
    the LLM from stacking duplicate modals when it retries the same change."""
    by_path: dict[str, dict] = {}
    for entry in (existing or []) + (new or []):
        by_path[entry["path"]] = entry
    return list(by_path.values())
