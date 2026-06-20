"""Import a garden model class from its stored source file.

A registered garden model is a ``.py`` file holding a bespoke MMM class. To run
it, the source must be **executed** (the class is defined by running the module
body). That is untrusted expert code, so this MUST happen kernel-side — the
garden ops (:mod:`mmm_framework.agents.model_ops`) call this from inside the
session kernel (sandboxed subprocess / container in the hosted profile), never
from the host API process. The host only ever reads registry metadata.

Loaded classes are cached by ``(resolved_path, mtime)`` so repeated fits in a
session don't re-exec the module; editing the file (new mtime) busts the cache.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from .contract import find_garden_class, validate_class

# (resolved_path, mtime, class_name) -> class. Bounded informally by the number
# of distinct (model, version) pairs a session touches.
_CLASS_CACHE: dict[tuple[str, float, str | None], type] = {}
_CACHE_MAX = 64


def _module_name_for(path: Path) -> str:
    """A stable, collision-free synthetic module name for an arbitrary source
    file (distinct paths -> distinct names so versions don't clobber each other
    in ``sys.modules``)."""
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return f"mmm_garden_{digest}"


def load_garden_class_from_path(
    source_path: str | Path, class_name: str | None = None
) -> type:
    """Import and return the garden model class defined in ``source_path``.

    Parameters
    ----------
    source_path:
        Path to the model ``.py`` file (typically a thread-local copy of the
        registered source — see ``agents.tools.load_garden_model``).
    class_name:
        Explicit class to pick out of the module. When omitted,
        :func:`~mmm_framework.garden.contract.find_garden_class` resolves it
        (a ``GARDEN_MODEL`` attribute, else the single BayesianMMM subclass).

    Raises
    ------
    FileNotFoundError
        If ``source_path`` does not exist.
    ValueError
        If no compatible class can be resolved, or the resolved class fails the
        static contract check.
    """
    path = Path(source_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"garden model source not found: {path}")

    key = (str(path), path.stat().st_mtime, class_name)
    cached = _CLASS_CACHE.get(key)
    if cached is not None:
        return cached

    mod_name = _module_name_for(path)
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / typing.get_type_hints resolve.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # executes untrusted body — kernel-side only
    except Exception as exc:  # noqa: BLE001 — surface the author's error cleanly
        sys.modules.pop(mod_name, None)
        raise ValueError(f"garden model source failed to import: {exc}") from exc

    cls = getattr(module, class_name) if class_name else find_garden_class(module)
    problems = validate_class(cls)
    if problems:
        raise ValueError(
            "garden model is not oracle-compatible: " + "; ".join(problems)
        )

    if len(_CLASS_CACHE) >= _CACHE_MAX:
        _CLASS_CACHE.clear()
    _CLASS_CACHE[key] = cls
    return cls


def clear_cache() -> None:
    """Drop the loaded-class cache (used by tests / after a re-publish)."""
    _CLASS_CACHE.clear()


def class_qualname(cls: Any) -> str:
    """Best-effort fully-qualified name for provenance in serialized metadata."""
    mod = getattr(cls, "__module__", "?")
    qn = getattr(cls, "__qualname__", getattr(cls, "__name__", "?"))
    return f"{mod}.{qn}"


__all__ = ["load_garden_class_from_path", "clear_cache", "class_qualname"]
