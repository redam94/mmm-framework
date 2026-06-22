"""Shared Model Garden registration core — used by BOTH the agent tool
(``tools.register_garden_model``) and the REST endpoint (``POST /model-garden``).

Lives under ``agents`` (not ``garden``) to avoid an import cycle: ``garden`` is
imported by ``agents`` (model_ops/loader), so this module — which needs the
workspace + sessions store — must sit on the ``agents`` side. Registration is
host-safe: the source is validated by **AST only** (never executed) and written
to the org-scoped garden store; the untrusted body runs only later, kernel-side,
during ``test``/``fit``.
"""

from __future__ import annotations

import ast
import json
from typing import Any


def static_class_name(source_code: str) -> tuple[str | None, str | None]:
    """Find the garden model class in source WITHOUT executing it (AST only).
    Returns ``(class_name, error)`` — exactly one is non-None."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return None, f"source has a syntax error: {e}"
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    explicit = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name):
            if any(
                isinstance(t, ast.Name) and t.id == "GARDEN_MODEL" for t in n.targets
            ):
                explicit = n.value.id
    if explicit:
        return explicit, None
    if len(classes) == 1:
        return classes[0], None
    if not classes:
        return None, "source defines no class (expected a BayesianMMM subclass)"
    return None, (
        f"source defines multiple classes {classes}; add `GARDEN_MODEL = YourClass` "
        "to say which one is the model"
    )


def static_model_kind(source_code: str, class_name: str | None) -> str:
    """Find a class's ``__garden_model_kind__`` literal WITHOUT executing the
    source (AST only). Returns the declared kind, or ``"mmm"`` when absent — the
    historical default. A non-MMM family (CFA, latent-class, …) declares e.g.
    ``__garden_model_kind__ = "cfa"`` in its class body to opt out of the
    MMM-specific contract gates."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return "mmm"
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    target = None
    if class_name:
        target = next((c for c in classes if c.name == class_name), None)
    elif len(classes) == 1:
        target = classes[0]
    if target is None:
        return "mmm"
    for node in target.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__garden_model_kind__"
            for t in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                return node.value.value
    return "mmm"


def register_garden_model_core(
    *,
    org_id: str,
    source_code: str,
    name: str,
    docs: str = "",
    version: int | None = None,
    tags: list | None = None,
    dataset_schema: dict | None = None,
    recommended_fit: dict | None = None,
    default_estimands: list | None = None,
    capabilities: list | None = None,
    config_schema: dict | None = None,
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    """Statically validate source, persist it + a manifest to the org-scoped
    garden store, and insert a ``draft`` registry row. Returns the created row.

    Raises ``ValueError`` for invalid source / a duplicate version — callers
    translate that into their own error shape (a ToolMessage or an HTTP 4xx).
    """
    from mmm_framework.agents import workspace as ws
    from mmm_framework.api import sessions as sessions_store
    from mmm_framework.garden.contract import GARDEN_CONTRACT_VERSION

    class_name, err = static_class_name(source_code)
    if err:
        raise ValueError(err)

    ver = (
        int(version)
        if version is not None
        else sessions_store.next_garden_version(org_id, name)
    )
    manifest = {
        "contract_version": GARDEN_CONTRACT_VERSION,
        "class_name": class_name,
        # Model family kind (AST-detected from ``__garden_model_kind__``; "mmm"
        # by default). Advisory metadata for discovery / UI — the authoritative
        # kind is read from the loaded class at fit time.
        "model_kind": static_model_kind(source_code, class_name),
        "dataset_schema": dataset_schema or {},
        "recommended_fit": recommended_fit or {},
        "tags": tags or [],
    }
    # Advisory estimand metadata (additive — readers use .get(..., default)). The
    # authoritative capability detection is runtime (model_capabilities); these
    # are for discovery + a model's declared default estimands. A garden model
    # also declares defaults directly via a class-level DEFAULT_ESTIMANDS attr.
    if default_estimands:
        manifest["default_estimands"] = list(default_estimands)
    if capabilities:
        manifest["capabilities"] = list(capabilities)
    # Bespoke per-model config schema (JSON Schema of the model's CONFIG_SCHEMA),
    # for rendering a dynamic params form in the UI — same way dataset_schema is
    # rendered. Additive: absent for models without bespoke params. Provided by
    # the caller (host registration is AST-only and cannot import the class).
    if config_schema:
        manifest["config_schema"] = dict(config_schema)
    gdir = ws.garden_dir(org_id, name, ver)
    src_path = gdir / "model.py"
    src_path.write_text(source_code, encoding="utf-8")
    (gdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return sessions_store.upsert_garden_model(
        org_id=org_id,
        name=name,
        version=ver,
        owner_user_id=owner_user_id,
        docs=docs or None,
        manifest=manifest,
        source_path=str(src_path),
    )


def read_garden_source(row: dict | None) -> str | None:
    """The stored source text for a registry row (or None if unavailable)."""
    if not row or not row.get("source_path"):
        return None
    try:
        from pathlib import Path

        return Path(row["source_path"]).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "static_class_name",
    "static_model_kind",
    "register_garden_model_core",
    "read_garden_source",
]
