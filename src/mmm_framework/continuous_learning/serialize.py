"""Persistence — JSON payloads for a Posterior, one ``.npz`` for a LearningState.

The heavy state (posterior draws, the accumulated panel, summary spend
vectors) is stored as **native npz arrays** — bit-exact float64 round-trips —
while the light config (channels, budget, pair signs, wave history, geo ids)
rides an embedded JSON string inside the same file. One file per learning
program: ``<workspace>/projects/<pid>/learning/<prog>/state.npz`` (the service
layer owns the path; this module only reads/writes).

Two layers:

* :func:`posterior_to_payload` / :func:`posterior_from_payload` — a JSON-safe
  dict for a :class:`~mmm_framework.continuous_learning.model.Posterior`
  (samples -> lists, pairs -> ``[[i, j], ...]``, pair_signs -> ``{"i,j": sign}``)
  for API snapshots and sessions rows.
* :func:`state_to_npz` / :func:`state_from_npz` — the full
  :class:`~mmm_framework.continuous_learning.loop.LearningState` (config, panel,
  summaries, posterior, history, geo_ids). Round-trip contract: a fixed-seed
  ``plan()`` on the reloaded state reproduces the original bit-identically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from .loop import LearningState, WaveRecord
from .model import Posterior

#: Highest schema version this READER understands. Writers stamp files
#: CONDITIONALLY (see :func:`_written_schema_version`): version 2 only when
#: the payload uses a v2 semantic (``likelihood != "normal"``,
#: ``time_effect != "none"``, or a persisted panel ``period_idx``), version 3
#: only when it uses a v3 semantic (an information discount
#: ``discount_half_life`` or a non-default ``spline_prior`` — an old reader
#: refitting such a program under the static/iid defaults would silently
#: change the model), else version 1 — so old files AND plain default-config
#: programs stay loadable by old readers, while a new-semantics file makes an
#: old reader refuse loudly instead of silently reinterpreting it.
SCHEMA_VERSION = 3


def _written_schema_version(
    likelihood: str,
    time_effect: str,
    has_period_idx: bool,
    *,
    discount_half_life: float | None = None,
    spline_prior: str = "iid",
) -> int:
    """The schema version to STAMP on a payload (conditional write)."""
    if discount_half_life is not None or spline_prior != "iid":
        return 3
    if likelihood != "normal" or time_effect != "none" or has_period_idx:
        return 2
    return 1


_SUMMARY_ARRAY_KEYS = ("spend_test", "spend_base")
_SUMMARY_SCALAR_KEYS = ("lift", "se", "scale")
_WAVE_FIELDS = {f.name for f in fields(WaveRecord)}


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays so ``json.dumps`` accepts them."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _signs_to_json(pair_signs: dict[tuple[int, int], str] | None):
    if pair_signs is None:
        return None
    return {f"{int(i)},{int(j)}": str(s) for (i, j), s in pair_signs.items()}


def _signs_from_json(d) -> dict[tuple[int, int], str] | None:
    if d is None:
        return None
    out: dict[tuple[int, int], str] = {}
    for key, sign in d.items():
        i, j = key.split(",")
        out[(int(i), int(j))] = str(sign)
    return out


# ── Posterior <-> JSON payload ─────────────────────────────────────────────────


def posterior_to_payload(post: Posterior) -> dict[str, Any]:
    """A JSON-safe dict capturing a :class:`Posterior` completely.

    Sample arrays become (nested) lists of float64 — Python's JSON float
    representation round-trips doubles exactly, so
    :func:`posterior_from_payload` reconstructs bit-identical samples.
    """
    return {
        "schema_version": _written_schema_version(
            post.likelihood, post.time_effect, has_period_idx=False
        ),
        "samples": {
            k: np.asarray(v, dtype=float).tolist() for k, v in post.samples.items()
        },
        "channels": list(post.channels),
        "pairs": [[int(i), int(j)] for i, j in post.pairs],
        "pair_signs": _signs_to_json(post.pair_signs) or {},
        "activation": post.activation,
        "likelihood": post.likelihood,
        "time_effect": post.time_effect,
        "spend_ref": (
            None
            if post.spend_ref is None
            else np.asarray(post.spend_ref, dtype=float).tolist()
        ),
        "diagnostics": _json_safe(post.diagnostics),
    }


def posterior_from_payload(d: dict[str, Any]) -> Posterior:
    """Inverse of :func:`posterior_to_payload`."""
    return Posterior(
        samples={
            k: np.asarray(v, dtype=float) for k, v in (d.get("samples") or {}).items()
        },
        channels=list(d["channels"]),
        pairs=[(int(i), int(j)) for i, j in d.get("pairs") or []],
        pair_signs=_signs_from_json(d.get("pair_signs")) or {},
        activation=str(d.get("activation", "hill")),
        likelihood=str(d.get("likelihood", "normal")),
        time_effect=str(d.get("time_effect", "none")),
        spend_ref=(
            None
            if d.get("spend_ref") is None
            else np.asarray(d["spend_ref"], dtype=float)
        ),
        diagnostics=dict(d.get("diagnostics") or {}),
    )


# ── LearningState <-> .npz ─────────────────────────────────────────────────────


def state_to_npz(state: LearningState, path: str | Path) -> str:
    """Persist a :class:`LearningState` to one compressed ``.npz`` file.

    Arrays (panel, posterior samples, summary spend vectors) are stored
    natively (exact float64); config/history/pair_signs/geo_ids ride an
    embedded JSON string under the ``meta`` key. Returns the written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {"center": np.asarray(state.center, dtype=float)}
    if state.spend_ref is not None:
        arrays["spend_ref"] = np.asarray(state.spend_ref, dtype=float)

    n_geo = None
    if state.data is not None:
        arrays["panel_spend"] = np.asarray(state.data["spend"], dtype=float)
        arrays["panel_geo_idx"] = np.asarray(state.data["geo_idx"], dtype=np.int64)
        arrays["panel_y"] = np.asarray(state.data["y"], dtype=float)
        if state.data.get("period_idx") is not None:
            arrays["panel_period_idx"] = np.asarray(
                state.data["period_idx"], dtype=np.int64
            )
        if state.data.get("row_week") is not None:
            arrays["panel_row_week"] = np.asarray(
                state.data["row_week"], dtype=np.int64
            )
        n_geo = int(state.data["n_geo"])

    summaries_extra: list[dict[str, Any]] = []
    if state.summaries:
        for key in _SUMMARY_ARRAY_KEYS:
            arrays[f"summary_{key}"] = np.stack(
                [np.asarray(s[key], dtype=float) for s in state.summaries]
            )
        for key in _SUMMARY_SCALAR_KEYS:
            arrays[f"summary_{key}"] = np.array(
                [float(s.get(key, 1.0)) for s in state.summaries], dtype=float
            )
        core = set(_SUMMARY_ARRAY_KEYS) | set(_SUMMARY_SCALAR_KEYS)
        for s in state.summaries:
            extra = {
                k: v
                for k, v in s.items()
                if k not in core and isinstance(v, (str, int, float, bool, type(None)))
            }
            summaries_extra.append(extra)

    post_meta = None
    if state.posterior is not None:
        post = state.posterior
        sample_names = list(post.samples.keys())
        for i, name in enumerate(sample_names):
            arrays[f"post_sample_{i:04d}"] = np.asarray(post.samples[name], dtype=float)
        if post.spend_ref is not None:
            arrays["post_spend_ref"] = np.asarray(post.spend_ref, dtype=float)
        post_meta = {
            "channels": list(post.channels),
            "pairs": [[int(i), int(j)] for i, j in post.pairs],
            "pair_signs": _signs_to_json(post.pair_signs) or {},
            "activation": post.activation,
            "likelihood": post.likelihood,
            "time_effect": post.time_effect,
            "diagnostics": _json_safe(post.diagnostics),
            "sample_names": sample_names,
            "has_spend_ref": post.spend_ref is not None,
        }

    has_period_idx = state.data is not None and state.data.get("period_idx") is not None
    meta = {
        # Conditional stamp: v2 IFF the state uses a v2 semantic (non-default
        # likelihood/time_effect on the state OR its posterior, or a persisted
        # period_idx), else v1 — old readers keep loading plain programs and
        # refuse loudly on files they would silently misread.
        "schema_version": max(
            _written_schema_version(
                state.likelihood,
                state.time_effect,
                has_period_idx,
                discount_half_life=state.discount_half_life,
                spline_prior=state.spline_prior,
            ),
            (
                _written_schema_version(
                    state.posterior.likelihood,
                    state.posterior.time_effect,
                    has_period_idx=False,
                )
                if state.posterior is not None
                else 1
            ),
        ),
        "channels": list(state.channels),
        "B": float(state.B),
        "value": float(state.value),
        "pairs": [[int(i), int(j)] for i, j in (state.pairs or [])],
        "pair_signs": _signs_to_json(state.pair_signs),
        "activation": state.activation,
        "likelihood": state.likelihood,
        "time_effect": state.time_effect,
        "mode": state.mode,
        "cap": None if state.cap is None else float(state.cap),
        "beta_scale": float(state.beta_scale),
        "gamma_scale": float(state.gamma_scale),
        "discount_half_life": (
            None
            if state.discount_half_life is None
            else float(state.discount_half_life)
        ),
        "spline_prior": state.spline_prior,
        "geo_ids": None if state.geo_ids is None else list(state.geo_ids),
        "n_geo": n_geo,
        "has_data": state.data is not None,
        "has_spend_ref": state.spend_ref is not None,
        "n_summaries": len(state.summaries),
        "summaries_extra": summaries_extra,
        "history": [_json_safe(asdict(w)) for w in state.history],
        "posterior": post_meta,
    }
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    return str(path)


def state_from_npz(path: str | Path) -> LearningState:
    """Reconstruct a :class:`LearningState` written by :func:`state_to_npz`."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        if int(meta.get("schema_version", 0)) > SCHEMA_VERSION:
            raise ValueError(
                f"state file schema_version {meta['schema_version']} is newer "
                f"than this reader ({SCHEMA_VERSION})"
            )

        state = LearningState(
            channels=list(meta["channels"]),
            center=np.asarray(z["center"], dtype=float),
            B=float(meta["B"]),
            value=float(meta["value"]),
            pairs=[(int(i), int(j)) for i, j in meta.get("pairs") or []],
            pair_signs=_signs_from_json(meta.get("pair_signs")),
            activation=str(meta.get("activation", "hill")),
            likelihood=str(meta.get("likelihood", "normal")),
            time_effect=str(meta.get("time_effect", "none")),
            mode=str(meta.get("mode", "fixed")),
            cap=None if meta.get("cap") is None else float(meta["cap"]),
            beta_scale=float(meta.get("beta_scale", 1.0)),
            gamma_scale=float(meta.get("gamma_scale", 0.8)),
            discount_half_life=(
                None
                if meta.get("discount_half_life") is None
                else float(meta["discount_half_life"])
            ),
            spline_prior=str(meta.get("spline_prior", "iid")),
            spend_ref=(
                np.asarray(z["spend_ref"], dtype=float)
                if meta.get("has_spend_ref")
                else None
            ),
        )
        state.geo_ids = (
            None if meta.get("geo_ids") is None else [str(g) for g in meta["geo_ids"]]
        )

        if meta.get("has_data"):
            state.data = {
                "spend": np.asarray(z["panel_spend"], dtype=float),
                "geo_idx": np.asarray(z["panel_geo_idx"], dtype=int),
                "y": np.asarray(z["panel_y"], dtype=float),
                "n_geo": int(meta["n_geo"]),
            }
            if "panel_period_idx" in z:  # absent in pre-time-effect files
                state.data["period_idx"] = np.asarray(z["panel_period_idx"], dtype=int)
            if "panel_row_week" in z:  # absent in pre-discount files
                state.data["row_week"] = np.asarray(z["panel_row_week"], dtype=int)

        n_summaries = int(meta.get("n_summaries", 0))
        if n_summaries:
            extras = meta.get("summaries_extra") or [{}] * n_summaries
            for m in range(n_summaries):
                summary: dict[str, Any] = dict(extras[m] if m < len(extras) else {})
                for key in _SUMMARY_ARRAY_KEYS:
                    summary[key] = np.asarray(z[f"summary_{key}"][m], dtype=float)
                for key in _SUMMARY_SCALAR_KEYS:
                    summary[key] = float(z[f"summary_{key}"][m])
                state.summaries.append(summary)

        post_meta = meta.get("posterior")
        if post_meta is not None:
            samples = {
                name: np.asarray(z[f"post_sample_{i:04d}"], dtype=float)
                for i, name in enumerate(post_meta["sample_names"])
            }
            state.posterior = Posterior(
                samples=samples,
                channels=list(post_meta["channels"]),
                pairs=[(int(i), int(j)) for i, j in post_meta.get("pairs") or []],
                pair_signs=_signs_from_json(post_meta.get("pair_signs")) or {},
                activation=str(post_meta.get("activation", "hill")),
                likelihood=str(post_meta.get("likelihood", "normal")),
                time_effect=str(post_meta.get("time_effect", "none")),
                spend_ref=(
                    np.asarray(z["post_spend_ref"], dtype=float)
                    if post_meta.get("has_spend_ref")
                    else None
                ),
                diagnostics=dict(post_meta.get("diagnostics") or {}),
            )

        for w in meta.get("history") or []:
            state.history.append(
                WaveRecord(**{k: v for k, v in w.items() if k in _WAVE_FIELDS})
            )
    return state
