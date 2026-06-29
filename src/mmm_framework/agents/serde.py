"""Defense-in-depth checkpoint serializer for the Oracle's ``AsyncSqliteSaver``.

LangGraph persists each agent-state checkpoint with ormsgpack (via
``JsonPlusSerializer``), which serializes numpy *arrays* but raises on numpy
*scalars* (``np.float64`` &c.) and on pandas ``Series``/``DataFrame``. A single
bare ``np.float64`` anywhere in a persisted ``AgentState`` channel — e.g. an
un-``float()``-cast statistic that a tool wrote into ``dashboard_data`` — would
otherwise crash the whole Oracle thread with::

    TypeError: Type is not msgpack serializable: numpy.float64

``agents.kernels._json_safe`` coerces model-op payloads at the source, but
several tool paths (``fit_mmm_model``, ``generate_model_defense_report``, and the
``delegate_to_expert`` / ``convene_review_panel`` sub-agents) fold values into
state without ever calling it. This serializer is the universal backstop:
``dumps_typed`` is the single serialization choke for every checkpoint write, so
wrapping it guarantees no state write can break persistence regardless of which
(possibly future) tool produced the value.

It delegates to the stock serializer on the happy path — so clean state is
byte-identical and native types ormsgpack already handles (``set``, ``Decimal``,
``datetime``, ndarray) keep full fidelity — and only deep-coerces the value on an
actual encode failure, then retries once (zero happy-path cost).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel


def _msgpack_coerce(o: Any) -> Any:
    """Convert msgpack-hostile leaves to native equivalents, recursing through
    plain containers AND pydantic models (incl. langchain ``BaseMessage``) so a
    numpy scalar buried inside a message's ``tool_calls`` / ``additional_kwargs``
    is reached too. Types ormsgpack already serializes (ndarray, ``Decimal``,
    ``set``, ``datetime``, ``bytes``) are left untouched. Never raises — a value
    it cannot coerce is returned as-is so the natural error surfaces on retry.

    Deliberately does NOT map NaN/Inf -> None: msgpack round-trips non-finite
    floats fine, and dropping them silently would lose data. That coercion is the
    frontend-facing job of ``_json_safe`` / the API JSON encoder, not the
    checkpoint serializer.
    """
    # numpy scalar (float64/float32/int64/bool_/datetime64/...) -> python scalar.
    if isinstance(o, np.generic):
        return o.item()
    # 0-d array -> scalar; real arrays survive via ormsgpack's EXT_NUMPY_ARRAY.
    if isinstance(o, np.ndarray):
        return o.item() if o.ndim == 0 else o
    if isinstance(o, dict):
        # numpy used as a dict KEY also raises -> coerce keys as well as values.
        return {_msgpack_coerce(k): _msgpack_coerce(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_msgpack_coerce(v) for v in o]
    if isinstance(o, tuple):
        return tuple(_msgpack_coerce(v) for v in o)
    if isinstance(o, (set, frozenset)):
        return type(o)(_msgpack_coerce(v) for v in o)
    # pandas, by type name so this module never imports pandas: Series/DataFrame
    # crash the encoder; Timestamp/NaT serialize but load back as a bare ISO
    # string (a security blocklist refuses Timestamp.fromisoformat), so normalize
    # them up front for write==read consistency.
    tname = type(o).__name__
    if tname in ("Series", "DataFrame"):
        return _msgpack_coerce(o.to_dict())
    if tname in ("Timestamp", "NaTType"):
        return str(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, BaseModel):  # pydantic, incl. langchain BaseMessage subtypes
        try:
            return o.model_copy(
                update={
                    name: _msgpack_coerce(getattr(o, name))
                    for name in type(o).model_fields
                }
            )
        except Exception:  # noqa: BLE001 - the net must never raise itself
            return o
    return o


class MsgpackSafeSerializer(JsonPlusSerializer):
    """``JsonPlusSerializer`` that never lets a stray numpy/pandas value break
    checkpoint persistence. ``dumps_typed`` is the only override; everything else
    (``loads_typed``, ``dumps``, ``loads``) is inherited unchanged."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        try:
            return super().dumps_typed(obj)
        except TypeError:
            # ormsgpack.MsgpackEncodeError IS a builtins.TypeError and, with
            # pickle_fallback off, JsonPlusSerializer re-raises it unchanged.
            # Coerce the offending value(s) and retry once; if it still cannot be
            # serialized the natural error propagates, exactly as before.
            return super().dumps_typed(_msgpack_coerce(obj))
