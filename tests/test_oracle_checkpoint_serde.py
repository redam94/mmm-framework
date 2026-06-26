"""Regression tests for the Oracle checkpoint serialization fix.

Background: the Oracle's LangGraph ``AsyncSqliteSaver`` serializes agent state
with ormsgpack via ``JsonPlusSerializer``, which handles numpy *arrays* but
raises ``TypeError: Type is not msgpack serializable: numpy.float64`` on numpy
*scalars*. A tool that wrote a bare ``np.float64`` into ``dashboard_data`` (the
confirmed trigger: the validation / posterior-predictive ops) crashed the whole
thread. The fix has two layers, both exercised here:

  1. ``agents.kernels._json_safe`` now converts ``np.float64`` (its numpy branch
     runs before the ``float`` branch — ``np.float64`` is a ``float`` subclass).
  2. ``agents.serde.MsgpackSafeSerializer`` backstops the checkpointer so no
     state write can break persistence regardless of which tool produced it.

Plus a source-level guard on ``PPCCheckResult`` (the confirmed offender).
"""

import math

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from mmm_framework.agents.kernels import _json_safe
from mmm_framework.agents.serde import MsgpackSafeSerializer, _msgpack_coerce


# --------------------------------------------------------------------------- #
# Guard: prove the failure mode still exists in the stock serializer, so these
# tests can't pass vacuously if a future dependency bump "fixes" it upstream.
# --------------------------------------------------------------------------- #
def test_stock_serializer_crashes_on_numpy_scalar():
    with pytest.raises(TypeError, match="msgpack serializable"):
        JsonPlusSerializer().dumps_typed({"roi": np.float64(2.5)})


# --------------------------------------------------------------------------- #
# Part 1: _json_safe actually coerces np.float64 (the float-subclass hole).
# --------------------------------------------------------------------------- #
def test_json_safe_coerces_numpy_float64():
    out = _json_safe({"roi": np.float64(2.5)})
    assert type(out["roi"]) is float
    assert out["roi"] == 2.5


@pytest.mark.parametrize(
    "value,expected_type",
    [
        (np.float64(1.5), float),
        (np.float32(2.0), float),
        (np.int64(7), int),
        (np.int32(3), int),
        (np.bool_(True), bool),
    ],
)
def test_json_safe_numpy_scalars_become_native(value, expected_type):
    out = _json_safe({"v": value})["v"]
    assert type(out) is expected_type
    assert not isinstance(out, np.generic)


def test_json_safe_nonfinite_numpy_floats_become_none():
    assert _json_safe({"v": np.float64("nan")})["v"] is None
    assert _json_safe({"v": np.float64("inf")})["v"] is None


def test_json_safe_preserves_real_arrays():
    arr = np.array([1.0, 2.0, 3.0])
    out = _json_safe({"arr": arr})["arr"]
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, arr)


def test_json_safe_output_is_checkpointable():
    """A dict run through _json_safe must now survive the stock serializer."""
    dirty = {
        "roi": np.float64(2.5),
        "n": np.int64(3),
        "ok": np.bool_(True),
        "nested": [{"p_value": np.float64(0.03)}],
    }
    clean = _json_safe(dirty)
    # round-trips through the stock serializer with no exception
    JsonPlusSerializer().loads_typed(JsonPlusSerializer().dumps_typed(clean))


# --------------------------------------------------------------------------- #
# Part 2: MsgpackSafeSerializer backstops the checkpointer.
# --------------------------------------------------------------------------- #
def _roundtrip(serde, obj):
    return serde.loads_typed(serde.dumps_typed(obj))


def test_serde_roundtrips_numpy_dashboard():
    serde = MsgpackSafeSerializer()
    dashboard = {
        "validation_ppc": {
            "checks": [
                {"observed_statistic": np.float64(1.2), "p_value": np.float64(0.03)}
            ]
        },
        "roi_metrics": [{"roi_mean": np.float64(2.1), "n": np.int64(5)}],
    }
    back = _roundtrip(serde, dashboard)
    pv = back["validation_ppc"]["checks"][0]["p_value"]
    assert type(pv) is float and pv == pytest.approx(0.03)
    assert type(back["roi_metrics"][0]["n"]) is int


def test_serde_preserves_real_arrays_and_nan():
    serde = MsgpackSafeSerializer()
    obj = {"arr": np.array([1.0, 2.0]), "hdi": [float("nan"), 1.0], "x": np.float64(9)}
    back = _roundtrip(serde, obj)
    assert isinstance(back["arr"], np.ndarray)
    assert np.array_equal(back["arr"], np.array([1.0, 2.0]))
    # the serde must NOT map NaN -> None (that is the frontend encoder's job)
    assert math.isnan(back["hdi"][0])
    assert back["x"] == 9.0


def test_serde_reaches_numpy_buried_in_a_message():
    """The saver serializes the whole checkpoint in one call, so a numpy scalar
    inside a message's tool_calls must be coerced too (BaseModel descent)."""
    serde = MsgpackSafeSerializer()
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "fit", "args": {"draws": np.int64(2000)}, "id": "c1"}],
        ),
        ToolMessage(content="done", tool_call_id="c1"),
    ]
    back = _roundtrip(serde, {"messages": msgs})
    types = [type(m).__name__ for m in back["messages"]]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage"]
    assert back["messages"][1].tool_calls[0]["args"]["draws"] == 2000


def test_serde_byte_identical_for_clean_state():
    """Zero behavior change on the happy path: clean state must serialize exactly
    like the stock serializer (only failures take the coerce-and-retry path)."""
    clean = {
        "messages": [HumanMessage(content="hello"), AIMessage(content="hi there")],
        "model_status": "ready",
        "dashboard_data": {"roi_metrics": [{"channel": "TV", "roi_mean": 2.5}]},
    }
    assert MsgpackSafeSerializer().dumps_typed(
        clean
    ) == JsonPlusSerializer().dumps_typed(clean)


def test_coerce_handles_dict_with_numpy_key():
    # numpy used as a dict KEY also crashes ormsgpack; coerce keys too.
    coerced = _msgpack_coerce({np.int64(1): np.float64(2.0)})
    assert coerced == {1: 2.0}
    assert all(type(k) is int for k in coerced)


# --------------------------------------------------------------------------- #
# Part 3: the confirmed offender is clean at source.
# --------------------------------------------------------------------------- #
def test_ppc_check_result_casts_numpy_at_source():
    from mmm_framework.validation.results import PPCCheckResult

    r = PPCCheckResult(
        check_name="mean",
        observed_statistic=np.float64(1.2),
        replicated_mean=np.float64(1.1),
        replicated_std=np.float64(0.2),
        p_value=np.float64(0.4),
        passed=np.bool_(True),
        description="test",
    )
    d = r.to_dict()
    for k in ("observed_statistic", "replicated_mean", "replicated_std", "p_value"):
        assert type(d[k]) is float, k
    assert type(d["passed"]) is bool
    # and the to_dict() output is checkpointable
    JsonPlusSerializer().dumps_typed(d)
