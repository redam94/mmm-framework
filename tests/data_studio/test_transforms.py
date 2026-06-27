"""Data Studio transform pipeline — every op on wide AND MFF-long frames,
replay/idempotency, and the structural-error vs data-error split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.data_studio.transforms import (
    TransformError,
    apply_pipeline,
    is_long_frame,
    user_columns,
)


def _wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": pd.date_range("2023-01-02", periods=10, freq="W-MON"),
            "sales": [100, 110, 120, 130, 140, 150, 160, 170, 180, 9999.0],
            "tv_spend": [10, 12, 11, 13, 12, np.nan, 13, 15, 14, 16.0],
            "notes": ["a"] * 10,
        }
    )


def _long() -> pd.DataFrame:
    weeks = list(pd.date_range("2023-01-02", periods=6, freq="W-MON"))
    return pd.DataFrame(
        {
            "Period": weeks * 2,
            "VariableName": ["sales"] * 6 + ["tv_spend"] * 6,
            "VariableValue": [
                100,
                110,
                120,
                130,
                140,
                5000.0,
                10,
                12,
                11,
                13,
                12,
                14.0,
            ],
        }
    )


WIDE_ROLES = {"week": "date", "sales": "kpi", "tv_spend": "media"}
LONG_ROLES = {"Period": "date", "sales": "kpi", "tv_spend": "media"}


# ── layout detection ──────────────────────────────────────────────────────────


def test_layout_detection():
    assert not is_long_frame(_wide())
    assert is_long_frame(_long())
    assert "sales" in user_columns(_wide())
    assert "sales" in user_columns(_long())  # variable names, not phys cols


# ── rename / drop reconcile roles ─────────────────────────────────────────────


def test_rename_reconciles_roles_wide_and_long():
    for df, roles in ((_wide(), WIDE_ROLES), (_long(), LONG_ROLES)):
        res = apply_pipeline(
            df, [{"op": "rename", "from": "sales", "to": "revenue"}], roles
        )
        assert "revenue" in user_columns(res.df)
        assert "sales" not in user_columns(res.df)
        assert res.roles.get("revenue") == "kpi"


def test_drop_columns_wide_and_long():
    res = apply_pipeline(
        _wide(), [{"op": "drop_columns", "columns": ["notes"]}], WIDE_ROLES
    )
    assert "notes" not in res.df.columns
    res = apply_pipeline(
        _long(), [{"op": "drop_columns", "columns": ["tv_spend"]}], LONG_ROLES
    )
    assert "tv_spend" not in user_columns(res.df)
    assert "tv_spend" not in res.roles


# ── value edits target the right cells on both shapes ─────────────────────────


def test_winsorize_targets_cell_wide_and_long():
    last = str(_wide()["week"].iloc[-1].date())
    res = apply_pipeline(
        _wide(),
        [{"op": "winsorize", "column": "sales", "periods": [last], "cap_value": 200.0}],
        WIDE_ROLES,
    )
    assert res.df["sales"].iloc[-1] == 200.0
    assert res.df["sales"].iloc[0] == 100.0  # other periods untouched

    last_l = str(_long()["Period"].iloc[5].date())
    res = apply_pipeline(
        _long(),
        [
            {
                "op": "winsorize",
                "column": "sales",
                "periods": [last_l],
                "cap_value": 150.0,
            }
        ],
        LONG_ROLES,
    )
    sales = res.df[res.df["VariableName"] == "sales"]["VariableValue"]
    assert sales.max() == 150.0


def test_impute_wide_and_long():
    last = str(_wide()["week"].iloc[-1].date())
    res = apply_pipeline(
        _wide(),
        [{"op": "impute", "column": "sales", "periods": [last], "value": 185.0}],
        WIDE_ROLES,
    )
    assert res.df["sales"].iloc[-1] == 185.0


def test_fill_missing_wide_and_long():
    res = apply_pipeline(
        _wide(), [{"op": "fill_missing", "strategy": "zero"}], WIDE_ROLES
    )
    assert res.df["tv_spend"].isna().sum() == 0
    assert res.df["tv_spend"].iloc[5] == 0.0
    # interpolate
    res = apply_pipeline(
        _wide(),
        [{"op": "fill_missing", "columns": ["tv_spend"], "strategy": "interpolate"}],
        WIDE_ROLES,
    )
    assert res.df["tv_spend"].isna().sum() == 0


def test_drop_duplicates_long_uses_var_period_grain():
    df = _long()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate one (var,period)
    res = apply_pipeline(dup, [{"op": "drop_duplicates"}], LONG_ROLES)
    assert len(res.df) == len(df)


def test_filter_rows_and_date_range():
    res = apply_pipeline(
        _wide(),
        [{"op": "filter_rows", "column": "sales", "operator": "<", "value": 9000}],
        WIDE_ROLES,
    )
    assert res.df["sales"].max() < 9000
    start = str(_wide()["week"].iloc[3].date())
    res = apply_pipeline(_wide(), [{"op": "date_range", "start": start}], WIDE_ROLES)
    assert len(res.df) == 7


def test_cast_and_parse_date():
    df = _wide().copy()
    df["sales"] = df["sales"].astype(str)
    res = apply_pipeline(
        df, [{"op": "cast", "column": "sales", "dtype": "number"}], WIDE_ROLES
    )
    assert pd.api.types.is_numeric_dtype(res.df["sales"])


def test_event_dummy_wide_and_long():
    p = str(_wide()["week"].iloc[2].date())
    res = apply_pipeline(
        _wide(), [{"op": "event_dummy", "name": "promo", "periods": [p]}], WIDE_ROLES
    )
    assert "promo" in res.df.columns
    assert res.df["promo"].sum() == 1.0
    assert res.roles.get("promo") == "control"

    res = apply_pipeline(
        _long(),
        [
            {
                "op": "event_dummy",
                "name": "promo",
                "periods": [str(_long()["Period"].iloc[1].date())],
            }
        ],
        LONG_ROLES,
    )
    promo = res.df[res.df["VariableName"] == "promo"]
    assert len(promo) == 6 and promo["VariableValue"].sum() == 1.0


# ── replay / idempotency / errors ─────────────────────────────────────────────


def test_replay_is_idempotent():
    steps = [
        {"op": "drop_columns", "columns": ["notes"]},
        {"op": "fill_missing", "strategy": "zero"},
        {"op": "winsorize", "column": "sales", "cap_value": 500.0},
    ]
    once = apply_pipeline(_wide(), steps, WIDE_ROLES).df
    twice = apply_pipeline(once, steps, WIDE_ROLES).df
    # applying the same value-clamping pipeline again changes nothing
    pd.testing.assert_frame_equal(
        once.reset_index(drop=True), twice.reset_index(drop=True)
    )


def test_unknown_op_raises_transform_error():
    with pytest.raises(TransformError):
        apply_pipeline(_wide(), [{"op": "teleport"}], WIDE_ROLES)


def test_missing_required_param_raises():
    with pytest.raises(TransformError):
        apply_pipeline(
            _wide(), [{"op": "winsorize", "column": "sales"}], WIDE_ROLES
        )  # no cap_value


def test_data_level_error_warns_not_crashes():
    # winsorize a column that doesn't exist -> recorded as a warning, frame intact
    res = apply_pipeline(
        _wide(), [{"op": "winsorize", "column": "ghost", "cap_value": 1.0}], WIDE_ROLES
    )
    assert res.warnings and "ghost" in res.warnings[0]
    assert "sales" in res.df.columns
