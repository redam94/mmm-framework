"""Data Studio service layer — EDA-on-frame (inline, JSON-safe figures),
commit-artifact melt, and the no-LLM commit_core state update."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.agents import workspace as W

    return W


def _wide(n=60, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "week": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
            "sales": 1000 + np.arange(n) * 5 + rng.normal(0, 20, n),
            "tv_spend": np.abs(rng.normal(50, 10, n)),
            "search_spend": np.abs(rng.normal(30, 8, n)),
        }
    )
    df.loc[n // 2, "tv_spend"] = 900.0  # spike → an outlier suggestion
    return df


def _stage(tid: str, df: pd.DataFrame, name="sales.csv"):
    from mmm_framework.data_studio import service as S

    raw = S.raw_dir(tid) / name
    if name.endswith(".csv"):
        df.to_csv(raw, index=False)
    return S.init_manifest(tid, str(raw), name, "dataset", raw.stat().st_size)


# ── loader + roles ────────────────────────────────────────────────────────────


def test_load_eda_panel_from_df_wide_and_long():
    from mmm_framework.eda import load_eda_panel_from_df

    p = load_eda_panel_from_df(_wide())
    assert p.kpi == "sales" and "tv_spend" in p.media

    long = (
        _wide()
        .melt(id_vars="week", var_name="VariableName", value_name="VariableValue")
        .rename(columns={"week": "Period"})
    )
    pl = load_eda_panel_from_df(long)
    assert pl.df_long is not None


def test_infer_roles(ws):
    from mmm_framework.data_studio import service as S

    roles = S.infer_roles(_wide())
    assert roles["sales"] == "kpi"
    assert roles["tv_spend"] == "media" and roles["search_spend"] == "media"
    assert roles["week"] == "date"


def _wide_ddmmyyyy(n=27) -> pd.DataFrame:
    # DD/MM/YYYY dates (days > 12 force day-first) — the strict loader crashes on these
    dates = [f"{(d % 28) + 1:02d}/11/2021" for d in range(n)]
    return pd.DataFrame(
        {
            "order_date": dates,
            "revenue": 1000 + np.arange(n) * 5.0,
            "tv_spend": np.abs(np.random.RandomState(0).normal(50, 10, n)),
            "search_spend": np.abs(np.random.RandomState(1).normal(30, 8, n)),
        }
    )


def test_infer_roles_survives_non_iso_dates(ws):
    """A DD/MM/YYYY date column must NOT silently null role inference."""
    from mmm_framework.data_studio import service as S

    roles = S.infer_roles(_wide_ddmmyyyy())
    assert roles.get("revenue") == "kpi"
    assert roles.get("tv_spend") == "media"
    assert roles.get("order_date") == "date"


def test_run_eda_does_not_crash_on_non_iso_dates(ws):
    """EDA on a non-ISO-date frame returns analyses, not a 500/exception."""
    import json

    from mmm_framework.data_studio import service as S

    df = _wide_ddmmyyyy()
    out = S.run_eda_on_frame(df, S.infer_roles(df), analyses=["overview", "outliers"])
    assert "overview" in out["analyses"]
    assert out["analyses"]["overview"]["figures"]
    json.dumps(out)


def test_commit_normalizes_ddmmyyyy_dates(ws):
    """Committing a DD/MM/YYYY frame writes parseable MFF-long dates."""
    from mmm_framework.data_studio import service as S

    tid = "t_commit_ddmm"
    _stage(tid, _wide_ddmmyyyy(), name="orders.csv")
    state = {
        "model_spec": {},
        "locked_fields": [],
        "pending_spec_changes": [],
        "dashboard_data": {},
    }
    err, _summary, update = S.commit_core(state, tid, reason="t")
    assert err is None, err
    df_long = pd.read_csv(update["dataset_path"])
    parsed = pd.to_datetime(df_long["Period"], errors="coerce")
    assert parsed.notna().all()  # every committed date parses


# ── EDA on frame: inline + json-safe ──────────────────────────────────────────


def test_run_eda_inline_figures_are_json_safe(ws):
    from mmm_framework.data_studio import service as S

    df = _wide()
    roles = S.infer_roles(df)
    out = S.run_eda_on_frame(
        df, roles, analyses=["overview", "correlation", "missingness", "outliers"]
    )
    # every figure carries inline data/layout (so PlotCard renders without /plots fetch)
    for sec in out["analyses"].values():
        for fig in sec["figures"]:
            assert "data" in fig and "layout" in fig and "key" in fig
    # the whole payload must round-trip JSON (no numpy scalars -> checkpointer-safe)
    json.dumps(out)
    assert out["issues"], "validation issues should be computed"
    assert out["outlier_suggestions"], "the planted spike should yield a suggestion"


def test_outlier_suggestions_map_to_steps(ws):
    from mmm_framework.data_studio import service as S
    from mmm_framework.eda import load_eda_panel_from_df

    df = _wide()
    panel = load_eda_panel_from_df(
        df,
        {
            "kpi": "sales",
            "media_channels": [{"name": "tv_spend"}, {"name": "search_spend"}],
        },
    )
    _report, suggestions, _damaged = S.outlier_suggestions(panel)
    assert suggestions
    steps = [s["step"] for s in suggestions if s["step"]]
    assert any(s["op"] in ("winsorize", "impute", "event_dummy") for s in steps)


# ── commit ────────────────────────────────────────────────────────────────────


def test_build_commit_artifact_melts_wide_to_mff_long(ws):
    from mmm_framework.data_studio import service as S

    tid = "t_commit_melt"
    _stage(tid, _wide())
    result = S.current_result(tid)
    path, df_long, spec = S.build_commit_artifact(tid, result)
    assert {"Period", "VariableName", "VariableValue"}.issubset(df_long.columns)
    assert set(df_long["VariableName"].unique()) == {
        "sales",
        "tv_spend",
        "search_spend",
    }
    assert spec["kpi"] == "sales"
    assert {c["name"] for c in spec["media_channels"]} == {"tv_spend", "search_spend"}
    assert spec["time_granularity"] == "weekly"
    assert path.endswith("data_studio_dataset.csv")


def test_commit_core_sets_spec_and_dashboard(ws):
    from mmm_framework.data_studio import service as S

    tid = "t_commit_core"
    _stage(tid, _wide())
    state = {
        "model_spec": {},
        "locked_fields": [],
        "pending_spec_changes": [],
        "dashboard_data": {},
    }
    err, summary, update = S.commit_core(state, tid, reason="t")
    assert err is None, err
    assert update["dataset_path"].endswith("data_studio_dataset.csv")
    assert update["model_spec"]["kpi"] == "sales"
    ds = update["dashboard_data"]["dataset"]
    assert ds["rows"] > 0 and "sales" in ds["variable_names"]
    assert update["dashboard_data"]["data_studio"]["committed"] is True
    # The no-LLM invariant: commit never appends a ToolMessage/AIMessage.
    assert "messages" not in update
    json.dumps(update)  # checkpointer-safe


def test_commit_requires_kpi_and_media(ws):
    from mmm_framework.data_studio import service as S

    tid = "t_commit_norole"
    # a frame with no kpi/media keywords -> heuristic finds nothing
    df = pd.DataFrame(
        {
            "week": pd.date_range("2023-01-02", periods=8, freq="W-MON"),
            "aaa": range(8),
            "bbb": range(8),
        }
    )
    _stage(tid, df)
    # blank roles
    S.set_pipeline(tid, [], {"week": "date"})
    state = {
        "model_spec": {},
        "locked_fields": [],
        "pending_spec_changes": [],
        "dashboard_data": {},
    }
    err, _summary, _update = S.commit_core(state, tid, reason="t")
    assert err and "KPI" in err


def test_commit_respects_locked_fields(ws):
    from mmm_framework.data_studio import service as S

    tid = "t_commit_lock"
    _stage(tid, _wide())
    # user locked kpi to a different value -> commit must defer, not overwrite
    state = {
        "model_spec": {"kpi": "search_spend"},
        "locked_fields": ["kpi"],
        "pending_spec_changes": [],
        "dashboard_data": {},
    }
    err, _summary, update = S.commit_core(state, tid, reason="t")
    assert err is None
    assert update["model_spec"]["kpi"] == "search_spend"  # lock preserved
    assert any(p["path"] == "kpi" for p in update["pending_spec_changes"])


@pytest.mark.slow
def test_wide_commit_fits_via_mff_branch(ws):
    """The headline guarantee: a cleaned WIDE upload, committed, fits through the
    standard MFF build_model branch (no spec['dataset'])."""
    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.data_studio import service as S

    tid = "t_commit_fit"
    _stage(tid, _wide(n=80, seed=3))
    state = {
        "model_spec": {
            "inference": {"draws": 40, "tune": 40, "chains": 2, "random_seed": 0}
        },
        "locked_fields": [],
        "pending_spec_changes": [],
        "dashboard_data": {},
    }
    err, _summary, update = S.commit_core(state, tid, reason="t")
    assert err is None
    spec = update["model_spec"]
    assert "dataset" not in spec  # MFF branch, not the native Dataset path
    mmm, results, info = build_and_fit(spec, update["dataset_path"])
    assert len(mmm.channel_names) == 2
