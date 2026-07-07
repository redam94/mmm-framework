"""Saved-model settings are a single source of truth for the load/list surfaces.

``agents.fitting.saved_model_settings`` reads the COMPLETE settings of a saved
model directory — the full spec from ``run_metadata.json`` when the model came
from ``build_and_fit``, else the model's own serialized configs — and
``load_model_core`` restores that spec into the session so the Model
Configuration the user sees can never silently diverge from the fitted model
(the "fourier terms missing after load" bug).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mmm_framework.agents.fitting import (
    saved_model_settings,
    settings_digest_markdown,
)

# ── Fixture builders ──────────────────────────────────────────────────────────


RUN_SPEC = {
    "kpi": "Sales",
    "kpi_level": "national",
    "time_granularity": "weekly",
    "media_channels": [
        {
            "name": "TV",
            "adstock": {"type": "weibull", "l_max": 12},
            "saturation": {"type": "logistic"},
        },
        {"name": "Digital"},
    ],
    "control_variables": [{"name": "price"}],
    "trend": {"type": "linear"},
    "seasonality": {"yearly": 4, "monthly": 0, "weekly": 0},
    "inference": {"method": "advi", "chains": 4, "draws": 1000, "tune": 1000},
    "media_prior_mode": "roi",
}


def _write_run_record(save_dir, spec=RUN_SPEC):
    save_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_name": save_dir.name,
        "kpi": spec["kpi"],
        "channels": [m["name"] for m in spec["media_channels"]],
        "controls": [c["name"] for c in spec["control_variables"]],
        "trend": spec["trend"]["type"],
        "seasonality": spec["seasonality"],
        "inference": {**spec["inference"], "target_accept": 0.85},
        "spec": spec,
    }
    (save_dir / "run_metadata.json").write_text(json.dumps(meta))
    return save_dir


def _write_serialized_configs(save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "model_config": {
            "seasonality": {"yearly": 3, "monthly": None, "weekly": None},
            "fit_method": "map",
            "n_chains": 2,
            "n_draws": 500,
            "n_tune": 500,
            "target_accept": 0.9,
            "likelihood": {"family": "normal", "link": "identity"},
            "media_prior_mode": "coefficient",
        },
        "trend_config": {"type": "piecewise"},
        "mff_config": {
            "kpi": {"name": "Revenue"},
            "media_channels": [
                {
                    "name": "TV",
                    "adstock": {"type": "geometric", "l_max": 8},
                    "saturation": {"type": "hill"},
                },
            ],
            "controls": [{"name": "price"}],
        },
    }
    (save_dir / "configs.json").write_text(json.dumps(configs))
    return save_dir


# ── saved_model_settings ──────────────────────────────────────────────────────


def test_run_record_settings_carry_fourier_and_method(tmp_path):
    save_dir = _write_run_record(tmp_path / "run_x")
    out = saved_model_settings(str(save_dir))

    assert out["spec"] == RUN_SPEC  # full spec restorable
    s = out["settings"]
    assert s["source"] == "run_record"
    assert s["seasonality"] == {"yearly": 4, "monthly": 0, "weekly": 0}
    assert s["inference"]["method"] == "advi"
    assert s["media_prior_mode"] == "roi"
    assert {
        "name": "TV",
        "adstock": "weibull",
        "l_max": 12,
        "saturation": "logistic",
    } in s["channels"]
    # defaults materialize for a channel that didn't spell them out
    assert {
        "name": "Digital",
        "adstock": "geometric",
        "l_max": 8,
        "saturation": "hill",
    } in s["channels"]


def test_serialized_configs_fallback(tmp_path):
    # A manual save_model has no run record — the model's OWN configs are the
    # ground truth and must still yield the full settings picture.
    save_dir = _write_serialized_configs(tmp_path / "manual")
    out = saved_model_settings(str(save_dir))

    assert out["spec"] is None
    s = out["settings"]
    assert s["source"] == "serialized_config"
    assert s["kpi"] == "Revenue"
    assert s["trend"] == "piecewise"
    assert s["seasonality"] == {"yearly": 3, "monthly": 0, "weekly": 0}
    assert s["inference"]["method"] == "map"
    assert s["channels"][0]["name"] == "TV"


def test_empty_dir_degrades_gracefully(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    out = saved_model_settings(str(empty))
    assert out["spec"] is None
    # digest still renders without raising
    assert isinstance(settings_digest_markdown(out["settings"]), str)


# ── settings_digest_markdown ──────────────────────────────────────────────────


def test_digest_names_fourier_terms_and_flags_approximate(tmp_path):
    save_dir = _write_run_record(tmp_path / "run_y")
    digest = settings_digest_markdown(saved_model_settings(str(save_dir))["settings"])
    assert "Fourier" in digest
    assert "yearly order 4" in digest
    assert "advi" in digest
    assert "approximate" in digest  # non-NUTS methods are flagged
    assert "TV (weibull/logistic)" in digest


def test_digest_seasonality_off_and_nuts():
    digest = settings_digest_markdown(
        {
            "kpi": "Sales",
            "trend": "linear",
            "seasonality": {"yearly": 0, "monthly": 0, "weekly": 0},
            "inference": {"method": "nuts", "chains": 4, "draws": 1000, "tune": 1000},
        }
    )
    assert "Seasonality (Fourier): off" in digest
    assert "nuts" in digest
    assert "4 chains × 1000 draws" in digest
    assert "approximate" not in digest


# ── load_model_core restores the saved spec ───────────────────────────────────


def _write_mff(path, n=40):
    periods = pd.date_range("2021-01-04", periods=n, freq="W-MON")
    dims = {
        "Geography": None,
        "Product": None,
        "Campaign": None,
        "Outlet": None,
        "Creative": None,
    }
    rows = []
    for i, p in enumerate(periods):
        iso = p.strftime("%Y-%m-%d")
        rows.append(
            {
                **dims,
                "Period": iso,
                "VariableName": "Sales",
                "VariableValue": 1000 + 10 * i + (i % 4) * 25,
            }
        )
        rows.append(
            {
                **dims,
                "Period": iso,
                "VariableName": "TV",
                "VariableValue": 100 + (i % 5) * 20,
            }
        )
        rows.append(
            {
                **dims,
                "Period": iso,
                "VariableName": "Digital",
                "VariableValue": 80 + (i % 3) * 15,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.mark.slow
def test_load_model_core_restores_saved_spec(tmp_path, monkeypatch):
    """MAP fit → auto-save → load with a DIVERGED session spec: the loaded
    model must come back with the spec it was fit with (seasonality Fourier
    terms + approximate method included), not the session's edited spec."""
    monkeypatch.chdir(tmp_path)
    # Skip the (slow, best-effort) HTML report step of build_and_fit.
    import mmm_framework.reporting.generator as gen

    monkeypatch.setattr(
        gen,
        "ReportBuilder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped in test")),
    )

    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.agents.tools import load_model_core

    dataset_path = _write_mff(tmp_path / "data.csv")
    fit_spec = {
        "kpi": "Sales",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "trend": {"type": "linear"},
        "seasonality": {"yearly": 2, "monthly": 0, "weekly": 0},
        "inference": {"method": "map", "metrics_draws": 0},
    }
    mmm, results, info = build_and_fit(fit_spec, dataset_path)
    assert results.approximate is True
    run_name = info["model_run"]["run_name"]
    assert info["model_run"]["model_path"] is not None

    # The session spec has since been edited away from what was fit.
    diverged = {
        "kpi": "Sales",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "seasonality": {"yearly": 0, "monthly": 0, "weekly": 0},
    }
    res = load_model_core(None, run_name, diverged, dataset_path)
    assert res["ok"], res["message"]
    # The saved run's spec is returned for session restore …
    assert res["spec"]["seasonality"]["yearly"] == 2
    assert res["spec"]["inference"]["method"] == "map"
    # … and the message tells the user exactly what was loaded.
    assert "Fourier" in res["message"]
    assert "yearly order 2" in res["message"]
    assert "map" in res["message"]
    assert "restored" in res["message"].lower()
