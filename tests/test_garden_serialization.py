"""Tests that a bespoke garden model (a BayesianMMM subclass) round-trips
through the generalized MMMSerializer: the saved metadata records its identity,
and load() reconstructs the SAME subclass (not the base BayesianMMM). Slow — it
fits an approximate model."""

from __future__ import annotations

import json

import pytest

from mmm_framework import load_mff
from mmm_framework.agents.fitting import _mff_config_from_spec, build_model
from mmm_framework.serialization import MMMSerializer
from mmm_framework.synth import generate_mff

_GARDEN_SRC = """
from mmm_framework.garden import CustomMMM


class RoundTripMMM(CustomMMM):
    \"\"\"A trivial bespoke model used to exercise serialization.\"\"\"


GARDEN_MODEL = RoundTripMMM
"""


def _spec_for(df, answer):
    chans = answer["channels"]
    allv = list(dict.fromkeys(df["VariableName"].tolist()))
    ctrls = [v for v in allv if v not in chans and v != "Sales"]
    return {
        "kpi": "Sales",
        "media_channels": [{"name": c} for c in chans],
        "control_variables": [{"name": c} for c in ctrls],
        "trend": {"type": "linear"},
        "seasonality": {"yearly": 0, "monthly": 0, "weekly": 0},
        "inference": {"method": "map", "chains": 1, "draws": 100, "tune": 100},
    }


@pytest.mark.slow
def test_garden_subclass_roundtrips(tmp_path):
    src = tmp_path / "model.py"
    src.write_text(_GARDEN_SRC, encoding="utf-8")

    df, answer = generate_mff("clean", seed=3, n_weeks=52)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    spec = _spec_for(df, answer)
    spec["garden_ref"] = {
        "name": "roundtrip",
        "version": 1,
        "source_path": str(src),
        "class_name": "RoundTripMMM",
        "contract_version": "1.0",
    }

    mmm = build_model(spec, str(csv))
    assert type(mmm).__name__ == "RoundTripMMM"
    assert getattr(mmm, "_garden_ref", None) is not None
    mmm.fit(method="map", random_seed=3)

    save_dir = tmp_path / "saved"
    MMMSerializer.save(mmm, save_dir)

    # metadata records provenance
    meta = json.loads((save_dir / "metadata.json").read_text())
    assert meta["garden_ref"]["name"] == "roundtrip"
    assert "model_class_qualname" in meta
    assert meta["format_version"] == "1.1"

    # saving again over an existing dir exercises the atomic swap
    MMMSerializer.save(mmm, save_dir)

    panel = load_mff(str(csv), _mff_config_from_spec(spec))
    reloaded = MMMSerializer.load(save_dir, panel)
    assert type(reloaded).__name__ == "RoundTripMMM"  # the SUBCLASS, not the base
    assert reloaded._trace is not None


@pytest.mark.slow
def test_load_falls_back_when_source_missing(tmp_path):
    """If a garden model's source can't be resolved at load (e.g. loaded in a
    different session), load() falls back to BayesianMMM with a warning rather
    than hard-failing — the trace still loads for inspection."""
    src = tmp_path / "model.py"
    src.write_text(_GARDEN_SRC, encoding="utf-8")
    df, answer = generate_mff("clean", seed=3, n_weeks=52)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    spec = _spec_for(df, answer)
    spec["garden_ref"] = {
        "name": "roundtrip",
        "version": 1,
        "source_path": str(src),
        "class_name": "RoundTripMMM",
        "contract_version": "1.0",
    }
    mmm = build_model(spec, str(csv))
    mmm.fit(method="map", random_seed=3)
    save_dir = tmp_path / "saved"
    MMMSerializer.save(mmm, save_dir)

    # Remove the source so the class can't be resolved on load.
    src.unlink()
    panel = load_mff(str(csv), _mff_config_from_spec(spec))
    with pytest.warns(UserWarning, match="garden model"):
        reloaded = MMMSerializer.load(save_dir, panel)
    assert type(reloaded).__name__ == "BayesianMMM"  # graceful fallback
    assert reloaded._trace is not None
