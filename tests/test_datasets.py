"""Bundled example datasets: ``load_example`` and friends.

Pins the zero-effort first-run contract (issue #146): the demo CSVs ship inside
the package and ``load_example`` returns a ready-to-fit panel in one call, with
a sealed answer key alongside it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework import (
    BayesianMMM,
    ModelConfigBuilder,
    PanelDataset,
    TrendConfig,
    TrendType,
    list_examples,
    load_example,
    load_example_answer_key,
)
from mmm_framework.datasets import EXAMPLES


def test_list_examples_covers_registry():
    names = list_examples()
    assert set(names) == set(EXAMPLES)
    assert names["national"] and names["geo"]  # non-empty descriptions


def test_national_example_shape():
    panel = load_example("national")
    assert isinstance(panel, PanelDataset)
    assert panel.n_channels == 7
    assert panel.n_controls == 6
    assert not panel.is_panel  # national -> single series
    assert list(panel.X_media.columns) == [
        "TV",
        "Search",
        "Social",
        "Display",
        "Video",
        "Radio",
        "Print",
    ]
    assert panel.n_obs == 104  # 104 weekly periods
    assert np.isfinite(panel.y.to_numpy()).all()


def test_geo_example_is_a_panel():
    panel = load_example("geo")
    assert isinstance(panel, PanelDataset)
    assert panel.is_panel
    assert panel.coords.n_geos == 8
    assert panel.n_channels == 4
    assert list(panel.X_media.columns) == ["TV", "Search", "Social", "Display"]


def test_as_frame_returns_raw_mff():
    df = load_example("national", as_frame=True)
    assert isinstance(df, pd.DataFrame)
    assert {"Period", "VariableName", "VariableValue"} <= set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["Period"])
    # raw frame keeps the decoy columns the built panel drops
    assert "noise_1" in set(df["VariableName"])


def test_loads_without_warnings():
    """A curated example should load cleanly — no contiguity or extra-variable noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        load_example("national")
        load_example("geo")


def test_answer_key_present_and_shaped():
    key = load_example_answer_key("national")
    assert key is not None
    assert set(key["channels"]) >= {"TV", "Search", "Social", "Display"}
    assert key["true_roas"]["TV"] > key["true_roas"]["Search"]  # brand > performance


def test_unknown_example_raises():
    with pytest.raises(ValueError, match="Unknown example"):
        load_example("does-not-exist")
    with pytest.raises(ValueError, match="Unknown example"):
        load_example_answer_key("does-not-exist")


@pytest.mark.slow
def test_national_example_fits_and_recovers_roi_ranking():
    """End-to-end: the bundled panel fits and recovers the causal ROI ranking.

    Not a magnitude claim (a single observational fit attenuates), but the
    brand channels (TV/Social/Video) should out-rank the performance channels
    (Search/Display) — the structural signal the sealed key encodes.
    """
    panel = load_example("national")
    config = (
        ModelConfigBuilder()
        .bayesian_numpyro()
        .with_chains(4)
        .with_draws(500)
        .with_tune(500)
        .build()
    )
    mmm = BayesianMMM(panel, config, TrendConfig(type=TrendType.LINEAR))
    results = mmm.fit(random_seed=42)
    assert results.diagnostics["rhat_max"] < 1.05

    decomp = mmm.compute_component_decomposition()
    contribution = decomp.media_by_channel.sum()
    spend = panel.X_media.sum()
    roi = contribution / spend
    assert np.isfinite(roi.to_numpy()).all()

    brand = roi[["TV", "Social", "Video"]].mean()
    performance = roi[["Search", "Display"]].mean()
    assert brand > performance
