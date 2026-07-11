"""Every extension model can feed the interactive results report.

The report pipeline reads a small duck-typed surface off a fitted model
(``X_media_raw`` / ``y_raw`` / ``time_idx`` / ``sample_channel_contributions``
+ a ``channel_contributions`` graph deterministic). These tests pin that
surface for all four extension families and — as a slow check — drive the real
``interactive_report_facts`` end to end for each.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework.mmm_extensions.builders import (
    CombinedModelConfigBuilder,
    MultivariateModelConfigBuilder,
)
from mmm_framework.mmm_extensions.config import (
    MediatorConfig,
    MediatorType,
    NestedModelConfig,
)
from mmm_framework.mmm_extensions.models.combined import CombinedMMM
from mmm_framework.mmm_extensions.models.multivariate import MultivariateMMM
from mmm_framework.mmm_extensions.models.nested import NestedMMM

N = 60
CHANNELS = ["TV", "Digital"]


@pytest.fixture(scope="module")
def media() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(100, 20, (N, 2)))


@pytest.fixture(scope="module")
def idx() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=N, freq="W-MON")


def _sales(media: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1)
    aware = 40 + 0.3 * media[:, 0] + rng.normal(0, 4, N)
    return 1000 + 4 * aware + 2 * media[:, 1] + rng.normal(0, 40, N)


def _outcomes(media: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2)
    return {
        "prod_a": 1000 + 2 * media[:, 0] + rng.normal(0, 30, N),
        "prod_b": 800 + 1.5 * media[:, 1] + rng.normal(0, 25, N),
    }


def _nested(media, idx):
    cfg = NestedModelConfig(
        mediators=(
            MediatorConfig(name="Awareness", mediator_type=MediatorType.FULLY_LATENT),
        )
    )
    return NestedMMM(media, _sales(media), CHANNELS, cfg, index=idx)


def _multivariate(media, idx):
    cfg = MultivariateModelConfigBuilder().with_outcomes("prod_a", "prod_b").build()
    return MultivariateMMM(
        X_media=media,
        outcome_data=_outcomes(media),
        channel_names=CHANNELS,
        config=cfg,
        index=idx,
    )


def _combined(media, idx):
    cfg = (
        CombinedModelConfigBuilder()
        .with_awareness_mediator("awareness")
        .map_channels_to_mediator("awareness", ["TV"])
        .with_outcomes("prod_a", "prod_b")
        .build()
    )
    return CombinedMMM(
        X_media=media,
        outcome_data=_outcomes(media),
        channel_names=CHANNELS,
        config=cfg,
        index=idx,
    )


BUILDERS = {"nested": _nested, "multivariate": _multivariate, "combined": _combined}


# ---------------------------------------------------------------------------
# Fast: graph surface + fitted contributions (MAP)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", list(BUILDERS))
def test_channel_contributions_registered(kind, media, idx):
    """Every extension graph registers the (obs, channel) contribution det."""
    m = BUILDERS[kind](media, idx)
    named = m.model.named_vars
    assert "channel_contributions" in named
    det = named["channel_contributions"]
    # obs × channel deterministic (2-D).
    assert det.ndim == 2


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_report_data_aliases(kind, media, idx):
    m = BUILDERS[kind](media, idx)
    assert m.X_media_raw.shape == (N, len(CHANNELS))
    assert m.y_raw.shape == (N,)
    assert np.array_equal(m.time_idx, np.arange(N))


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_sample_channel_contributions_shape_and_finite(kind, media, idx):
    m = BUILDERS[kind](media, idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(method="map", random_seed=0)
    c = m.sample_channel_contributions(max_draws=1, random_seed=0)
    assert c.shape[1:] == (N, len(CHANNELS))
    assert np.isfinite(c).all()
    # A counterfactual scenario (media doubled) re-evaluates the curve.
    c2 = m.sample_channel_contributions(
        X_media=m.X_media_raw * 2.0, max_draws=1, random_seed=0
    )
    assert c2.shape == c.shape and np.isfinite(c2).all()


def test_unfitted_contributions_raise(media, idx):
    m = _nested(media, idx)
    with pytest.raises(ValueError, match="not fitted|fit"):
        m.sample_channel_contributions()


# ---------------------------------------------------------------------------
# Slow: full interactive_report_facts end to end (ADVI)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("kind", list(BUILDERS))
def test_interactive_report_facts_end_to_end(kind, media, idx):
    from mmm_framework.reporting.interactive.facts import interactive_report_facts

    m = BUILDERS[kind](media, idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = m.fit(method="advi", draws=200, random_seed=0)
    f = interactive_report_facts(
        m,
        results,
        max_draws=80,
        curve_max_draws=30,
        n_prior_samples=40,
        random_seed=42,
    )
    # Core sections all populate.
    assert f["meta"]["channels"] == CHANNELS
    assert f["meta"]["n_periods"] == N
    assert set(f["contrib"]["draws_b64"]) == set(CHANNELS)
    assert set(f["curves"]["draws_b64"]) == set(CHANNELS)
    assert len(f["headline"]["channels"]) == len(CHANNELS)
    # Predictive-fit + LOO-PIT (single-KPI / primary-outcome predict) present.
    assert f["fit"]["series"]
    assert (f["ppc_stats"] or {}).get("loo_pit") is not None
    # Every embedded number is finite/JSON-safe.
    import json

    json.dumps(f)
