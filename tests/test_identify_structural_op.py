"""Wiring + end-to-end tests for the structural-identification op.

Fast: the gate refuses a non-parametric (legacy mix-weight) fit. Slow: a real
tiny parametric geometric + logistic national fit runs the full op — the anchor
extraction, the byte-mirror self-check, and the per-parameter (β, α, ψ)
identification.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.agents.model_ops import (
    _structural_anchor,
    _structural_self_check,
    identify_structural_parameters,
)


class _LegacyMMM:
    """Minimal stand-in for a fit without parametric adstock."""

    channel_names = ["TV", "Search"]
    use_parametric_adstock = False
    n_cells = 1


class _PanelMMM:
    """Parametric, but multi-cell (geo panel) → v1 refuses."""

    channel_names = ["TV"]
    use_parametric_adstock = True
    n_cells = 2


def test_gate_refuses_legacy_non_parametric():
    anchor, reason = _structural_anchor(_LegacyMMM(), "TV")
    assert anchor is None
    assert "non-parametric" in reason or "parametric" in reason


def test_gate_unknown_channel():
    anchor, reason = _structural_anchor(_LegacyMMM(), "Nope")
    assert anchor is None
    assert "not in" in reason


def test_gate_refuses_multi_cell_panel():
    anchor, reason = _structural_anchor(_PanelMMM(), "TV")
    assert anchor is None
    assert "single-cell" in reason or "national" in reason


def test_self_check_fails_closed_on_mismatch():
    """The byte-mirror self-check must FAIL when the model's contribution does
    not match the numpy forward op (fail-closed → Tier 2 withheld)."""
    rng = np.random.default_rng(0)
    T, C = 24, 2
    x_media = np.abs(rng.normal(50, 10, (T, C))) + 1.0
    anchor = {
        "channel_index": 0,
        "x_media": x_media,
        "draws": {
            "beta": np.clip(rng.normal(1.5, 0.3, 400), 1e-3, None),
            "alpha": np.clip(rng.normal(0.5, 0.08, 400), 0.0, 0.95),
            "lam": np.clip(rng.normal(1.0, 0.2, 400), 1e-3, None),
        },
        "op_spend": float(x_media[:, 0].mean()),
        "raw_max": float(x_media[:, 0].max() * 1.2),
        "y_std": 100.0,
        "l_max": 8,
        "normalize": True,
    }

    class _BadMMM:
        def sample_channel_contributions(self, X_media=None, max_draws=None, random_seed=None):
            # deliberately unrelated noise → must not byte-mirror the forward op
            n = int(max_draws or 20)
            return rng.normal(0.0, 1e6, (n, T, C))

    ok, detail = _structural_self_check(_BadMMM(), anchor, n_draws=20, random_seed=1)
    assert ok is False
    assert detail


def test_op_requires_model():
    out = identify_structural_parameters(
        None, dataset_path="x", kpi="Sales", channel="TV"
    )
    assert out.get("error")


def test_get_adstock_alpha_prefers_parametric_over_legacy():
    """The kind-aware fix: parametric ``adstock_alpha_<ch>`` must win over the
    legacy ``adstock_<ch>`` mix weight; a legacy-only posterior still resolves."""
    from types import SimpleNamespace

    from mmm_framework.reporting.helpers.adstock import _get_adstock_alpha

    def _var(v):
        return SimpleNamespace(values=np.full(50, v))

    both = {"adstock_alpha_TV": _var(0.7), "adstock_TV": _var(0.2)}
    assert float(np.mean(_get_adstock_alpha(both, "TV"))) == pytest.approx(0.7)

    legacy_only = {"adstock_TV": _var(0.4)}
    assert float(np.mean(_get_adstock_alpha(legacy_only, "TV"))) == pytest.approx(0.4)

    assert _get_adstock_alpha({}, "TV") is None


@pytest.mark.slow
def test_identify_structural_end_to_end(tmp_path):
    """Parametric geometric + logistic national fit → the op extracts the anchor,
    passes the byte-mirror self-check, and reports per-parameter identification."""
    import logging

    logging.disable(logging.CRITICAL)
    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.synth import generate_mff

    df, _ = generate_mff("realistic", seed=5, n_weeks=130)  # national
    path = str(tmp_path / "nat.csv")
    df.to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "media_channels": [
            {
                "name": n,
                "adstock": {"type": "geometric", "l_max": 8},
                "saturation": {"type": "logistic"},
            }
            for n in ["TV", "Search", "Social", "Display"]
        ],
        "control_variables": [],
        "inference": {"draws": 60, "tune": 60, "chains": 2, "random_seed": 0},
        "seasonality": {"yearly": 2},
        "trend": {"type": "linear"},
    }
    mmm, results, _ = build_and_fit(spec, path)
    assert mmm.use_parametric_adstock

    # the anchor extracts the three structural posteriors
    anchor, reason = _structural_anchor(mmm, "TV")
    assert anchor is not None, reason
    assert set(anchor["draws"]) == {"beta", "alpha", "lam"}
    assert anchor["op_spend"] > 0 and anchor["raw_max"] > 0 and anchor["y_std"] > 0

    out = identify_structural_parameters(
        mmm,
        dataset_path=path,
        kpi="Sales",
        channel="TV",
        levels=[0.5, 1.0, 1.5],
        duration=16,
        max_draws=60,
        self_check_draws=40,
    )
    assert not out.get("error")
    payload = out["dashboard"]["structural_identification"]
    assert payload["structural_gated"] is True
    # the numpy forward op must byte-mirror the fitted graph
    assert payload["self_check"]["passed"] is True, payload["self_check"]
    struct = payload["structural"]
    assert struct is not None
    for p in ("beta", "alpha", "lam"):
        assert p in struct["params"]
        d = struct["params"][p]
        assert d["prior_sd"] > 0
        if d["claimed"]:
            assert d["post_sd"] is not None and d["post_sd"] > 0
            assert 0.0 <= d["contraction"] <= 1.0
    # a 3-level schedule with blocks >= cooldown can claim the curve
    assert payload["n_levels"] >= 3
    assert struct["upper_bound"] is True
    # a structural table is attached
    assert out.get("tables")


@pytest.mark.slow
def test_identify_structural_gate_off_hill_saturation(tmp_path):
    """A hill-saturation fit gates Tier 2 off but still returns the reduced-form
    curve power (the schedule still traces the saturation with >=3 levels)."""
    import logging

    logging.disable(logging.CRITICAL)
    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.synth import generate_mff

    df, _ = generate_mff("realistic", seed=6, n_weeks=120)
    path = str(tmp_path / "nat.csv")
    df.to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        # default saturation is hill → Tier 2 gate refuses (logistic only)
        "media_channels": [{"name": n} for n in ["TV", "Search", "Social"]],
        "control_variables": [],
        "inference": {"draws": 40, "tune": 40, "chains": 2, "random_seed": 0},
        "seasonality": {"yearly": 2},
        "trend": {"type": "linear"},
    }
    mmm, results, _ = build_and_fit(spec, path)

    out = identify_structural_parameters(
        mmm, dataset_path=path, kpi="Sales", channel="TV", levels=[0.5, 1.0, 1.5]
    )
    assert not out.get("error")
    payload = out["dashboard"]["structural_identification"]
    assert payload["structural_gated"] is False
    assert payload["structural"] is None
    assert "logistic" in payload["structural_gate_reason"]
    # reduced-form schedule still computed (>=3 levels trace the curve)
    assert payload["n_levels"] >= 3
