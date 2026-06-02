"""Tests for the geo-based identification diagnostic (P2-1, honestly scoped).

Reports whether cross-geo spend variation is sufficient to support geo-level
inference; does not change the model. Fast (fake geo model, no MCMC).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mmm_framework.validation import geo_spend_variation_diagnostic


def _geo_model(per_geo_spend: dict[str, list[float]], obs_per_geo: int = 10):
    """Build a fake multi-geo model: each channel's per-geo spend is given."""
    channels = list(per_geo_spend)
    n_geos = len(next(iter(per_geo_spend.values())))
    rows_media = []
    geo_idx = []
    for g in range(n_geos):
        for _ in range(obs_per_geo):
            geo_idx.append(g)
            rows_media.append([per_geo_spend[ch][g] for ch in channels])
    return SimpleNamespace(
        has_geo=True,
        n_geos=n_geos,
        geo_idx=np.array(geo_idx),
        X_media_raw=np.array(rows_media, dtype=float),
        channel_names=channels,
    )


class TestGeoVariationDiagnostic:
    def test_uniform_channel_flagged_weak_varying_channel_sufficient(self):
        model = _geo_model(
            {
                "TV": [100.0, 500.0, 50.0],  # strong cross-geo variation
                "Radio": [100.0, 100.0, 100.0],  # uniform -> no identifying variation
            }
        )
        diag = geo_spend_variation_diagnostic(model)
        by_ch = {c.channel: c for c in diag.channels}
        assert by_ch["TV"].sufficient is True
        assert by_ch["Radio"].sufficient is False
        assert by_ch["Radio"].cv_across_geos < 0.01
        assert "Radio" in diag.weak_channels and "TV" not in diag.weak_channels
        assert diag.has_identifying_variation is True
        # Honest framing: variation is only *quasi*-experimental.
        assert "exogenous" in diag.caveat

    def test_national_model_raises(self):
        national = SimpleNamespace(has_geo=False, n_geos=1)
        with pytest.raises(ValueError, match="multi-geo"):
            geo_spend_variation_diagnostic(national)

    def test_to_dict_shape(self):
        model = _geo_model({"TV": [100.0, 300.0, 50.0]})
        d = geo_spend_variation_diagnostic(model).to_dict()
        assert d["n_geos"] == 3
        assert d["channels"][0]["channel"] == "TV"
        assert "caveat" in d and "weak_channels" in d

    def test_zero_spend_channel_handled(self):
        # A channel with zero spend everywhere must not divide-by-zero; cv=0,
        # flagged as insufficient.
        model = _geo_model({"TV": [100.0, 300.0, 50.0], "Dead": [0.0, 0.0, 0.0]})
        diag = geo_spend_variation_diagnostic(model)
        dead = next(c for c in diag.channels if c.channel == "Dead")
        assert dead.cv_across_geos == 0.0
        assert dead.sufficient is False
        assert "Dead" in diag.weak_channels
