"""One correct carryover reader, delegating to the model's own kernel (#218).

Five places derived "how long does a channel keep working?" and disagreed. The
reader behind the reports and the experiment optimiser was wrong five ways at
once: family-blind, dropped Weibull and no-adstock channels entirely, reported a
legacy Beta MIXTURE WEIGHT as a decay rate, collapsed the posterior before a
convex transform, and always used ``l_max = 8``.

These tests use SYNTHETIC traces, not fits. The kernel is a deterministic
function of the posterior parameters, so a fit would add sampling noise without
adding evidence — and the properties worth pinning (family-awareness, exact
agreement with the shipped transform, per-draw vs collapsed) are exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.transforms.adstock import adstock_weights
from mmm_framework.transforms.carryover import (
    carryover_half_life,
    posterior_carryover_kernels,
)

# ---------------------------------------------------------------------------
# a minimal fake model carrying a real trace-shaped posterior
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, type_, l_max, normalize=True):
        self.type = type_
        self.l_max = l_max
        self.normalize = normalize


def _model(channels: dict[str, tuple], *, n_draws: int = 64, legacy: bool = False):
    """`channels` maps name -> (AdstockType, l_max, {param: array-or-scalar})."""
    import xarray as xr

    from mmm_framework.config.enums import AdstockType

    data = {}
    cfgs = {}
    for ch, (kind, l_max, params) in channels.items():
        cfgs[ch] = _Cfg(kind, l_max)
        for p, v in params.items():
            arr = np.broadcast_to(np.asarray(v, dtype=float), (n_draws,)).copy()
            name = p if p.startswith("adstock_") else f"adstock_{p}_{ch}"
            data[name] = xr.DataArray(arr.reshape(1, n_draws), dims=("chain", "draw"))

    class _Trace:
        posterior = xr.Dataset(data)

    class _M:
        _trace = _Trace()
        channel_names = list(channels)
        use_parametric_adstock = not legacy
        adstock_alphas = [0.0, 0.3, 0.5, 0.7, 0.9]

        def _get_adstock_config(self, ch):
            return cfgs[ch]

    _ = AdstockType  # imported for the caller's convenience
    return _M()


@pytest.fixture()
def AT():
    from mmm_framework.config.enums import AdstockType

    return AdstockType


# ---------------------------------------------------------------------------
# exact agreement with the model's own kernel
# ---------------------------------------------------------------------------


class TestDelegatesToTheShippedTransform:
    @pytest.mark.parametrize("l_max", [8, 12, 26])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_geometric_matches_adstock_weights_exactly(self, AT, l_max, normalize):
        m = _model({"TV": (AT.GEOMETRIC, l_max, {"alpha": 0.6})})
        m._get_adstock_config("TV").normalize = normalize
        k = posterior_carryover_kernels(m)["TV"]
        want = adstock_weights("geometric", l_max, alpha=0.6, normalize=normalize)
        np.testing.assert_allclose(k.kernel[0], want, rtol=0, atol=1e-12)
        assert k.kernel.shape[1] == l_max

    def test_delayed_is_family_aware(self, AT):
        """The old reader built alpha**lags and put a delayed peak at lag 0."""
        m = _model({"TV": (AT.DELAYED, 12, {"alpha": 0.6, "theta": 3.0})})
        k = posterior_carryover_kernels(m)["TV"]
        want = adstock_weights("delayed", 12, alpha=0.6, theta=3.0, normalize=True)
        np.testing.assert_allclose(k.kernel[0], want, rtol=0, atol=1e-12)
        assert int(np.argmax(k.mean_kernel)) == 3, "the peak must not be at lag 0"

    def test_weibull_is_present_not_dropped(self, AT):
        """Weibull writes shape/scale, matched no prefix, and vanished."""
        m = _model({"Radio": (AT.WEIBULL, 10, {"shape": 2.0, "scale": 3.0})})
        k = posterior_carryover_kernels(m)["Radio"]
        assert k.status == "ok" and k.family == "weibull"
        want = adstock_weights("weibull", 10, shape=2.0, scale=3.0, normalize=True)
        np.testing.assert_allclose(k.kernel[0], want, rtol=0, atol=1e-12)
        assert k.alpha_mean is None, "Weibull has no alpha; reporting one is a lie"

    def test_none_renders_a_unit_impulse_not_an_absence(self, AT):
        m = _model({"Direct": (AT.NONE, 8, {})})
        k = posterior_carryover_kernels(m)["Direct"]
        assert k.status == "ok" and k.family == "none"
        np.testing.assert_array_equal(k.mean_kernel, [1.0])

    def test_lmax_is_per_channel_not_always_eight(self, AT):
        m = _model(
            {
                "A": (AT.GEOMETRIC, 8, {"alpha": 0.5}),
                "B": (AT.GEOMETRIC, 26, {"alpha": 0.5}),
            }
        )
        ks = posterior_carryover_kernels(m)
        assert ks["A"].l_max == 8 and ks["B"].l_max == 26
        assert ks["B"].kernel.shape[1] == 26

    def test_every_requested_channel_is_returned(self, AT):
        m = _model(
            {
                "geo": (AT.GEOMETRIC, 8, {"alpha": 0.5}),
                "wei": (AT.WEIBULL, 8, {"shape": 2.0, "scale": 2.0}),
                "non": (AT.NONE, 8, {}),
            }
        )
        assert set(posterior_carryover_kernels(m)) == {"geo", "wei", "non"}


# ---------------------------------------------------------------------------
# per-draw, not collapsed
# ---------------------------------------------------------------------------


def test_per_draw_differs_from_collapsing_alpha_first(AT):
    """mean(alpha) ** lags is NOT mean(alpha ** lags).

    Measured on a real posterior, the old collapse understated the lag-5 weight
    7x. This pins that the reader does not do it.
    """
    rng = np.random.default_rng(0)
    alphas = np.clip(rng.normal(0.5, 0.22, 256), 0.01, 0.98)
    m = _model({"TV": (AT.GEOMETRIC, 12, {"alpha": alphas})}, n_draws=256)
    k = posterior_carryover_kernels(m)["TV"]

    per_draw = k.mean_kernel
    collapsed = adstock_weights(
        "geometric", 12, alpha=float(alphas.mean()), normalize=True
    )
    assert not np.allclose(per_draw, collapsed, rtol=1e-3)
    # the collapse understates the tail
    assert per_draw[5] > collapsed[5] * 1.5


# ---------------------------------------------------------------------------
# legacy: a mixture weight is not a decay rate
# ---------------------------------------------------------------------------


class TestLegacyBlend:
    def test_reconstructs_the_blend_and_refuses_to_call_it_alpha(self):
        m = _model({"TV": (None, 8, {"adstock_TV": 0.07})}, legacy=True)
        k = posterior_carryover_kernels(m)["TV"]

        assert k.status == "legacy_blend"
        assert k.alpha_mean is None, (
            "the stored value is a Beta MIXTURE WEIGHT between two fixed alphas; "
            "reporting it as a decay rate is the bug this closes"
        )
        # the reconstructed kernel is front-loaded, not a 0.07-decay
        assert k.mean_kernel[0] > 0.5
        assert k.truncated_tail_mass > 0, "legacy adstock is IIR; state the tail"
        assert "IIR" in k.note

    def test_a_legacy_fit_without_the_mixture_is_reported_not_dropped(self):
        m = _model({"TV": (None, 8, {})}, legacy=True)
        k = posterior_carryover_kernels(m)["TV"]
        assert k.status == "missing_params" and "adstock_TV" in k.note


# ---------------------------------------------------------------------------
# the canonical half-life
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_monotone_in_alpha(self):
        """The shipped implementations were NOT.

        They special-cased the first lag with a different origin, so geometric
        0.5 gave 1.00 while 0.7 gave 0.82 — more carryover reported as a shorter
        half-life.
        """
        hl = [
            carryover_half_life(
                adstock_weights("geometric", 8, alpha=a, normalize=True)[None, :]
            )[0]
            for a in (0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
        ]
        assert all(b > a for a, b in zip(hl, hl[1:])), hl

    def test_continuous_across_the_old_discontinuity(self):
        """The old branches jumped at cum[0] == 0.5, i.e. around alpha = 0.5."""
        xs = np.linspace(0.46, 0.54, 17)
        hl = [
            carryover_half_life(
                adstock_weights("geometric", 8, alpha=float(a), normalize=True)[None, :]
            )[0]
            for a in xs
        ]
        assert max(abs(np.diff(hl))) < 0.05

    def test_truncation_bias_is_real_and_reported(self, AT):
        """A horizon read off a truncated kernel understates the true one."""
        short = carryover_half_life(
            adstock_weights("geometric", 8, alpha=0.95, normalize=True)[None, :]
        )[0]
        long = carryover_half_life(
            adstock_weights("geometric", 200, alpha=0.95, normalize=True)[None, :]
        )[0]
        assert long > 3 * short

        m = _model({"TV": (AT.GEOMETRIC, 8, {"alpha": 0.95})})
        assert posterior_carryover_kernels(m)["TV"].truncated_tail_mass > 0.05

    def test_a_humped_kernel_has_a_half_life_past_its_peak(self):
        k = adstock_weights("delayed", 12, alpha=0.6, theta=3.0, normalize=True)
        assert carryover_half_life(k[None, :])[0] > int(np.argmax(k))

    def test_degenerate_kernels_are_nan_not_zero(self):
        out = carryover_half_life(np.zeros((2, 5)))
        assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# the consumers that were provably broken
# ---------------------------------------------------------------------------


class TestConsumers:
    def test_compute_adstock_weights_returns_weibull_channels(self, AT):
        from mmm_framework.reporting.helpers.adstock import compute_adstock_weights

        m = _model(
            {
                "TV": (AT.GEOMETRIC, 8, {"alpha": 0.6}),
                "Radio": (AT.WEIBULL, 10, {"shape": 2.0, "scale": 3.0}),
            }
        )
        res = compute_adstock_weights(m)
        assert set(res) == {"TV", "Radio"}, "Weibull channels used to vanish"
        assert res["Radio"].family == "weibull"
        assert res["Radio"].l_max == 10, "l_max used to be hardcoded to 8"
        assert np.isnan(res["Radio"].alpha_mean)

    def test_compute_adstock_weights_is_family_aware(self, AT):
        from mmm_framework.reporting.helpers.adstock import compute_adstock_weights

        m = _model({"TV": (AT.DELAYED, 12, {"alpha": 0.6, "theta": 3.0})})
        r = compute_adstock_weights(m)["TV"]
        assert int(np.argmax(r.decay_weights)) == 3
        assert r.family == "delayed"

    def test_deck_half_life_is_not_unconditionally_none(self, AT):
        """It called a 2-arg function with 3 positional args -> TypeError -> None."""
        from mmm_framework.reporting.deck.builder import _half_life_weeks

        m = _model({"TV": (AT.GEOMETRIC, 8, {"alpha": 0.7})})
        got = _half_life_weeks(m, "TV")
        assert got is not None and got > 0

    def test_deck_half_life_works_for_weibull_too(self, AT):
        from mmm_framework.reporting.deck.builder import _half_life_weeks

        m = _model({"Radio": (AT.WEIBULL, 10, {"shape": 2.0, "scale": 3.0})})
        assert _half_life_weeks(m, "Radio") is not None


def test_unfitted_model_raises():
    class _M:
        _trace = None
        channel_names = ["TV"]

    with pytest.raises(ValueError, match="not fitted"):
        posterior_carryover_kernels(_M())
