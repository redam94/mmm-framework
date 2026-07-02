"""Unit tests for ``continuous_learning.arms.arm_shares`` — the outbound bridge
from a continuous-learning arms posterior to the breakout-weighted MMM's share
calibration (a ``ShareMeasurement``-shaped payload: mean shares + the empirical
ALR log-ratio covariance over posterior draws).

All tests run on a fabricated :class:`Posterior` (the ``_fake_posterior`` idiom
from ``test_continuous_learning.py``) — no MCMC.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.continuous_learning.arms import arm_shares, expand_arms
from mmm_framework.continuous_learning.model import Posterior, pair_name

# The Hill activation at _SPEND_REF with the fixture's kappa/alpha is ~0.6, so
# setting every Search pair's gamma to -0.9 drives the weakest arm's zero-out
# response (beta ~0.6 plus ~ -1.8 * 0.6 of interaction) firmly non-positive.
_CANNIBAL_GAMMA = -0.9

# Search split three ways + a plain Display channel -> 4 arms; the Search group
# is arms [0, 1, 2] with within-parent pairs carrying the gamma sites.
_ARMS = {"Search": ["Brand", "NonBrand", "Shopping"]}
_PAIRS = [(0, 1), (0, 2), (1, 2)]
_NAME_MAP = {
    "Brand": "Search_Brand",
    "NonBrand": "Search_NonBrand",
    "Shopping": "Search_Shopping",
}
_SPEND_REF = np.array([0.8, 0.8, 0.8, 0.8])


def _fake_posterior(n: int = 200, seed: int = 0, gamma: float = 0.0):
    """A Gaussian-ish posterior over the arms surface — no MCMC needed."""
    spec = expand_arms(["Search", "Display"], _ARMS)
    rng = np.random.default_rng(seed)
    k = spec.n_arms
    base_beta = np.array([2.0, 1.2, 0.6, 1.5])
    s = {
        "beta": np.abs(base_beta + 0.1 * rng.standard_normal((n, k))),
        "kappa": np.abs(0.6 + 0.05 * rng.standard_normal((n, k))),
        "alpha": np.clip(1.5 + 0.1 * rng.standard_normal((n, k)), 0.5, 5),
    }
    for i, j in _PAIRS:
        s[pair_name(spec.channels, (i, j))] = gamma + 0.02 * rng.standard_normal(n)
    post = Posterior(samples=s, channels=spec.channels, pairs=_PAIRS, pair_signs={})
    return post, spec


def test_shares_sum_to_one_and_names_are_mapped():
    post, spec = _fake_posterior()
    out = arm_shares(
        post, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, draws=100
    )
    assert out["channel"] == "Search"
    assert out["breakouts"] == ["Search_Brand", "Search_NonBrand", "Search_Shopping"]
    assert out["distribution"] == "logistic_normal"
    shares = np.asarray(out["shares"])
    assert shares.shape == (3,)
    assert np.all(shares > 0)
    assert float(shares.sum()) == pytest.approx(1.0, abs=1e-9)
    # the biggest beta (Brand) carries the biggest share at a common spend point
    assert int(np.argmax(shares)) == 0
    assert out["source"]["mode"] == "zero_out"
    assert out["source"]["spend_ref"] == [float(s) for s in _SPEND_REF]


def test_payload_round_trips_through_share_measurement():
    from mmm_framework.calibration.likelihood import ShareMeasurement

    post, spec = _fake_posterior()
    out = arm_shares(
        post, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, draws=100
    )
    meas = ShareMeasurement.from_dict(out)
    assert list(meas.breakouts) == out["breakouts"]
    assert meas.distribution == "logistic_normal"
    assert np.asarray(meas.log_ratio_cov).shape == (2, 2)


def test_missing_name_map_key_raises():
    post, spec = _fake_posterior()
    incomplete = {k: v for k, v in _NAME_MAP.items() if k != "Shopping"}
    with pytest.raises(ValueError, match="Shopping"):
        arm_shares(post, spec, "Search", _SPEND_REF, breakout_name_map=incomplete)


def test_bad_inputs_raise():
    post, spec = _fake_posterior()
    with pytest.raises(ValueError, match="not in parents"):
        arm_shares(post, spec, "Nope", _SPEND_REF, breakout_name_map=_NAME_MAP)
    with pytest.raises(ValueError, match="mode"):
        arm_shares(
            post, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, mode="huh"
        )
    with pytest.raises(ValueError, match="shape"):
        arm_shares(
            post, spec, "Search", np.array([0.8, 0.8]), breakout_name_map=_NAME_MAP
        )
    # an unsplit parent has a single arm -> no shares to export
    with pytest.raises(ValueError, match=">= 2"):
        arm_shares(post, spec, "Display", _SPEND_REF, breakout_name_map=_NAME_MAP)


def test_zero_out_vs_main_effect_differ_when_gamma_nonzero():
    # With sibling cannibalization (gamma < 0) the zero-out contrast includes the
    # interaction the main-effect read ignores -> the shares differ. (Kept mild
    # enough that every arm's zero-out response stays positive.)
    post_g, spec = _fake_posterior(gamma=-0.15)
    zo = arm_shares(
        post_g, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, draws=100
    )
    me = arm_shares(
        post_g,
        spec,
        "Search",
        _SPEND_REF,
        breakout_name_map=_NAME_MAP,
        mode="main_effect",
        draws=100,
    )
    assert not np.allclose(zo["shares"], me["shares"], atol=1e-3)

    # With gamma == 0 the zero-out contrast IS the main effect -> they coincide.
    post_0, spec = _fake_posterior(gamma=0.0)
    zo0 = arm_shares(
        post_0, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, draws=100
    )
    me0 = arm_shares(
        post_0,
        spec,
        "Search",
        _SPEND_REF,
        breakout_name_map=_NAME_MAP,
        mode="main_effect",
        draws=100,
    )
    np.testing.assert_allclose(zo0["shares"], me0["shares"], atol=1e-2)


def test_log_ratio_cov_is_positive_definite():
    post, spec = _fake_posterior()
    out = arm_shares(
        post, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, draws=150
    )
    cov = np.asarray(out["log_ratio_cov"])
    assert cov.shape == (2, 2)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(cov).min()) > 0


def _contaminate(post, rows) -> None:
    """Give the draws in ``rows`` strong sibling cannibalization, so their
    zero-out responses go non-positive (the draws have no valid share)."""
    for pair in _PAIRS:
        post.samples[pair_name(post.channels, pair)][np.asarray(rows)] = _CANNIBAL_GAMMA


def test_zero_out_excludes_contaminated_draws_and_warns():
    """A MINORITY of strongly-cannibalizing draws must be EXCLUDED from the
    shares and the ALR covariance, not floored into them: a single floored
    draw contributes z ~= log(eps) ~= -27 against a clean z spread of O(0.5),
    inflating the exported cov diagonal by orders of magnitude (which silently
    down-weights the share evidence in the consuming MMM)."""
    post_clean, spec = _fake_posterior(n=500, seed=3)
    out_clean = arm_shares(
        post_clean, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, draws=500
    )
    assert out_clean["source"]["n_excluded"] == 0

    post_bad, _ = _fake_posterior(n=500, seed=3)
    _contaminate(post_bad, [10, 250, 490])  # 3/500 = 0.6% of draws
    with pytest.warns(UserWarning, match=r"excluded 3/500"):
        out_bad = arm_shares(
            post_bad, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, draws=500
        )
    assert out_bad["source"]["n_excluded"] == 3
    assert out_bad["source"]["n_draws"] == 497

    # The contamination test: dropping 3 draws barely moves the covariance, so
    # the exported diagonal must stay within ~2x of the clean-draws diagonal
    # (the old eps-floor path inflated it ~20-40x).
    d_clean = np.diag(np.asarray(out_clean["log_ratio_cov"]))
    d_bad = np.diag(np.asarray(out_bad["log_ratio_cov"]))
    assert np.all(d_bad <= 2.0 * d_clean), (d_bad, d_clean)
    assert np.all(d_bad >= 0.5 * d_clean), (d_bad, d_clean)
    # ... and the shares barely move either.
    np.testing.assert_allclose(out_bad["shares"], out_clean["shares"], atol=0.02)


def test_zero_out_raises_when_exclusions_exceed_20_percent():
    post, spec = _fake_posterior(n=500, seed=5)
    _contaminate(post, np.arange(150))  # 30% of draws
    with pytest.raises(ValueError, match="ill-defined"):
        arm_shares(
            post, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP, draws=500
        )


def test_zero_out_requires_ten_surviving_draws():
    # 2/11 excluded (18% -- under the 20% gate) leaves 9 survivors < 10.
    post, spec = _fake_posterior(n=11, seed=5)
    _contaminate(post, [0, 6])
    with pytest.raises(ValueError, match="at least 10"):
        with pytest.warns(UserWarning, match="excluded 2/11"):
            arm_shares(post, spec, "Search", _SPEND_REF, breakout_name_map=_NAME_MAP)


def test_main_effect_mode_is_unaffected_by_cannibalizing_gammas():
    """main_effect reads beta_i * act(s_ref)_i only -- strictly positive -- so
    the same contaminated posterior exports without exclusions or warnings."""
    import warnings as _warnings

    post, spec = _fake_posterior(n=500, seed=3)
    _contaminate(post, np.arange(150))
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        out = arm_shares(
            post,
            spec,
            "Search",
            _SPEND_REF,
            breakout_name_map=_NAME_MAP,
            mode="main_effect",
            draws=500,
        )
    assert not [w for w in rec if "arm_shares" in str(w.message)]
    assert out["source"]["n_excluded"] == 0
    assert out["source"]["n_draws"] == 500


def test_shares_are_inverse_alr_of_mean_log_ratios():
    """Location/cov consistency: the exported shares are the softmax of the
    MEAN log-ratios over the surviving draws, so the consumer's observed
    z_hat = ALR(shares) is exactly mean(z) -- the MvNormal location matches
    the covariance's draws (NOT the arithmetic mean of the share draws)."""
    spec = expand_arms(["Search", "Display"], _ARMS)
    k, n = spec.n_arms, 12
    # Two alternating known beta vectors, constant activation, no interactions
    # -> per-draw shares are exactly beta / sum(beta) over the Search arms.
    beta = np.where(
        np.arange(n)[:, None] % 2 == 0,
        np.array([2.0, 1.0, 0.5, 1.0]),
        np.array([1.0, 2.0, 0.5, 1.0]),
    )
    s = {
        "beta": beta,
        "kappa": np.full((n, k), 0.6),
        "alpha": np.full((n, k), 1.5),
    }
    for pair in _PAIRS:
        s[pair_name(spec.channels, pair)] = np.zeros(n)
    post = Posterior(samples=s, channels=spec.channels, pairs=_PAIRS, pair_signs={})

    out = arm_shares(
        post,
        spec,
        "Search",
        _SPEND_REF,
        breakout_name_map=_NAME_MAP,
        mode="main_effect",
    )

    share_draws = beta[:, :3] / beta[:, :3].sum(axis=1, keepdims=True)
    z = np.log(share_draws[:, :-1] / share_draws[:, -1:])
    z_bar = z.mean(axis=0)
    expected = np.exp(np.concatenate([z_bar, [0.0]]))
    expected = expected / expected.sum()
    np.testing.assert_allclose(out["shares"], expected, atol=1e-9)

    # the observed z_hat the MvNormal consumer derives IS mean(z) ...
    sh = np.asarray(out["shares"])
    np.testing.assert_allclose(np.log(sh[:-1] / sh[-1]), z_bar, atol=1e-9)
    # ... and that is NOT the arithmetic mean of the share draws.
    assert not np.allclose(out["shares"], share_draws.mean(axis=0), atol=1e-4)


def test_subsampling_caps_draws_and_is_seedable():
    post, spec = _fake_posterior(n=300)
    out = arm_shares(
        post,
        spec,
        "Search",
        _SPEND_REF,
        breakout_name_map=_NAME_MAP,
        draws=50,
        rng=7,
    )
    assert out["source"]["n_draws"] == 50
    assert float(np.sum(out["shares"])) == pytest.approx(1.0, abs=1e-9)
    # same seed -> identical subsample -> identical payload
    out2 = arm_shares(
        post,
        spec,
        "Search",
        _SPEND_REF,
        breakout_name_map=_NAME_MAP,
        draws=50,
        rng=7,
    )
    assert out2["shares"] == out["shares"]
    assert out2["log_ratio_cov"] == out["log_ratio_cov"]
