"""The numpy saturation must reproduce the graph's, guards included.

Epic #180 builds a design matrix out of the PyTensor graph. That is only sound if
``frequentist._transforms.saturate`` computes exactly what
``model.base._apply_saturation_pt`` computes — otherwise ``X @ theta`` is not the
model's ``mu`` and every comparison between a frequentist and a Bayesian fit is
measuring the gap between two saturation definitions rather than between two
estimators.

The two drifts these tests exist to prevent were both real in the public numpy
helpers before this module existed:

* ``transforms.saturation.logistic_saturation`` clips ``x`` where the graph clips
  the *exponent*, so the two diverge by ~2e-9 once ``lam * x > 20``;
* ``transforms.saturation.root_saturation`` returns ``0`` at a zero-spend row
  where the graph returns ``1e-9 ** k`` — 3.16e-05 for ``k = 0.5``.

Both are far above the 1e-12 tolerance the design-matrix equivalence test needs.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest

from mmm_framework.config.enums import SaturationType
from mmm_framework.frequentist._transforms import SATURATION_PARAMS, saturate
from mmm_framework.model.base import _apply_saturation_pt

# Includes exact zero (flighted weeks), a subnormal-ish value, and the region
# beyond the logistic exponent clip.
X = np.array([0.0, 1e-12, 1e-6, 0.25, 0.5, 1.0, 2.0, 5.0])

CASES = [
    (SaturationType.LOGISTIC, {"sat_lam": 0.5}),
    (SaturationType.LOGISTIC, {"sat_lam": 2.7}),
    # lam * x exceeds 20 on the tail of X -- where the graph's exponent clip bites
    # and the public numpy twin does not.
    (SaturationType.LOGISTIC, {"sat_lam": 25.0}),
    (SaturationType.LOGISTIC, {"sat_lam": 60.0}),
    (SaturationType.HILL, {"sat_half": 0.4, "sat_slope": 1.5}),
    (SaturationType.HILL, {"sat_half": 0.9, "sat_slope": 0.6}),
    (SaturationType.MICHAELIS_MENTEN, {"sat_half": 0.4}),
    (SaturationType.TANH, {"sat_half": 0.4}),
    (SaturationType.ROOT, {"sat_exponent": 0.5}),
    (SaturationType.ROOT, {"sat_exponent": 0.8}),
    (SaturationType.NONE, {}),
]


def _graph(kind: SaturationType, params: dict) -> np.ndarray:
    return _apply_saturation_pt(pt.constant(X), kind, params).eval()


@pytest.mark.parametrize(
    ("kind", "params"), CASES, ids=lambda v: getattr(v, "value", None)
)
def test_matches_the_graph_exactly(kind, params):
    """Bit-for-bit agreement, not merely 'close enough'."""
    np.testing.assert_allclose(
        saturate(X, kind, **params), _graph(kind, params), rtol=0, atol=0
    )


def test_every_saturation_type_is_covered():
    """A new SaturationType must not silently fall through to identity.

    This is exactly how ``PosteriorForecaster._saturate`` shipped a bug: it
    handled four families and returned its input unchanged for ``ROOT``, so a
    root-saturation model was backtested with no saturation at all.
    """
    assert set(SATURATION_PARAMS) == set(SaturationType)
    covered = {kind for kind, _ in CASES}
    assert covered == set(SaturationType)


def test_unknown_kind_raises_rather_than_returning_the_input():
    with pytest.raises(ValueError, match="Unknown saturation type"):
        saturate(X, "not_a_saturation_type")  # type: ignore[arg-type]


def test_root_does_not_collapse_to_zero_at_zero_spend():
    """The graph's ``maximum(x, 1e-9)`` clamp is load-bearing, not cosmetic."""
    out = saturate(np.array([0.0]), SaturationType.ROOT, sat_exponent=0.5)
    assert out[0] == pytest.approx(1e-9**0.5)
    assert out[0] != 0.0


def test_logistic_saturates_at_the_exponent_clip():
    """Beyond ``lam * x = 20`` the graph stops moving; the mirror must too."""
    big = saturate(np.array([1.0, 10.0]), SaturationType.LOGISTIC, sat_lam=60.0)
    assert big[0] == pytest.approx(1.0 - np.exp(-20))
    assert big[0] == big[1]


def test_broadcasts_per_draw_parameters():
    """PosteriorForecaster evaluates (n_obs, n_draws) with per-draw parameters."""
    x2d = X[:, None] * np.ones((1, 4))
    lam = np.array([0.5, 1.0, 2.0, 4.0])
    got = saturate(x2d, SaturationType.LOGISTIC, sat_lam=lam[None, :])
    assert got.shape == (len(X), 4)
    for j, single in enumerate(lam):
        np.testing.assert_allclose(
            got[:, j], saturate(X, SaturationType.LOGISTIC, sat_lam=single)
        )
