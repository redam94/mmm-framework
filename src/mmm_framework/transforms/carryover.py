"""The posterior carryover kernel, read once and correctly (#218).

Five places in this codebase derived "how long does a channel keep working?" and
they did not agree. ``reporting/helpers/adstock.py`` was the worst of them, and
everything it fed was wrong:

* **Family-blind.** It built ``alpha_mean ** lags`` unconditionally. On a
  *delayed* channel with a fitted ``theta`` of 3.1 it reported the peak at lag 0;
  the true peak is lag 3. ``theta`` appeared nowhere in the file.
* **Weibull and no-adstock channels vanished.** They register
  ``adstock_shape_``/``adstock_scale_`` or no RV at all, matched none of the
  four name prefixes it probed, and were dropped with a log line — so a channel
  simply disappeared from the carryover table.
* **A legacy fit's Beta MIX WEIGHT was reported as a decay rate.** A mix of
  0.070 was rendered as ``alpha = 0.070`` (half-life 0.26 weeks) when the actual
  blended kernel is ``[0.752, 0.048, 0.043, ...]`` — 24.8% carryover.
* **It collapsed the posterior before a convex transform.** ``mean(alpha)`` then
  ``**lags`` is not ``mean(alpha ** lags)``: measured on a real posterior, the
  lag-5 weight was understated **7x** and the mean half-life by 41%.
* **``l_max`` was always 8**, because the lookup probed a field name the panel
  does not have. A channel declared with ``l_max=26`` was silently truncated.

This module is the single reader. It **delegates** to
:func:`mmm_framework.transforms.adstock.adstock_weights` — the same function the
model graph uses — rather than reimplementing a kernel for the sixth time, so
"agrees with the model" holds by construction rather than by test.

It lives in ``transforms/`` and not ``reporting/helpers/`` deliberately:
``adstock_weights`` is already a dependency of ``model/``, ``validation/``,
``planning/`` and ``reporting/``, and putting the reader in the plotting layer
would make ``model/base.py`` import upward into it.

**Half-life is truncation-biased and says so.** A kernel truncated at ``l_max``
cannot express mass beyond it: geometric ``alpha=0.9`` gives 2.19 at ``l_max=8``
against 5.59 at ``l_max=200``. Every result therefore carries
``truncated_tail_mass``, and any surface rendering a horizon is expected to
render that beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .adstock import adstock_weights

__all__ = [
    "CarryoverKernel",
    "carryover_half_life",
    "posterior_carryover_kernels",
]

#: Parameter names each family reads out of the trace, in `adstock_<name>_<ch>`.
_FAMILY_PARAMS: dict[str, tuple[str, ...]] = {
    "geometric": ("alpha",),
    "delayed": ("alpha", "theta"),
    "weibull": ("shape", "scale"),
    "none": (),
}


@dataclass
class CarryoverKernel:
    """A channel's posterior carryover kernel, per draw.

    Attributes
    ----------
    kernel : np.ndarray
        ``(n_draws, l_max)`` normalized-or-not weights, lag 0 first. Per draw —
        never a kernel built from a collapsed parameter, which is a different
        and wrong quantity for any transform convex in its parameters.
    status : {'ok', 'legacy_blend', 'unsupported', 'missing_params'}
        ``legacy_blend`` reconstructs the two-alpha mixture at a stated
        truncation (legacy adstock is IIR, so no exact ``l_max`` exists).
        ``unsupported``/``missing_params`` mean no kernel could be read —
        reported rather than silently dropping the channel.
    truncated_tail_mass : float
        Fraction of an *untruncated* kernel's mass falling beyond ``l_max``.
        Load-bearing: a half-life read off a truncated kernel understates the
        true one, and this is the size of that understatement.
    """

    channel: str
    family: str
    l_max: int
    normalize: bool
    kernel: np.ndarray
    status: str = "ok"
    truncated_tail_mass: float = 0.0
    note: str = ""
    #: Only meaningful for geometric/delayed. ``None`` for weibull (which has no
    #: alpha), for ``none``, and for legacy blends — where the stored quantity is
    #: a MIXTURE WEIGHT, not a decay rate, and surfacing it as one was the bug.
    alpha_mean: float | None = None
    lags: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lags:
            self.lags = list(range(self.kernel.shape[-1]))

    @property
    def mean_kernel(self) -> np.ndarray:
        """Posterior mean of the kernel, per lag (mean OF the transform)."""
        return np.asarray(self.kernel, dtype=float).mean(axis=0)

    def half_life(self) -> np.ndarray:
        """Per-draw half-life. See :func:`carryover_half_life`."""
        return carryover_half_life(self.kernel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "family": self.family,
            "l_max": int(self.l_max),
            "normalize": bool(self.normalize),
            "status": self.status,
            "truncated_tail_mass": float(self.truncated_tail_mass),
            "alpha_mean": self.alpha_mean,
            "lags": list(self.lags),
            "mean_kernel": [float(v) for v in self.mean_kernel],
            "note": self.note,
        }


def carryover_half_life(kernel: np.ndarray) -> np.ndarray:
    """Lag at which the cumulative kernel first reaches half its total, per draw.

    Linearly interpolated between lags, so the result is continuous in the
    parameters rather than jumping by whole lags.

    This is a *cumulative-50%* definition. It is NOT the same quantity as
    ``log(0.5)/log(alpha)`` (a geometric decay constant), nor as "first lag whose
    weight is under half of lag 0", nor as "number of lags with weight >= 1%".
    Those are four different functions of a kernel — at geometric ``alpha=0.9``
    they give 6.58, 2.19, 7 and 8 — and this codebase shipped all four under the
    single word "half-life". This one is canonical because it is the only one
    that means the same thing for a humped (delayed/Weibull) kernel as for a
    monotone one.

    **Truncation-biased**: computed on a kernel cut at ``l_max``, it understates
    the untruncated half-life. Read it beside ``truncated_tail_mass``.
    """
    k = np.atleast_2d(np.asarray(kernel, dtype=float))
    out = np.empty(k.shape[0])
    for i, row in enumerate(k):
        total = row.sum()
        if not np.isfinite(total) or total <= 0:
            out[i] = np.nan
            continue
        cum = np.cumsum(row) / total
        # Lag k occupies the half-open interval [k, k+1), so cum[k] is the share
        # landed by TIME k+1. One branch for every j, which matters: the shipped
        # implementations special-cased j == 0 with a different origin, making
        # the function DISCONTINUOUS at cum[0] == 0.5 and non-monotone in alpha
        # (geometric 0.5 -> 1.00 but 0.7 -> 0.82, i.e. more carryover reported as
        # a SHORTER half-life). This form is continuous and monotone.
        j = int(np.searchsorted(cum, 0.5))
        if j >= len(cum):  # pragma: no cover - cum ends at 1.0
            out[i] = float(len(cum))
            continue
        prev = cum[j - 1] if j > 0 else 0.0
        span = cum[j] - prev
        out[i] = j + ((0.5 - prev) / span if span > 0 else 0.0)
    return out


def _tail_mass(family: str, l_max: int, params: dict[str, float]) -> float:
    """Mass of the untruncated kernel beyond ``l_max``.

    Computed by evaluating the same family far out and comparing, so it stays
    correct if a family's shape changes.
    """
    far = max(int(l_max) * 8, int(l_max) + 40)
    try:
        long = adstock_weights(family, far, normalize=False, **params)
    except Exception:  # pragma: no cover - defensive
        return 0.0
    total = float(np.sum(long))
    if total <= 0:
        return 0.0
    return float(np.sum(long[l_max:]) / total)


def posterior_carryover_kernels(
    model: Any,
    channels: list[str] | None = None,
    *,
    max_draws: int = 500,
) -> dict[str, CarryoverKernel]:
    """Per-draw carryover kernels for a fitted model, one per channel.

    Delegates to :func:`~mmm_framework.transforms.adstock.adstock_weights`, the
    function the model graph itself uses, so the reported kernel cannot drift
    from the fitted one.

    Parameters
    ----------
    model : BayesianMMM
        A fitted model.
    channels : list of str, optional
        Defaults to every channel.
    max_draws : int
        Thin the posterior to at most this many draws. Kernel construction is a
        Python loop over draws, so this bounds the cost.

    Returns
    -------
    dict[str, CarryoverKernel]
        **Every requested channel is present**, including ones whose kernel
        could not be read — with a ``status`` saying why. The previous reader
        dropped those channels entirely, so a Weibull channel simply vanished
        from the carryover table with only a log line.
    """
    from ..model.base import _ADSTOCK_KIND

    trace = getattr(model, "_trace", None)
    if trace is None:
        raise ValueError("Model not fitted. Call fit() first.")
    posterior = trace.posterior
    # A posterior with no chain/draw dims means nothing sampled — possible when
    # every channel is NONE-adstock, or on a hand-built trace. Degrade to a
    # single nominal draw rather than KeyError: the families that need no
    # parameters (none, and the missing-params branches) still have an answer.
    sizes = getattr(posterior, "sizes", {})
    n_draws = int(sizes.get("chain", 1) * sizes.get("draw", 1)) or 1
    take = np.arange(n_draws)
    if n_draws > max_draws:
        take = np.linspace(0, n_draws - 1, max_draws).astype(int)

    def get(name: str) -> np.ndarray | None:
        if name not in posterior:
            return None
        arr = posterior[name].values
        return arr.reshape(n_draws, *arr.shape[2:])[take]

    names = list(channels or getattr(model, "channel_names", []) or [])
    out: dict[str, CarryoverKernel] = {}

    legacy = not getattr(model, "use_parametric_adstock", True)
    for ch in names:
        if legacy:
            out[ch] = _legacy_kernel(model, ch, get, len(take))
            continue
        try:
            cfg = model._get_adstock_config(ch)
            family = _ADSTOCK_KIND.get(cfg.type, "geometric")
            l_max = int(cfg.l_max)
            normalize = bool(cfg.normalize)
        except Exception as exc:
            out[ch] = CarryoverKernel(
                channel=ch,
                family="unknown",
                l_max=1,
                normalize=False,
                kernel=np.ones((len(take), 1)),
                status="unsupported",
                note=f"adstock config unreadable: {type(exc).__name__}: {exc}",
            )
            continue

        if family == "none":
            # A unit impulse, NOT an absent channel. The old reader dropped it.
            out[ch] = CarryoverKernel(
                channel=ch,
                family="none",
                l_max=1,
                normalize=normalize,
                kernel=np.ones((len(take), 1)),
                status="ok",
                note="no adstock configured; all effect lands in the same period",
            )
            continue

        want = _FAMILY_PARAMS.get(family, ("alpha",))
        params = {p: get(f"adstock_{p}_{ch}") for p in want}
        if any(v is None for v in params.values()):
            missing = [p for p, v in params.items() if v is None]
            out[ch] = CarryoverKernel(
                channel=ch,
                family=family,
                l_max=l_max,
                normalize=normalize,
                kernel=np.full((len(take), l_max), np.nan),
                status="missing_params",
                note=f"trace has no adstock_{missing[0]}_{ch}",
            )
            continue

        kern = np.empty((len(take), l_max))
        for d in range(len(take)):
            kw = {p: float(v[d]) for p, v in params.items()}
            kern[d] = adstock_weights(family, l_max, normalize=normalize, **kw)

        mean_kw = {p: float(np.mean(v)) for p, v in params.items()}
        out[ch] = CarryoverKernel(
            channel=ch,
            family=family,
            l_max=l_max,
            normalize=normalize,
            kernel=kern,
            status="ok",
            truncated_tail_mass=_tail_mass(family, l_max, mean_kw),
            alpha_mean=mean_kw.get("alpha"),
        )
    return out


def _legacy_kernel(model: Any, ch: str, get, n_take: int) -> CarryoverKernel:
    """The legacy fixed-alpha BLEND, reconstructed per draw.

    ``adstock_<ch>`` is a Beta **mixture weight** between two fixed alphas, not a
    decay rate. The previous reader matched that name with its ``"adstock_"``
    prefix and reported the mixture weight as ``alpha`` — so a mix of 0.070 was
    rendered as a 0.26-week half-life against a true 24.8% carryover.

    Legacy adstock is IIR (applied recursively), so there is no exact ``l_max``;
    it is truncated at a stated constant and the omitted mass is reported.
    """
    _TRUNC = 26
    mix = get(f"adstock_{ch}")
    alphas = list(getattr(model, "adstock_alphas", []) or [])
    if mix is None or len(alphas) < 2:
        return CarryoverKernel(
            channel=ch,
            family="legacy_blend",
            l_max=_TRUNC,
            normalize=True,
            kernel=np.full((n_take, _TRUNC), np.nan),
            status="missing_params",
            note=f"legacy fit has no adstock_{ch} mixture weight",
        )
    a_lo, a_hi = float(alphas[0]), float(alphas[-1])
    lags = np.arange(_TRUNC, dtype=float)
    lo = a_lo**lags
    hi = a_hi**lags
    m = np.asarray(mix, dtype=float).reshape(-1, 1)
    kern = (1.0 - m) * lo[None, :] + m * hi[None, :]
    kern = kern / kern.sum(axis=1, keepdims=True)
    tail = float(a_hi**_TRUNC)  # geometric remainder of the slower component
    return CarryoverKernel(
        channel=ch,
        family="legacy_blend",
        l_max=_TRUNC,
        normalize=True,
        kernel=kern,
        status="legacy_blend",
        truncated_tail_mass=tail,
        alpha_mean=None,  # a mixture weight is NOT a decay rate
        note=(
            f"legacy two-alpha blend ({a_lo:g}/{a_hi:g}) truncated at {_TRUNC} "
            "lags; legacy adstock is IIR so no exact l_max exists"
        ),
    )
