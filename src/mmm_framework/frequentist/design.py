"""Out-of-graph design matrix for fixed (adstock, saturation).

Hold each channel's adstock and saturation parameters fixed and the core MMM is
**linear in everything else**: the transformed media become constant columns, and
the model's mean is exactly ``X @ theta``. That is what makes a closed-form ridge
solve (#185) and a convex QP (#187) possible, and it is verified rather than
assumed — ``tests/frequentist/test_design_equivalence.py`` builds the PyTensor
graph, evaluates its mean at a fixed parameter point, and asserts agreement to
1e-12 across every supported transform, trend and panel shape.

The design here is deliberately *narrow and loud*. The core model has grown a
number of terms that are **not** linear given fixed transforms — a GP trend whose
basis depends on an estimated lengthscale, exponentiated group coefficients,
time-varying coefficients, a frequency gain applied inside the media input. Each
of those raises :class:`NotImplementedError` naming the feature, because the
alternative — dropping the term and fitting the rest — would return a number that
looks like the model's answer and is not.

See ``technical-docs/frequentist-estimation.md`` for the full derivation,
including why media columns are scaled into ROI space and why the geo dummies are
identified only by the penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.enums import LikelihoodFamily, ModelSpecification
from ..model.base import _ADSTOCK_KIND
from ._transforms import SATURATION_PARAMS, adstock_panel, saturate

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from ..data_loader import PanelDataset
    from ..model.base import BayesianMMM

__all__ = ["DesignMatrix", "build_design_matrix", "UnsupportedModelError"]


class UnsupportedModelError(NotImplementedError):
    """A model term is not linear given fixed transforms.

    Raised instead of silently dropping the term. Carries ``feature`` so callers
    (the transform search, the agent) can report *which* configuration blocked the
    frequentist path rather than a generic refusal.
    """

    def __init__(self, feature: str, reason: str):
        self.feature = feature
        self.reason = reason
        super().__init__(
            f"{feature} is not supported by the frequentist estimation path: "
            f"{reason}. Fit this model with the Bayesian path instead "
            "(see technical-docs/frequentist-estimation.md)."
        )


@dataclass(frozen=True)
class DesignMatrix:
    """A linear design for a fixed (adstock, saturation) point.

    Attributes:
        X: ``(n_obs, k)`` design. ``X @ theta`` reproduces the graph's ``mu`` on
            the standardized outcome scale.
        y: The standardized outcome the graph's likelihood sees.
        columns: Column names, ``k`` of them.
        blocks: Block name (``"intercept"``, ``"trend"``, ``"seasonality"``,
            ``"geo"``, ``"product"``, ``"media"``, ``"controls"``, ``"events"``)
            to the column slice it occupies.
        penalize: ``(k,)`` boolean. ``True`` where the ridge penalty applies.
            The intercept is never penalized; trend and seasonality basis
            coefficients are structural and unpenalized by default; media,
            controls, geo and product are penalized. Geo/product identification
            *depends* on this — see the module docs.
        param_map: Column index to ``(trace_variable_name, position)``, so a
            solved ``theta`` can be scattered back into the graph's variable names
            when assembling an ``InferenceData``.
        roi_scale: Channel to the constant ``c_c`` with
            ``beta_<ch> = c_c * theta_<ch>``. ``1.0`` unless the channel takes the
            ROI parameterization, in which case ``theta`` *is* the channel's ROI.
        scaling: The standardization used (``y_mean``, ``y_std``, ``media_max``,
            ``control_mean``, ``control_std``), matching ``_prepare_data`` so the
            penalty is defined against a pinned scale.
    """

    X: "NDArray[np.floating]"
    y: "NDArray[np.floating]"
    columns: list[str]
    blocks: dict[str, slice]
    penalize: "NDArray[np.bool_]"
    param_map: dict[int, tuple[str, int]]
    roi_scale: dict[str, float] = field(default_factory=dict)
    scaling: dict[str, Any] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_params(self) -> int:
        return int(self.X.shape[1])


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def _reject_unsupported(mmm: "BayesianMMM") -> None:
    """Refuse every configuration that is not linear given fixed transforms.

    Ordered roughly by how likely a user is to hit it, so the first message they
    see names the thing they actually configured.
    """
    trend_type = str(getattr(mmm.trend_config.type, "value", mmm.trend_config.type))
    if trend_type == "gaussian_process":
        raise UnsupportedModelError(
            "Gaussian-process trend",
            "the HSGP basis weights depend on the estimated lengthscale and "
            "amplitude, so the trend is not linear in its coefficients once "
            "adstock and saturation are fixed",
        )

    if not getattr(mmm, "use_parametric_adstock", True):
        raise UnsupportedModelError(
            "Legacy (non-parametric) adstock",
            "it blends two precomputed fixed-alpha series through an estimated "
            "Beta mixture and normalizes by the adstocked max rather than the "
            "raw max — a different model family",
        )

    spec = getattr(mmm.model_config, "specification", ModelSpecification.ADDITIVE)
    if spec == ModelSpecification.MULTIPLICATIVE:
        raise UnsupportedModelError(
            "Multiplicative specification",
            "it is linear on standardized log(y), but the back-transform makes "
            "intervals asymmetric and predict() semantics diverge; deferred",
        )

    family = getattr(mmm._likelihood_config, "family", LikelihoodFamily.NORMAL)
    if family != LikelihoodFamily.NORMAL:
        raise UnsupportedModelError(
            f"Likelihood family {getattr(family, 'value', family)!r}",
            "least squares is the Gaussian-likelihood estimator; other families "
            "are not a penalized linear solve",
        )

    if mmm._selection_active():
        raise UnsupportedModelError(
            "Control selection (horseshoe / spike-slab / LASSO)",
            "those are adaptive or L1 penalties, not the L2 the closed-form "
            "ridge solves; the CVXPY path (#187) is where LASSO belongs",
        )

    if getattr(mmm.hierarchical_config, "vary_media_by_geo", False) and mmm.has_geo:
        raise UnsupportedModelError(
            "Per-geo media coefficients (vary_media_by_geo)",
            "the per-geo betas are exponentiated and their pooling strength is "
            "itself estimated, so a ridge dummy expansion would fix by penalty "
            "what the model estimates",
        )

    if getattr(mmm, "_reach_freq", None):
        raise UnsupportedModelError(
            "Reach/frequency channels",
            "the frequency gain multiplies the media column before adstock and "
            "carries its own shape parameters, so fixing adstock and saturation "
            "is not enough to make the term linear",
        )

    if getattr(mmm, "experiments", None):
        raise UnsupportedModelError(
            "Experiment calibration",
            "the estimands are linear in beta, but their 1/se^2 weighting has no "
            "defensible scale against the ridge penalty. Refusing rather than "
            "dropping them, because dropping discards randomized evidence the "
            "Bayesian fit uses",
        )

    for ch in mmm.channel_names:
        cfg = mmm.mff_config.get_media_config(ch)
        if getattr(cfg, "time_varying", False):
            raise UnsupportedModelError(
                f"Time-varying coefficient on {ch!r}",
                "the random-walk prior is what identifies beta_t; plain ridge "
                "would not",
            )
        if getattr(cfg, "parent_channel", None):
            raise UnsupportedModelError(
                f"Grouped media prior on {ch!r}",
                "the group coefficient is exponentiated, so it is not linear",
            )

    if getattr(mmm, "_price_lever", None) is not None or getattr(
        mmm, "_promo_levers", None
    ):
        raise UnsupportedModelError(
            "Price / promotion levers",
            "supported in principle (log-price is a linear column) but not yet "
            "wired; the promo carryover parameter also leaves the linear family "
            "when adstock_lmax > 1",
        )

    if getattr(mmm, "event_features", np.empty((0, 0))).shape[1] > 0:
        raise UnsupportedModelError(
            "Event / holiday effects",
            "linear in principle but not yet wired into the design matrix",
        )

    if getattr(mmm.model_config, "channel_interactions", None):
        raise UnsupportedModelError(
            "Cross-channel interactions",
            "linear in principle (a product of two constant columns) but not yet "
            "wired into the design matrix",
        )


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #


def build_design_matrix(
    panel: "PanelDataset",
    alpha: dict[str, dict[str, float]],
    lam: dict[str, dict[str, float]],
    *,
    model_config: Any,
    trend_config: Any,
) -> DesignMatrix:
    """Build the linear design implied by a fixed (adstock, saturation) point.

    Args:
        panel: The panel the model would be fitted on.
        alpha: Per-channel adstock parameters, e.g.
            ``{"TV": {"alpha": 0.6}}`` for geometric,
            ``{"TV": {"alpha": 0.6, "theta": 2.0}}`` for delayed, or
            ``{"TV": {"shape": 2.0, "scale": 3.0}}`` for Weibull. A channel whose
            adstock type is ``none`` takes an empty dict.
        lam: Per-channel saturation parameters keyed as the graph names them —
            ``sat_lam`` (logistic), ``sat_half``/``sat_slope`` (hill),
            ``sat_half`` (michaelis_menten, tanh), ``sat_exponent`` (root).
        model_config: The :class:`ModelConfig` the Bayesian path would use. Read
            for the likelihood family, media prior mode and seasonality.
        trend_config: The :class:`TrendConfig` the Bayesian path would use.

    Returns:
        The :class:`DesignMatrix`, whose ``X @ theta`` reproduces the graph's
        ``mu`` for the corresponding parameter point.

    Raises:
        UnsupportedModelError: If any configured term is not linear given fixed
            transforms. The message names the feature.
        KeyError: If a channel is missing a parameter its family requires.

    Note:
        The model object is constructed but its PyMC graph is **never built** —
        ``_prepare_data`` runs during ``__init__`` and ``.model`` is lazy, so this
        is cheap and touches no PyTensor.
    """
    from ..model.base import BayesianMMM

    mmm = BayesianMMM(panel, model_config, trend_config)
    _reject_unsupported(mmm)

    n = mmm.n_obs
    ti = mmm.time_idx
    cols: list["NDArray[np.floating]"] = []
    names: list[str] = []
    penal: list[bool] = []
    blocks: dict[str, slice] = {}
    pmap: dict[int, tuple[str, int]] = {}

    def _add(block: str, col, name: str, penalized: bool, param: str, pos: int):
        pmap[len(cols)] = (param, pos)
        cols.append(np.asarray(col, dtype=float))
        names.append(name)
        penal.append(penalized)

    def _close(block: str, start: int) -> None:
        if len(cols) > start:
            blocks[block] = slice(start, len(cols))

    # -- intercept: never penalized, and it is what identifies the geo dummies --
    start = len(cols)
    _add("intercept", np.ones(n), "intercept", False, "intercept", 0)
    _close("intercept", start)

    # -- trend --------------------------------------------------------------
    start = len(cols)
    trend_type = str(getattr(trend_config.type, "value", trend_config.type))
    t_scaled = mmm.t_scaled
    if trend_type == "linear":
        _add("trend", t_scaled[ti], "trend_slope", False, "trend_slope", 0)
    elif trend_type == "piecewise":
        s = np.asarray(mmm.trend_features["changepoints"], dtype=float)
        A = np.asarray(mmm.trend_features["changepoint_matrix"], dtype=float)
        t_unique = np.linspace(0, 1, mmm.n_periods)
        _add("trend", t_unique[ti], "trend_k", False, "trend_k", 0)
        # `trend_m` is a second constant column, exactly collinear with the
        # intercept. Ridge would identify the split arbitrarily, so it is folded
        # into the (unpenalized) intercept and reported as 0.
        for j in range(len(s)):
            _add(
                "trend",
                (A[:, j] * (t_unique - s[j]))[ti],
                f"trend_delta[{j}]",
                False,
                "trend_delta",
                j,
            )
    elif trend_type == "spline":
        basis = np.asarray(mmm.trend_features["spline_basis"], dtype=float)
        # The graph demeans the trend OUTPUT; demeaning the basis COLUMNS is
        # algebraically identical and is the correct design form. It also breaks
        # the exact collinearity with the intercept that a partition-of-unity
        # B-spline basis would otherwise introduce.
        centred = basis - basis.mean(axis=0, keepdims=True)
        for j in range(centred.shape[1]):
            _add(
                "trend", centred[:, j][ti], f"spline_coef[{j}]", False, "spline_coef", j
            )
    _close("trend", start)

    # -- seasonality --------------------------------------------------------
    start = len(cols)
    for sname, feats in mmm.seasonality_features.items():
        F = np.asarray(feats, dtype=float)[ti]
        for j in range(F.shape[1]):
            _add(
                "seasonality",
                F[:, j],
                f"season_{sname}[{j}]",
                False,
                f"season_{sname}",
                j,
            )
    _close("seasonality", start)

    # -- geo / product ------------------------------------------------------
    # Penalized dummies against an unpenalized intercept: that is what makes the
    # split unique (the graph has no per-geo intercept) and is the ridge analogue
    # of the hierarchy's shrinkage.
    for level, active, idx, count in (
        (
            "geo",
            mmm.has_geo and mmm.hierarchical_config.pool_across_geo,
            mmm.geo_idx,
            mmm.n_geos,
        ),
        (
            "product",
            mmm.has_product and mmm.hierarchical_config.pool_across_product,
            mmm.product_idx,
            mmm.n_products,
        ),
    ):
        start = len(cols)
        if active:
            for g in range(count):
                _add(
                    level,
                    (idx == g).astype(float),
                    f"{level}_effect[{g}]",
                    True,
                    f"{level}_effect",
                    g,
                )
        _close(level, start)

    # -- media --------------------------------------------------------------
    start = len(cols)
    X_norm = mmm._prepare_raw_media_for_model()
    roi_scale: dict[str, float] = {}
    for c, ch in enumerate(mmm.channel_names):
        acfg = mmm._get_adstock_config(ch)
        kind = _ADSTOCK_KIND.get(acfg.type, "geometric")
        x_ad = adstock_panel(
            X_norm[:, c],
            kind,
            acfg.l_max,
            time_idx=ti,
            cell_idx=mmm.cell_idx,
            n_cells=mmm.n_cells,
            normalize=acfg.normalize,
            **alpha.get(ch, {}),
        )
        sat_kind = mmm._get_saturation_config(ch).type
        sat_params = lam.get(ch, {})
        missing = set(SATURATION_PARAMS[sat_kind]) - set(sat_params)
        if missing:
            raise KeyError(
                f"Channel {ch!r} has {sat_kind.value} saturation and needs "
                f"{sorted(missing)} in `lam`."
            )
        col = saturate(x_ad, sat_kind, **sat_params)

        # ROI parameterization: beta_c = c_c * roi_c with c_c known once the
        # transforms are fixed. Scaling the column makes the solved coefficient
        # the channel's ROI, so one penalty means the same thing across channels
        # regardless of their spend.
        c_c = 1.0
        divisor = mmm._roi_mode_divisor(ch, c)
        roi_mode = getattr(model_config, "media_prior_mode", "coefficient") == "roi"
        if roi_mode and divisor is not None and kind != "none":
            c_c = float(divisor / (mmm.y_std * col.sum() + 1e-9))
        roi_scale[ch] = c_c
        _add("media", col * c_c, f"media_{ch}", True, f"beta_{ch}", c)
    _close("media", start)

    # -- controls -----------------------------------------------------------
    start = len(cols)
    if mmm.n_controls > 0:
        for c, cn in enumerate(mmm.control_names):
            _add(
                "controls",
                mmm.X_controls[:, c],
                f"control_{cn}",
                True,
                "beta_controls",
                c,
            )
    _close("controls", start)

    return DesignMatrix(
        X=np.column_stack(cols),
        y=np.asarray(mmm.y, dtype=float),
        columns=names,
        blocks=blocks,
        penalize=np.asarray(penal, dtype=bool),
        param_map=pmap,
        roi_scale=roi_scale,
        scaling={
            "y_mean": mmm.y_mean,
            "y_std": mmm.y_std,
            "media_max": dict(mmm._media_raw_max),
            "control_mean": getattr(mmm, "control_mean", None),
            "control_std": getattr(mmm, "control_std", None),
        },
    )
