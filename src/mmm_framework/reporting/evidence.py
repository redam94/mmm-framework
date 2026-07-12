"""Evidence tiers + identifiability gate — one visual language for trust.

A posterior credible interval tells you what to believe *after* seeing the data;
it does not tell you whether that belief came from the **data** or the **prior**,
nor whether a per-channel effect is even *separable* from the others. A skeptical
statistician will (rightly) discount every reported number until those two things
are explicit. This module makes them explicit and — crucially — makes every report
say it the **same way**.

Two annotations ride on every client-facing channel ROI / estimand:

* an **evidence tier** — where the number's credibility comes from:

  - ``experiment-validated`` : the channel's effect was calibrated against a
    randomized experiment folded into the fit — the one true causal anchor.
  - ``model-identified``     : the data moved the channel's parameter off its
    prior *and* the channel is separately identifiable. A genuine model finding,
    not yet experimentally confirmed.
  - ``prior-dominated``      : the posterior barely moved off the prior (thin
    data, a sign-constrained prior, or collinearity). The number is mostly the
    analyst's assumption wearing a model — a placeholder, not a finding.

* an **identifiability flag** — whether the channel can be *separated* from the
  others at all. Two always-on collinear channels (Search & Shopping) can have a
  solid *total* contribution yet fragile *individual* effects; when the flag
  trips, the honest statement is "Search and Shopping cannot be separately
  identified in this data", not two confident, invented numbers.

The tier is derived from experiment coverage + the prior→posterior learning
diagnostics (:mod:`mmm_framework.diagnostics.learning`); the flag is derived from
per-channel collinearity (variance inflation over the channels' contributions).

This module is the **single source of truth** so the classic report, the augur
readout, and the interactive report all render one chip. It is pure-Python and
unit-testable without a fit: callers pass in the pieces (experiment channels, a
learning frame, a collinearity map) and get back a :class:`ChannelEvidence` per
channel plus HTML / label helpers.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "EvidenceTier",
    "ChannelEvidence",
    "DEFAULT_CONTRACTION_MIN",
    "DEFAULT_VIF_MAX",
    "channel_evidence",
    "evidence_for_model",
    "collinearity_from_matrix",
    "evidence_chip_html",
    "evidence_legend_html",
    "EVIDENCE_CHIP_CSS",
    "tier_from_string",
]


class EvidenceTier(str, Enum):
    """Where a reported number's credibility comes from (ordered strongest→weakest)."""

    EXPERIMENT_VALIDATED = "experiment-validated"
    MODEL_IDENTIFIED = "model-identified"
    PRIOR_DOMINATED = "prior-dominated"


# Display metadata per tier — ONE place, reused by every renderer so the visual
# language stays consistent. ``augur`` maps to the existing augur ``tier-chip``
# classes (sage / steel / rust); ``css`` maps to the classic-report chip classes
# defined in ``EVIDENCE_CHIP_CSS``.
_TIER_META: dict[EvidenceTier, dict[str, Any]] = {
    EvidenceTier.EXPERIMENT_VALIDATED: {
        "label": "Experiment-validated",
        "short": "Validated",
        "css": "ev-experiment",
        "augur": "t-scale",  # sage / green
        "order": 0,
        "gloss": (
            "Calibrated against a randomized experiment folded into the fit — "
            "the strongest causal anchor available."
        ),
    },
    EvidenceTier.MODEL_IDENTIFIED: {
        "label": "Model-identified",
        "short": "Modeled",
        "css": "ev-model",
        "augur": "t-hold",  # steel / blue
        "order": 1,
        "gloss": (
            "The data moved this effect off its prior and the channel is "
            "separately identifiable — a genuine model finding, not yet "
            "experimentally confirmed."
        ),
    },
    EvidenceTier.PRIOR_DOMINATED: {
        "label": "Prior-dominated",
        "short": "Prior-driven",
        "css": "ev-prior",
        "augur": "t-reduce",  # rust / red
        "order": 2,
        "gloss": (
            "The posterior barely moved off its prior — this number reflects the "
            "assumed prior more than the data. Treat it as a placeholder until "
            "an experiment or more data confirms it."
        ),
    },
}

# Verdict/contraction gate thresholds (conventions, tunable — not law).
# Below ``DEFAULT_CONTRACTION_MIN`` prior→posterior contraction, the data did not
# meaningfully pin the parameter → prior-dominated.
DEFAULT_CONTRACTION_MIN = 0.10
# Above ``DEFAULT_VIF_MAX`` variance inflation, the channel is not separately
# identifiable from its co-linear partners.
DEFAULT_VIF_MAX = 5.0

# Parameter-name prefixes that carry a channel's effect. A learning-frame row
# whose base name (before any ``[geo]`` index) is ``<prefix><channel>`` is
# attributed to that channel.
_CHANNEL_PARAM_PREFIXES = ("beta_", "roi_", "beta_media_")

# Learning verdicts that mean "the location moved a lot even if the width did
# not" — these are NOT prior-dominated (the evidence dominated the location).
_STRONG_LEARNING_VERDICTS = frozenset({"strong", "moderate", "relocated"})


def tier_from_string(value: str | EvidenceTier) -> EvidenceTier:
    """Coerce a tier value (enum or its string) to :class:`EvidenceTier`."""
    if isinstance(value, EvidenceTier):
        return value
    return EvidenceTier(str(value))


def _finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _human_join(items: Sequence[str]) -> str:
    """Oxford-comma join: ``[a] -> "a"``, ``[a,b] -> "a and b"``,
    ``[a,b,c] -> "a, b, and c"``."""
    items = [str(i) for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


@dataclass
class ChannelEvidence:
    """The trust annotation for one channel's reported effect.

    Attributes
    ----------
    channel:
        Channel name.
    tier:
        The :class:`EvidenceTier`.
    identified:
        ``False`` when the channel is collinear with another and cannot be
        separately identified (its individual effect is fragile even if the
        combined effect is solid).
    collinear_with:
        Channel names this one is collinear with (drives the plain-language
        caveat when ``identified`` is ``False``).
    contraction:
        Worst prior→posterior contraction across the channel's parameter(s), or
        ``None`` when not computed.
    learning_verdict:
        The learning verdict driving the tier ("prior-dominated", "strong", …).
    vif:
        Variance-inflation factor for the channel over the channels' contributions.
    experiment:
        ``True`` when a randomized experiment for this channel was folded into
        the fit.
    """

    channel: str
    tier: EvidenceTier
    identified: bool = True
    collinear_with: list[str] = field(default_factory=list)
    contraction: float | None = None
    learning_verdict: str | None = None
    vif: float | None = None
    experiment: bool = False

    @property
    def gated(self) -> bool:
        """``True`` when the number must be *caveated* rather than shown as a
        naked confident figure — prior-dominated, or not separately identifiable.
        (Experiment-validated numbers are never gated by collinearity: the
        experiment measures the effect directly.)"""
        if self.tier is EvidenceTier.EXPERIMENT_VALIDATED:
            return False
        return self.tier is EvidenceTier.PRIOR_DOMINATED or not self.identified

    @property
    def label(self) -> str:
        return str(_TIER_META[self.tier]["label"])

    @property
    def short_label(self) -> str:
        return str(_TIER_META[self.tier]["short"])

    @property
    def gloss(self) -> str:
        return str(_TIER_META[self.tier]["gloss"])

    @property
    def css_class(self) -> str:
        return str(_TIER_META[self.tier]["css"])

    @property
    def augur_class(self) -> str:
        return str(_TIER_META[self.tier]["augur"])

    def caveat(self) -> str | None:
        """Plain-language identifiability caveat, or ``None`` when separately
        identified. Experiment-validated channels are never caveated here."""
        if self.identified or self.tier is EvidenceTier.EXPERIMENT_VALIDATED:
            return None
        if self.collinear_with:
            others = _human_join(self.collinear_with)
            return (
                f"{self.channel} and {others} cannot be separately identified in "
                f"this data — their individual effects are collinear, so the "
                f"combined contribution is far more trustworthy than either "
                f"channel's number alone."
            )
        return (
            f"{self.channel}'s effect cannot be reliably separated from the other "
            f"channels in this data; read its individual number with caution."
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (used by the interactive report + agent tools)."""
        return {
            "channel": self.channel,
            "tier": self.tier.value,
            "label": self.label,
            "short_label": self.short_label,
            "gloss": self.gloss,
            "identified": bool(self.identified),
            "collinear_with": list(self.collinear_with),
            "contraction": _finite_or_none(self.contraction),
            "learning_verdict": self.learning_verdict,
            "vif": _finite_or_none(self.vif),
            "experiment": bool(self.experiment),
            "gated": bool(self.gated),
            "caveat": self.caveat(),
        }


# =============================================================================
# Computation
# =============================================================================


def _channel_learning(
    learning: "pd.DataFrame | None", channel: str
) -> tuple[float | None, str | None]:
    """Worst (least-learned) contraction + its verdict for a channel's parameters.

    Matches learning-frame rows whose base parameter name (before any ``[geo]``
    index) is ``<prefix><channel>`` for the channel-effect prefixes, and returns
    the row with the SMALLEST contraction (the worst case dominates the tier).
    """
    if learning is None or getattr(learning, "empty", True):
        return None, None
    if "parameter" not in learning.columns:
        return None, None
    targets = {f"{p}{channel}" for p in _CHANNEL_PARAM_PREFIXES}
    worst_c: float | None = None
    worst_verdict: str | None = None
    for _, row in learning.iterrows():
        param = str(row.get("parameter", ""))
        base = param.split("[", 1)[0]
        if base not in targets:
            continue
        c = _finite_or_none(row.get("contraction"))
        verdict = row.get("verdict")
        verdict = str(verdict) if verdict is not None else None
        if c is None:
            # Undetermined width (degenerate prior) — remember the verdict but
            # keep looking for a finite contraction.
            if worst_verdict is None:
                worst_verdict = verdict
            continue
        if worst_c is None or c < worst_c:
            worst_c = c
            worst_verdict = verdict
    return worst_c, worst_verdict


def channel_evidence(
    channels: Sequence[str],
    *,
    experiment_channels: Iterable[str] | None = None,
    learning: "pd.DataFrame | None" = None,
    collinearity: Mapping[str, Mapping[str, Any]] | None = None,
    contraction_min: float = DEFAULT_CONTRACTION_MIN,
    vif_max: float = DEFAULT_VIF_MAX,
) -> dict[str, ChannelEvidence]:
    """Assign an :class:`EvidenceTier` + identifiability flag to every channel.

    Parameters
    ----------
    channels:
        Channel names to annotate.
    experiment_channels:
        Channels whose effect was calibrated against a randomized experiment
        folded into the fit (→ ``experiment-validated``).
    learning:
        A prior→posterior learning frame from
        :meth:`BayesianMMM.compute_parameter_learning` (columns ``parameter``,
        ``contraction``, ``verdict``). Used to detect ``prior-dominated``
        channels. ``None`` → no channel is downgraded on the learning axis.
    collinearity:
        ``{channel: {"vif": float, "collinear_with": [names]}}`` — per-channel
        variance inflation over the channels' contributions (see
        :func:`collinearity_from_matrix`). A channel with ``vif >= vif_max`` is
        flagged *not separately identified*. ``None`` → every channel identified.
    contraction_min, vif_max:
        Gate thresholds (conventions, tunable).

    Returns
    -------
    dict[str, ChannelEvidence]
        One entry per channel.
    """
    exp_set = {str(c) for c in (experiment_channels or [])}
    coll = collinearity or {}
    out: dict[str, ChannelEvidence] = {}
    for ch in channels:
        ch = str(ch)
        c, verdict = _channel_learning(learning, ch)

        # --- identifiability (collinearity) flag ---
        info = coll.get(ch) or {}
        vif = _finite_or_none(info.get("vif"))
        collinear_with = [str(x) for x in (info.get("collinear_with") or [])]
        identified = True
        if vif is not None and vif >= vif_max:
            identified = False
        elif collinear_with:
            # A collinear partner was named without a VIF (duck-typed map).
            identified = False

        # --- evidence tier ---
        if ch in exp_set:
            tier = EvidenceTier.EXPERIMENT_VALIDATED
        else:
            is_prior = verdict == "prior-dominated" or (
                verdict not in _STRONG_LEARNING_VERDICTS
                and c is not None
                and c < contraction_min
            )
            tier = (
                EvidenceTier.PRIOR_DOMINATED
                if is_prior
                else EvidenceTier.MODEL_IDENTIFIED
            )

        out[ch] = ChannelEvidence(
            channel=ch,
            tier=tier,
            identified=identified,
            collinear_with=collinear_with,
            contraction=c,
            learning_verdict=verdict,
            vif=vif,
            experiment=ch in exp_set,
        )
    return out


def _model_media_matrix(model: Any, n_channels: int) -> "np.ndarray | None":
    """The model's ``(n_obs, n_channels)`` raw media design for the collinearity
    check — prefers ``X_media_raw`` then ``X_media``. ``None`` when a single
    channel or neither attribute aligns to the channel list."""
    if n_channels < 2:
        return None
    for attr in ("X_media_raw", "X_media"):
        X = getattr(model, attr, None)
        if X is None:
            continue
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            continue
        if X.ndim == 2 and X.shape[1] == n_channels and X.shape[0] >= 3:
            return X
    return None


def evidence_for_model(
    model: Any,
    channels: Sequence[str],
    *,
    collinearity_matrix: "np.ndarray | None" = None,
    prior_samples: int = 400,
    random_seed: int = 0,
) -> dict[str, ChannelEvidence]:
    """Assemble :func:`channel_evidence` inputs straight from a fitted model.

    This is the **single gathering path** shared by the report extractor and the
    fit-time persistence that feeds the live Performance/Estimands dashboard, so a
    channel's tier reads identically in the report and the dashboard (issue #124).
    It folds three model signals:

    * *experiment coverage* — ``model.experiments`` (→ ``experiment-validated``);
    * *prior→posterior learning* — ``model.compute_parameter_learning``
      (best-effort; a failure there just omits the prior-dominated downgrade);
    * *per-channel collinearity* — the model's raw media design, or a
      caller-supplied ``collinearity_matrix`` (e.g. a contribution-series
      fallback when the model does not expose its media design).

    Returns one :class:`ChannelEvidence` per channel. Pure aside from reading
    model attributes; raises nothing the underlying learning call would not.
    """
    chans = [str(c) for c in channels]

    # experiment-validated: channels folded into this fit as calibration.
    exp_channels: set[str] = set()
    for exp in getattr(model, "experiments", None) or []:
        ch = getattr(exp, "channel", None)
        if ch is not None:
            exp_channels.add(str(ch))

    # prior-dominated: prior→posterior contraction per channel parameter.
    learning = None
    fn = getattr(model, "compute_parameter_learning", None)
    if callable(fn) and getattr(model, "_trace", None) is not None:
        try:
            learning = fn(prior_samples=prior_samples, random_seed=random_seed)
        except Exception:  # noqa: BLE001 — learning is best-effort
            learning = None

    # identifiability: per-channel collinearity over the media design.
    collinearity = None
    mat = collinearity_matrix
    if mat is None:
        mat = _model_media_matrix(model, len(chans))
    if mat is not None:
        collinearity = collinearity_from_matrix(mat, chans)

    return channel_evidence(
        chans,
        experiment_channels=exp_channels,
        learning=learning,
        collinearity=collinearity,
    )


def collinearity_from_matrix(
    matrix: np.ndarray,
    channels: Sequence[str],
    *,
    vif_max: float = DEFAULT_VIF_MAX,
    pair_corr_min: float = 0.9,
) -> dict[str, dict[str, Any]]:
    """Per-channel variance-inflation + named collinear partners.

    ``matrix`` is ``(n_obs, n_channels)`` — the channels' contributions (preferred)
    or their adstocked/raw media. For each channel column we regress it on all the
    others (with an intercept) → ``R^2`` → ``vif = 1/(1-R^2)``, and we name the
    partner channel(s) whose pairwise ``|corr|`` with it is ``>= pair_corr_min``
    (so the caveat reads "Search and Shopping cannot be separated", not just a
    bare VIF). Degenerate (near-constant) columns get ``vif = 1`` and no partner.

    Returns ``{channel: {"vif": float, "r2": float, "collinear_with": [names]}}``.
    """
    X = np.asarray(matrix, dtype=float)
    n_ch = len(channels)
    out: dict[str, dict[str, Any]] = {}
    if X.ndim != 2 or X.shape[1] != n_ch or X.shape[0] < 3 or n_ch < 2:
        # Nothing to separate (single channel) or ill-shaped — everyone identified.
        return {
            str(ch): {"vif": 1.0, "r2": 0.0, "collinear_with": []} for ch in channels
        }

    # Pairwise correlations (guard constant columns → 0 correlation).
    stds = X.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    for j, ch in enumerate(channels):
        ch = str(ch)
        y = X[:, j]
        if stds[j] <= 1e-12:
            out[ch] = {"vif": 1.0, "r2": 0.0, "collinear_with": []}
            continue
        others_idx = [k for k in range(n_ch) if k != j]
        Xo = X[:, others_idx]
        yc = y - y.mean()
        sst = float(np.sum(yc * yc))
        r2 = 0.0
        if sst > 1e-12 and Xo.shape[1] > 0:
            A = np.column_stack([np.ones_like(y), Xo - Xo.mean(axis=0)])
            try:
                coef, *_ = np.linalg.lstsq(A, yc, rcond=None)
                resid = yc - A @ coef
                r2 = float(np.clip(1.0 - np.sum(resid * resid) / sst, 0.0, 1.0))
            except np.linalg.LinAlgError:
                r2 = 0.0
        vif = float(1.0 / max(1.0 - r2, 1e-12)) if r2 < 1.0 else float("inf")
        partners = [
            str(channels[k])
            for k in others_idx
            if abs(float(corr[j, k])) >= pair_corr_min
        ]
        # Only name partners when the channel is actually flagged (vif high).
        if vif < vif_max:
            partners = []
        out[ch] = {"vif": vif, "r2": r2, "collinear_with": partners}
    return out


# =============================================================================
# Rendering
# =============================================================================


def _chip_view(ev: "ChannelEvidence | Mapping[str, Any]") -> dict[str, Any]:
    """Normalize a :class:`ChannelEvidence` OR its ``to_dict()`` form (the shape
    the bundle carries) to the fields the renderers need."""
    if isinstance(ev, ChannelEvidence):
        return {
            "augur_class": ev.augur_class,
            "css_class": ev.css_class,
            "short_label": ev.short_label,
            "gloss": ev.gloss,
            "identified": ev.identified,
            "caveat": ev.caveat(),
        }
    tier = tier_from_string(ev.get("tier", EvidenceTier.MODEL_IDENTIFIED))
    meta = _TIER_META[tier]
    return {
        "augur_class": meta["augur"],
        "css_class": meta["css"],
        "short_label": ev.get("short_label") or meta["short"],
        "gloss": ev.get("gloss") or meta["gloss"],
        "identified": bool(ev.get("identified", True)),
        "caveat": ev.get("caveat"),
    }


def evidence_chip_html(
    ev: "ChannelEvidence | Mapping[str, Any] | None",
    *,
    theme: str = "classic",
    show_caveat: bool = True,
) -> str:
    """Inline chip HTML for one channel's evidence — the shared visual language.

    Accepts a :class:`ChannelEvidence` or its ``to_dict()`` mapping (what the
    report bundle carries). ``theme="classic"`` uses the ``.evidence-chip``
    classes from :data:`EVIDENCE_CHIP_CSS`; ``theme="augur"`` reuses the augur
    ``.tier-chip`` classes (sage / steel / rust) already in :mod:`augur_theme`.
    When the channel is not separately identified and ``show_caveat`` is set, a
    second amber chip ("not separately identified") is appended with the
    plain-language caveat as its ``title``.
    """
    if ev is None:
        return ""
    v = _chip_view(ev)
    if theme == "augur":
        chip = (
            f'<span class="tier-chip {v["augur_class"]}" '
            f'title="{_html.escape(v["gloss"])}">{_html.escape(v["short_label"])}</span>'
        )
        flag_cls = "t-test"  # gold — a caution, not a verdict
        prefix = "tier-chip"
    else:
        chip = (
            f'<span class="evidence-chip {v["css_class"]}" '
            f'title="{_html.escape(v["gloss"])}">{_html.escape(v["short_label"])}</span>'
        )
        flag_cls = "ev-uncertain"
        prefix = "evidence-chip"
    # The "not separately identified" flag rides on the plain-language caveat,
    # which is ``None`` for identified channels AND for experiment-validated ones
    # (the experiment measured the effect directly, so observational collinearity
    # doesn't undermine it).
    if show_caveat and v["caveat"]:
        chip += (
            f'<span class="{prefix} {flag_cls}" '
            f'title="{_html.escape(v["caveat"])}">not separately identified</span>'
        )
    return chip


def evidence_legend_html(*, theme: str = "classic") -> str:
    """A one-line legend explaining the three tiers — render once per report so
    the color language is defined before it is used."""
    order = sorted(EvidenceTier, key=lambda t: _TIER_META[t]["order"])
    items = "".join(
        evidence_chip_html(
            ChannelEvidence(channel="", tier=t), theme=theme, show_caveat=False
        )
        + f'<span class="evidence-legend-gloss">{_html.escape(_TIER_META[t]["gloss"])}</span>'
        for t in order
    )
    return f'<div class="evidence-legend">{items}</div>'


# Classic-report CSS for the chips + legend. The augur readout reuses its own
# ``.tier-chip`` classes, so this block is only injected by the classic path.
EVIDENCE_CHIP_CSS = """
.evidence-chip{display:inline-flex;align-items:center;gap:.35em;padding:.12em .55em;
  border-radius:999px;font-size:.72rem;font-weight:600;line-height:1.4;white-space:nowrap;
  vertical-align:middle;}
.evidence-chip::before{content:"";width:6px;height:6px;border-radius:50%;
  background:currentColor;flex:none;}
.evidence-chip.ev-experiment{background:#e7f2ea;color:#2f6a45;}
.evidence-chip.ev-model{background:#e7eef2;color:#3a5a75;}
.evidence-chip.ev-prior{background:#f5e7e2;color:#7a3525;}
.evidence-chip.ev-uncertain{background:#f5ecd8;color:#8a6408;}
.evidence-legend{display:flex;flex-direction:column;gap:.4rem;margin:.6rem 0 1rem;
  font-size:.8rem;}
.evidence-legend>*{display:flex;align-items:center;}
.evidence-legend .evidence-chip{margin-right:.6rem;flex:none;min-width:5.5rem;justify-content:center;}
.evidence-legend-gloss{color:#5a5148;}
"""
