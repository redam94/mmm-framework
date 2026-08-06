"""When does a channel's effect land — and can this model even say? (issue #224)

"TV pays back in 3 weeks" is the most CFO-legible number an MMM can emit, and it
rests on the model's **least identified** parameter: the adstock decay sits on
an equifinality ridge with saturation and the coefficient, its default prior
differs by 2x in implied half-life depending on which constructor built the
config, and the fitted kernel is truncated at ``l_max`` and renormalized — which
makes every horizon read off it *structurally optimistic* (geometric ``α=0.8``
at ``l_max=8`` reads t90≈5.8 against an untruncated 9.3).

So this module ships ONE named quantity with its epistemics attached, not three
under one word:

* :func:`channel_payback` — **response timing**: the per-draw interpolated lag
  where the cumulative carryover kernel crosses a threshold (``t50``, ``t90``).
  Intervals are the ETI **of the per-draw transform**, never the transform of a
  mean parameter (the crossing is convex in ``α``; the two differ). Every
  result carries the truncated tail mass, the configured ``l_max``, a
  carryover-learning verdict (did the data move the prior at all?), and a
  residual-autocorrelation gate.
* :func:`payback_breakeven` — the **finance** sense: the lag at which the
  cumulative discounted dollar return on a dollar of spend reaches 1. Needs a
  value per KPI unit and **refuses** on efficiency-measured channels (an
  impressions-denominated contribution has no dollar-in to pay back) and when
  no valuation resolves — never a silent ``value_per_kpi=1.0``.

Nothing here re-derives a kernel: :func:`channel_payback` consumes
:func:`mmm_framework.transforms.carryover.posterior_carryover_kernels`, the one
family-aware per-draw reader (#218), so "agrees with the model" holds by
construction.

**Refusals by name.** Three model families cannot answer this question with a
single carryover kernel, and each fails in a different direction:

* ``StructuralNestedMMM`` with AR(1) mediators — persistence lives in the
  mediator state's ρ, on a stated ridge with the per-channel α; a kernel-only
  horizon ignores the slow path entirely.
* Dual-stock brand models (``LongTermBrandMMM`` and copies) — they register the
  FAST activation decay under ``adstock_alpha_<ch>`` while a tightly-primed slow
  brand stock (Beta(47,3), ~11-week retention) carries the long tail; a payback
  read off the registered name is dramatically too short while the model itself
  says the opposite.
* Extension models (Nested / Multivariate / Combined) — one hardcoded geometric
  family at a fixed ``l_max=8``, with mediated paths whose timing the media
  kernel does not describe.

Each refusal names the family and the reason; a payback that cannot be computed
honestly is reported as a refusal, not as a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_THRESHOLDS",
    "TAIL_MASS_CAVEAT_MIN",
    "ChannelPayback",
    "PaybackResult",
    "channel_payback",
    "payback_breakeven",
]

#: The cumulative-share thresholds reported by default: t50 (the canonical
#: half-life, issue #218) and t90 ("when has ~all of it landed").
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 0.9)

#: Above this untruncated-tail mass, the truncation caveat is promoted from a
#: field to a sentence. At the default l_max=8 a geometric alpha=0.8 leaves
#: 13.4% of its mass beyond the window and reads t90≈5.8 against a true 9.3 —
#: the size of bias worth a sentence, not a footnote.
TAIL_MASS_CAVEAT_MIN = 0.10

#: Carryover-parameter contraction below which the payback is prior-dominated
#: (mirrors reporting.evidence.DEFAULT_CONTRACTION_MIN so the two surfaces
#: cannot disagree about what "the data moved the prior" means).
_CONTRACTION_MIN = 0.10

#: Learning verdicts that protect against the contraction-based downgrade —
#: the location moved even if the width did not (reporting.evidence's set).
_STRONG_LEARNING_VERDICTS = frozenset({"strong", "moderate", "relocated"})


@dataclass(frozen=True)
class ChannelPayback:
    """One channel's response-timing payback, with its epistemics attached.

    ``status`` is the headline:

    * ``"ok"`` — a per-draw horizon with an interval.
    * ``"downgraded"`` — a number exists but at least one gate fired
      (prior-dominated carryover, autocorrelated residuals, material truncated
      tail); read ``caveats``.
    * ``"refused"`` — no honest number exists; ``reason`` says why.
    """

    channel: str
    status: str
    reason: str = ""
    family: str = ""
    l_max: int = 0
    normalize: bool = True
    basis: str = "kernel"
    truncated_tail_mass: float = 0.0
    #: Per-threshold summaries: ``{"t50": {"mean", "lower", "upper"}, ...}``.
    #: ``lower``/``upper`` are None when the interval collapsed (#249).
    horizons: dict[str, dict[str, float | None]] = field(default_factory=dict)
    interval_mass: float | None = None
    #: "credible" or "confidence" — a bootstrap-derived payback interval is a
    #: CONFIDENCE interval and must say so (diagnostics.provenance).
    interval_kind: str = "credible"
    interval_collapsed: bool = False
    n_draws: int = 0
    #: Worst-case carryover-parameter learning for this channel.
    learning_contraction: float | None = None
    learning_verdict: str | None = None
    prior_dominated: bool = False
    #: When basis="counterfactual": the kernel-basis t50 mean, so the
    #: disagreement between the two bases is measured once and reported.
    kernel_t50_mean: float | None = None
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "status": self.status,
            "reason": self.reason,
            "family": self.family,
            "l_max": int(self.l_max),
            "normalize": bool(self.normalize),
            "basis": self.basis,
            "truncated_tail_mass": float(self.truncated_tail_mass),
            "horizons": self.horizons,
            "interval_mass": self.interval_mass,
            "interval_kind": self.interval_kind,
            "interval_collapsed": bool(self.interval_collapsed),
            "n_draws": int(self.n_draws),
            "learning_contraction": self.learning_contraction,
            "learning_verdict": self.learning_verdict,
            "prior_dominated": bool(self.prior_dominated),
            "kernel_t50_mean": self.kernel_t50_mean,
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class PaybackResult:
    """Per-channel payback horizons plus the run-level provenance.

    The trap this shape exists to avoid: a bare "TV pays back in 3 weeks" reads
    as a measurement of the least identified parameter in the model, from a
    truncated-and-renormalized kernel that makes it look shorter than truth,
    from a posterior mean that hides how little the prior moved. Nothing renders
    without (a) a per-draw interval, (b) the learning verdict, (c) the truncated
    tail mass, and (d) the stated basis — all carried here.
    """

    channels: dict[str, ChannelPayback]
    basis: str
    thresholds: tuple[float, ...]
    interval_mass: float
    interval_kind: str
    #: {"ljung_box_p", "lag", "autocorrelated"} — the model-level residual
    #: autocorrelation gate. ``autocorrelated=None`` means the test could not
    #: run, which is reported rather than read as a pass.
    autocorrelation: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": {ch: p.to_dict() for ch, p in self.channels.items()},
            "basis": self.basis,
            "thresholds": [float(t) for t in self.thresholds],
            "interval_mass": float(self.interval_mass),
            "interval_kind": self.interval_kind,
            "autocorrelation": dict(self.autocorrelation),
            "caveats": list(self.caveats),
        }


def _eti(draws: np.ndarray, mass: float) -> tuple[float, float]:
    lo = float(np.percentile(draws, 100.0 * (1.0 - mass) / 2.0))
    hi = float(np.percentile(draws, 100.0 * (1.0 + mass) / 2.0))
    return lo, hi


def _fit_diagnostics(model: Any) -> dict:
    diag = getattr(model, "_fit_diagnostics", None)
    return diag if isinstance(diag, dict) else {}


def _interval_kind(model: Any) -> str:
    from ..diagnostics.provenance import is_frequentist

    return "confidence" if is_frequentist(_fit_diagnostics(model)) else "credible"


def _refusal(model: Any) -> str | None:
    """The family-level reason no kernel payback exists, or ``None``.

    Ordered specific → generic so the message names the actual mechanism
    rather than the base class.
    """
    # Dual-stock brand models: detected by their registered variables, not by
    # class name — a user-authored copy of LongTermBrandMMM has the same trap
    # under a different name. `adstock_alpha_<ch>` there is only the FAST stock.
    trace = getattr(model, "_trace", None)
    post = getattr(trace, "posterior", None) if trace is not None else None
    if post is not None:
        names = set(getattr(post, "data_vars", []) or [])
        if "long_term_fraction" in names or "brand_retention" in names:
            return (
                "this model carries a second, slow brand stock "
                "('brand_retention' / 'long_term_fraction' are in its "
                "posterior) and registers only the fast activation decay under "
                "'adstock_alpha_<channel>'. A payback horizon read off that "
                "kernel is dramatically too short while the model's own "
                "long-term split says the opposite. Use the long-term split "
                "surface instead."
            )

    # StructuralNestedMMM with AR(1) mediator/factor dynamics: persistence
    # lives in the state's rho, on a stated ridge with the per-channel alpha.
    cfg = getattr(model, "config", None)
    for spec_list in ("mediators", "latent_factors"):
        specs = getattr(cfg, spec_list, None) or []
        for spec in specs:
            dyn = getattr(getattr(spec, "dynamics", None), "value", None) or getattr(
                spec, "dynamics", None
            )
            if str(dyn).lower() == "ar1":
                return (
                    f"this structural model gives '{getattr(spec, 'name', '?')}' "
                    "AR(1) dynamics, so carryover persistence lives in the "
                    "state's rho — on a stated equifinality ridge with the "
                    "per-channel adstock alpha. A kernel-only payback ignores "
                    "the slow mediated path entirely; there is no single "
                    "kernel to read a horizon from."
                )

    # Extension models (Nested / Multivariate / Combined / Structural without
    # AR(1)): one hardcoded geometric family at fixed l_max, with mediated
    # paths the media kernel does not time. MRO check by NAME so this module
    # never imports the extension stack.
    mro_names = {getattr(b, "__name__", "") for b in type(model).__mro__}
    if "BaseExtendedMMM" in mro_names:
        return (
            f"{type(model).__name__} is an extension model: its media transform "
            "is one hardcoded geometric family at a fixed l_max=8, and its "
            "mediated/cross-outcome paths carry effect on timings the media "
            "kernel does not describe. A single-kernel payback would be "
            "wrong in a direction that depends on the structure; refused "
            "rather than guessed."
        )
    return None


def _carryover_learning(learning: Any, channel: str) -> tuple[float | None, str | None]:
    """Worst-case (least-learned) contraction + verdict over the channel's
    CARRYOVER parameters.

    The evidence-tier machinery attributes only effect-size parameters
    (``beta_``/``roi_``) to a channel; payback's credibility rests on
    ``adstock_alpha_/theta_/shape_/scale_`` instead, which that filter drops
    before the tier decision (issue #224). Delegates to
    :func:`mmm_framework.reporting.evidence.carryover_learning` so the two
    surfaces share one attribution rule.
    """
    from ..reporting.evidence import carryover_learning

    return carryover_learning(learning, channel)


def channel_payback(
    model: Any,
    channels: list[str] | None = None,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    hdi_prob: float = 0.90,
    max_draws: int = 500,
    basis: str = "kernel",
    learning: Any = None,
    autocorrelation: dict[str, Any] | None = None,
) -> PaybackResult:
    """Per-channel response-timing payback: interpolated cumulative-share
    crossing lags, per draw, with intervals and every gate the number needs.

    Parameters
    ----------
    model : BayesianMMM
        A fitted model. Extension / structural-AR(1) / dual-stock families are
        refused by name — see the module docstring.
    channels : list of str, optional
        Defaults to every channel.
    thresholds : tuple of float
        Cumulative-share crossings to report; ``(0.5, 0.9)`` → ``t50``/``t90``.
    hdi_prob : float
        Interval mass. Equal-tailed over the per-draw crossing lags — the ETI
        **of the transform**, never the transform of the mean (``log(0.5)/
        log(α)`` is convex, so those differ; no public function here derives a
        horizon from ``mean(alpha)``).
    max_draws : int
        Posterior thinning for the kernel construction.
    basis : {"kernel", "counterfactual"}
        ``"kernel"`` (default) reads the fitted carryover kernel — exact under
        a linear response, cheap (no posterior-predictive pass). The graph is
        ``beta·sat(adstock(x))``, so the true incremental profile is
        ``beta·sat'(a_t)·w_lag``; ``"counterfactual"`` measures that with a
        posterior-predictive pulse per channel (N extra passes) and reports the
        kernel-basis disagreement alongside.
    learning : pd.DataFrame | False, optional
        A precomputed ``compute_parameter_learning()`` frame; computed here
        (best-effort) when omitted. Pass ``False`` to skip the learning gate
        entirely (cheap-snapshot paths) — the verdict then reads ``None``,
        which renders as untested, never as passed.
    autocorrelation : dict, optional
        A precomputed ``{"ljung_box_p", "lag", "autocorrelated"}`` residual
        check (e.g. from a forecast's ``caveat_fields``), optionally with
        ``ppc_acf1_*`` keys; computed here (Ljung-Box + one predict() pass for
        the PPC lag-1 check) when omitted. Pass ``{}`` to skip — reported as
        UNTESTED, never as passed.

    Returns
    -------
    PaybackResult
        Every requested channel is present — as a horizon, a downgrade, or a
        named refusal. Nothing is silently dropped.
    """
    if basis not in ("kernel", "counterfactual"):
        raise ValueError(f"basis must be 'kernel' or 'counterfactual', got {basis!r}")
    if getattr(model, "_trace", None) is None:
        raise ValueError("Model not fitted. Call fit() first.")

    interval_kind = _interval_kind(model)
    names = list(channels or getattr(model, "channel_names", []) or [])

    # Family-level refusal: one reason, applied to every channel, so the
    # result still carries per-channel entries a table can render.
    family_reason = _refusal(model)
    if family_reason is not None:
        refused = {
            ch: ChannelPayback(
                channel=ch,
                status="refused",
                reason=family_reason,
                basis=basis,
                interval_kind=interval_kind,
            )
            for ch in names
        }
        return PaybackResult(
            channels=refused,
            basis=basis,
            thresholds=tuple(float(t) for t in thresholds),
            interval_mass=float(hdi_prob),
            interval_kind=interval_kind,
            caveats=[family_reason],
        )

    from ..transforms.carryover import (
        carryover_crossing_lags,
        posterior_carryover_kernels,
    )

    kernels = posterior_carryover_kernels(model, names, max_draws=max_draws)

    # -- gates computed once per model ------------------------------------
    if learning is False:
        learning = None  # explicit skip: verdict None -> renders as untested
    elif learning is None:
        fn = getattr(model, "compute_parameter_learning", None)
        if callable(fn):
            try:
                learning = fn(prior_samples=400, random_seed=0)
            except Exception:  # noqa: BLE001 — learning is best-effort
                learning = None

    if autocorrelation is None:
        from .forecast import _residual_autocorrelation

        autocorrelation = dict(_residual_autocorrelation(model))
        # The residual Ljung-Box is necessary but not sufficient: measured on
        # `adstock_misspec` (NUTS, 156 weeks) it reads p=0.16 while the
        # posterior-predictive lag-1 autocorrelation check is extreme at
        # p=0.004 — the PPC compares the DATA's persistence against replicated
        # draws and catches what whitened residuals hide. One predict() pass
        # (~1s); skipped when the caller supplied a precomputed dict.
        autocorrelation.update(_ppc_acf1(model, max_draws=max_draws))
    autocorr_fired = bool(autocorrelation.get("autocorrelated")) or bool(
        autocorrelation.get("ppc_acf1_extreme")
    )

    global_caveats: list[str] = []
    if autocorr_fired:
        bits = []
        if autocorrelation.get("ppc_acf1_extreme"):
            bp = autocorrelation.get("ppc_acf1_bayes_p")
            bits.append(
                "the posterior-predictive lag-1 autocorrelation check is "
                f"extreme (Bayesian p={bp:.3g})"
                if isinstance(bp, (int, float))
                else "the posterior-predictive lag-1 autocorrelation check is "
                "extreme"
            )
        if autocorrelation.get("autocorrelated"):
            p = autocorrelation.get("ljung_box_p")
            p_s = f"{p:.3g}" if isinstance(p, (int, float)) else "n/a"
            bits.append(f"training residuals are autocorrelated (Ljung-Box p={p_s})")
        global_caveats.append(
            "The model fails to reproduce week-to-week persistence — "
            + " and ".join(bits)
            + " — which is the signature of a misspecified carryover window. "
            "A kernel truncated at l_max cannot express mass beyond it, so on "
            "such fits the reported horizon is BIASED SHORT: read every "
            "payback here as a lower bound, and prefer re-fitting with a "
            "longer or more flexible carryover before acting on it."
        )
    elif (
        autocorrelation.get("autocorrelated") is None
        and autocorrelation.get("ppc_acf1_extreme") is None
    ):
        global_caveats.append(
            "Neither autocorrelation check could run on this fit; the "
            "carryover-misspecification gate is UNTESTED, not passed."
        )
    if interval_kind == "confidence":
        global_caveats.append(
            "Frequentist fit: payback intervals are bootstrap confidence "
            "intervals — statements about the estimator's sampling "
            "variability, not probability over the horizon."
        )

    out: dict[str, ChannelPayback] = {}
    for ch in names:
        K = kernels.get(ch)
        if K is None or K.status in ("unsupported", "missing_params"):
            out[ch] = ChannelPayback(
                channel=ch,
                status="refused",
                reason=(K.note if K is not None else "no kernel readable"),
                family=(K.family if K is not None else "unknown"),
                basis=basis,
                interval_kind=interval_kind,
            )
            continue

        if K.family == "none":
            # A unit impulse: all effect lands in the same period. That IS the
            # payback answer, not a refusal.
            out[ch] = ChannelPayback(
                channel=ch,
                status="ok",
                family="none",
                l_max=1,
                normalize=K.normalize,
                basis=basis,
                horizons={
                    _t_name(t): {"mean": 0.0, "lower": 0.0, "upper": 0.0}
                    for t in thresholds
                },
                interval_mass=float(hdi_prob),
                interval_kind=interval_kind,
                n_draws=int(K.kernel.shape[0]),
                caveats=["No adstock configured; the effect has no carryover."],
            )
            continue

        kern = K.kernel
        if basis == "counterfactual":
            profile = _counterfactual_profile(model, ch, K.l_max, max_draws)
            kernel_t50 = float(np.nanmean(carryover_crossing_lags(kern, 0.5)))
            if profile is not None:
                kern = profile
            else:
                # The pulse pass failed; fall back to the kernel and say so
                # rather than silently switching basis.
                kernel_t50 = None
        else:
            kernel_t50 = None

        caveats: list[str] = []
        n_draws = int(kern.shape[0])

        # Per-draw crossings for every threshold; the interval is the ETI of
        # the per-draw values.
        horizons: dict[str, dict[str, float | None]] = {}
        collapsed = n_draws < 2
        for t in thresholds:
            lags = carryover_crossing_lags(kern, float(t))
            lags = lags[np.isfinite(lags)]
            if lags.size == 0:
                horizons[_t_name(t)] = {"mean": None, "lower": None, "upper": None}
                continue
            mean = float(lags.mean())
            if lags.size < 2:
                horizons[_t_name(t)] = {"mean": mean, "lower": None, "upper": None}
                collapsed = True
                continue
            lo, hi = _eti(lags, hdi_prob)
            # A tiny-but-nonzero interval renders as an identical pair once
            # rounded, which misleads exactly as much as a zero one (#249).
            if f"{lo:.3f}" == f"{hi:.3f}":
                horizons[_t_name(t)] = {"mean": mean, "lower": None, "upper": None}
                collapsed = True
            else:
                horizons[_t_name(t)] = {"mean": mean, "lower": lo, "upper": hi}
        if collapsed:
            caveats.append(
                "The interval collapsed onto the point estimate — this fit "
                "produced no spread to summarise (an approximate MAP/ADVI fit "
                "does this by construction). The point estimate stands; the "
                "absent interval is not evidence of precision."
            )

        # Carryover learning: did the data move the adstock prior at all?
        contraction, verdict = _carryover_learning(learning, ch)
        prior_dominated = verdict == "prior-dominated" or (
            verdict not in _STRONG_LEARNING_VERDICTS
            and contraction is not None
            and contraction < _CONTRACTION_MIN
        )
        if prior_dominated:
            caveats.append(
                "The carryover parameters barely moved off their prior "
                f"(contraction {contraction:.2f} — verdict: {verdict}). This "
                "horizon reflects the ASSUMED adstock prior more than the "
                "data; two defensible priors ship with half-lives differing "
                "2x, so treat it as a placeholder until an experiment or more "
                "flighting variation pins it."
                if contraction is not None
                else "The carryover parameters' learning could not be "
                "determined; this horizon may reflect the prior more than "
                "the data."
            )

        tail = float(K.truncated_tail_mass)
        if tail >= TAIL_MASS_CAVEAT_MIN:
            caveats.append(
                f"{tail:.0%} of the untruncated kernel's mass falls beyond the "
                f"configured l_max={K.l_max} and was renormalized INSIDE the "
                "window — the true horizon is longer than reported. Refit "
                "with a larger l_max to measure how much."
            )

        if K.status == "legacy_blend":
            caveats.append(K.note)
        if basis == "counterfactual" and kernel_t50 is None:
            caveats.append(
                "The counterfactual pulse pass failed for this channel; "
                "numbers shown are KERNEL-basis."
            )

        status = "ok"
        if prior_dominated or autocorr_fired or tail >= TAIL_MASS_CAVEAT_MIN:
            status = "downgraded"

        out[ch] = ChannelPayback(
            channel=ch,
            status=status,
            family=K.family,
            l_max=int(K.l_max),
            normalize=bool(K.normalize),
            basis=(
                "kernel" if basis == "counterfactual" and kernel_t50 is None else basis
            ),
            truncated_tail_mass=tail,
            horizons=horizons,
            interval_mass=float(hdi_prob),
            interval_kind=interval_kind,
            interval_collapsed=collapsed,
            n_draws=n_draws,
            learning_contraction=contraction,
            learning_verdict=verdict,
            prior_dominated=prior_dominated,
            kernel_t50_mean=kernel_t50,
            caveats=caveats,
        )

    return PaybackResult(
        channels=out,
        basis=basis,
        thresholds=tuple(float(t) for t in thresholds),
        interval_mass=float(hdi_prob),
        interval_kind=interval_kind,
        autocorrelation=dict(autocorrelation),
        caveats=global_caveats,
    )


def _t_name(threshold: float) -> str:
    """``0.5 -> "t50"``, ``0.9 -> "t90"``."""
    return f"t{int(round(float(threshold) * 100))}"


def _ppc_acf1(model: Any, *, max_draws: int = 500) -> dict[str, Any]:
    """Posterior-predictive lag-1 autocorrelation check, in gate shape.

    ``P(acf1(y_rep) >= acf1(y_obs))``; extreme when outside (0.05, 0.95) —
    the same statistic and decision the interactive report's ``_ppc_stat_facts``
    and the validator's ``AutocorrelationCheck`` use, computed here from one
    ``predict()`` pass so the payback gate does not depend on a report having
    been built first. Best-effort: any failure returns ``{... : None}``, which
    the caller reports as UNTESTED rather than passed.
    """
    out: dict[str, Any] = {"ppc_acf1_bayes_p": None, "ppc_acf1_extreme": None}
    try:
        pred = model.predict(return_original_scale=True, random_seed=0)
        rep = np.asarray(pred.y_pred_samples, dtype=float)
        obs = np.asarray(model.y_raw, dtype=float)
        if rep.ndim != 2 or rep.shape[0] < 20 or obs.size < 4:
            return out
        if rep.shape[0] > max_draws:
            take = np.linspace(0, rep.shape[0] - 1, max_draws).astype(int)
            rep = rep[take]

        def _acf1(x: np.ndarray) -> np.ndarray:
            a, b = x[..., :-1], x[..., 1:]
            am = a - a.mean(axis=-1, keepdims=True)
            bm = b - b.mean(axis=-1, keepdims=True)
            num = (am * bm).sum(axis=-1)
            den = np.sqrt((am**2).sum(axis=-1) * (bm**2).sum(axis=-1))
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / den

        obs_v = float(_acf1(obs))
        rep_v = _acf1(rep)
        rep_v = rep_v[np.isfinite(rep_v)]
        if not np.isfinite(obs_v) or rep_v.size < 20:
            return out
        p = float(np.mean(rep_v >= obs_v))
        out["ppc_acf1_bayes_p"] = p
        out["ppc_acf1_extreme"] = bool(p < 0.05 or p > 0.95)
    except Exception:  # noqa: BLE001 — a gate must never fail the payback
        pass
    return out


def _counterfactual_profile(
    model: Any, channel: str, l_max: int, max_draws: int
) -> np.ndarray | None:
    """Per-draw incremental response profile of a one-period spend pulse.

    Measures ``beta·sat'(a_t)·w_lag`` — the true incremental timing under the
    fitted nonlinear response — by differencing two posterior-predictive
    contribution passes with a shared seed. Returns ``(n_draws, l_max)``
    profiles (may be non-normalized; crossing lags normalize per draw), or
    ``None`` when the pulse pass cannot run.
    """
    try:
        X = np.asarray(getattr(model, "X_media_raw"), dtype=float)
        names = list(getattr(model, "channel_names", []))
        c = names.index(channel)
        n_obs = X.shape[0]
        # Pulse at a point with room for the full window, sized at the
        # channel's mean nonzero spend — a representative marginal pulse.
        t0 = max(0, n_obs - int(l_max) - 1)
        nz = X[:, c][X[:, c] > 0]
        pulse = float(nz.mean()) if nz.size else float(X[:, c].mean() or 1.0)
        Xp = X.copy()
        Xp[t0, c] += pulse
        base = model.sample_channel_contributions(
            X_media=X, max_draws=max_draws, random_seed=0
        )
        up = model.sample_channel_contributions(
            X_media=Xp, max_draws=max_draws, random_seed=0
        )
        delta = np.asarray(up)[:, :, c] - np.asarray(base)[:, :, c]  # (D, obs)
        prof = delta[:, t0 : t0 + int(l_max)]
        if prof.shape[1] < 1:
            return None
        # Clip tiny negatives from numerical noise; a profile that is all
        # non-positive cannot yield a crossing and returns None.
        prof = np.clip(prof, 0.0, None)
        if not np.any(prof.sum(axis=1) > 0):
            return None
        return prof
    except Exception:  # noqa: BLE001 — the caller reports the fallback
        return None


@dataclass(frozen=True)
class BreakevenResult:
    """Finance-sense payback for one channel: when cumulative discounted
    dollar return on a dollar of spend reaches 1."""

    channel: str
    status: str
    reason: str = ""
    #: Per-draw break-even lag summaries; ``prob_never`` is the posterior mass
    #: on "this channel never repays its spend within the window".
    breakeven_mean: float | None = None
    breakeven_lower: float | None = None
    breakeven_upper: float | None = None
    prob_never: float | None = None
    interval_mass: float | None = None
    interval_kind: str = "credible"
    value_per_kpi: float | None = None
    value_source: str | None = None
    discount_rate_annual: float = 0.0
    l_max: int = 0
    truncated_tail_mass: float = 0.0
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def payback_breakeven(
    model: Any,
    channels: list[str] | None = None,
    *,
    value_per_kpi: float | None = None,
    value_source: str | None = None,
    discount_rate_annual: float = 0.0,
    hdi_prob: float = 0.90,
    max_draws: int = 300,
) -> dict[str, BreakevenResult]:
    """Finance-sense payback: the lag at which a dollar of spend has returned a
    discounted dollar, per draw.

    Composition: the channel's per-draw ROI (KPI units per divisor unit) ×
    ``value_per_kpi`` gives dollars back per dollar; the fitted carryover
    kernel times WHEN it lands; :mod:`.discount` prices the delay. Break-even
    for draw d is the first lag where ``ROI_d · value · Σ_{k≤lag} w_k · disc_k``
    reaches 1. Draws that never reach 1 inside the kernel window are reported as
    ``prob_never``, not dropped.

    Two refusals, both inherited rather than invented:

    * **Efficiency-measured channels** (``MetricMeta.is_monetary`` False): an
      impressions-denominated contribution has no dollar-in to pay back (#221).
    * **No resolved valuation**: converting KPI units to dollars with a silent
      ``1.0`` is the exact defect :mod:`mmm_framework.finance` exists to
      prevent. Raises :class:`~mmm_framework.finance.UnresolvedValueError`.
    """
    from ..finance import UnresolvedValueError

    if value_per_kpi is None:
        raise UnresolvedValueError("Finance-sense payback break-even")
    if getattr(model, "_trace", None) is None:
        raise ValueError("Model not fitted. Call fit() first.")

    interval_kind = _interval_kind(model)
    names = list(channels or getattr(model, "channel_names", []) or [])

    family_reason = _refusal(model)
    if family_reason is not None:
        return {
            ch: BreakevenResult(channel=ch, status="refused", reason=family_reason)
            for ch in names
        }

    from ..reporting.helpers.measurement import resolve_channel_divisor
    from ..transforms.carryover import posterior_carryover_kernels
    from .discount import discount_weights

    kernels = posterior_carryover_kernels(model, names, max_draws=max_draws)

    # One contribution pass for every channel's ROI draws.
    contrib = model.sample_channel_contributions(max_draws=max_draws, random_seed=0)
    contrib = np.asarray(contrib, dtype=float)  # (D, obs, C)
    all_names = list(getattr(model, "channel_names", []))

    out: dict[str, BreakevenResult] = {}
    for ch in names:
        resolved = resolve_channel_divisor(model, ch)
        if not resolved.meta.is_monetary:
            out[ch] = BreakevenResult(
                channel=ch,
                status="refused",
                reason=(
                    f"{ch} is measured in {resolved.meta.divisor_units} — an "
                    "efficiency metric with no dollar of spend to pay back. "
                    "Provide a spend column or CPM/CPC cost basis to convert."
                ),
                interval_kind=interval_kind,
            )
            continue
        K = kernels.get(ch)
        if K is None or K.status in ("unsupported", "missing_params"):
            out[ch] = BreakevenResult(
                channel=ch,
                status="refused",
                reason=(K.note if K is not None else "no kernel readable"),
                interval_kind=interval_kind,
            )
            continue
        spend = float(resolved.total)
        if not resolved.found or spend <= 0:
            out[ch] = BreakevenResult(
                channel=ch,
                status="refused",
                reason="no positive spend to pay back",
                interval_kind=interval_kind,
            )
            continue

        c_idx = all_names.index(ch)
        roi_draws = contrib[:, :, c_idx].sum(axis=1) / spend  # (D,)
        D = min(roi_draws.shape[0], K.kernel.shape[0])
        roi_draws = roi_draws[:D]
        kern = K.kernel[:D]
        disc = discount_weights(kern.shape[1], rate_annual=discount_rate_annual)

        be = np.full(D, np.nan)
        for d in range(D):
            total = kern[d].sum()
            if not np.isfinite(total) or total <= 0:
                continue
            cum_return = (
                float(roi_draws[d])
                * float(value_per_kpi)
                * np.cumsum(kern[d] * disc)
                / total
            )
            hit = np.nonzero(cum_return >= 1.0)[0]
            if hit.size:
                j = int(hit[0])
                prev = cum_return[j - 1] if j > 0 else 0.0
                span = cum_return[j] - prev
                be[d] = j + ((1.0 - prev) / span if span > 0 else 0.0)
        finite = be[np.isfinite(be)]
        prob_never = float(1.0 - finite.size / D) if D else None

        caveats: list[str] = []
        tail = float(K.truncated_tail_mass)
        if tail >= TAIL_MASS_CAVEAT_MIN:
            caveats.append(
                f"{tail:.0%} of the kernel's mass falls beyond l_max={K.l_max}; "
                "returns landing there are not counted, so prob_never is "
                "overstated and the break-even lag understated."
            )
        if finite.size == 0:
            out[ch] = BreakevenResult(
                channel=ch,
                status="never",
                reason="no posterior draw repays the spend within the window",
                prob_never=prob_never,
                interval_mass=float(hdi_prob),
                interval_kind=interval_kind,
                value_per_kpi=float(value_per_kpi),
                value_source=value_source,
                discount_rate_annual=float(discount_rate_annual),
                l_max=int(K.l_max),
                truncated_tail_mass=tail,
                caveats=caveats,
            )
            continue
        mean = float(finite.mean())
        if finite.size < 2:
            lo = hi = None
        else:
            lo, hi = _eti(finite, hdi_prob)
            if f"{lo:.3f}" == f"{hi:.3f}":
                lo = hi = None
        out[ch] = BreakevenResult(
            channel=ch,
            status="ok",
            breakeven_mean=mean,
            breakeven_lower=lo,
            breakeven_upper=hi,
            prob_never=prob_never,
            interval_mass=float(hdi_prob),
            interval_kind=interval_kind,
            value_per_kpi=float(value_per_kpi),
            value_source=value_source,
            discount_rate_annual=float(discount_rate_annual),
            l_max=int(K.l_max),
            truncated_tail_mass=tail,
            caveats=caveats,
        )
    return out
