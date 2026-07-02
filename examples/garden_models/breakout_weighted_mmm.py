"""Breakout-Weighted MMM (Model Garden example) — a principled, regularized
replacement for the in-house "PSO weight optimizer" that picks channel-breakout
weights before a saturation curve.

The problem it replaces
-----------------------
A channel (say TV) is reported as several *breakout* impression sub-streams
(Premium / Standard / Remnant; or brand / non-brand for Search). A common
in-house pattern fits a particle-swarm optimizer (PSO) to choose weights
``w_k`` that form a weighted impression aggregate ``Σ_k w_k·I_{k,t}`` fed into
the channel's adstock → saturation pipeline, subject only to the sum-preserving
constraint ``Σ_k w_k·S_k = Σ_k S_k`` (``S_k = Σ_t I_{k,t}``; the share-weighted
mean of the weights is 1). The PSO minimizes **in-sample MSE**.

That procedure **overfits to noise**:

* No regularization. With K breakouts there are K−1 free weights (after the one
  linear constraint), each interacting *multiplicatively* with a nonlinear
  saturation curve. A more flexible reparameterization of the same inputs *must*
  lower in-sample MSE — mechanical, not evidence of signal.
* The constraint pins only the *level*, not the *mix*: it is an identification
  device, not a regularizer. Meeting it launders an overfit mix as discipline.
* No uncertainty. A point estimate of ``w`` cannot answer the only question that
  matters: *is a breakout's weight meaningfully different from equal-weighting,
  or is it within noise?*
* Two-stage / generated-regressor bias: weights are chosen in an outer loop, the
  response fit conditional on them, so weight uncertainty never propagates into
  the reported ROI/contribution (which become over-confident).
* An identifiability ceiling no optimizer can beat: breakouts within a channel
  are usually highly collinear (bought together, scale together). The honest
  output there is a *wide posterior*, which a point optimizer cannot produce.

The principled replacement
--------------------------
Keep the **exact** PSO functional form, but make the weights partial-pooled
Bayesian random effects that shrink toward equal-weighting (``w_k = 1``), with
the between-breakout spread ``τ`` *estimated*:

    logtau_C ~ HalfNormal(σ_w)                 # between-breakout log-SD
    z_C      ~ Normal(0, 1, shape=K)           # non-centered offsets
    w_raw    = exp(logtau_C · z_C)             # logmu = 0 (the constraint sets the level)
    w_C      = w_raw · (Σ_k S_k) / (Σ_k w_raw_k·S_k)   # EXACT sum-preserving renorm
    I_C(t)   = Σ_k w_C,k · I_{k,t}             # weighted impression aggregate
    contribution_C(t) = β_C · sat( adstock( I_C(t) / M_C ) )

This mirrors the framework's per-geo partial-pooled effectiveness
(:meth:`BayesianMMM._build_channel_betas_geo`), applied across *breakouts within
a channel* instead of *geos*. Because ``logtau`` is estimated, the model
**self-regularizes**: if the breakouts don't truly differ, the data pull
``τ → 0`` and the model collapses to the plain equal-weight channel model (the
nested null) — it *learns* there is no granular signal rather than inventing
one. If they do differ (and vary independently enough to be identified), the
weights move *with credible intervals*, and everything is estimated jointly in
one PyMC graph so weight uncertainty propagates into ROI/contribution.

Why the LogNormal+renormalize parameterization (vs. a Dirichlet on shares):
identical NUTS geometry to the proven per-geo hierarchy; an *exact* constraint
(the renorm is exact, not analytic-only); and a single, finite, bounded
falsification statistic — ``τ → 0`` says "breakouts don't differ" — instead of
a Dirichlet ``α → ∞`` boundary.

Identification & the exact nesting (load-bearing)
-------------------------------------------------
The weighted aggregate is normalized by ``M_C = max_t Σ_k I_{k,t}`` — the
**unweighted** aggregate max. At ``w_k = 1`` the re-mix factor
``(Σ_k w_k I_k)/(Σ_k I_k) ≡ 1``, so the channel input equals the plain channel
whose column is the pre-summed ``Σ_k I_{k,t}`` (same normalization). Hence
``τ = 0`` reproduces the equal-weight channel model in the likelihood — the
nested null. (It is *not* byte-identical: the extra ``logtau``/``z`` RVs still
exist, exactly as for the per-geo ``τ``.)

The real prerequisite for reading breakouts apart is **independent variation in
their exposure**. If the sub-streams share a flighting calendar, only the
share-weighted level is pinned and the mix is unidentified — the honest model
then reports wide posteriors (see the ``breakout_collinear`` synthetic world),
the opposite of the PSO's false confidence. MAP is unstable for variance-
component (``τ``) models; **fit with NUTS** (the non-centered parameterization is
mandatory) — a MAP/VI fit's weights are not to be trusted.

Implementation
--------------
A garden model that overrides only ``_prepare_data`` (build a *virtual* grouped-
channel axis plus a private raw-impression matrix and the totals ``S_k``),
``_build_coords`` (add the per-channel breakout dims), and the small base hook
``_channel_media_input`` (the weight block + re-mix). Everything else — adstock,
saturation, trend, seasonality, geo/product pooling, controls, the likelihood,
ROI/decomposition reporting, serialization, the estimand engine — is inherited.
It stays ``__garden_model_kind__ == "mmm"`` so all MMM reporting applies.

Data contract
-------------
The breakout sub-streams are ordinary media columns (tagged
``DatasetRole.PREDICTOR``); ``model_params.breakout_groups`` maps a *virtual*
channel name to its sub-stream columns, e.g.
``{"TV": ["TV_Premium", "TV_Standard", "TV_Remnant"]}``. Channels not listed stay
plain. ``breakout_groups`` round-trips through ``model_params`` so a reloaded
model rebuilds the same virtual axis (the saved panel keeps the sub-stream
columns).
"""

from __future__ import annotations

import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pydantic import BaseModel, Field, model_validator

from mmm_framework.config.roles import DatasetRole
from mmm_framework.estimands.registry import latent_scalar
from mmm_framework.garden import CustomMMM


class BreakoutWeightedParams(BaseModel):
    """Bespoke, settable, defaulted configuration for :class:`BreakoutWeightedMMM`
    (its ``CONFIG_SCHEMA``). Read off ``self.model_params`` in ``_prepare_data`` /
    the media-input hook."""

    #: Virtual channel -> its impression sub-stream columns, e.g.
    #: ``{"TV": ["TV_Premium", "TV_Standard", "TV_Remnant"]}``. Each list must
    #: have >= 2 members (a 1-member group is just a plain channel and is
    #: demoted with a warning); the parent name must NOT collide with an existing
    #: media column. Channels not listed here are modeled as plain channels.
    breakout_groups: dict[str, list[str]] = Field(default_factory=dict)

    #: ``σ_w`` — the prior scale (``HalfNormal``) on each channel's between-
    #: breakout log-SD ``logtau``. Smaller = stronger shrinkage toward equal-
    #: weighting. Default 0.3 matches the per-geo ``media_geo_sigma`` default.
    breakout_weight_sigma: float = Field(default=0.3, gt=0)

    #: Share calibrations: observed within-channel share compositions folded in
    #: as likelihood terms on the model-implied ``breakout_share_<C>`` simplex
    #: (e.g. exported from a continuous-learning program via
    #: ``continuous_learning.arms.arm_shares``). Each entry is a
    #: :class:`mmm_framework.calibration.likelihood.ShareMeasurement` dict:
    #: ``{channel, breakouts, shares, log_ratio_cov | concentration,
    #: distribution, name, source}``. The entry's ``channel`` must be a
    #: ``breakout_groups`` parent and its ``breakouts`` must ALL come from that
    #: group's columns; at build time the order must match the model's breakout
    #: order EXACTLY (the ALR covariance is order-dependent).
    share_calibrations: list[dict] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_share_calibrations(self) -> "BreakoutWeightedParams":
        """Round-trip each share-calibration entry through ``ShareMeasurement``
        so a bad spec fails at build time with a clear message, and check the
        entry targets a real breakout group with that group's own columns."""
        if not self.share_calibrations:
            return self
        # Import inside the validator to keep the schema import-light (the
        # manifest form / JSON-schema path never needs the calibration module).
        from mmm_framework.calibration.likelihood import ShareMeasurement

        normalized: list[dict] = []
        for i, entry in enumerate(self.share_calibrations):
            try:
                meas = ShareMeasurement.from_dict(dict(entry))
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"share_calibrations[{i}]: {e}") from e
            if meas.channel not in self.breakout_groups:
                raise ValueError(
                    f"share_calibrations[{i}]: channel {meas.channel!r} is not a "
                    f"breakout_groups parent {sorted(self.breakout_groups)}."
                )
            group_cols = set(self.breakout_groups[meas.channel])
            unknown = [b for b in meas.breakouts if b not in group_cols]
            if unknown:
                raise ValueError(
                    f"share_calibrations[{i}]: breakouts {unknown} are not "
                    f"sub-streams of {meas.channel!r} "
                    f"{self.breakout_groups[meas.channel]}."
                )
            normalized.append(meas.to_dict())
        object.__setattr__(self, "share_calibrations", normalized)
        return self


class BreakoutWeightedMMM(CustomMMM):
    """An MMM that learns partial-pooled per-breakout impression weights feeding a
    single shared saturation curve per channel — the regularized, uncertainty-
    quantified replacement for a PSO breakout-weight optimizer.

    Bespoke parameters live in :class:`BreakoutWeightedParams` (the
    ``CONFIG_SCHEMA``). It overrides only ``_prepare_data``, ``_build_coords`` and
    the ``_channel_media_input`` hook; ``fit`` / serialization / the estimand
    engine / reporting are inherited.
    """

    #: It IS an MMM (channels, spend, ROI) — keep the kind ``"mmm"`` so the full
    #: MMM reporting / compat tiers apply.
    __garden_model_kind__ = "mmm"

    #: Bespoke, defaulted, validated configuration (read via ``self.model_params``).
    CONFIG_SCHEMA = BreakoutWeightedParams

    #: Data contract: an outcome + >= 1 media predictor (the sub-streams are
    #: ordinary PREDICTOR columns grouped by ``model_params.breakout_groups``).
    REQUIRED_ROLES = (DatasetRole.TARGET, DatasetRole.PREDICTOR)

    # -- data ----------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Build the normal MMM views via the base, then re-express the media axis
        as a *virtual* (grouped) channel axis.

        After ``super()._prepare_data()`` the media columns are the raw sub-stream
        + plain columns. Here we (a) validate ``breakout_groups``; (b) keep a
        private raw-impression matrix ``X_breakout_raw`` and the per-channel
        totals ``S_k`` for the in-graph re-mix; and (c) rebuild ``X_media_raw`` /
        ``channel_names`` / ``_media_raw_max`` over the virtual channels (a
        breakout channel's column is the UNWEIGHTED sum of its sub-streams), so
        every inherited reader (``channel_contributions`` dims, ROI, decomposition,
        ``validate_instance``, ``sample_channel_contributions``) sees a standard
        channel axis.
        """
        super()._prepare_data()  # y, X_media (sub-streams + plain), controls, …

        # The in-graph re-mix lives on the base parametric-adstock path (the
        # legacy fixed-alpha blend has no per-channel media-input hook), so this
        # model is always parametric.
        if not self.use_parametric_adstock and self.model_params.breakout_groups:
            warnings.warn(
                "BreakoutWeightedMMM requires the parametric-adstock path for its "
                "breakout re-mix; forcing use_parametric_adstock=True.",
                stacklevel=2,
            )
        self.use_parametric_adstock = True

        groups_in = dict(self.model_params.breakout_groups or {})
        orig_names = list(self.channel_names)
        orig_X = self.X_media_raw  # (n_obs, n_raw_media)
        name_to_col = {n: i for i, n in enumerate(orig_names)}

        # --- validate + normalize the group spec -------------------------------
        grouped: list[str] = []
        groups: dict[str, list[str]] = {}
        for parent, subs in groups_in.items():
            subs = [str(s) for s in subs]
            if len(subs) < 2:
                warnings.warn(
                    f"breakout group '{parent}' has < 2 members {subs}; its "
                    "column(s) are modeled as plain channels.",
                    stacklevel=2,
                )
                continue
            if parent in name_to_col:
                raise ValueError(
                    f"breakout parent '{parent}' collides with an existing media "
                    f"column; use a new virtual-channel name. Columns: {orig_names}"
                )
            for s in subs:
                if s not in name_to_col:
                    raise ValueError(
                        f"breakout sub-stream '{s}' (parent '{parent}') is not a "
                        f"media column. Available media columns: {orig_names}"
                    )
                if s in grouped:
                    raise ValueError(
                        f"sub-stream '{s}' appears in more than one breakout group."
                    )
                grouped.append(s)
            groups[parent] = subs

        grouped_set = set(grouped)
        plain = [n for n in orig_names if n not in grouped_set]
        virtual = plain + list(groups.keys())

        # --- private raw-impression matrix + per-channel column slices / totals -
        self._breakout_names: dict[str, list[str]] = {
            p: list(s) for p, s in groups.items()
        }
        sub_order = [s for p in groups for s in groups[p]]
        self._breakout_sub_order = sub_order
        if sub_order:
            self.X_breakout_raw = np.column_stack(
                [orig_X[:, name_to_col[s]] for s in sub_order]
            ).astype(np.float64)
        else:
            self.X_breakout_raw = np.zeros((self.n_obs, 0), dtype=np.float64)

        self._breakout_col_idx: dict[str, list[int]] = {}
        self._breakout_totals: dict[str, np.ndarray] = {}
        off = 0
        for p in groups:
            k = len(groups[p])
            self._breakout_col_idx[p] = list(range(off, off + k))
            off += k
            self._breakout_totals[p] = np.array(
                [float(orig_X[:, name_to_col[s]].sum()) for s in groups[p]]
            )

        # --- rebuild the virtual (aggregate) media axis ------------------------
        agg_cols: list[np.ndarray] = []
        new_max: dict[str, float] = {}
        for v in virtual:
            if v in self._breakout_names:
                col = orig_X[:, [name_to_col[s] for s in self._breakout_names[v]]].sum(
                    axis=1
                )
            else:
                col = orig_X[:, name_to_col[v]]
            agg_cols.append(col)
            new_max[v] = float(col.max())
        self.X_media_raw = np.column_stack(agg_cols).astype(np.float64)
        self.channel_names = list(virtual)
        self.n_channels = len(virtual)
        self._media_raw_max = new_max

        # Keep the legacy fixed-alpha caches coherent with the virtual axis (the
        # parametric path is what fit uses, but some inherited readers key on
        # ``_media_max`` — recompute over the aggregate so they never see stale
        # sub-stream keys).
        self._media_max = {}
        self.X_media_adstocked = {}
        for alpha in self.adstock_alphas:
            ad = self._geometric_adstock_per_cell(self.X_media_raw, alpha)
            for cc, ch in enumerate(virtual):
                self._media_max[ch] = max(
                    self._media_max.get(ch, 0.0), float(ad[:, cc].max())
                )
        for alpha in self.adstock_alphas:
            ad = self._geometric_adstock_per_cell(self.X_media_raw, alpha)
            nd = np.zeros_like(ad)
            for cc, ch in enumerate(virtual):
                nd[:, cc] = ad[:, cc] / (self._media_max[ch] + 1e-8)
            self.X_media_adstocked[alpha] = nd
        self._scaling_params["media_max"] = self._media_max.copy()

    # -- model ---------------------------------------------------------------

    def _build_coords(self) -> dict:
        """Add a ``breakout_<C>`` coordinate per breakout channel (the rest of the
        graph is the inherited base build)."""
        coords = super()._build_coords()
        for parent, subs in getattr(self, "_breakout_names", {}).items():
            coords[f"breakout_{parent}"] = list(subs)
        return coords

    def _channel_media_input(self, c, channel_name, X_media_raw_data):
        """Override the base media-input hook: for a breakout channel, return the
        partial-pooled **weighted impression aggregate** (normalized) instead of
        the plain column; plain channels are unchanged.

        Builds the weight block (``logtau``/``z`` → exact sum-preserving renorm)
        and the re-mix factor ``(Σ_k w_k I_{k,t}) / (Σ_k I_{k,t})`` from the raw
        sub-stream impressions. At ``w_k = 1`` the factor is identically 1, so the
        channel reduces to the plain pre-summed channel (the nested null). Called
        inside the ``pm.Model`` context (parametric path).
        """
        if channel_name not in getattr(self, "_breakout_names", {}):
            return X_media_raw_data[:, c]

        model = pm.modelcontext(None)
        # Create the shared raw-impression Data container once per build.
        if "X_breakout_raw" in model.named_vars:
            x_brk = model["X_breakout_raw"]
        else:
            x_brk = pm.Data("X_breakout_raw", self.X_breakout_raw)

        C = channel_name
        subs = self._breakout_names[C]
        S = self._breakout_totals[C]
        S_sum = float(S.sum())
        S_t = pt.as_tensor_variable(S)
        sig_w = float(self.model_params.breakout_weight_sigma)

        # Non-centered log-space hierarchy (mirrors `_build_channel_betas_geo`,
        # with the level omitted because the constraint sets it).
        logtau = pm.HalfNormal(f"breakout_logtau_{C}", sigma=sig_w)
        z = pm.Normal(f"breakout_z_{C}", 0.0, 1.0, dims=f"breakout_{C}")
        w_raw = pt.exp(logtau * z)

        # Sum-preserving constraint, EXACT & differentiable: Σ_k w_k S_k = Σ_k S_k.
        w = pm.Deterministic(
            f"breakout_weights_{C}",
            w_raw * (S_sum / pt.dot(w_raw, S_t)),
            dims=f"breakout_{C}",
        )
        for i, bk in enumerate(subs):
            pm.Deterministic(f"breakout_weight_{C}_{bk}", w[i])
        # Effective contribution share of each breakout (w_k S_k / Σ_j w_j S_j).
        pm.Deterministic(
            f"breakout_share_{C}", (w * S_t) / pt.dot(w, S_t), dims=f"breakout_{C}"
        )

        idx = self._breakout_col_idx[C]
        I_brk = x_brk[:, idx]  # (n_obs, K) raw impressions
        mix_factor = (I_brk @ w) / (I_brk.sum(axis=1) + 1e-12)  # ≡1 at w=1
        return X_media_raw_data[:, c] * mix_factor

    # -- calibration ----------------------------------------------------------

    def _add_experiment_likelihoods(self, channel_handles: dict[str, dict]) -> None:
        """Attach the configured share calibrations, then the base scalar
        experiment likelihoods.

        Each ``model_params.share_calibrations`` entry is compared to the
        channel's ``breakout_share_<C>`` Deterministic (guaranteed built —
        ``_channel_media_input`` runs before the base build reaches this hook)
        via :func:`~mmm_framework.calibration.likelihood.attach_share_likelihood`.

        The measurement's ``breakouts`` must match the MODEL's breakout order
        for that channel EXACTLY (``self._breakout_names[C]``): the ALR
        log-ratio covariance is defined w.r.t. the ordered breakouts, so a
        reordered measurement would need a re-derived covariance — we require
        the exact order rather than silently reindexing. A parent channel that
        also carries a scalar :class:`ExperimentMeasurement` is flagged with a
        double-counting warning (a level readout + a share vector from the SAME
        program enter the posterior twice with correlated errors).
        """
        from mmm_framework.calibration.likelihood import (
            ShareMeasurement,
            attach_share_likelihood,
        )

        entries = list(self.model_params.share_calibrations or [])
        if entries:
            model = pm.modelcontext(None)
            scalar_channels = {exp.channel for exp in (self.experiments or [])}
            used_names: set[str] = set()
            for i, entry in enumerate(entries):
                meas = ShareMeasurement.from_dict(entry)
                C = meas.channel
                expected = list(getattr(self, "_breakout_names", {}).get(C, []))
                if not expected:
                    raise ValueError(
                        f"share calibration targets {C!r}, which is not a "
                        f"breakout channel of this model "
                        f"{sorted(getattr(self, '_breakout_names', {}))}."
                    )
                if list(meas.breakouts) != expected:
                    raise ValueError(
                        f"share calibration for {C!r} must list the model's "
                        f"breakouts in the model's order exactly — expected "
                        f"{expected}, got {list(meas.breakouts)}. (The ALR "
                        "covariance is order-dependent; re-derive it rather "
                        "than reordering.)"
                    )
                if C in scalar_channels:
                    warnings.warn(
                        f"Channel {C!r} has BOTH a scalar experiment "
                        "measurement and a share calibration. If both derive "
                        "from the same program/wave, the evidence enters the "
                        "posterior twice with correlated errors (level via the "
                        "scalar readout, mix via the shares) — prefer one "
                        "parent-level readout + one share vector per program.",
                        stacklevel=2,
                    )
                share_expr = model[f"breakout_share_{C}"]
                base_name = meas.default_node_name(i)
                name = base_name
                bump = 2
                while name in used_names or f"{name}_model_share" in used_names:
                    name = f"{base_name}_{bump}"
                    bump += 1
                used_names.add(name)
                used_names.add(f"{name}_model_share")
                attach_share_likelihood(name, share_expr, meas)

        super()._add_experiment_likelihoods(channel_handles)

    # -- estimands + reporting ----------------------------------------------

    def _default_estimands(self) -> list:
        """Channel-level ROI/contribution (unchanged, on the virtual-channel axis)
        + one named per-breakout *weight* estimand (mean + HDI — the credible
        interval a PSO point estimate cannot give) + the per-channel between-
        breakout spread ``logtau`` (→0 means "breakouts don't differ"). Built
        dynamically because the breakout names are known only after
        ``_prepare_data`` (mirrors the LCA class-size estimands).
        """
        ests: list = ["contribution_roi", "marginal_roas", "contribution"]
        for C, subs in getattr(self, "_breakout_names", {}).items():
            for bk in subs:
                ests.append(
                    latent_scalar(
                        f"breakout_weight_{C}_{bk}",
                        var=f"breakout_weight_{C}_{bk}",
                        kind="breakout_weight",
                        units="weight (share-mean=1)",
                        causal_assumptions=(
                            f"Partial-pooled multiplicative weight on breakout {bk} "
                            f"of {C}; the share-weighted mean of a channel's weights "
                            "is 1 by construction. An interval covering 1 means the "
                            "breakout is indistinguishable from equal-weighting."
                        ),
                    )
                )
            ests.append(
                latent_scalar(
                    f"breakout_spread_{C}",
                    var=f"breakout_logtau_{C}",
                    kind="breakout_spread",
                    units="log-SD",
                    causal_assumptions=(
                        f"Between-breakout effectiveness spread for {C}; a posterior "
                        "concentrating near 0 nests the equal-weight model."
                    ),
                )
            )
        return [self._resolve_estimand(e) for e in ests]

    def breakout_weights_summary(self, hdi_prob: float = 0.94):
        """Posterior breakout-weight table (mean + HDI per sub-stream, the
        impression-share ``S_k``, and the per-channel ``logtau`` spread).

        The headline artifact the PSO cannot produce: an *interval* on every
        weight, so "this breakout is more effective" can be stated honestly (or
        flagged as indistinguishable from equal-weighting when the interval
        covers 1).
        """
        import arviz as az
        import pandas as pd

        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        post = self._trace.posterior
        rows = []
        for C, subs in self._breakout_names.items():
            S = self._breakout_totals[C]
            S_sum = float(S.sum())
            tau_mean = float(post[f"breakout_logtau_{C}"].mean())
            tau_hdi = az.hdi(
                self._trace, var_names=[f"breakout_logtau_{C}"], hdi_prob=hdi_prob
            )[f"breakout_logtau_{C}"].values
            for i, bk in enumerate(subs):
                v = f"breakout_weight_{C}_{bk}"
                mean = float(post[v].mean())
                hdi = az.hdi(self._trace, var_names=[v], hdi_prob=hdi_prob)[v].values
                rows.append(
                    {
                        "channel": C,
                        "breakout": bk,
                        "impression_share": float(S[i] / S_sum),
                        "weight_mean": mean,
                        "hdi_low": float(hdi[0]),
                        "hdi_high": float(hdi[1]),
                        "covers_equal": bool(hdi[0] <= 1.0 <= hdi[1]),
                        "logtau_mean": tau_mean,
                        "logtau_hdi_low": float(tau_hdi[0]),
                        "logtau_hdi_high": float(tau_hdi[1]),
                    }
                )
        return pd.DataFrame(rows)


# Disambiguate for the Model Garden loader (a module may define helper classes).
GARDEN_MODEL = BreakoutWeightedMMM


# ---------------------------------------------------------------------------
# Worked-example data: role-tagged Datasets for the three breakout worlds, plus
# the equal-weight (pre-summed) baseline panel the breakout model nests at τ=0.
# ---------------------------------------------------------------------------


def breakout_dataset(
    scenario: str = "breakout_heterogeneous", *, seed: int | None = None
):
    """Build a role-tagged :class:`Dataset` for :class:`BreakoutWeightedMMM` from
    one of the breakout synthetic worlds, returning ``(dataset, scenario, groups)``.

    KPI → TARGET, every media column (the TV sub-streams + plain channels) →
    PREDICTOR, Price → CONTROL. ``groups`` is the ``breakout_groups`` mapping to
    pass as ``model_params``.
    """
    import pandas as pd

    from mmm_framework.config.dataset import DatasetSchema, RoleBinding
    from mmm_framework.dataset import Dataset
    from mmm_framework.synth import dgp

    sc = dgp.build(scenario, seed=seed)
    table = pd.DataFrame({"Period": sc.weeks})
    table["Sales"] = sc.y.to_numpy()
    for c in sc.channels:
        table[c] = sc.spend[c].to_numpy()
    table["Price"] = sc.controls["Price"].to_numpy()

    bindings = [RoleBinding(name="Sales", role=DatasetRole.TARGET)]
    bindings += [RoleBinding(name=c, role=DatasetRole.PREDICTOR) for c in sc.channels]
    bindings.append(RoleBinding(name="Price", role=DatasetRole.CONTROL))
    schema = DatasetSchema(bindings=bindings, time_col="Period", frequency="W")
    return Dataset.from_wide(table, schema), sc, sc.notes["breakout_groups"]


def breakout_aggregated_panel(scenario):
    """The equal-weight baseline: a :class:`PanelDataset` where each breakout
    channel's sub-streams are summed into a single channel column (so a plain
    :class:`BayesianMMM` on it IS the ``τ = 0`` nested null of the breakout
    model). Accepts a scenario object or name.
    """
    import pandas as pd

    from mmm_framework.config import (
        ControlVariableConfig,
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset
    from mmm_framework.synth import dgp

    sc = dgp.build(scenario) if isinstance(scenario, str) else scenario
    groups = sc.notes["breakout_groups"]
    grouped = {s for subs in groups.values() for s in subs}

    spend = {}
    channels = []
    for parent, subs in groups.items():
        spend[parent] = sc.spend[subs].sum(axis=1).to_numpy()
        channels.append(parent)
    for c in sc.channels:
        if c not in grouped:
            spend[c] = sc.spend[c].to_numpy()
            channels.append(c)
    X_media = pd.DataFrame({c: spend[c] for c in channels}, columns=channels)

    controls = list(sc.controls.columns)
    coords = PanelCoordinates(
        periods=sc.weeks,
        geographies=None,
        products=None,
        channels=channels,
        controls=controls,
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
            for c in channels
        ],
        controls=[
            ControlVariableConfig(name=c, dimensions=[DimensionType.PERIOD])
            for c in controls
        ],
    )
    return PanelDataset(
        y=pd.Series(sc.y.to_numpy(), name="Sales"),
        X_media=X_media,
        X_controls=sc.controls.copy(),
        coords=coords,
        index=sc.weeks,
        config=config,
    )


if __name__ == "__main__":
    # Standalone smoke test: fit on the heterogeneous breakout world (NUTS) and
    # report the recovered breakout weights (with HDIs) vs the planted truth.
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    print("Building the heterogeneous breakout world (TV split 3 ways)…")
    dataset, sc, groups = breakout_dataset("breakout_heterogeneous")
    mmm = BreakoutWeightedMMM(
        dataset,
        ModelConfig(use_parametric_adstock=True),
        TrendConfig(type=TrendType.LINEAR),
        model_params={"breakout_groups": groups},
    )
    print(f"Fitting BreakoutWeightedMMM on {mmm.n_obs} weeks (NUTS)…")
    mmm.fit(draws=500, tune=1000, chains=4, target_accept=0.9, random_seed=11)

    truth = sc.notes["true_weights"]
    print("\nRecovered breakout weights (true in parens; share-mean = 1):")
    for _, r in mmm.breakout_weights_summary().iterrows():
        print(
            f"  {r['breakout']:<14} weight={r['weight_mean']:+.3f} "
            f"[{r['hdi_low']:+.2f}, {r['hdi_high']:+.2f}]  "
            f"(true {truth[r['breakout']]:+.3f})  covers_equal={r['covers_equal']}"
        )
    tau = float(mmm._trace.posterior["breakout_logtau_TV"].mean())
    print(
        f"\nbetween-breakout spread  logtau_TV (mean) = {tau:.3f}  "
        f"(true {sc.notes['true_logtau']:.3f})"
    )
