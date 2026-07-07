"""Bayesian Latent Class Analysis (Model Garden example) — a second **non-MMM**
family, structurally distinct from the CFA: instead of continuous latent factors
it posits discrete latent **classes** (segments).

LCA is a *mixture* measurement model: each observation belongs to one of ``K``
unobserved classes, and each class has its own profile of item-endorsement
probabilities over a set of **binary indicators**. There are no channels, no
spend, no single KPI. It answers:

    "Are there distinct latent segments in these binary responses, how large is
     each, and how does each segment respond to each item?"

How it plugs in (the non-MMM contract)
--------------------------------------
Like the CFA, it subclasses :class:`~mmm_framework.garden.CustomMMM` and declares
``__garden_model_kind__ = "latent_class"`` to opt out of the MMM-specific garden
gates (channels, ``beta_<channel>``, channel read-ops / compat tiers). It
overrides only ``_prepare_data`` (assemble the binary indicator matrix) and
``_build_model`` (the mixture graph), reusing ``fit`` / serialization / the
estimand engine. Its quantities of interest — **class sizes** (mixing
proportions) and the per-class item profiles — are surfaced as estimands and a
``class_profile_summary()`` table, the same way the CFA surfaces loadings.

Likelihood + identification
---------------------------
The discrete class labels are **integrated out** (no discrete latent sampling,
which NUTS can't do): each observation's log-likelihood is the log-mixture

    log p(yᵢ) = logsumexp_k [ log πₖ + Σⱼ yᵢⱼ·log θₖⱼ + (1−yᵢⱼ)·log(1−θₖⱼ) ]

added with a ``pm.Potential``. ``π`` (mixing proportions) is a softmax of an
**ordered** logit vector, which pins the class order by size and so resolves the
label-switching symmetry; ``θₖⱼ = P(itemⱼ = 1 | classₖ)`` are ``Beta`` priors.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pydantic import BaseModel, Field
from pymc.distributions.transforms import ordered

from mmm_framework.estimands.registry import latent_scalar
from mmm_framework.garden import CustomMMM


class LCAConfig(BaseModel):
    """Bespoke, settable configuration for :class:`BayesianLCA` (its CONFIG_SCHEMA)."""

    n_classes: int = Field(default=2, ge=2)
    #: Beta(a, b) prior on each item-endorsement probability θₖⱼ. Beta(1,1) is
    #: uniform; raise for stronger pull toward 0.5 (less separated classes).
    item_prior_a: float = Field(default=1.0, gt=0)
    item_prior_b: float = Field(default=1.0, gt=0)
    #: Spread of the ordered class-logit prior (controls how unequal class sizes
    #: may be a priori). Larger ⇒ more imbalanced segments allowed.
    class_logit_sigma: float = Field(default=1.5, gt=0)
    #: Indicators that aren't already 0/1 are thresholded at this value (``> t``);
    #: ``None`` assumes the columns are already binary.
    binarize_threshold: float | None = None

    model_config = {"extra": "forbid"}


class BayesianLCA(CustomMMM):
    """Latent class analysis as an oracle-compatible non-MMM garden model."""

    __garden_model_kind__ = "latent_class"

    #: Bespoke configuration (read via ``self.model_params``).
    CONFIG_SCHEMA = LCAConfig

    # -- data ----------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Read and binarize the indicator matrix from the role-tagged dataset
        (every measured column), then set the model-agnostic attributes via
        :meth:`CustomMMM._set_non_mmm_defaults`.

        ``self.dataset.observed()`` replaces the old ``pd.concat([y, X_media,
        X_controls])`` hack with one role-aware call (same columns, same order).
        """
        observed = self.dataset.observed()
        self.item_names = [str(c) for c in observed.columns]
        Y = observed.to_numpy(dtype=np.float64)

        thr = self.model_params.binarize_threshold
        if thr is not None:
            Y = (Y > float(thr)).astype(np.float64)
        elif not np.isin(np.unique(Y), (0.0, 1.0)).all():
            # Default binarization at the per-item median if not already 0/1.
            Y = (Y > np.median(Y, axis=0)).astype(np.float64)
        self.items = Y
        self.n_obs, self.n_items = Y.shape

        # Model-agnostic attributes the base contract / estimand engine read.
        self._set_non_mmm_defaults()

    # -- model ---------------------------------------------------------------

    def _build_model(self) -> pm.Model:
        cfg = self.model_params
        Y = self.items
        K, J = cfg.n_classes, self.n_items
        coords = {
            "obs": np.arange(self.n_obs),
            "klass": [f"C{k + 1}" for k in range(K)],
            "item": self.item_names,
        }
        with pm.Model(coords=coords) as model:
            # Mixing proportions via a softmax of an ORDERED logit vector — the
            # ordering pins class order by size and resolves label-switching.
            class_logits = pm.Normal(
                "class_logits",
                mu=0.0,
                sigma=cfg.class_logit_sigma,
                shape=K,
                transform=ordered,
                initval=np.linspace(-1.0, 1.0, K),
            )
            pi = pm.Deterministic(
                "class_sizes", pt.special.softmax(class_logits), dims="klass"
            )
            for k in range(K):  # named scalar class sizes -> estimand-addressable
                pm.Deterministic(f"class_size_{k + 1}", pi[k])

            # Per-class item-endorsement probabilities θ (K x J).
            theta = pm.Beta(
                "item_prob",
                alpha=cfg.item_prior_a,
                beta=cfg.item_prior_b,
                shape=(K, J),
                dims=("klass", "item"),
            )
            pm.Deterministic("class_profile", theta, dims=("klass", "item"))

            # Marginal mixture log-likelihood (class labels integrated out).
            Yt = pt.as_tensor_variable(Y)
            log_theta = pt.log(theta)  # (K, J)
            log_1m = pt.log1p(-theta)  # (K, J)
            # comp[i, k] = Σ_j y_ij·logθ_kj + (1−y_ij)·log(1−θ_kj)
            comp = pt.dot(Yt, log_theta.T) + pt.dot(1.0 - Yt, log_1m.T)  # (n_obs, K)
            weighted = comp + pt.log(pi)[None, :]
            logp = pt.logsumexp(weighted, axis=1)  # (n_obs,)
            pm.Potential("lca_loglik", pt.sum(logp))

            # Per-obs class responsibilities (posterior membership) for reporting.
            pm.Deterministic(
                "class_responsibility",
                pt.special.softmax(weighted, axis=1),
                dims=("obs", "klass"),
            )

        return model

    # -- estimands + reporting ----------------------------------------------

    def _default_estimands(self):
        """Class sizes are the headline estimands (the segment shares). Dynamic in
        ``n_classes``, so declared here rather than as a static class attribute."""
        K = self.model_params.n_classes
        return [
            latent_scalar(
                f"class_size_{k + 1}",
                var=f"class_size_{k + 1}",
                kind="class_size",
                units="proportion",
                causal_assumptions=f"Posterior share of the population in class C{k + 1}.",
            )
            for k in range(K)
        ]

    def class_profile_summary(self, hdi_prob: float = 0.94):
        """Per-(class, item) endorsement probability P(item=1 | class) — mean +
        HDI. The interpretable LCA output (the non-MMM analogue of the CFA's
        loadings table); also feeds the report's latent-structure section."""
        import arviz as az  # noqa: F401 - kept for other az uses
        from mmm_framework.utils.arviz_compat import hdi_dataset
        import pandas as pd

        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        prof = self._trace.posterior["class_profile"]
        mean = prof.mean(("chain", "draw")).values  # (K, J)
        hdi = hdi_dataset(self._trace, hdi_prob, var_names=["class_profile"])[
            "class_profile"
        ].values  # (K, J, 2)
        sizes = self._trace.posterior["class_sizes"].mean(("chain", "draw")).values
        rows = []
        for k in range(mean.shape[0]):
            for j, item in enumerate(self.item_names):
                rows.append(
                    {
                        "class": f"C{k + 1}",
                        "size": float(sizes[k]),
                        "item": item,
                        "prob": float(mean[k, j]),
                        "hdi_low": float(hdi[k, j, 0]),
                        "hdi_high": float(hdi[k, j, 1]),
                    }
                )
        return pd.DataFrame(rows)


GARDEN_MODEL = BayesianLCA


def synthetic_lca_panel(n: int = 600, seed: int = 11):
    """A :class:`PanelDataset` of 6 binary indicators with a KNOWN 2-class
    structure: a 35% class endorses items 1–3 (p=0.85) and rejects 4–6 (p=0.15),
    a 65% class does the reverse. Indicators are listed as kpi + ``media_channels``
    (the LCA treats every observed column as a binary item uniformly)."""
    import pandas as pd

    from mmm_framework.config import (
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset

    rng = np.random.default_rng(seed)
    true_sizes = np.array([0.35, 0.65])
    # profiles[class, item] = P(item = 1 | class)
    profiles = np.array(
        [
            [0.85, 0.85, 0.85, 0.15, 0.15, 0.15],  # class A (segment 1)
            [0.15, 0.15, 0.15, 0.85, 0.85, 0.85],  # class B (segment 2)
        ]
    )
    z = rng.choice(2, size=n, p=true_sizes)
    Y = (rng.random((n, 6)) < profiles[z]).astype(int)
    cols = [f"q{j + 1}" for j in range(6)]
    df = pd.DataFrame(Y, columns=cols)
    periods = pd.date_range("2021-01-04", periods=n, freq="W-MON")
    media = cols[1:]
    config = MFFConfig(
        kpi=KPIConfig(name=cols[0], dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD]) for c in media
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=df[cols[0]],
        X_media=df[media],
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=media,
            controls=None,
        ),
        index=periods,
        config=config,
    )
    return panel, true_sizes, profiles


if __name__ == "__main__":
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    panel, true_sizes, profiles = synthetic_lca_panel()
    print("Fitting BayesianLCA (MAP) on synthetic 2-class binary data…")
    mmm = BayesianLCA(
        panel,
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params={"n_classes": 2},
    )
    mmm.fit(method="map", random_seed=11)

    print(f"\nTrue class sizes (sorted): {np.sort(true_sizes)}")
    summary = mmm.class_profile_summary()
    print("\nRecovered class profiles  P(item=1 | class):")
    print(
        summary.pivot(index="item", columns="class", values="prob").round(2).to_string()
    )

    est = mmm.evaluate_estimands()  # class-size estimands
    print("\nClass-size estimands:")
    for name, r in est.items():
        print(f"  {name:14s} mean={r.mean:.3f}  ({r.status})")

    # HTML report: a non-MMM (latent-class) model auto-routes to the latent-
    # structure section (class profiles + class-size cards); channel sections off.
    import tempfile
    from pathlib import Path

    from mmm_framework.reporting import MMMReportGenerator

    gen = MMMReportGenerator(model=mmm)
    html = gen.render()
    assert 'id="factor-analysis"' in html, "report missing the latent-structure section"
    assert 'id="channel-roi"' not in html, "channel/ROI section should be gated off"
    report_path = gen.to_html(Path(tempfile.gettempdir()) / "bayesian_lca_report.html")
    print(f"\nHTML report written to {report_path} (latent-structure section).")
