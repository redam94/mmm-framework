"""The model matrix behind the graph-fingerprint contract (src-refactor PR 0.1).

An IMPORTABLE fixture module — PR 0.2 (serializer round-trip) and every Phase 4
PR import this registry rather than re-implementing their own case list, so the
two cannot drift. Nothing here is test-local: import it as

    sys.path.insert(0, "<repo>/tests/contracts")
    import model_matrix

``CASES`` maps case-name -> a zero-arg builder returning a **built, UNFITTED**
model (``BayesianMMM``-family, or a ``BaseExtendedMMM`` whose ``.model`` is the
``pm.Model``). For the names listed in ``REFUSAL_CASES`` the builder RAISES —
the refusal IS the contract, and the golden pins the exception type + message.

Probe worlds are the synthetic DGPs with fixed seeds (one per panel shape,
cached), so every case is deterministic:

- national: ``dgp.make_clean(seed=0, n_weeks=60)``
- geo: ``dgp_geo.make_geo_clean(seed=20, n_weeks=40)``
- geo x product: ``dgp_geo.make_geo_product(seed=22, n_weeks=40)``
- reach/frequency: ``dgp.make_reach_frequency(seed=15, n_weeks=60)``
- agent CSV: ``synth.mff.scenario_to_mff`` of the national world, materialized
  deterministically under ``graph_fingerprints/_data/`` (checked in; the
  builder regenerates the identical bytes if the file is missing).

Scenario objects are cached; each case builds a FRESH panel (``Scenario.panel()``
constructs a new ``MFFConfig`` each call) so per-case config mutations cannot
leak between cases.

``UNBUILDABLE`` records capabilities with NO reachable construction path (with
the evidence), so absence from ``CASES`` is explicit and golden-pinned. It is
empty today: every capability on the PR 0.1 must-add list was buildable.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable

CASES: dict[str, Callable[[], Any]] = {}

#: Cases whose builder RAISES by contract: name -> (exception type name,
#: message fragment). The golden JSON pins the full message; this metadata
#: lets the test (and any importer) know the call must not return a model.
REFUSAL_CASES: dict[str, tuple[str, str]] = {
    "binomial_refused": (
        "NotImplementedError",
        "does not fit the 'binomial' likelihood directly",
    ),
}

#: Capabilities with no reachable construction path (name -> reason with
#: evidence). Golden-pinned by the test so absence is explicit. Empty: every
#: capability on the must-add list had a construction path.
UNBUILDABLE: dict[str, str] = {}

#: Deterministic MFF CSV for the agent build_model(spec, csv) case.
AGENT_CSV = (
    Path(__file__).resolve().parent / "graph_fingerprints" / "_data" / "agent_world.csv"
)


def _case(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    def register(fn: Callable[[], Any]) -> Callable[[], Any]:
        assert name not in CASES, f"duplicate case {name!r}"
        CASES[name] = fn
        return fn

    return register


# ── shared probe worlds (one per panel shape, cached) ────────────────────────


@functools.lru_cache(maxsize=None)
def _national_scenario():
    from mmm_framework.synth import dgp

    return dgp.make_clean(seed=0, n_weeks=60)


@functools.lru_cache(maxsize=None)
def _geo_scenario():
    from mmm_framework.synth import dgp_geo

    return dgp_geo.make_geo_clean(seed=20, n_weeks=40)


@functools.lru_cache(maxsize=None)
def _geo_product_scenario():
    from mmm_framework.synth import dgp_geo

    return dgp_geo.make_geo_product(seed=22, n_weeks=40)


@functools.lru_cache(maxsize=None)
def _reach_frequency_scenario():
    from mmm_framework.synth import dgp

    return dgp.make_reach_frequency(seed=15, n_weeks=60)


def _national_panel():
    """A FRESH national panel (new MFFConfig every call — mutations don't leak)."""
    return _national_scenario().panel()


def _model(panel, model_config=None, trend_config=None):
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    return BayesianMMM(
        panel,
        model_config or ModelConfig(),
        trend_config or TrendConfig(type=TrendType.LINEAR),
    )


# ── base matrix: defaults, trend families, saturation, adstock ───────────────


@_case("default_national")
def default_national():
    return _model(_national_panel())


def _trend_case(trend_type):
    from mmm_framework.model import TrendConfig

    return _model(_national_panel(), trend_config=TrendConfig(type=trend_type))


def _register_trend_cases() -> None:
    from mmm_framework.model import TrendType

    for tt in TrendType:  # none, linear, piecewise, spline, gaussian_process
        CASES[f"trend_{tt.name.lower()}"] = functools.partial(_trend_case, tt)


def _saturation_case(name: str):
    from mmm_framework.config import SaturationConfig

    factory = getattr(SaturationConfig, name)
    panel = _national_panel()
    for ch in panel.config.media_channels:
        ch.saturation = factory()
    return _model(panel)


def _register_saturation_cases() -> None:
    # Every SaturationType, via its SaturationConfig factory (default priors).
    for name in ("logistic", "hill", "michaelis_menten", "tanh", "root", "none"):
        CASES[f"saturation_{name}"] = functools.partial(_saturation_case, name)


def _adstock_case(name: str):
    from mmm_framework.config import AdstockConfig

    factory = getattr(AdstockConfig, name)
    panel = _national_panel()
    for ch in panel.config.media_channels:
        ch.adstock = factory()
    return _model(panel)


def _register_adstock_cases() -> None:
    for name in ("geometric", "delayed", "weibull", "none"):
        CASES[f"adstock_{name}"] = functools.partial(_adstock_case, name)


_register_trend_cases()
_register_saturation_cases()
_register_adstock_cases()


# ── panel shapes ─────────────────────────────────────────────────────────────


@_case("geo_panel")
def geo_panel():
    return _model(_geo_scenario().panel())


@_case("geo_vary_media_by_geo")
def geo_vary_media_by_geo():
    from mmm_framework.config import HierarchicalConfig, ModelConfig

    return _model(
        _geo_scenario().panel(),
        ModelConfig(hierarchical=HierarchicalConfig(vary_media_by_geo=True)),
    )


@_case("geo_product_panel")
def geo_product_panel():
    return _model(_geo_product_scenario().panel())


# ── media-prior modes and explicit priors ────────────────────────────────────


@_case("media_prior_mode_coefficient")
def media_prior_mode_coefficient():
    from mmm_framework.config import ModelConfig

    return _model(_national_panel(), ModelConfig(media_prior_mode="coefficient"))


@_case("media_prior_mode_roi")
def media_prior_mode_roi():
    from mmm_framework.config import ModelConfig

    return _model(_national_panel(), ModelConfig(media_prior_mode="roi"))


@_case("explicit_media_control_priors")
def explicit_media_control_priors():
    from mmm_framework.config import PriorConfig
    from mmm_framework.config.enums import PriorType

    panel = _national_panel()
    panel.config.media_channels[0].coefficient_prior = PriorConfig.half_normal(
        sigma=0.75
    )
    panel.config.controls[0].coefficient_prior = PriorConfig(
        distribution=PriorType.NORMAL, params={"mu": 0.0, "sigma": 0.4}
    )
    return _model(panel)


@_case("grouped_media_priors")
def grouped_media_priors():
    from mmm_framework.config import ModelConfig

    panel = _national_panel()
    for ch in panel.config.media_channels:
        if ch.name in ("Search", "Social", "Display"):
            ch.parent_channel = "digital"
    return _model(panel, ModelConfig(use_grouped_media_priors=True))


# ── likelihood / specification / adstock-engine variants ─────────────────────


@_case("student_t_likelihood")
def student_t_likelihood():
    from mmm_framework.config import ModelConfig
    from mmm_framework.config.likelihood import LikelihoodConfig

    return _model(
        _national_panel(), ModelConfig(likelihood=LikelihoodConfig.student_t())
    )


@_case("multiplicative")
def multiplicative():
    from mmm_framework.config import ModelConfig
    from mmm_framework.config.enums import ModelSpecification

    return _model(
        _national_panel(),
        ModelConfig(specification=ModelSpecification.MULTIPLICATIVE),
    )


@_case("legacy_blend_adstock")
def legacy_blend_adstock():
    from mmm_framework.config import ModelConfig

    return _model(_national_panel(), ModelConfig(use_parametric_adstock=False))


@_case("binomial_refused")
def binomial_refused():
    """The refusal IS the contract: a binomial family with no trials wiring —
    the built-in additive model refuses at graph build (model/base.py,
    ``_build_observation``) rather than silently mis-fitting. Forcing ``.model``
    here makes the builder itself raise ``NotImplementedError``."""
    from mmm_framework.config import ModelConfig
    from mmm_framework.config.enums import LikelihoodFamily
    from mmm_framework.config.likelihood import LikelihoodConfig

    model = _model(
        _national_panel(),
        ModelConfig(likelihood=LikelihoodConfig(family=LikelihoodFamily.BINOMIAL)),
    )
    model.model  # noqa: B018 — force the lazy graph build; raises by contract
    return model


# ── the must-add list (each maps to an invariant with no other coverage) ─────


@_case("events")
def events():
    from mmm_framework.config import EventsConfig, EventSpec, ModelConfig

    cfg = EventsConfig(
        custom_events=[
            EventSpec(name="Launch", dates=["2021-06-21"], post_weeks=2, decay=0.5)
        ]
    )
    return _model(_national_panel(), ModelConfig(events=cfg))


def _add_controls(panel, columns: dict[str, "Any"]) -> None:
    """Append deterministic extra control columns to a national panel (the
    clean world carries only Price; some levers need more controls)."""
    from mmm_framework.config import ControlVariableConfig

    panel.X_controls = panel.X_controls.assign(**columns)
    panel.coords.controls = list(panel.coords.controls) + list(columns)
    dims = list(panel.config.controls[0].dimensions)
    panel.config.controls.extend(
        ControlVariableConfig(name=name, dimensions=dims) for name in columns
    )


@_case("price_promotions")
def price_promotions():
    import numpy as np

    from mmm_framework.config import ModelConfig, PriceConfig, PromoConfig

    panel = _national_panel()
    # A deterministic promo flag column so the promo lever has a variable to
    # promote (Price stays the price lever).
    rng = np.random.default_rng(123)
    promo = (rng.uniform(size=len(panel.index)) < 0.25).astype(float)
    _add_controls(panel, {"Promo": promo})
    return _model(
        panel,
        ModelConfig(
            price=PriceConfig(variable="Price", reference="median"),
            promotions=[PromoConfig(variable="Promo", adstock_lmax=4)],
        ),
    )


@_case("channel_interactions")
def channel_interactions():
    from mmm_framework.config import ChannelInteraction, ModelConfig

    return _model(
        _national_panel(),
        ModelConfig(
            channel_interactions=[
                ChannelInteraction(
                    channel_a="TV", channel_b="Search", expected_sign="positive"
                )
            ]
        ),
    )


@_case("control_selection_horseshoe")
def control_selection_horseshoe():
    """Selection is gated on >= 2 SELECTABLE controls (``_selection_active``,
    model/base.py) — with the clean world's single Price control a horseshoe
    config is a silent no-op and the fingerprint collides with the default.
    Extra deterministic controls make the selection prior actually wire."""
    import numpy as np

    from mmm_framework.config import ControlSelectionConfig, ModelConfig

    panel = _national_panel()
    rng = np.random.default_rng(456)
    n = len(panel.index)
    _add_controls(
        panel,
        {
            "Macro1": rng.normal(0.0, 1.0, n),
            "Macro2": rng.normal(0.0, 1.0, n),
            "Macro3": rng.normal(0.0, 1.0, n),
        },
    )
    return _model(
        panel,
        ModelConfig(control_selection=ControlSelectionConfig(method="horseshoe")),
    )


@_case("reach_frequency")
def reach_frequency():
    from mmm_framework.config import ModelConfig, ReachFrequencyConfig

    return _model(
        _reach_frequency_scenario().panel(),
        ModelConfig(
            reach_frequency=[
                ReachFrequencyConfig(channel="TV", frequency_column="Frequency")
            ]
        ),
    )


@_case("time_varying_betas")
def time_varying_betas():
    panel = _national_panel()
    panel.config.media_channels[0].time_varying = True  # TV drifts
    return _model(panel)


@_case("cpm_impressions_channel")
def cpm_impressions_channel():
    """An impressions-measured channel with a cpm cost constant. The golden is
    EXPECTED to equal ``default_national``: the measurement descriptor drives
    ROI/efficiency reporting, never the response graph — this case pins that
    a refactor does not start leaking it into the model."""
    from mmm_framework.config.enums import MeasurementUnit

    panel = _national_panel()
    ch = panel.config.media_channels[1]  # Search
    ch.measurement_unit = MeasurementUnit.IMPRESSIONS
    ch.cpm = 5.0
    return _model(panel)


@_case("confounder_controls")
def confounder_controls():
    from mmm_framework.config import CausalControlRole

    panel = _national_panel()
    panel.config.controls[0].causal_role = CausalControlRole.CONFOUNDER
    return _model(panel)


@_case("agent_spec_priors")
def agent_spec_priors():
    """The agent ``build_model(spec, csv)`` path with ``priors.*`` writes —
    both a media prior (adstock_alpha) and a control coefficient prior, the
    two spec routes that actually reach the graph."""
    from mmm_framework.agents.fitting import build_model

    sc = _national_scenario()
    _ensure_agent_csv()
    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "media_channels": [
            {
                "name": c,
                "adstock": {"type": "geometric"},
                "saturation": {"type": "hill"},
            }
            for c in sc.channels
        ],
        "control_variables": [{"name": c} for c in sc.controls.columns],
        "priors": {
            "media": {
                "TV": {
                    "adstock_alpha": {
                        "distribution": "beta",
                        "params": {"alpha": 1, "beta": 3},
                    }
                }
            },
            "controls": {
                "Price": {
                    "coefficient": {
                        "distribution": "normal",
                        "params": {"mu": 0.0, "sigma": 0.4},
                    }
                }
            },
        },
    }
    return build_model(spec, str(AGENT_CSV))


@_case("garden_subclass")
def garden_subclass():
    """A Model Garden ``CustomMMM`` subclass via the ``garden_ref`` spec path
    (PR 0.2's serializer contract needs it: load() must reconstruct the SAME
    subclass, never quietly demote a bespoke model to BayesianMMM). A garden
    identity is a label, not a structural change — the graph fingerprints like
    any other model built from the same spec."""
    from mmm_framework.agents.fitting import build_model

    sc = _national_scenario()
    _ensure_agent_csv()
    src = AGENT_CSV.parent / "garden_roundtrip.py"
    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "media_channels": [
            {
                "name": c,
                "adstock": {"type": "geometric"},
                "saturation": {"type": "hill"},
            }
            for c in sc.channels
        ],
        "control_variables": [{"name": c} for c in sc.controls.columns],
        "garden_ref": {
            "name": "contract-roundtrip",
            "version": 1,
            "source_path": str(src),
            "class_name": "ContractRoundTripMMM",
            "contract_version": "1.0",
        },
    }
    return build_model(spec, str(AGENT_CSV))


def _ensure_agent_csv() -> None:
    """Materialize the deterministic agent MFF CSV (identical bytes each time:
    the scenario is seed-pinned and ``scenario_to_mff`` is pure)."""
    if AGENT_CSV.exists():
        return
    from mmm_framework.synth import mff as synth_mff

    AGENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    synth_mff.scenario_to_mff(_national_scenario()).to_csv(AGENT_CSV, index=False)


# ── BaseExtendedMMM configurations (required before PR 6) ────────────────────


def _extension_inputs():
    import numpy as np
    import pandas as pd

    n = 52
    rng = np.random.default_rng(0)
    media = np.abs(rng.normal(100, 20, (n, 2)))
    idx = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    r = np.random.default_rng(1)
    aware = 40 + 0.3 * media[:, 0] + r.normal(0, 4, n)
    sales = 1000 + 4 * aware + 2 * media[:, 1] + r.normal(0, 40, n)
    return n, media, idx, sales, r


@_case("extension_nested")
def extension_nested():
    from mmm_framework.mmm_extensions.config import (
        MediatorConfig,
        MediatorType,
        NestedModelConfig,
    )
    from mmm_framework.mmm_extensions.models import NestedMMM

    _, media, idx, sales, _r = _extension_inputs()
    cfg = NestedModelConfig(
        mediators=(
            MediatorConfig(name="Awareness", mediator_type=MediatorType.FULLY_LATENT),
        )
    )
    model = NestedMMM(media, sales, ["TV", "Digital"], cfg, index=idx)
    model.model  # noqa: B018 — force the lazy pm.Model build (unfitted)
    return model


@_case("extension_multivariate")
def extension_multivariate():
    from mmm_framework.mmm_extensions.builders import (
        MultivariateModelConfigBuilder,
        OutcomeConfigBuilder,
        cannibalization_effect,
    )
    from mmm_framework.mmm_extensions.models import MultivariateMMM

    n, media, _idx, _sales, r = _extension_inputs()
    cfg = (
        MultivariateModelConfigBuilder()
        .add_outcome(
            OutcomeConfigBuilder("sales_a")
            .with_positive_media_effects(sigma=0.5)
            .build()
        )
        .add_outcome(
            OutcomeConfigBuilder("sales_b")
            .with_positive_media_effects(sigma=0.5)
            .build()
        )
        .add_cross_effect(cannibalization_effect(source="sales_b", target="sales_a"))
        .build()
    )
    outcomes = {
        "sales_a": 1000 + r.normal(0, 100, n),
        "sales_b": 800 + r.normal(0, 80, n),
    }
    model = MultivariateMMM(media, outcomes, ["TV", "Digital"], cfg)
    model.model  # noqa: B018 — force the lazy pm.Model build (unfitted)
    return model
