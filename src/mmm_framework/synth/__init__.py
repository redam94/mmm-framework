"""Synthetic marketing worlds with known causal ground truth.

Originally built as the stress-test battery in ``tests/synth`` (which now
re-exports from here), these data-generating processes produce *realistic*
synthetic MMM data: confounded budgets, collinear channels, misspecified
carryover/saturation, structural breaks, data-entry defects, and geo panels
with heterogeneous effectiveness — each with a recorded causal answer key.

Entry points
------------
``dgp.SCENARIOS`` / ``dgp.build``
    National violation worlds (``"realistic"``, ``"clean"``,
    ``"unobserved_confounding"``, ...).
``dgp_geo.SCENARIOS`` / ``dgp_geo.build``
    Geography and geography x product panel worlds.
``generate_mff``
    Flatten any scenario to a Master Flat File dataset + JSON answer key.
"""

from . import dgp, dgp_geo
from .dgp import PRIORITY, SCENARIOS, Scenario, build
from .dgp_geo import SCENARIOS as GEO_SCENARIOS
from .dgp_geo import GeoScenario
from .mff import (
    MFF_COLUMNS,
    MIN_WEEKS,
    generate_mff,
    geo_scenario_to_mff,
    make_awareness_survey,
    scenario_to_mff,
    truth_summary,
)

__all__ = [
    "dgp",
    "dgp_geo",
    "Scenario",
    "GeoScenario",
    "SCENARIOS",
    "GEO_SCENARIOS",
    "PRIORITY",
    "build",
    "MFF_COLUMNS",
    "MIN_WEEKS",
    "generate_mff",
    "scenario_to_mff",
    "geo_scenario_to_mff",
    "truth_summary",
    "make_awareness_survey",
]
