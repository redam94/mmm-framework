"""update_model_setting's consumed-paths registry, extended beyond ``priors.*``.

``unconsumed_spec_path`` must reject any model_spec write that ``build_model``
would silently drop (an unread key) or silently default (a typo'd enum value),
and accept every path the builder actually consumes. This pins the registry so
a new spec read without a registry entry — or vice versa — fails a test instead
of shipping the accept-then-no-op bug class again (see the 2026-07-02
"agent-set priors silently ignored" fix).
"""

import pytest

from mmm_framework.agents.fitting import unconsumed_spec_path

SPEC = {
    "kpi": "Sales",
    "media_channels": [
        {"name": "TV", "adstock": {"type": "geometric", "l_max": 8}},
        {"name": "Digital"},
    ],
    "control_variables": [{"name": "price"}],
}


def check(path: str, value):
    return unconsumed_spec_path(path.split("."), value, SPEC)


ACCEPTED = [
    ("kpi", "Revenue"),
    ("kpi_level", "geo"),
    ("kpi_level", "National"),  # canonicalized before storage by the tool
    ("time_granularity", "daily"),
    ("media_prior_mode", "coefficient"),
    ("skip_quality_gate", True),
    ("inference.draws", 2000),
    ("inference.method", "map"),
    ("inference.method", "advi"),
    ("inference.method", "fullrank_advi"),
    ("inference.method", "pathfinder"),
    ("inference.method", "NUTS"),  # canonicalized case-insensitively
    ("inference.metrics_draws", 0),
    ("trend.type", "piecewise"),
    ("trend.n_changepoints", 10),
    ("seasonality.yearly", 4),
    ("likelihood.family", "student_t"),
    ("likelihood.params", {"nu": 5}),
    ("media_channels.TV.adstock.type", "weibull"),
    ("media_channels.TV.adstock.l_max", 13),
    ("media_channels.TV.saturation.type", "logistic"),
    ("media_channels.TV.measurement_unit", "impressions"),
    ("media_channels.TV.cpm", 5.5),
    ("media_channels.TV.cpc", 0.8),
    ("media_channels.TV.spend_column", "TV_spend"),
    ("control_variables.price.role", "confounder"),
    # free-form: validated downstream at build time
    ("estimands", ["contribution_roi"]),
    ("garden_ref", {"org": "acme", "name": "custom"}),
    ("model_params", {"number_of_trials": 500}),
    ("dataset", {"path": "d.csv"}),
    ("experiments", []),
    ("experiment_ids", ["e1"]),
    # priors.* delegates to the existing registry
    ("priors.media.TV.roi", {"median": 1.2, "sigma": 0.6}),
    ("priors.intercept.sigma", 0.3),
]


@pytest.mark.parametrize("path,value", ACCEPTED, ids=[p for p, _ in ACCEPTED])
def test_consumed_paths_accepted(path, value):
    assert check(path, value) is None


REJECTED = [
    # unknown top-level keys (silently dropped today)
    ("vary_media_by_geo", True, "never reads top-level"),
    ("sampler", "numpyro", "never reads top-level"),
    # typo'd subkeys
    ("inference.samples", 2000, "inference"),
    ("trend.knots", 5, "trend"),
    ("seasonality.quarterly", 2, "seasonality"),
    ("likelihood.sigma", 1.0, "likelihood"),
    # enum values that silently fall back to a default at build time
    ("time_granularity", "dailly", "silently fall back"),
    ("kpi_level", "regional", "silently fall back"),
    ("media_prior_mode", "rio", "silently fall back"),
    ("trend.type", "quadratic", "trend"),
    # a typo'd fit method would only fail at fit time — reject up front
    ("inference.method", "pathfnder", "fit method"),
    ("inference.method", "vi", "fit method"),
    ("media_channels.TV.adstock.type", "weibul", "geometric"),
    ("media_channels.TV.saturation.type", "hilll", "hill"),
    ("media_channels.TV.measurement_unit", "views", "measurement unit"),
    # unknown channel / control names (typo detection)
    ("media_channels.Radio.adstock.type", "geometric", "unknown media channel"),
    ("control_variables.weather.role", "confounder", "unknown control"),
    # per-item fields the builder never reads
    ("media_channels.TV.budget", 1000, "never reads it"),
    ("control_variables.price.allow_negative", False, "priors.controls"),
    # priors delegation still rejects unread prior paths
    ("priors.intercept.stdev", 0.3, "intercept"),
    ("priors.gamma", 1.0, "set a specific prior"),
]


@pytest.mark.parametrize(
    "path,value,fragment", REJECTED, ids=[p for p, _, _ in REJECTED]
)
def test_unconsumed_paths_rejected(path, value, fragment):
    err = check(path, value)
    assert err is not None, f"{path} should have been rejected"
    assert fragment.lower() in err.lower(), err


def test_whole_dict_writes_validate_their_keys():
    # Setting an entire object validates each contained leaf, like the
    # priors registry does.
    assert check("trend", {"type": "spline", "n_knots": 7}) is None
    err = check("trend", {"type": "spline", "knots": 7})
    assert err is not None and "knots" in err


def test_channel_name_check_skipped_when_spec_has_no_channels():
    # Mirrors unconsumed_prior_path: an empty spec can't typo-check names.
    err = unconsumed_spec_path(
        ["media_channels", "TV", "adstock", "l_max"], 6, {"media_channels": []}
    )
    assert err is None
