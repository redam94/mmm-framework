"""Tests for Phase A2: threading the dataset schema through the serializer,
the garden manifest (AST detection), and the data-fit advisory warnings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_framework import BayesianMMM
from mmm_framework.config import (
    ControlVariableConfig,
    DatasetRole,
    DatasetSchema,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.config.dataset import RoleBinding
from mmm_framework.data_loader import load_mff


def _make_panel(n: int = 10):
    mff = MFFConfig(
        kpi=KPIConfig(name="Sales"),
        media_channels=[
            MediaChannelConfig(name="TV"),
            MediaChannelConfig(name="Digital"),
        ],
        controls=[ControlVariableConfig(name="Price")],
    )
    periods = pd.date_range("2023-01-01", periods=n, freq="W")
    rng = np.random.default_rng(1)
    rows = []
    for p in periods:
        for var in ["Sales", "TV", "Digital", "Price"]:
            rows.append(
                {
                    "Period": p.strftime("%Y-%m-%d"),
                    "Geography": "National",
                    "Product": "All",
                    "Campaign": "All",
                    "Outlet": "All",
                    "Creative": "All",
                    "VariableName": var,
                    "VariableValue": float(rng.uniform(1.0, 100.0)),
                }
            )
    return load_mff(pd.DataFrame(rows), mff)


# --------------------------------------------------------------------------- #
# Serializer round-trip (no fit required — exercise _collect_metadata directly)
# --------------------------------------------------------------------------- #
def test_collect_metadata_records_dataset_schema():
    from mmm_framework.serialization import MMMSerializer

    model = BayesianMMM(_make_panel(), ModelConfig())
    meta = MMMSerializer._collect_metadata(model)
    assert "dataset_schema" in meta
    assert "dataset_schema_version" in meta
    roles = {b["name"]: b["role"] for b in meta["dataset_schema"]["bindings"]}
    assert roles == {
        "Sales": "target",
        "TV": "predictor",
        "Digital": "predictor",
        "Price": "control",
    }


def test_retag_round_trip():
    ds = _make_panel().as_dataset()
    saved = ds.schema.model_dump(mode="json")
    validated = DatasetSchema.model_validate(saved)
    ds2 = ds.retag(validated)
    assert ds2.schema.target_names == ["Sales"]
    assert np.allclose(
        ds2.observed().to_numpy(np.float64), ds.observed().to_numpy(np.float64)
    )


def test_retag_rejects_unknown_columns():
    ds = _make_panel().as_dataset()
    bad = DatasetSchema(
        bindings=[RoleBinding(name="NotAColumn", role=DatasetRole.INDICATOR)]
    )
    try:
        ds.retag(bad)
    except ValueError as e:
        assert "not present" in str(e)
    else:  # pragma: no cover
        raise AssertionError("retag should reject names absent from the table")


# --------------------------------------------------------------------------- #
# Garden manifest — AST detection of the data contract
# --------------------------------------------------------------------------- #
def test_static_dataset_requirements_reads_required_roles():
    from mmm_framework.agents.garden_registry import static_dataset_requirements

    source = (
        "from mmm_framework.garden import CustomMMM\n"
        "from mmm_framework.config.roles import DatasetRole\n"
        "class M(CustomMMM):\n"
        "    REQUIRED_ROLES = (DatasetRole.TARGET, DatasetRole.PREDICTOR)\n"
        "    REQUIRED_DATASET_CAPABILITIES = ('GEO_PANEL',)\n"
    )
    out = static_dataset_requirements(source, "M")
    assert out["required_roles"] == ["target", "predictor"]
    assert out["required_capabilities"] == ["GEO_PANEL"]


def test_static_dataset_requirements_empty_when_undeclared():
    from mmm_framework.agents.garden_registry import static_dataset_requirements

    source = (
        "from mmm_framework.garden import CustomMMM\n"
        "class M(CustomMMM):\n"
        "    __garden_model_kind__ = 'cfa'\n"
    )
    assert static_dataset_requirements(source, "M") == {}


# --------------------------------------------------------------------------- #
# Advisory data-fit warnings
# --------------------------------------------------------------------------- #
def test_garden_schema_warnings_indicator_role():
    from mmm_framework.agents.tools import _garden_schema_warnings

    warn = _garden_schema_warnings({"required_roles": ["indicator"]}, {}, None)
    assert "INDICATOR" in warn
    assert "ROI/budget" in warn


def test_garden_schema_warnings_trials_capability():
    from mmm_framework.agents.tools import _garden_schema_warnings

    warn = _garden_schema_warnings(
        {"required_capabilities": ["HAS_TRIALS"]}, {"media_channels": ["TV"]}, None
    )
    assert "trials" in warn.lower()


def test_garden_schema_warnings_empty_for_plain_mmm():
    from mmm_framework.agents.tools import _garden_schema_warnings

    assert (
        _garden_schema_warnings({}, {"media_channels": ["TV", "Digital"]}, None) == ""
    )
