"""Tests for the flexible role-tagged :class:`Dataset` container (Phase A1).

Covers: the ``PanelDataset`` ↔ ``Dataset`` round-trip, the MMM read-surface views,
the ``DatasetSchema``/``DatasetRole`` mapping, the ``observed()`` indicator frame,
and the model-side ``DATASET_SCHEMA`` / ``REQUIRED_ROLES`` / ``dataset_capabilities``
scaffolding (mirroring the ``CONFIG_SCHEMA`` extensibility pattern).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework import BayesianMMM, Dataset, PanelDataset
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


def _make_panel(n: int = 12, seed: int = 0) -> PanelDataset:
    """A tiny national MMM panel: KPI=Sales, media=[TV, Digital], control=[Price]."""
    mff = MFFConfig(
        kpi=KPIConfig(name="Sales"),
        media_channels=[
            MediaChannelConfig(name="TV"),
            MediaChannelConfig(name="Digital"),
        ],
        controls=[ControlVariableConfig(name="Price")],
    )
    periods = pd.date_range("2023-01-01", periods=n, freq="W")
    rng = np.random.default_rng(seed)
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
# Adapters + views
# --------------------------------------------------------------------------- #
class TestPanelRoundTrip:
    def test_from_panel_views_match(self):
        panel = _make_panel()
        ds = panel.as_dataset()
        assert isinstance(ds, Dataset)
        assert np.allclose(ds.y.values, panel.y.values)
        assert list(ds.X_media.columns) == list(panel.X_media.columns)
        assert np.allclose(ds.X_media.values, panel.X_media.values)
        assert np.allclose(ds.X_controls.values, panel.X_controls.values)
        assert ds.n_obs == panel.n_obs
        assert ds.n_channels == panel.n_channels
        assert ds.n_controls == panel.n_controls

    def test_as_panel_round_trips(self):
        panel = _make_panel()
        panel2 = panel.as_dataset().as_panel()
        assert np.allclose(panel2.y.values, panel.y.values)
        assert list(panel2.X_media.columns) == list(panel.X_media.columns)
        assert np.allclose(panel2.X_media.values, panel.X_media.values)
        assert np.allclose(panel2.X_controls.values, panel.X_controls.values)

    def test_observed_matches_concat_order(self):
        """observed() must equal the legacy [y, X_media, X_controls] concat."""
        panel = _make_panel()
        ds = panel.as_dataset()
        legacy = pd.concat(
            [panel.y.to_frame(), panel.X_media, panel.X_controls], axis=1
        )
        observed = ds.observed()
        assert list(observed.columns) == list(legacy.columns)
        assert np.allclose(observed.to_numpy(np.float64), legacy.to_numpy(np.float64))


# --------------------------------------------------------------------------- #
# Schema / roles
# --------------------------------------------------------------------------- #
class TestDatasetSchema:
    def test_from_mff_role_mapping(self):
        panel = _make_panel()
        schema = DatasetSchema.from_mff(panel.config)
        assert schema.target_names == ["Sales"]
        assert schema.predictor_names == ["TV", "Digital"]
        assert schema.control_names == ["Price"]
        assert schema.indicator_names == []

    def test_to_mff_preserves_mmm_roles(self):
        schema = DatasetSchema.from_mff(_make_panel().config)
        mff = schema.to_mff()
        assert mff.kpi.name == "Sales"
        assert mff.media_names == ["TV", "Digital"]
        assert mff.control_names == ["Price"]

    def test_duplicate_binding_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            DatasetSchema(
                bindings=[
                    RoleBinding(name="X", role=DatasetRole.INDICATOR),
                    RoleBinding(name="X", role=DatasetRole.INDICATOR),
                ]
            )

    def test_indicator_schema_observed(self):
        """A natively indicator-tagged Dataset returns those columns from observed()."""
        n = 8
        table = pd.DataFrame(
            {"q1": np.arange(n, dtype=float), "q2": np.arange(n, dtype=float)[::-1]},
            index=pd.RangeIndex(n),
        )
        schema = DatasetSchema(
            bindings=[
                RoleBinding(name="q1", role=DatasetRole.INDICATOR),
                RoleBinding(name="q2", role=DatasetRole.INDICATOR),
            ]
        )
        from mmm_framework.data_loader import PanelCoordinates

        coords = PanelCoordinates(
            periods=pd.date_range("2023-01-01", periods=n, freq="W")
        )
        ds = Dataset(table=table, schema=schema, index=table.index, coords=coords)
        assert list(ds.observed().columns) == ["q1", "q2"]
        assert ds.columns_for(DatasetRole.INDICATOR) == ["q1", "q2"]


# --------------------------------------------------------------------------- #
# Model-side declaration
# --------------------------------------------------------------------------- #
class TestModelDatasetContract:
    def test_mmm_accepts_panel_and_dataset(self):
        panel = _make_panel()
        cfg = ModelConfig()
        m_panel = BayesianMMM(panel, cfg)
        m_ds = BayesianMMM(panel.as_dataset(), cfg)
        assert isinstance(m_panel.panel, PanelDataset)
        assert isinstance(m_panel.dataset, Dataset)
        # A Dataset input still yields a PanelDataset view + identical channels.
        assert isinstance(m_ds.panel, PanelDataset)
        assert m_panel.channel_names == m_ds.channel_names == ["TV", "Digital"]
        assert np.allclose(m_panel.y_raw, m_ds.y_raw)

    def test_base_model_declares_no_requirements(self):
        assert BayesianMMM.DATASET_SCHEMA is None
        assert BayesianMMM.REQUIRED_ROLES == ()
        assert BayesianMMM.REQUIRED_DATASET_CAPABILITIES == ()

    def test_required_roles_gate_raises(self):
        """A model requiring a role the data lacks fails fast with a clear error."""

        class IndicatorOnlyModel(BayesianMMM):
            REQUIRED_ROLES = (DatasetRole.INDICATOR,)

        with pytest.raises(ValueError, match="requires dataset roles"):
            IndicatorOnlyModel(_make_panel(), ModelConfig())

    def test_required_roles_gate_passes_for_present_roles(self):
        class TargetPredictorModel(BayesianMMM):
            REQUIRED_ROLES = (DatasetRole.TARGET, DatasetRole.PREDICTOR)

        m = TargetPredictorModel(_make_panel(), ModelConfig())
        assert m.channel_names == ["TV", "Digital"]

    def test_dataset_capabilities(self):
        panel = _make_panel()
        caps = BayesianMMM.dataset_capabilities(panel.as_dataset())
        # National panel, no indicators/trials → no flags.
        assert caps == set()
