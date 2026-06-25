"""Tests for the pre-fit data-quality gate (Phase 1 / D1, QW-1).

``agents.fitting.run_data_quality_gate`` wires the EDA validator into the fit
path: error-tier issues (e.g. date gaps that silently shift adstock carryover)
block a fit; warn-tier issues never block; an explicit ``skip_quality_gate``
overrides; and a gate failure must not itself break fitting.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mmm_framework.agents.fitting import run_data_quality_gate


def _write_mff(path, periods, *, drop_idx=None):
    rows = []
    for i, p in enumerate(periods):
        if drop_idx is not None and i == drop_idx:
            continue  # create a gap in the otherwise-contiguous cadence
        iso = p.strftime("%Y-%m-%d")
        rows.append({"Period": iso, "VariableName": "Sales", "VariableValue": 100 + i})
        rows.append({"Period": iso, "VariableName": "TV", "VariableValue": 10 + i})
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_gate_passes_clean_dataset(tmp_path):
    periods = pd.date_range("2021-01-04", periods=16, freq="W-MON")
    path = _write_mff(tmp_path / "clean.csv", periods)
    # Should not raise.
    run_data_quality_gate(path, {})


def test_gate_blocks_on_date_gap(tmp_path):
    periods = pd.date_range("2021-01-04", periods=16, freq="W-MON")
    path = _write_mff(tmp_path / "gap.csv", periods, drop_idx=8)
    with pytest.raises(ValueError, match="data-quality"):
        run_data_quality_gate(path, {})


def test_gate_respects_skip_override(tmp_path):
    periods = pd.date_range("2021-01-04", periods=16, freq="W-MON")
    path = _write_mff(tmp_path / "gap.csv", periods, drop_idx=8)
    # Explicit override -> no raise despite the gap.
    run_data_quality_gate(path, {"skip_quality_gate": True})


def test_gate_failure_does_not_block_fitting(tmp_path):
    # A nonexistent path makes the gate machinery fail; it must warn, not raise,
    # so a gate bug never takes down fitting (loader-level checks still apply).
    missing = str(tmp_path / "nope.csv")
    with pytest.warns(UserWarning, match="Data-quality gate skipped"):
        run_data_quality_gate(missing, {})
