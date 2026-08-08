"""Serializer round-trip contract over the model matrix (src-refactor PR 0.2).

``serialization.py`` writes instance attributes directly at 13 sites — 15
distinct attributes, 11 by assignment and 4 by in-place container mutation.
Any state that moves behind a collaborator in Phase 4 breaks ``load``, and the
pre-existing coverage was mostly ``Mock*`` unit tests (37 mock occurrences
against 2 real ``BayesianMMM`` constructions). This is the real gate:

* a REAL save → load → ``predict()`` round-trip over the fingerprint matrix's
  national, geo, roi-mode, extension and garden-subclass cases;
* ``predict()`` equality at **rtol=0** — the loaded model must reproduce the
  fitted model's predictions bit-for-bit, not approximately;
* ``metadata.json`` **field-set** equality against a checked-in snapshot
  (``tests/contracts/serializer_metadata_fields.json``), so an attribute that
  silently stops being persisted fails CI with the field named. Regeneration
  only behind ``MMM_REGEN_SERIALIZER_FIELDS=1``.
* the garden case must load back as the SAME subclass — a bespoke model
  quietly demoted to ``BayesianMMM`` on load is an identity bug, not a detail.

Fits are MAP (seconds each); the file is unmarked (fast tier) per the Phase 0
rule that a safety net gating nothing gates nothing.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "contracts"))

FIELDS_PATH = ROOT / "contracts" / "serializer_metadata_fields.json"
REGEN = os.environ.get("MMM_REGEN_SERIALIZER_FIELDS") == "1"

#: The spec's named coverage: national, geo, roi-mode, garden subclass, and
#: two BaseExtendedMMM configurations.
ROUNDTRIP_CASES = [
    "default_national",
    "geo_panel",
    "media_prior_mode_roi",
    "garden_subclass",
    "extension_nested",
    "extension_multivariate",
]


def _fit_map(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            model.fit(method="map", random_seed=7)
    return model


@pytest.fixture(scope="module")
def fitted_cases():
    import model_matrix as M

    out = {}
    for name in ROUNDTRIP_CASES:
        out[name] = _fit_map(M.CASES[name]())
    return out


def _roundtrip(model, tmp_path, name):
    from mmm_framework.serialization import MMMSerializer

    save_dir = tmp_path / name
    MMMSerializer.save(model, str(save_dir))
    meta = json.loads((save_dir / "metadata.json").read_text())
    panel = getattr(model, "panel", None)
    loaded = MMMSerializer.load(str(save_dir), panel)
    return loaded, meta


class TestRoundTrip:
    @pytest.mark.parametrize("name", ROUNDTRIP_CASES)
    def test_save_load_predict_bit_identical(self, fitted_cases, tmp_path, name):
        """The number a report renders tomorrow (from a loaded model) must be
        THE number the fit produced today — rtol=0, not 'close'."""
        model = fitted_cases[name]
        loaded, _meta = _roundtrip(model, tmp_path, name)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = model.predict(random_seed=11)
            b = loaded.predict(random_seed=11)

        pred_a = np.asarray(a.y_pred_mean, dtype=float)
        pred_b = np.asarray(b.y_pred_mean, dtype=float)
        assert pred_a.shape == pred_b.shape
        both_nan = np.isnan(pred_a) & np.isnan(pred_b)
        assert np.array_equal(pred_a[~both_nan], pred_b[~both_nan]), (
            f"{name}: loaded-model predictions differ from the fitted "
            f"model's (max abs diff "
            f"{np.nanmax(np.abs(pred_a - pred_b)):.3e}); the serializer "
            "dropped or transformed state"
        )

    def test_garden_subclass_identity_survives_load(self, fitted_cases, tmp_path):
        """load() must reconstruct the bespoke class, not the base."""
        model = fitted_cases["garden_subclass"]
        loaded, meta = _roundtrip(model, tmp_path, "garden_identity")
        assert (
            type(loaded).__name__ == "ContractRoundTripMMM"
        ), f"garden model demoted to {type(loaded).__name__} on load"
        assert type(loaded).__name__ == type(model).__name__


class TestMetadataFieldSet:
    def test_metadata_field_sets_match_the_snapshot(self, fitted_cases, tmp_path):
        """Every metadata.json key, per case, against the checked-in snapshot.

        Additions fail until recorded (a new persisted field is a contract
        event); removals fail until removed deliberately (a field that stops
        being persisted breaks whoever reads it). Regenerate with
        MMM_REGEN_SERIALIZER_FIELDS=1.
        """
        observed = {}
        for name in ROUNDTRIP_CASES:
            _loaded, meta = _roundtrip(fitted_cases[name], tmp_path, f"meta_{name}")
            observed[name] = sorted(meta.keys())

        if REGEN or not FIELDS_PATH.exists():
            FIELDS_PATH.write_text(json.dumps(observed, indent=2) + "\n")
            if REGEN:
                pytest.skip("snapshot regenerated; rerun without the env var")

        snapshot = json.loads(FIELDS_PATH.read_text())
        assert set(observed) == set(
            snapshot
        ), f"case set drifted: {sorted(set(observed) ^ set(snapshot))}"
        problems = []
        for name in sorted(observed):
            added = sorted(set(observed[name]) - set(snapshot[name]))
            removed = sorted(set(snapshot[name]) - set(observed[name]))
            if added:
                problems.append(f"{name}: NEW metadata fields {added}")
            if removed:
                problems.append(f"{name}: metadata fields GONE {removed}")
        assert not problems, (
            "metadata.json field sets drifted from "
            "tests/contracts/serializer_metadata_fields.json:\n  "
            + "\n  ".join(problems)
            + "\n(deliberate change? MMM_REGEN_SERIALIZER_FIELDS=1 and review "
            "the diff)"
        )
