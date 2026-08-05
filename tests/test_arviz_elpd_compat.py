"""ELPD field access across the arviz 1.x generic rename.

arviz 1.x made ``ELPDData`` one container for every information criterion, so
``elpd_loo`` / ``p_loo`` / ``loo_i`` became ``elpd`` / ``p`` / ``elpd_i``. Both
call sites that read the old spellings sat inside a broad ``except``, so the
rename did not crash anything — it made LOO **silently missing**:

* ``validation/validator.py`` logged "LOO-CV computation failed" on every
  validation run, and model comparison came back empty.
* ``validation/spec_curve.py`` logged "spec-curve LOO failed" for every spec, so
  ``compute_loo=True`` produced ``SpecFit.loo is None`` throughout.

Neither had a test, which is why both survived the arviz upgrade. These pin the
shim against both spellings so a future rename fails loudly here instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.utils.arviz_compat import (
    elpd_field,
    elpd_pareto_k_threshold,
    elpd_scalar,
    has_waic,
)


class _Modern:
    """arviz >= 1.0: generic field names, one container per criterion."""

    elpd = -68.16
    p = 16.6
    se = 7.6
    elpd_i = np.arange(5, dtype=float)
    pareto_k = np.array([0.1, 0.2, 0.9, 0.3, 0.95])
    good_k = 0.697


class _Legacy:
    """arviz < 1.0: metric-specific field names."""

    elpd_loo = -68.16
    p_loo = 16.6
    se = 7.6
    loo_i = np.arange(5, dtype=float)
    pareto_k = np.array([0.1, 0.2, 0.9, 0.3, 0.95])


class _LegacyWaic:
    elpd_waic = -70.0
    p_waic = 18.0
    se = 8.0
    waic_i = np.arange(3, dtype=float)


class TestBothSpellings:
    @pytest.mark.parametrize("obj", [_Modern(), _Legacy()])
    def test_scalars_read_the_same(self, obj):
        assert elpd_scalar(obj, "elpd") == pytest.approx(-68.16)
        assert elpd_scalar(obj, "p") == pytest.approx(16.6)
        assert elpd_scalar(obj, "se") == pytest.approx(7.6)

    @pytest.mark.parametrize("obj", [_Modern(), _Legacy()])
    def test_pointwise_reads_the_same(self, obj):
        assert elpd_field(obj, "pointwise").shape == (5,)

    def test_waic_spellings_resolve_too(self):
        assert elpd_scalar(_LegacyWaic(), "elpd") == pytest.approx(-70.0)
        assert elpd_scalar(_LegacyWaic(), "p") == pytest.approx(18.0)
        assert elpd_field(_LegacyWaic(), "pointwise").shape == (3,)


class TestAbsence:
    def test_missing_field_is_none_not_zero(self):
        """The distinction the caller needs: "this arviz does not expose it" is
        not "it is zero", and a zero elpd would silently rank models."""

        class _Bare:
            pass

        assert elpd_scalar(_Bare(), "elpd") is None
        assert elpd_field(_Bare(), "pointwise") is None
        assert elpd_field(_Bare(), "elpd", default="sentinel") == "sentinel"

    def test_unknown_field_name_raises(self):
        with pytest.raises(KeyError, match="Unknown ELPD field"):
            elpd_field(_Modern(), "not_a_field")

    def test_zero_dim_containers_unwrap(self):
        """arviz returns several of these as 0-d DataArrays, not floats."""

        class _Wrapped:
            elpd = np.array(-12.5)

        assert elpd_scalar(_Wrapped(), "elpd") == pytest.approx(-12.5)


class TestParetoKThreshold:
    def test_uses_the_fits_own_good_k(self):
        """arviz 1.x computes a sample-size-dependent threshold. At small draw
        counts the honest bar is BELOW the historical 0.7 constant, so hard-
        coding 0.7 undercounts bad observations."""
        assert elpd_pareto_k_threshold(_Modern()) == pytest.approx(0.697)

    def test_falls_back_when_absent(self):
        assert elpd_pareto_k_threshold(_Legacy()) == pytest.approx(0.7)
        assert elpd_pareto_k_threshold(_Legacy(), 0.5) == pytest.approx(0.5)

    def test_threshold_changes_the_bad_k_count(self):
        k = _Modern.pareto_k
        assert int((k > elpd_pareto_k_threshold(_Modern())).sum()) == 2
        assert int((k > 0.99).sum()) == 0


class TestWaicAvailability:
    def test_reports_what_this_arviz_actually_has(self):
        """arviz 1.x removed WAIC outright in favour of PSIS-LOO. There is no
        shim for a metric that no longer exists; callers must say so rather than
        report a blank or quietly substitute LOO."""
        import arviz as az

        assert has_waic() is hasattr(az, "waic")


class TestAgainstRealArviz:
    """The shim has to work on the object arviz actually returns, not just on
    the stand-ins above — which is the gap the original bug lived in."""

    def test_reads_a_real_loo_result(self):
        az = pytest.importorskip("arviz")
        data = az.load_arviz_data("centered_eight")
        loo = az.loo(data, pointwise=True)

        elpd = elpd_scalar(loo, "elpd")
        assert elpd is not None and np.isfinite(elpd)
        assert elpd_scalar(loo, "p") is not None
        assert elpd_scalar(loo, "se") is not None
        assert elpd_field(loo, "pointwise") is not None

        thresh = elpd_pareto_k_threshold(loo)
        assert 0.0 < thresh <= 0.7
