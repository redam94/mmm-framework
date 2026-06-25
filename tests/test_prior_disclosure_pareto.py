"""Prior disclosure + LOO Pareto-k guard (Phase 2 / I3+I4)."""

from __future__ import annotations


# --- I4: Pareto-k warning helper -------------------------------------------
def test_pareto_k_warning_none_when_clean():
    from mmm_framework.validation.validator import pareto_k_warning

    assert pareto_k_warning(0) is None
    assert pareto_k_warning(None) is None


def test_pareto_k_warning_message_when_bad():
    from mmm_framework.validation.validator import pareto_k_warning

    msg = pareto_k_warning(3)
    assert msg is not None
    assert "Pareto k" in msg
    assert "3 observation" in msg


# --- I3: default-prior disclosure in the report ----------------------------
def test_methodology_section_discloses_default_prior():
    from mmm_framework.reporting.config import ReportConfig, SectionConfig
    from mmm_framework.reporting.data_extractors import MMMDataBundle
    from mmm_framework.reporting.sections import MethodologySection

    section = MethodologySection(
        data=MMMDataBundle(),
        config=ReportConfig(),
        section_config=SectionConfig(enabled=True),
    )
    html = section.render()
    assert "non-negative prior" in html
    assert "roi_prior" in html
    # The disclosure renders even with no model_specification on the bundle.
    assert "negative or zero effect" in html


def test_normal_roi_prior_admits_negative_beta():
    """Negative-effect priors are already configurable via a channel's roi_prior
    (a Normal PriorConfig), which the model honors over the default Gamma."""
    from mmm_framework.model.base import _sample_from_prior_config
    from mmm_framework.config import PriorConfig
    from mmm_framework.config.enums import PriorType
    import pymc as pm

    normal_prior = PriorConfig(distribution=PriorType.NORMAL, params={"mu": 0.0, "sigma": 1.0})
    with pm.Model():
        rv = _sample_from_prior_config(
            "beta_test", normal_prior, lambda: pm.Gamma("beta_test", mu=1.5, sigma=1.0)
        )
        # A Normal RV is unbounded (admits negative values); Gamma would not be.
        assert rv.owner.op.name == "normal"
