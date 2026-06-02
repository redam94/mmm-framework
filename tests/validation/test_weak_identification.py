"""Tests for weak-identification detection (P2-2).

Honestly scoped: this DETECTS and REPORTS collinear channel clusters and
recommends grouped priors; it does not change the model. Conditioning on a fake
model object keeps these fast (no MCMC).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mmm_framework.validation.channel_diagnostics import ChannelDiagnostics


def _model(X_media, channel_names, media_groups=None):
    # _trace=None makes the convergence checker fall back to defaults gracefully.
    return SimpleNamespace(
        X_media_raw=np.asarray(X_media),
        channel_names=list(channel_names),
        _trace=None,
        media_groups=media_groups or {},
    )


def _collinear_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    a = base + rng.normal(0, 0.02, n)
    b = 2.0 * base + rng.normal(0, 0.02, n)  # collinear with A
    c = rng.normal(0, 1, n)  # independent
    return np.column_stack([a, b, c])


class TestCollinearClusterDetection:
    def test_detects_collinear_pair_excludes_independent(self):
        res = ChannelDiagnostics(_model(_collinear_data(), ["A", "B", "C"])).run_all()
        cluster_sets = [set(c.channels) for c in res.collinear_clusters]
        assert {"A", "B"} in cluster_sets
        assert res.weak_identification_warning is True
        # The independent channel is not dragged into a cluster.
        assert all("C" not in c.channels for c in res.collinear_clusters)
        # Surfaced as an explicit identifiability issue.
        assert any("cannot be separated" in s for s in res.identifiability_issues)

    def test_condition_number_high_under_collinearity(self):
        res = ChannelDiagnostics(_model(_collinear_data(), ["A", "B", "C"])).run_all()
        assert res.condition_number is not None
        assert res.condition_number > 30  # ill-conditioned design

    def test_independent_channels_no_cluster(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, size=(120, 3))
        res = ChannelDiagnostics(_model(X, ["A", "B", "C"])).run_all()
        assert res.collinear_clusters == []
        assert res.weak_identification_warning is False
        assert res.grouped_prior_recommendations == []

    def test_grouped_prior_recommendation_names_shared_parent(self):
        res = ChannelDiagnostics(
            _model(
                _collinear_data(),
                ["A", "B", "C"],
                media_groups={"social": ["A", "B"]},
            )
        ).run_all()
        assert res.grouped_prior_recommendations
        assert any("social" in r for r in res.grouped_prior_recommendations)

    def test_grouped_prior_recommendation_without_group(self):
        res = ChannelDiagnostics(_model(_collinear_data(), ["A", "B", "C"])).run_all()
        assert res.grouped_prior_recommendations
        assert any("grouped prior" in r for r in res.grouped_prior_recommendations)

    def test_to_dict_includes_weak_identification(self):
        res = ChannelDiagnostics(_model(_collinear_data(), ["A", "B", "C"])).run_all()
        d = res.to_dict()
        assert d["weak_identification_warning"] is True
        assert d["collinear_clusters"]
        assert "condition_number" in d
        assert d["grouped_prior_recommendations"]
