"""Backend deferred-infra slice: fit quota enforcement (O3) + request-id (O4)."""

from __future__ import annotations

import pytest


# ----- O3: fit quota enforcement --------------------------------------------
def test_fit_quota_blocks_at_limit(monkeypatch):
    from mmm_framework.auth import plans

    free = plans.get_plan("free")
    quota = free.monthly_fit_quota
    assert quota is not None  # the free tier is metered

    monkeypatch.setattr(plans, "entitlements_for_org", lambda org, db_path=None: free)

    # Under quota -> allowed.
    monkeypatch.setattr(plans.store, "count_org_fits_since", lambda *a, **k: quota - 1)
    plans.assert_within_fit_quota("org1")

    # At/over quota -> blocked.
    monkeypatch.setattr(plans.store, "count_org_fits_since", lambda *a, **k: quota)
    with pytest.raises(plans.PlanLimitError, match="quota"):
        plans.assert_within_fit_quota("org1")


def test_fit_quota_unlimited_plan_never_blocks(monkeypatch):
    from mmm_framework.auth import plans

    ent = plans.get_plan("enterprise")  # monthly_fit_quota is None (unlimited)
    assert ent.monthly_fit_quota is None
    monkeypatch.setattr(plans, "entitlements_for_org", lambda org, db_path=None: ent)
    monkeypatch.setattr(plans.store, "count_org_fits_since", lambda *a, **k: 10**6)
    plans.assert_within_fit_quota("org1")  # no raise


# ----- O4: request-id correlation -------------------------------------------
def test_request_id_minted_and_echoed():
    from fastapi.testclient import TestClient

    from mmm_framework.api.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.headers.get("x-request-id")  # minted

        r2 = client.get("/health", headers={"X-Request-ID": "corr-abc-123"})
        assert r2.headers.get("x-request-id") == "corr-abc-123"  # echoed
