"""Cross-org tenant isolation for the root classic API (Phase 1.3).

Exercises the real routes with MMM_AUTH_ENABLED via a get_auth_settings override
+ two minted org tokens, asserting org A cannot read org B's records (IDOR) and
that list endpoints are org-scoped. Also unit-tests the storage org filter.
"""

import pytest

from mmm_framework.auth import service
from mmm_framework.auth.config import AuthSettings, get_auth_settings


def _token(settings, org):
    return service.issue_tokens(
        user_id=f"u-{org}",
        org_id=org,
        email=f"{org}@example.com",
        org_role="owner",
        settings=settings,
    ).access_token


@pytest.fixture
def auth_client(test_client):
    settings = AuthSettings(enabled=True, secret="z" * 32)
    test_client.app.dependency_overrides[get_auth_settings] = lambda: settings
    yield test_client, settings
    test_client.app.dependency_overrides.pop(get_auth_settings, None)


def _hdr(settings, org):
    return {"Authorization": f"Bearer {_token(settings, org)}"}


def test_data_idor_and_list_scoping(auth_client, sample_mff_csv):
    client, settings = auth_client
    ha, hb = _hdr(settings, "orgA"), _hdr(settings, "orgB")

    up = client.post(
        "/data/upload",
        headers=ha,
        files={"file": ("d.csv", sample_mff_csv, "text/csv")},
    )
    assert up.status_code in (200, 201), up.text
    data_id = up.json()["data_id"]

    # by-id IDOR: owner 200, other org 404
    assert client.get(f"/data/{data_id}", headers=ha).status_code == 200
    assert client.get(f"/data/{data_id}", headers=hb).status_code == 404
    # cross-org delete also 404
    assert client.delete(f"/data/{data_id}", headers=hb).status_code == 404

    # list scoping: only the owning org sees it
    a_ids = {d["data_id"] for d in client.get("/data", headers=ha).json()["datasets"]}
    b_ids = {d["data_id"] for d in client.get("/data", headers=hb).json()["datasets"]}
    assert data_id in a_ids
    assert data_id not in b_ids


def test_storage_org_filter_unit(mock_settings):
    from fastapi import HTTPException

    from storage import DEFAULT_ORG_ID, StorageService, assert_org_owns

    st = StorageService(mock_settings)
    st.save_data(b"x,y\n1,2\n", "a.csv", org_id="orgA")
    st.save_data(b"x,y\n3,4\n", "b.csv", org_id="orgB")

    a = st.list_data(org_id="orgA")
    assert len(a) == 1
    assert all((d.get("org_id") or DEFAULT_ORG_ID) == "orgA" for d in a)

    # legacy records (no org_id) read as DEFAULT_ORG_ID
    assert_org_owns(None, DEFAULT_ORG_ID)  # ok, no raise
    assert_org_owns("orgA", None)  # dev principal (org None) → no-op
    with pytest.raises(HTTPException):
        assert_org_owns("orgA", "orgB")


def test_backfill_stamps_legacy_records(mock_settings):
    from storage import StorageService, backfill_org_id

    st = StorageService(mock_settings)
    meta = st.save_data(b"x,y\n1,2\n", "legacy.csv")  # no org_id (legacy)
    data_id = meta["data_id"]

    # invisible to a real org before backfill
    assert st.list_data(org_id="orgZ") == []
    counts = backfill_org_id(st, "orgZ")
    assert counts["data"] >= 1
    # now visible to orgZ, and idempotent
    assert data_id in {d["data_id"] for d in st.list_data(org_id="orgZ")}
    assert backfill_org_id(st, "orgZ")["data"] == 0
