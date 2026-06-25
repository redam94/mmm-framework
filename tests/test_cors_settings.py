"""CORS allowlist resolution (Phase 3 / S1)."""

from __future__ import annotations

from mmm_framework.api.main import cors_settings


def test_default_is_localhost_with_credentials(monkeypatch):
    monkeypatch.delenv("MMM_CORS_ORIGINS", raising=False)
    origins, creds = cors_settings()
    assert creds is True
    assert "http://localhost:5173" in origins
    assert "*" not in origins  # never wildcard-with-credentials


def test_wildcard_disables_credentials(monkeypatch):
    monkeypatch.setenv("MMM_CORS_ORIGINS", "*")
    origins, creds = cors_settings()
    assert origins == ["*"]
    assert creds is False  # the only valid wildcard combination


def test_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("MMM_CORS_ORIGINS", "https://a.example.com, https://b.example.com")
    origins, creds = cors_settings()
    assert origins == ["https://a.example.com", "https://b.example.com"]
    assert creds is True
