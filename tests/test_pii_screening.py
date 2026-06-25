"""PII screening (Phase 3 / G3)."""

from __future__ import annotations

import pandas as pd

from mmm_framework.security import scan_dataframe_for_pii, scan_text


def test_detects_email_phone_ssn():
    findings = scan_text("reach a@b.com or 415-555-1234; ssn 123-45-6789")
    kinds = {f.kind for f in findings}
    assert {"email", "phone", "ssn"} <= kinds


def test_credit_card_requires_luhn():
    valid = scan_text("card 4111 1111 1111 1111")  # valid Luhn test card
    assert any(f.kind == "credit_card" for f in valid)
    invalid = scan_text("ref 4111 1111 1111 1112")  # fails Luhn
    assert not any(f.kind == "credit_card" for f in invalid)


def test_samples_are_masked():
    findings = scan_text("a@b.com")
    assert findings and findings[0].kind == "email"
    assert "b.com" not in findings[0].sample  # not the raw value


def test_dataframe_scan_targets_string_columns():
    df = pd.DataFrame(
        {"email": ["x@y.com", "z@w.com"], "spend": [100.0, 200.0]}
    )
    findings = scan_dataframe_for_pii(df)
    assert any(f.location == "email" and f.kind == "email" for f in findings)
    assert not any(f.location == "spend" for f in findings)


def test_clean_dataframe_no_findings():
    df = pd.DataFrame({"channel": ["TV", "Search"], "spend": [1.0, 2.0]})
    assert scan_dataframe_for_pii(df) == []
