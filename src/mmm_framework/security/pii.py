"""PII screening for ingested data.

Client marketing files can carry personal data (emails, phones, government IDs,
card numbers) that an aggregate MMM never needs. This surfaces it BEFORE it is
stored/modeled so a human can redact or confirm — a governance control, not a
hard block. Pure-Python (regex + Luhn); no extra dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Conservative patterns: aim for high precision (few false positives) over recall.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CC_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IPV4 = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
)


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum — reduces credit-card false positives from any long number."""
    d = [int(c) for c in digits if c.isdigit()]
    if not (13 <= len(d) <= 19):
        return False
    checksum = 0
    parity = len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        checksum += n
    return checksum % 10 == 0


@dataclass
class PIIFinding:
    location: str  # column name, or "text"
    kind: str  # email | phone | ssn | credit_card | ip_address
    n_matches: int
    sample: str  # one masked example


_MASK_KEEP = 2


def _mask(value: str) -> str:
    v = str(value)
    if len(v) <= _MASK_KEEP:
        return "*" * len(v)
    return v[:_MASK_KEEP] + "*" * (len(v) - _MASK_KEEP)


def scan_text(text: str, location: str = "text") -> list[PIIFinding]:
    """Scan a single string for PII patterns."""
    findings: list[PIIFinding] = []
    checks = [
        ("email", _EMAIL.findall(text)),
        ("phone", _PHONE.findall(text)),
        ("ssn", _SSN.findall(text)),
        ("ip_address", _IPV4.findall(text)),
    ]
    cc = [m for m in _CC_CANDIDATE.findall(text) if _luhn_ok(m)]
    if cc:
        checks.append(("credit_card", cc))
    for kind, matches in checks:
        if matches:
            findings.append(
                PIIFinding(
                    location=location,
                    kind=kind,
                    n_matches=len(matches),
                    sample=_mask(matches[0]),
                )
            )
    return findings


def scan_dataframe_for_pii(df, columns=None, max_rows: int = 1000) -> list[PIIFinding]:
    """Scan (a sample of) a DataFrame's string columns for PII.

    Returns one :class:`PIIFinding` per (column, kind). ``max_rows`` caps how many
    rows are scanned per column for speed; ``columns`` restricts the scan.
    """
    findings: list[PIIFinding] = []
    cols = list(columns) if columns is not None else list(df.columns)
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        # Only object/string-like columns can carry these patterns.
        if getattr(series, "dtype", None) is not None and series.dtype != object:
            try:
                if not series.map(lambda v: isinstance(v, str)).any():
                    continue
            except Exception:
                continue
        values = series.dropna().astype(str).head(max_rows)
        blob = "\n".join(values.tolist())
        if not blob:
            continue
        for f in scan_text(blob, location=str(col)):
            findings.append(f)
    return findings
