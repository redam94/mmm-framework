"""Version-consistency gate for the docs site (issue #228).

``docs/changelog.html`` is hand-authored and nothing generated it from
``CHANGELOG.md`` — so it rotted silently: 1.1.0 and 1.2.0 both shipped while
the site still announced 1.0.0, and the ``==X.Y.Z`` pin examples scattered
across six pages each had to be found by grep at release time. What this pins:

* the changelog page announces the ``pyproject.toml`` version (which itself
  must match ``mmm_framework.__version__`` — already gated by
  ``test_api_contracts``);
* exactly one release carries the ``Current`` chip;
* every concrete ``mmm-framework==X.Y.Z`` pin across ``docs/*.html`` names
  the released version (the ``==&lt;version&gt;`` placeholder is exempt).

A mutated version fails naming both values and every stale ``file:line``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
CHANGELOG_HTML = DOCS_DIR / "changelog.html"

_PIN_RE = re.compile(r"mmm-framework==(\d+\.\d+\.\d+)")
_VERSION_PROSE_RE = re.compile(r"\bversion (\d+\.\d+\.\d+)\b")


def _released_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_changelog_page_announces_the_released_version() -> None:
    version = _released_version()
    text = CHANGELOG_HTML.read_text(encoding="utf-8")
    announced = _VERSION_PROSE_RE.search(text)
    assert announced, "changelog.html no longer states a version in prose"
    assert announced.group(1) == version, (
        f"docs/changelog.html announces {announced.group(1)} but "
        f"pyproject.toml says {version} — the release checklist's mirror "
        "step was skipped"
    )


def test_exactly_one_current_chip() -> None:
    text = CHANGELOG_HTML.read_text(encoding="utf-8")
    chips = re.findall(r'class="release-tag">\s*Current\s*<', text)
    assert len(chips) == 1, (
        f"docs/changelog.html carries {len(chips)} 'Current' release chips; "
        "exactly one release may be current"
    )


def test_every_concrete_pin_names_the_released_version() -> None:
    version = _released_version()
    stale: list[str] = []
    for page in sorted(DOCS_DIR.glob("*.html")):
        for lineno, line in enumerate(
            page.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for m in _PIN_RE.finditer(line):
                if m.group(1) != version:
                    stale.append(
                        f"{page.relative_to(ROOT)}:{lineno}: "
                        f"mmm-framework=={m.group(1)} (released: {version})"
                    )
    assert not stale, (
        "stale version pins in docs (sweep them per the release checklist):\n  "
        + "\n  ".join(stale)
    )


def test_prose_version_statements_match() -> None:
    """about.html's citation line and any other 'version X.Y.Z' prose."""
    version = _released_version()
    stale: list[str] = []
    for page in sorted(DOCS_DIR.glob("*.html")):
        for lineno, line in enumerate(
            page.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for m in _VERSION_PROSE_RE.finditer(line):
                if m.group(1) != version:
                    stale.append(
                        f"{page.relative_to(ROOT)}:{lineno}: "
                        f"'version {m.group(1)}' (released: {version})"
                    )
    assert not stale, "stale prose version statements in docs:\n  " + "\n  ".join(stale)
