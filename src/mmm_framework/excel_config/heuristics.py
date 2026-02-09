"""
Heuristic classification of MFF variable names into roles.

Uses keyword matching and statistical properties to guess whether
a variable is a KPI, media channel, control, or should be excluded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import VariableRole


# =============================================================================
# Keyword patterns for role classification
# =============================================================================

# Patterns are checked case-insensitively against the variable name.
# Order matters within each list — more specific patterns first.

KPI_PATTERNS: list[str] = [
    r"\bsales\b",
    r"\brevenue\b",
    r"\bconversions?\b",
    r"\borders?\b",
    r"\bleads?\b",
    r"\bregistrations?\b",
    r"\bdownloads?\b",
    r"\bsignups?\b",
    r"\bsubscri",
    r"\bkpi\b",
    r"\btarget\b",
    r"\bresponse\b",
    r"\btransactions?\b",
    r"\bprofit\b",
    r"\bincome\b",
    r"\bunits?\b",
    r"\bvolume\b",
]

MEDIA_PATTERNS: list[str] = [
    # Channels
    r"\btv\b",
    r"\btelevision\b",
    r"\bradio\b",
    r"\bprint\b",
    r"\bdigital\b",
    r"\bdisplay\b",
    r"\bsearch\b",
    r"\bsocial\b",
    r"\bfacebook\b",
    r"\bmeta\b",
    r"\bgoogle\b",
    r"\byoutube\b",
    r"\btiktok\b",
    r"\binstagram\b",
    r"\btwitter\b",
    r"\bprogrammatic\b",
    r"\bvideo\b",
    r"\bemail\b",
    r"\bsms\b",
    r"\booh\b",
    r"\boutdoor\b",
    r"\bcinema\b",
    r"\bpodcast\b",
    r"\bspotify\b",
    r"\baffiliate\b",
    r"\binfluencer\b",
    r"\bnative\b",
    r"\bbanner\b",
    r"\bpaid\b",
    # Metrics that suggest media
    r"\bspend\b",
    r"\bimpressions?\b",
    r"\bgrps?\b",
    r"\btrps?\b",
    r"\bclicks?\b",
    r"\bcpm\b",
    r"\bcpc\b",
    r"\bctr\b",
    r"\breach\b",
    r"\bfrequency\b",
    r"\badvertis",
    r"\bmedia\b",
    r"\bcampaign\b",
    r"\bad_\b",
    r"\bads?\b",
]

CONTROL_PATTERNS: list[str] = [
    r"\bprice\b",
    r"\bpricing\b",
    r"\bpromo",
    r"\bdiscount\b",
    r"\bdistribution\b",
    r"\bweather\b",
    r"\btemperature\b",
    r"\bholiday\b",
    r"\bseasonalit",
    r"\bcompetitor\b",
    r"\bunemployment\b",
    r"\bgdp\b",
    r"\bcpi\b",
    r"\bindex\b",
    r"\bcovid\b",
    r"\bpandemic\b",
    r"\btrend\b",
    r"\bevent\b",
    r"\bstore.?count\b",
    r"\binflation\b",
    r"\binterest.?rate\b",
    r"\bpopulation\b",
    r"\bseasonal\b",
    r"\bwages?\b",
    r"\binventory\b",
    r"\bstock\b",
    r"\bweekday\b",
    r"\bweekend\b",
]

# Variables that are likely allocation weights or auxiliary data
AUXILIARY_PATTERNS: list[str] = [
    r"\bweight\b",
    r"\ballocation\b",
    r"\bpopulation\b",
    r"\bshare\b",
]


@dataclass
class VariableStats:
    """Summary statistics for a variable, used for heuristic classification."""

    mean: float
    std: float
    min_val: float
    max_val: float
    zero_pct: float  # Fraction of values that are zero
    n_obs: int
    coverage_pct: float  # % of expected dimension combos with data


def _match_patterns(name: str, patterns: list[str]) -> bool:
    """Check if a variable name matches any pattern in the list."""
    # Replace underscores and hyphens with spaces so \b word boundaries work
    name_lower = name.lower().replace("_", " ").replace("-", " ")
    return any(re.search(pat, name_lower) for pat in patterns)


def _score_by_stats(stats: VariableStats | None) -> str | None:
    """
    Use statistical properties as a fallback classification signal.

    Returns a role string hint or None if inconclusive.
    """
    if stats is None:
        return None

    # High zero percentage + high variance → likely media (sparse spending)
    if stats.zero_pct > 0.3 and stats.std > stats.mean * 0.5:
        return "media"

    # Very low variance → likely a control or index
    if stats.std < stats.mean * 0.05 and stats.mean != 0:
        return "control"

    return None


def classify_variable(
    name: str,
    stats: VariableStats | None = None,
) -> VariableRole:
    """
    Classify a variable name into a role using keyword heuristics.

    Parameters
    ----------
    name : str
        The variable name from the MFF VariableName column.
    stats : VariableStats, optional
        Summary statistics for the variable. Used as a fallback signal.

    Returns
    -------
    VariableRole
        The guessed role: KPI, MEDIA, CONTROL, or AUXILIARY.
    """
    # Check keyword patterns in priority order
    if _match_patterns(name, KPI_PATTERNS):
        return VariableRole.KPI

    if _match_patterns(name, MEDIA_PATTERNS):
        return VariableRole.MEDIA

    if _match_patterns(name, CONTROL_PATTERNS):
        return VariableRole.CONTROL

    if _match_patterns(name, AUXILIARY_PATTERNS):
        return VariableRole.AUXILIARY

    # Fallback to statistical heuristics
    stat_hint = _score_by_stats(stats)
    if stat_hint == "media":
        return VariableRole.MEDIA
    if stat_hint == "control":
        return VariableRole.CONTROL

    # Default: exclude (analyst must decide)
    return VariableRole.AUXILIARY


def generate_display_name(variable_name: str) -> str:
    """
    Generate a human-friendly display name from a variable name.

    Examples
    --------
    >>> generate_display_name("tv_spend")
    'TV Spend'
    >>> generate_display_name("PaidSearch_Impressions")
    'Paid Search Impressions'
    >>> generate_display_name("GDP")
    'GDP'
    """
    # Replace underscores and hyphens with spaces
    name = variable_name.replace("_", " ").replace("-", " ")

    # Split on camelCase boundaries
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

    # Capitalize each word, but keep common acronyms uppercase
    acronyms = {"tv", "gdp", "cpi", "sms", "ooh", "cpm", "cpc", "ctr", "grp", "trp", "roi"}
    words = name.split()
    result = []
    for word in words:
        if word.lower() in acronyms:
            result.append(word.upper())
        else:
            result.append(word.capitalize())

    return " ".join(result)
