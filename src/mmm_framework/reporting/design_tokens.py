"""
Design tokens for MMM Framework reports.

This module provides the canonical design token values used across
both documentation (docs/shared/styles.css) and generated reports.
It serves as the single source of truth for styling consistency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DesignTokens:
    """
    Canonical design tokens for the MMM Framework.

    Values are aligned with docs/shared/styles.css to ensure
    visual consistency between documentation pages and generated reports.
    """

    # Colors
    primary: str = "#8fa86a"
    primary_dark: str = "#6d8a4a"
    accent: str = "#6a8fa8"
    accent_dark: str = "#4a6d8a"
    background: str = "#fafbf9"
    background_alt: str = "#f0f2ed"
    surface: str = "#ffffff"
    text: str = "#2d3a2d"
    text_muted: str = "#5a6b5a"
    border: str = "#d4ddd4"
    success: str = "#6abf8a"
    warning: str = "#d4a86a"
    danger: str = "#c97067"

    # Shadows (aligned with docs/shared/styles.css)
    shadow_sm: str = "0 2px 8px rgba(45, 58, 45, 0.06)"
    shadow_md: str = "0 8px 24px rgba(45, 58, 45, 0.08)"
    shadow_lg: str = "0 16px 48px rgba(45, 58, 45, 0.12)"

    # Fonts (Source Sans 3 to match documentation)
    font_serif: str = "'DM Serif Display', serif"
    font_sans: str = "'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif"
    font_mono: str = "'JetBrains Mono', monospace"

    # Transitions
    transition_smooth: str = "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"


# Default instance for easy importing
TOKENS = DesignTokens()
