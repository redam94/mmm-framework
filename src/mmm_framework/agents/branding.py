"""Client branding & user preferences for agent output (plots + reports).

Branding lives in the sessions-store ``preferences`` table:
``preferences(scope=<project_id>, key="branding")`` for per-client branding,
``preferences(scope="global", key="branding_defaults")`` for deployment-wide
defaults. The agent recalls it via the ``get_preferences`` tool; plots pick it
up automatically at store time (``apply_brand_colors``); reports map it onto
the reporting ``ColorScheme``/``ChannelColors``.

Brand colors are applied HOST-SIDE on the captured figure JSON — never inside
``_normalize_figure`` — because that function runs inside subprocess/container
kernels where the preferences DB is (deliberately) unreachable.
"""

from __future__ import annotations

import colorsys
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$")


def _valid_hex(value: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    if not _HEX_RE.match(v):
        raise ValueError(f"not a hex color: {value!r}")
    return v.lower()


class BrandColors(BaseModel):
    primary: str | None = None
    secondary: str | None = None
    accent: str | None = None
    palette: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("primary", "secondary", "accent")
    @classmethod
    def _check_hex(cls, v):
        return _valid_hex(v)

    @field_validator("palette")
    @classmethod
    def _check_palette(cls, v):
        return [_valid_hex(c) for c in v if c]


class BrandFonts(BaseModel):
    heading: str | None = None
    body: str | None = None


class Branding(BaseModel):
    """Per-client branding (or global defaults). All fields optional so a
    partially-extracted brand still validates."""

    client_name: str | None = None
    colors: BrandColors = Field(default_factory=BrandColors)
    logo_url: str | None = None
    fonts: BrandFonts = Field(default_factory=BrandFonts)
    footer_text: str | None = None
    source_url: str | None = None
    source: str = "manual"  # "manual" | "extracted"
    confirmed: bool = True

    model_config = {"extra": "ignore"}


def resolve_branding(thread_id: str | None) -> dict[str, Any] | None:
    """The effective branding for a session: project branding, falling back to
    the global defaults. Returns a plain dict (or None)."""
    from mmm_framework.api import sessions as sessions_store

    try:
        project_id = sessions_store.resolve_project_id(thread_id)
        branding = sessions_store.get_project_branding(project_id)
        if branding:
            return branding
        global_default = sessions_store.get_preference("global", "branding_defaults")
        return global_default if isinstance(global_default, dict) else None
    except Exception:
        return None


def brand_palette(branding: dict[str, Any] | None) -> list[str]:
    """The plot colorway implied by a branding dict (may be empty)."""
    if not branding:
        return []
    colors = branding.get("colors") or {}
    palette = [c for c in (colors.get("palette") or []) if c]
    if not palette:
        palette = [
            c
            for c in (
                colors.get("primary"),
                colors.get("secondary"),
                colors.get("accent"),
            )
            if c
        ]
    return palette


def is_active(branding: dict[str, Any] | None) -> bool:
    """Branding drives output styling only once confirmed (extracted-but-
    unconfirmed proposals must not silently restyle deliverables)."""
    return (
        bool(branding)
        and bool(branding.get("confirmed"))
        and bool(brand_palette(branding))
    )


def apply_brand_colors(fig_json: dict, branding: dict[str, Any] | None) -> dict:
    """Recolor a captured Plotly figure JSON with the brand palette.

    Pure-JSON, host-side: sets ``layout.colorway`` and remaps any trace colors
    that are members of the design palette (``tools._PALETTE``) onto the brand
    palette by index (cyclic). Unknown/custom trace colors are left alone —
    the analyst may have encoded meaning in them."""
    palette = brand_palette(branding)
    if not palette or not isinstance(fig_json, dict):
        return fig_json

    from mmm_framework.agents.tools import _PALETTE

    index_of = {c.lower(): i for i, c in enumerate(_PALETTE)}

    def _remap(color: Any) -> Any:
        if isinstance(color, str) and color.lower() in index_of:
            return palette[index_of[color.lower()] % len(palette)]
        return color

    layout = fig_json.setdefault("layout", {})
    layout["colorway"] = palette
    for trace in fig_json.get("data") or []:
        if not isinstance(trace, dict):
            continue
        for holder_key in ("marker", "line"):
            holder = trace.get(holder_key)
            if isinstance(holder, dict) and "color" in holder:
                c = holder["color"]
                if isinstance(c, str):
                    holder["color"] = _remap(c)
                elif isinstance(c, list):
                    holder["color"] = [_remap(x) for x in c]
    return fig_json


def _shade(hex_color: str, factor: float) -> str:
    """Darken (<1) or lighten (>1) a hex color via HLS lightness scaling."""
    v = hex_color.lstrip("#")
    if len(v) == 3:
        v = "".join(ch * 2 for ch in v)
    r, g, b = (int(v[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, lum, s = colorsys.rgb_to_hls(r, g, b)
    lum = max(0.0, min(1.0, lum * factor))
    r, g, b = colorsys.hls_to_rgb(h, lum, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def branding_to_color_scheme(branding: dict[str, Any] | None):
    """Map branding onto the reporting ``ColorScheme`` (defaults where the
    brand doesn't specify)."""
    from mmm_framework.reporting.config import ColorScheme

    colors = (branding or {}).get("colors") or {}
    primary = colors.get("primary")
    accent = colors.get("accent") or colors.get("secondary")
    if not primary:
        return ColorScheme()
    return ColorScheme(
        primary=primary,
        primary_dark=_shade(primary, 0.75),
        accent=accent or _shade(primary, 1.25),
        accent_dark=_shade(accent, 0.75) if accent else _shade(primary, 0.95),
    )


def branding_to_channel_colors(branding: dict[str, Any] | None, channels: list[str]):
    """Assign brand palette colors to media channels in order (cyclic)."""
    from mmm_framework.reporting.config import ChannelColors

    palette = brand_palette(branding)
    if not palette or not channels:
        return ChannelColors()
    return ChannelColors(
        colors={ch: palette[i % len(palette)] for i, ch in enumerate(channels)}
    )
