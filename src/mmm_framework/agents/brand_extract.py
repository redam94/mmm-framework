"""Extract client branding (colors, logo, fonts, name) from a website.

Runs in the API process only (a host-side tool like ``search_knowledge_base``)
— never inside the execution kernel. The fetch is SSRF-guarded:

* http/https only, no userinfo, ports 80/443;
* every DNS resolution of the host must be a GLOBAL address (rejects
  loopback, RFC1918, link-local 169.254.x — the cloud-metadata vector — ULA,
  multicast, reserved) — literal IPs are checked the same way;
* redirects are NOT followed automatically: each ``Location`` is re-vetted
  through the same guard, max 3 hops;
* the body is streamed with a hard byte cap and a content-type check.

Residual risk: the address check happens immediately before connect (httpx
re-resolves) — a pathologically fast DNS rebind could slip through. Accepted
for the dev posture; in the hosted profile the feature is OFF unless
``MMM_BRAND_FETCH_ALLOW`` is set (egress is denied there anyway).

``parse_brand_html`` is a pure function (no network) so the extraction
heuristics are unit-testable on fixtures.
"""

from __future__ import annotations

import colorsys
import ipaddress
import os
import re
import socket
from collections import Counter
from typing import Any
from urllib.parse import urljoin, urlsplit

_FALSE = ("", "0", "false", "no")

_MAX_REDIRECTS = 3
_DEFAULT_TIMEOUT = 8.0
_DEFAULT_MAX_BYTES = 2_000_000


class BrandExtractError(ValueError):
    """User-facing extraction failure (bad URL, blocked target, fetch error)."""


# ── SSRF guard ────────────────────────────────────────────────────────────────


def _assert_public_http(url: str) -> None:
    """Raise BrandExtractError unless ``url`` is a public http(s) URL whose
    host resolves exclusively to globally-routable addresses."""
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise BrandExtractError(f"Invalid URL: {url!r}") from exc
    if parts.scheme not in ("http", "https"):
        raise BrandExtractError(
            f"Only http/https URLs are allowed (got {parts.scheme!r})."
        )
    if "@" in (parts.netloc or ""):
        raise BrandExtractError("URLs with embedded credentials are not allowed.")
    host = parts.hostname
    if not host:
        raise BrandExtractError(f"Invalid URL (no host): {url!r}")
    port = parts.port
    if port not in (None, 80, 443):
        raise BrandExtractError(f"Port {port} is not allowed (80/443 only).")

    try:
        infos = socket.getaddrinfo(host, port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise BrandExtractError(f"Could not resolve host {host!r}.") from exc
    addrs = {info[4][0] for info in infos}
    if not addrs:
        raise BrandExtractError(f"Could not resolve host {host!r}.")
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError as exc:
            raise BrandExtractError(f"Unresolvable address for {host!r}.") from exc
        if not ip.is_global:
            raise BrandExtractError(
                f"Host {host!r} resolves to a non-public address ({addr}) — refusing."
            )


def _assert_enabled() -> None:
    from mmm_framework.agents.profile import is_hosted

    if is_hosted() and (
        os.environ.get("MMM_BRAND_FETCH_ALLOW", "").strip().lower() in _FALSE
    ):
        raise BrandExtractError(
            "Brand extraction from websites is disabled in the hosted profile "
            "(set MMM_BRAND_FETCH_ALLOW=1 to enable). Enter the brand colors "
            "manually instead."
        )


def _fetch(
    url: str, *, timeout: float, max_bytes: int, accept_css: bool = False
) -> str:
    """SSRF-guarded GET returning text, following at most 3 manually-vetted
    redirects, streaming with a byte cap."""
    import httpx

    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        _assert_public_http(current)
        with httpx.Client(follow_redirects=False, timeout=timeout) as client:
            with client.stream(
                "GET",
                current,
                headers={"User-Agent": "mmm-framework-brand-extract/1.0"},
            ) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        raise BrandExtractError("Redirect without a Location header.")
                    current = urljoin(current, loc)
                    continue
                if resp.status_code >= 400:
                    raise BrandExtractError(
                        f"Fetch failed with HTTP {resp.status_code} for {current}"
                    )
                ctype = (resp.headers.get("content-type") or "").lower()
                allowed = ("text/html", "application/xhtml") + (
                    ("text/css",) if accept_css else ()
                )
                if ctype and not any(ctype.startswith(a) for a in allowed):
                    raise BrandExtractError(
                        f"Unsupported content-type {ctype!r} at {current}"
                    )
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise BrandExtractError(
                            f"Page exceeds the {max_bytes} byte fetch cap."
                        )
                    chunks.append(chunk)
                return b"".join(chunks).decode(resp.encoding or "utf-8", "replace")
    raise BrandExtractError("Too many redirects.")


# ── HTML parsing (pure, fixture-testable) ─────────────────────────────────────

_META_RE = re.compile(r"<meta\s+[^>]*>", re.IGNORECASE | re.DOTALL)
_LINK_RE = re.compile(r"<link\s+[^>]*>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r"""([a-zA-Z-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.IGNORECASE | re.DOTALL)
_STYLE_ATTR_RE = re.compile(r"""style\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)
_HEX_COLOR_RE = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
_RGB_COLOR_RE = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
_FONT_FAMILY_RE = re.compile(r"font-family\s*:\s*([^;}{]+)", re.IGNORECASE)
_GENERIC_FONTS = {
    "serif",
    "sans-serif",
    "monospace",
    "cursive",
    "fantasy",
    "system-ui",
    "inherit",
    "initial",
    "unset",
    "-apple-system",
    "blinkmacsystemfont",
}


def _attrs(tag: str) -> dict[str, str]:
    return {
        m.group(1).lower(): (m.group(2) if m.group(2) is not None else m.group(3))
        for m in _ATTR_RE.finditer(tag)
    }


def _norm_hex(color: str) -> str:
    v = color.lstrip("#").lower()
    if len(v) == 3:
        v = "".join(ch * 2 for ch in v)
    return f"#{v}"


def _is_brandable(hex_color: str) -> bool:
    """Drop near-white / near-black / grey colors — they're chrome, not brand."""
    v = hex_color.lstrip("#")
    r, g, b = (int(v[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    _h, lum, sat = colorsys.rgb_to_hls(r, g, b)
    return 0.12 <= lum <= 0.92 and sat >= 0.18


def _saturation(hex_color: str) -> float:
    v = hex_color.lstrip("#")
    r, g, b = (int(v[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    _, _, s = colorsys.rgb_to_hls(r, g, b)
    return s


def _abs_http_url(candidate: str | None, base_url: str) -> str | None:
    if not candidate:
        return None
    absolute = urljoin(base_url, candidate.strip())
    if urlsplit(absolute).scheme in ("http", "https"):
        return absolute
    return None


def parse_brand_html(html: str, base_url: str) -> dict[str, Any]:
    """Extract a branding proposal from raw HTML. Pure function — no network.

    Returns the Branding dict shape with ``source="extracted"``,
    ``confirmed=False``.
    """
    metas = [_attrs(m.group(0)) for m in _META_RE.finditer(html)]
    links = [_attrs(m.group(0)) for m in _LINK_RE.finditer(html)]

    def _meta(*names: str) -> str | None:
        for m in metas:
            key = (m.get("property") or m.get("name") or "").lower()
            if key in names and m.get("content"):
                return m["content"].strip()
        return None

    # Client name: og:site_name > og:title > <title>
    client_name = _meta("og:site_name") or _meta("og:title")
    if not client_name:
        t = _TITLE_RE.search(html)
        if t:
            client_name = re.sub(r"\s+", " ", t.group(1)).strip() or None
    if client_name and len(client_name) > 80:
        client_name = client_name[:80].rsplit(" ", 1)[0]

    # Logo: og:image > apple-touch-icon > rel=icon
    logo = _abs_http_url(_meta("og:image"), base_url)
    if not logo:
        for rel_pref in ("apple-touch-icon", "icon", "shortcut icon"):
            for lk in links:
                if rel_pref in (lk.get("rel") or "").lower() and lk.get("href"):
                    logo = _abs_http_url(lk["href"], base_url)
                    break
            if logo:
                break

    # Colors: theme-color first, then frequency-ranked CSS colors.
    css_text = " ".join(m.group(1) for m in _STYLE_BLOCK_RE.finditer(html))
    css_text += " " + " ".join(
        (m.group(1) or m.group(2) or "") for m in _STYLE_ATTR_RE.finditer(html)
    )
    counts: Counter[str] = Counter()
    for m in _HEX_COLOR_RE.finditer(css_text):
        counts[_norm_hex(m.group(0))] += 1
    for m in _RGB_COLOR_RE.finditer(css_text):
        r, g, b = (min(255, int(m.group(i))) for i in (1, 2, 3))
        counts[f"#{r:02x}{g:02x}{b:02x}"] += 1

    theme_color = _meta("theme-color")
    if theme_color and _HEX_COLOR_RE.fullmatch(theme_color.strip()):
        theme_color = _norm_hex(theme_color.strip())
    else:
        theme_color = None

    ranked = [c for c, _ in counts.most_common() if _is_brandable(c)]
    palette: list[str] = []
    if theme_color and _is_brandable(theme_color):
        palette.append(theme_color)
    for c in ranked:
        if c not in palette:
            palette.append(c)
        if len(palette) >= 6:
            break

    primary = palette[0] if palette else theme_color
    secondary = palette[1] if len(palette) > 1 else None
    accent = (
        max(palette[1:], key=_saturation, default=None) if len(palette) > 1 else None
    )

    # Fonts: first non-generic families seen in font-family declarations.
    fonts: list[str] = []
    for m in _FONT_FAMILY_RE.finditer(css_text):
        for fam in m.group(1).split(","):
            name = fam.strip().strip("'\"").strip()
            if not name or name.lower() in _GENERIC_FONTS:
                continue
            if name not in fonts:
                fonts.append(name)
        if len(fonts) >= 4:
            break

    return {
        "client_name": client_name,
        "colors": {
            "primary": primary,
            "secondary": secondary,
            "accent": accent,
            "palette": palette,
        },
        "logo_url": logo,
        "fonts": {
            "heading": fonts[0] if fonts else None,
            "body": fonts[1] if len(fonts) > 1 else (fonts[0] if fonts else None),
        },
        "footer_text": None,
        "source_url": base_url,
        "source": "extracted",
        "confirmed": False,
    }


def extract_brand_from_url(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Fetch ``url`` (SSRF-guarded) and extract a branding proposal. Also pulls
    ONE same-origin stylesheet (size/time-capped, same guard) when the inline
    CSS yields no usable palette."""
    _assert_enabled()
    if "://" not in url:
        url = f"https://{url}"
    html = _fetch(url, timeout=timeout, max_bytes=max_bytes)
    brand = parse_brand_html(html, url)

    if not brand["colors"]["palette"]:
        origin = urlsplit(url)
        for lk in (_attrs(m.group(0)) for m in _LINK_RE.finditer(html)):
            if "stylesheet" not in (lk.get("rel") or "").lower() or not lk.get("href"):
                continue
            css_url = _abs_http_url(lk["href"], url)
            if not css_url or urlsplit(css_url).netloc != origin.netloc:
                continue
            try:
                css = _fetch(
                    css_url, timeout=timeout, max_bytes=max_bytes, accept_css=True
                )
            except BrandExtractError:
                break
            richer = parse_brand_html(f"<style>{css}</style>", url)
            if richer["colors"]["palette"]:
                brand["colors"] = richer["colors"]
                if not brand["fonts"]["heading"]:
                    brand["fonts"] = richer["fonts"]
            break

    return brand
