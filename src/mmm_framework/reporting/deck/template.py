"""Low-level python-pptx helpers for filling a *designed* template deck.

The template is a finished, hand-laid-out deck (named ``Text``/``Shape``/``Image``
auto-shapes with example content), not a placeholder layout. So filling it means:
locate a shape by its **stable label text** (e.g. ``"RETURN / $1"``) or by
geometry relative to a label, replace the text **preserving the template's run
formatting**, swap example chart images for model-rendered PNGs, and trim the
template's extra channel rows/slides down to the model's channel count.

python-pptx is imported lazily so importing :mod:`reporting.deck` never requires
it — only :func:`mmm_framework.reporting.deck.builder.build_pptx` does.
"""

from __future__ import annotations

import io
from typing import Any, Callable


def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def shape_text(shape) -> str:
    """The shape's text, or '' if it has none."""
    if shape.has_text_frame:
        return shape.text_frame.text or ""
    return ""


def set_text(shape, text: str) -> bool:
    """Replace a shape's text while keeping the template's formatting.

    Keeps the first paragraph's first run (its font / size / color / bold) and
    rewrites only its characters, deleting any other runs and paragraphs — so the
    designed look is preserved. Returns ``False`` if the shape can't hold text.
    """
    if not shape.has_text_frame:
        return False
    tf = shape.text_frame
    paras = tf.paragraphs
    if not paras:
        return False
    first = paras[0]
    # delete extra paragraphs (keep paragraph[0]'s XML element)
    for extra in list(paras[1:]):
        extra._p.getparent().remove(extra._p)
    runs = first.runs
    if runs:
        runs[0].text = str(text)
        for r in list(runs[1:]):
            r._r.getparent().remove(r._r)
    else:
        first.add_run().text = str(text)
    return True


def iter_text_shapes(slide):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            yield sh


def find_by_label(slide, label: str):
    """First shape whose (normalized) text equals ``label``."""
    target = _norm(label)
    for sh in iter_text_shapes(slide):
        if _norm(sh.text_frame.text) == target:
            return sh
    return None


def find_by_prefix(slide, prefix: str):
    """First shape whose (normalized) text starts with ``prefix``."""
    target = _norm(prefix)
    for sh in iter_text_shapes(slide):
        if _norm(sh.text_frame.text).startswith(target):
            return sh
    return None


def _emu(v) -> int:
    return int(v) if v is not None else 0


def shapes_below(slide, anchor, *, left_tol_in: float = 0.6, max_n: int = 4) -> list:
    """Text shapes positioned directly below ``anchor`` and roughly left-aligned
    with it, ordered top-to-bottom — i.e. the value/sub lines of a KPI card whose
    label is ``anchor``.
    """
    from pptx.util import Inches

    a_left = _emu(anchor.left)
    a_top = _emu(anchor.top)
    tol = int(Inches(left_tol_in))
    out = []
    for sh in iter_text_shapes(slide):
        if sh is anchor:
            continue
        if abs(_emu(sh.left) - a_left) <= tol and _emu(sh.top) > a_top:
            out.append(sh)
    out.sort(key=lambda s: _emu(s.top))
    return out[:max_n]


def fill_card(slide, label: str, value: str | None, sub: str | None = None) -> bool:
    """Fill a labeled KPI card: the shape with text ``label`` keeps its label;
    the next shape below becomes ``value`` and the one after becomes ``sub``.
    """
    anchor = find_by_label(slide, label)
    if anchor is None:
        return False
    below = shapes_below(slide, anchor)
    if value is not None and len(below) >= 1:
        set_text(below[0], value)
    if sub is not None and len(below) >= 2:
        set_text(below[1], sub)
    return True


def replace_image(
    slide,
    png: bytes,
    *,
    match: Callable[[Any], bool] | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> bool:
    """Replace the picture(s) in a region with ``png``.

    Finds picture shapes (optionally filtered by ``match``), records the geometry
    of the first, removes them all (the template stacks 2–3 layered copies per
    chart), and adds a single new picture at that geometry. ``region`` (EMU
    left/top/width/height) overrides the geometry. Returns ``False`` if no
    picture matched.
    """
    pics = [
        sh
        for sh in slide.shapes
        if sh.shape_type is not None
        and "PICTURE" in str(sh.shape_type)
        and (match is None or match(sh))
    ]
    if not pics:
        return False
    first = pics[0]
    geom = region or (first.left, first.top, first.width, first.height)
    for sh in pics:
        sh._element.getparent().remove(sh._element)
    slide.shapes.add_picture(
        io.BytesIO(png), geom[0], geom[1], width=geom[2], height=geom[3]
    )
    return True


def pictures_in_region(slide, left, top, width, height, *, tol_in: float = 0.3):
    """Predicate factory: matches pictures whose top-left is within ``tol_in`` of
    a target region's top-left (to target one chart when a slide has several)."""
    from pptx.util import Inches

    tol = int(Inches(tol_in))
    lo_l, lo_t = int(left) - tol, int(top) - tol
    hi_l, hi_t = int(left) + tol, int(top) + tol

    def _match(sh):
        return lo_l <= _emu(sh.left) <= hi_l and lo_t <= _emu(sh.top) <= hi_t

    return _match


def delete_shape(shape) -> None:
    shape._element.getparent().remove(shape._element)


def delete_slide(prs, index: int) -> None:
    """Remove a slide from the presentation by index (drops the relationship and
    the sldId entry; the part is orphaned, which PowerPoint tolerates)."""
    xml_slides = prs.slides._sldIdLst
    ids = list(xml_slides)
    if 0 <= index < len(ids):
        xml_slides.remove(ids[index])


__all__ = [
    "shape_text",
    "set_text",
    "iter_text_shapes",
    "find_by_label",
    "find_by_prefix",
    "shapes_below",
    "fill_card",
    "replace_image",
    "pictures_in_region",
    "delete_shape",
    "delete_slide",
]
