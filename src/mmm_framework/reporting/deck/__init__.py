"""Automated slide-deck generation from a fitted MMM.

Two template-independent layers (no AI, no PowerPoint dependency):

* :mod:`charts` — matplotlib renderers that turn model-derived numbers into PNG
  images (the centerpiece is the saturation chart that shades the
  breakthrough / optimal / saturation spend zones from
  :func:`mmm_framework.reporting.helpers.compute_response_zones`).
* :mod:`engine` — a deterministic per-slide *data* engine: it computes every
  slide's numbers, tables, and chart images directly from the model, and marks
  which slides are deck-level summaries. AI insights (per slide) and the
  whole-deck synthesis are layered on later, and the python-pptx builder fills a
  user-supplied template from this :class:`~engine.Deck`.
"""

from __future__ import annotations

from .engine import Deck, Slide, build_deck

__all__ = ["Deck", "Slide", "build_deck"]
