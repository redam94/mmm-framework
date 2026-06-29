"""The agentic layer of the slide deck: per-slide AI insights + a whole-deck
synthesis, generated from each slide's *deterministic* facts (never invented).

The deck engine (``reporting.deck``) computes every number and chart with no AI
and attaches a deterministic ``notes`` string to each slide. Here we turn those
facts into prose:

* **per-slide insight** — one grounded `llm.invoke` per channel deep-dive slide,
  writing the channel's narrative from its zone facts;
* **synthesis** — one `llm.invoke` over *all* slide facts, writing the report
  headline (the single most important finding + the top reallocation move).

These are direct, grounded LLM calls (a system prompt + the slide facts), not a
full agent thread — the same lightweight pattern as the Model Garden copilot.
The result is the ``insights`` map consumed by
``reporting.deck.builder.build_pptx``. Everything is best-effort: an LLM failure
degrades to a deck with no narrative, never a broken job.
"""

from __future__ import annotations

from typing import Any

_INSIGHT_SYS = (
    "You are a senior marketing-effectiveness analyst writing one slide's insight "
    "for a client MMM readout. Be concise, quantitative, and decision-oriented. "
    "Ground every claim in the facts provided — never invent numbers. One insight "
    "of 1–2 sentences (~35 words). No preamble, no bullet markers, no markdown."
)
_SYNTH_SYS = (
    "You are a senior marketing-effectiveness analyst writing the HEADLINE of a "
    "client MMM readout. Synthesize across the whole deck. Be specific and "
    "decision-oriented; ground every claim in the facts — never invent numbers. "
    "No preamble, no markdown."
)

# slide kinds that get a per-slide insight (the channel deep-dives)
_PER_SLIDE_KINDS = {"saturation"}


def slide_key(slide: Any) -> str:
    """Stable insight key for a slide: channel deep-dives → ``channel:<name>``,
    the executive summary → ``headline``, the reallocation → ``reallocation``."""
    kind = slide.kind
    if kind == "saturation":
        ch = (slide.metrics or {}).get("zones", {}).get("channel")
        return f"channel:{ch}" if ch else f"saturation:{slide.title}"
    if kind == "executive_summary":
        return "headline"
    if kind == "optimization":
        return "reallocation"
    return kind


def light_metrics(slide: Any) -> dict[str, Any]:
    """Scalar-only metrics for insight context (drop big grids/arrays)."""
    out: dict[str, Any] = {}
    for k, v in (slide.metrics or {}).items():
        if k == "zones" and isinstance(v, dict):
            out["zone"] = {
                kk: v.get(kk)
                for kk in (
                    "current_zone",
                    "recommendation",
                    "current_roi",
                    "current_mroi",
                    "optimal_spend",
                    "headroom_to_optimal",
                    "break_even",
                )
            }
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict) and len(v) <= 12:
            out[k] = {
                kk: vv
                for kk, vv in v.items()
                if isinstance(vv, (int, float, str, bool))
            }
    return out


def _clean(text: Any) -> str:
    """Extract plain text from an LLM reply, then collapse whitespace.

    A LangChain message's ``.content`` may be a plain string OR a list of
    content blocks (``[{"type": "text", "text": "..."}, ...]`` — Anthropic and
    some OpenAI-compatible servers). Stringifying that list dumps the dict repr
    (and any ``extras``/metadata) into the slide, so we pull only the text.
    """
    s = text.content if hasattr(text, "content") else text
    if isinstance(s, list):
        parts = []
        for blk in s:
            if isinstance(blk, dict):
                parts.append(str(blk.get("text") or blk.get("content") or ""))
            elif isinstance(blk, str):
                parts.append(blk)
        s = " ".join(p for p in parts if p)
    elif not isinstance(s, str):
        s = str(s)
    return " ".join(s.split()).strip().strip('"')


def _brief(metrics: dict | None) -> str:
    if not metrics:
        return ""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            parts.append(
                f"{k}={{" + ", ".join(f"{kk}={vv}" for kk, vv in v.items()) + "}"
            )
        else:
            parts.append(f"{k}={v}")
    return "; ".join(parts)


def generate_deck_insights(
    notes: list[dict], llm, *, max_per_slide: int = 14
) -> dict[str, str]:
    """Per-slide insights (channel deep-dives) + a synthesized headline.

    ``notes`` is the light per-slide list from the ``slide_deck_notes`` model-op
    (key / kind / title / notes / metrics / is_summary). ``llm`` is any
    LangChain chat model (``.invoke``). Returns the ``insights`` map for
    ``build_pptx``. Sync (so it composes with the agent tool and with
    ``asyncio.to_thread`` in the API job)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    insights: dict[str, str] = {}

    # all slide facts, for both per-slide grounding and the synthesis
    facts_blob = "\n".join(
        f"- {n.get('title')} [{n.get('kind')}]: {n.get('notes', '')}"
        + (f"  ({_brief(n.get('metrics'))})" if n.get("metrics") else "")
        for n in notes
    )

    calls = 0
    for n in notes:
        if n.get("is_summary") or n.get("kind") not in _PER_SLIDE_KINDS:
            continue
        if calls >= max_per_slide:
            break
        prompt = (
            f"Slide: {n.get('title')}\n"
            f"Facts: {n.get('notes', '')}\n"
            f"Metrics: {_brief(n.get('metrics'))}\n\n"
            "Write the one-line insight for this channel slide: what its return / "
            "next-dollar (marginal) ROI / zone imply, and the specific spend action."
        )
        try:
            r = llm.invoke(
                [SystemMessage(content=_INSIGHT_SYS), HumanMessage(content=prompt)]
            )
            txt = _clean(r)
            if txt:
                insights[n["key"]] = txt
            calls += 1
        except Exception:
            # If the very first call fails the LLM is almost certainly
            # unreachable (mis-config / endpoint down); short-circuit rather than
            # retrying every slide, so the deck still renders promptly with no
            # narrative instead of waiting out N×retry backoffs.
            if calls == 0:
                return insights
            continue

    # synthesis -> the report headline
    try:
        sp = (
            f"The deck's slide facts:\n{facts_blob}\n\n"
            "Write the report HEADLINE standfirst: 2–3 sentences stating the single "
            "most important finding and the highest-value reallocation move across "
            "all channels (which to scale up, which to pull back, and why)."
        )
        r = llm.invoke([SystemMessage(content=_SYNTH_SYS), HumanMessage(content=sp)])
        txt = _clean(r)
        if txt:
            insights["headline"] = txt
    except Exception:
        pass

    return insights


__all__ = ["generate_deck_insights", "slide_key", "light_metrics"]
