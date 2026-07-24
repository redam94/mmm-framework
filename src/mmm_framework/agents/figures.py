"""Figure normalization + execution-error formatting, shared with the kernel impls.

Extracted from ``agents/tools.py`` (2026-07-24 packaging split) into a module
with NO langchain/langgraph imports: the subprocess/container kernel bootstrap
(``kernels.py::_build_startup_source``) imports :func:`_normalize_figure`
INSIDE the kernel, and the sandboxed kernel image ships only the lean core
closure (deploy/kernel/requirements.lock has no LLM stack). ``tools.py``
re-exports everything here, so host-side imports are unchanged.
"""

from __future__ import annotations

# Design-consistent palette (indigo / teal / amber / rose / emerald / violet / sky …)
_PALETTE = [
    "#4f46e5",
    "#0d9488",
    "#f59e0b",
    "#e11d48",
    "#059669",
    "#7c3aed",
    "#0284c7",
    "#b45309",
    "#6366f1",
    "#0f766e",
]
# Default Plotly Express / graph_objects colors we want to remap
_PLOTLY_DEFAULTS = {
    "#636efa": 0,
    "#ef553b": 1,
    "#00cc96": 2,
    "#ab63fa": 3,
    "#ffa15a": 4,
    "#19d3f3": 5,
    "#ff6692": 6,
    "#b6e880": 7,
    "#ff97ff": 8,
    "#fecb52": 9,
}


def _normalize_figure(fig):
    """Remap default colors, fix margins and suppress overlapping bar labels."""
    color_map: dict = {}
    next_idx = [0]

    def _remap(c: str) -> str:
        if not isinstance(c, str):
            return c
        key = c.lower()
        if key not in color_map:
            if key in _PLOTLY_DEFAULTS:
                color_map[key] = _PALETTE[_PLOTLY_DEFAULTS[key] % len(_PALETTE)]
            else:
                color_map[key] = _PALETTE[next_idx[0] % len(_PALETTE)]
                next_idx[0] += 1
        return color_map[key]

    for trace in fig.data:
        # Remap solid string colors on the marker
        mc = getattr(getattr(trace, "marker", None), "color", None)
        if isinstance(mc, str):
            trace.marker.color = _remap(mc)
        elif isinstance(mc, (list, tuple)):
            # Array of colors — remap each unique color
            trace.marker.color = [_remap(c) if isinstance(c, str) else c for c in mc]
        # Also remap line color
        lc = getattr(getattr(trace, "line", None), "color", None)
        if isinstance(lc, str):
            trace.line.color = _remap(lc)

    # Fix bar chart text overlap: hide labels that don't fit
    has_bar = any(getattr(t, "type", "") in ("bar",) for t in fig.data)

    fig.update_layout(
        colorway=_PALETTE,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1f2937"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f9fafb",
        margin=dict(t=90, l=70, r=40, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(size=11, color="#374151"),
        ),
    )
    if has_bar:
        fig.update_layout(uniformtext=dict(minsize=9, mode="hide"))

    return fig


def format_execution_error(
    traceback_str: str,
    *,
    is_name_error: bool = False,
    missing_name: str | None = None,
) -> str:
    """Format an ``execute_python`` failure identically for the in-process and
    (future) subprocess kernels.

    The literal ``Error executing code`` substring is **load-bearing**: the
    ``/chat`` capture loop keys ``is_error`` off it (``api/main.py``) and the
    portable ``.py`` export marks errored cells with it (``session_export.py``).
    When the failure is a ``NameError``, append the self-healing hint (the warm
    namespace persists only within a live session). The caller prepends any
    captured stdout.
    """
    out = f"Error executing code:\n{traceback_str}"
    if is_name_error:
        ref = f"`{missing_name}`" if missing_name else "a variable"
        out += (
            f"\n\nHint: variables persist across execute_python calls only "
            f"within a live session. {ref} from an earlier call is gone — the "
            f"kernel may have been reset (e.g. a server restart). The dataset is "
            f"auto-loaded as `df` and `dataset_path` is set, so reload/rebuild "
            f"what you need; or call `load_result('name')` if you saved it "
            f"earlier with `save_result('name', obj)`."
        )
    return out
