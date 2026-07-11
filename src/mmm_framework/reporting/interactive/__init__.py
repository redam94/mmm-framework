"""Interactive MMM Results Report — embedded-posterior, recompute-in-browser.

See :mod:`.facts` (extraction), :mod:`.insights` (narrative),
:mod:`.script` (client-side engine) and :mod:`.generator` (HTML shell).
"""

from .facts import interactive_report_facts
from .generator import InteractiveReportGenerator
from .insights import INTERACTIVE_INSIGHT_SLOTS, build_interactive_insights

__all__ = [
    "InteractiveReportGenerator",
    "interactive_report_facts",
    "build_interactive_insights",
    "INTERACTIVE_INSIGHT_SLOTS",
]
