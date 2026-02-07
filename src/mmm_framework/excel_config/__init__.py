"""
Excel-based model configuration for MMM Framework.

Provides tools to generate pre-filled Excel templates from MFF data
and parse completed templates back into framework configuration objects.

Usage
-----
Generate a template from MFF data:

    >>> from mmm_framework.excel_config import TemplateGenerator
    >>> path = TemplateGenerator.from_mff("my_data.csv")
    >>> # Edit the generated Excel file, then:

Parse a completed template:

    >>> from mmm_framework.excel_config import TemplateParser
    >>> mff_config, model_config, trend_config = TemplateParser.parse("config.xlsx")

Lower-level discovery (for programmatic inspection):

    >>> from mmm_framework.excel_config import discover_mff
    >>> discovery = discover_mff("my_data.csv")
    >>> for v in discovery.variables:
    ...     print(f"{v.name}: {v.role} ({', '.join(v.dimensions)})")
"""

from .generator import (
    DiscoveredVariable,
    MFFDiscovery,
    TemplateGenerator,
    discover_mff,
)
from .heuristics import (
    VariableStats,
    classify_variable,
    generate_display_name,
)
from .parser import (
    TemplateParseError,
    TemplateParser,
    TemplateValidationError,
)

__all__ = [
    # Generator
    "TemplateGenerator",
    "discover_mff",
    "DiscoveredVariable",
    "MFFDiscovery",
    # Parser
    "TemplateParser",
    "TemplateParseError",
    "TemplateValidationError",
    # Heuristics
    "classify_variable",
    "generate_display_name",
    "VariableStats",
]
