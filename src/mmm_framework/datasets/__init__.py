"""Bundled example datasets for a zero-effort first run.

    >>> from mmm_framework.datasets import load_example
    >>> panel = load_example("national")

``load_example`` and ``list_examples`` are also re-exported from the top-level
package (``from mmm_framework import load_example``).
"""

from .examples import (
    EXAMPLES,
    ExampleSpec,
    list_examples,
    load_example,
    load_example_answer_key,
)

__all__ = [
    "EXAMPLES",
    "ExampleSpec",
    "list_examples",
    "load_example",
    "load_example_answer_key",
]
