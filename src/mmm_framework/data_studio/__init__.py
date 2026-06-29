"""Data Studio — staged upload → interactive EDA → clean → convert-to-dataset.

A pre-fit data-preparation layer for the Oracle session. A raw upload is staged
(NOT yet the working dataset); the user runs interactive EDA against it and
builds an ordered, replayable transform pipeline (rename/drop/cast/parse-date/
fill-missing/dedup/filter/winsorize/impute/event-dummy). On commit the cleaned
frame is materialised as MFF-long CSV and promoted to the session's
``dataset_path`` with role-derived spec keys — with no chat/LLM round-trip.

* :mod:`~mmm_framework.data_studio.transforms` — pure, shape-aware pipeline ops.
* :mod:`~mmm_framework.data_studio.service` — staging manifest IO, EDA-on-frame
  (reuses :mod:`mmm_framework.eda`), and the commit-artifact builder.

The package is free of LangChain/LangGraph imports so it unit-tests in isolation
(the same boundary as :mod:`mmm_framework.agents.spec_locks`).
"""

from __future__ import annotations

from .transforms import (
    PipelineResult,
    TransformError,
    apply_pipeline,
    is_long_frame,
    user_columns,
)

__all__ = [
    "PipelineResult",
    "TransformError",
    "apply_pipeline",
    "is_long_frame",
    "user_columns",
]
