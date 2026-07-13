"""Bundled example MFF datasets for a zero-effort first run.

The library ships two small, ready-to-model Master Flat File (MFF) datasets so a
new user can fit a real model without hand-authoring any data::

    from mmm_framework import load_example
    panel = load_example("national")   # a ready PanelDataset

Both datasets come from the framework's synthetic worlds and ship a *sealed
answer key* (each channel's true causal contribution and ROAS), so an estimate
can be graded against ground truth::

    truth = load_example_answer_key("national")
    truth["true_roas"]["TV"]

The raw long-format frame is available with ``as_frame=True`` for anyone who
wants to build their own :class:`~mmm_framework.config.MFFConfig`.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from importlib import resources

import pandas as pd

from ..config import (
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MFFConfig,
    create_geo_media_config,
    create_national_media_config,
)
from ..data_loader import PanelDataset, load_mff

__all__ = [
    "ExampleSpec",
    "EXAMPLES",
    "load_example",
    "list_examples",
    "load_example_answer_key",
]


@dataclass(frozen=True)
class ExampleSpec:
    """Declarative description of a bundled example dataset."""

    key: str
    csv: str
    kpi: str
    media: tuple[str, ...]
    controls: tuple[str, ...]
    geo: bool
    truth: str | None
    description: str


# Registry of bundled examples. The media/control variable sets are chosen to be
# *faithful* to each world's data-generating process — every real confounder is
# included — so the fitted estimates are unbiased and directly comparable to the
# shipped answer key. (The synthetic worlds also carry decoy ``noise_*`` columns
# and an on-path ``brand_awareness`` mediator; both are deliberately excluded.)
EXAMPLES: dict[str, ExampleSpec] = {
    "national": ExampleSpec(
        key="national",
        csv="national_mff.csv",
        kpi="Sales",
        media=("TV", "Search", "Social", "Display", "Video", "Radio", "Print"),
        controls=(
            "category_demand",
            "distribution",
            "price",
            "competitor_promo",
            "weather",
            "holiday",
        ),
        geo=False,
        truth="national_truth.json",
        description=(
            "104 weeks of national weekly data — 7 media channels and 6 controls "
            "(including the two demand confounders), Sales KPI. Ships a sealed answer key."
        ),
    ),
    "geo": ExampleSpec(
        key="geo",
        csv="geo_mff.csv",
        kpi="Sales",
        media=("TV", "Search", "Social", "Display"),
        controls=("Price",),
        geo=True,
        truth="geo_truth.json",
        description=(
            "91 weeks across 8 DMAs — 4 geo-level media channels and a Price control, "
            "geo-level Sales KPI. Ships a per-DMA answer key."
        ),
    ),
}


def _resource_text(filename: str) -> str:
    """Read a bundled data file, working from a wheel install or the source tree."""
    return (resources.files(__package__) / "data" / filename).read_text(
        encoding="utf-8"
    )


def list_examples() -> dict[str, str]:
    """Return ``{name: one-line description}`` for every bundled example."""
    return {name: spec.description for name, spec in EXAMPLES.items()}


def load_example(
    name: str = "national", *, as_frame: bool = False
) -> PanelDataset | pd.DataFrame:
    """Load a bundled example dataset, ready to model.

    Parameters
    ----------
    name:
        Which example to load — ``"national"`` (default) or ``"geo"``. See
        :func:`list_examples`.
    as_frame:
        If ``True``, return the raw long-format MFF :class:`pandas.DataFrame`
        instead of a built :class:`~mmm_framework.data_loader.PanelDataset`
        (useful for building your own config).

    Returns
    -------
    PanelDataset or pandas.DataFrame
        A ready-to-fit panel (default), or the raw MFF frame when
        ``as_frame=True``.

    Examples
    --------
    >>> from mmm_framework import load_example, BayesianMMM, ModelConfigBuilder, TrendConfig
    >>> panel = load_example("national")
    >>> panel.n_channels
    7
    """
    try:
        spec = EXAMPLES[name]
    except KeyError:
        raise ValueError(
            f"Unknown example {name!r}. Available: {sorted(EXAMPLES)}"
        ) from None

    df = pd.read_csv(io.StringIO(_resource_text(spec.csv)), parse_dates=["Period"])
    if as_frame:
        return df

    # Keep only the variables this example models. The synthetic worlds also carry
    # decoy ``noise_*`` columns and an on-path mediator; dropping them here means a
    # curated example loads without an "extra variables ignored" notice.
    modeled = {spec.kpi, *spec.media, *spec.controls}
    df = df[df["VariableName"].isin(modeled)].reset_index(drop=True)

    dims = [DimensionType.PERIOD] + ([DimensionType.GEOGRAPHY] if spec.geo else [])
    make_media = create_geo_media_config if spec.geo else create_national_media_config
    config = MFFConfig(
        kpi=KPIConfig(name=spec.kpi, dimensions=dims),
        media_channels=[make_media(m) for m in spec.media],
        controls=[
            ControlVariableConfig(name=c, dimensions=dims) for c in spec.controls
        ],
    )
    return load_mff(df, config)


def load_example_answer_key(name: str = "national") -> dict | None:
    """Load the sealed ground-truth answer key for an example, if one ships.

    Returns the parsed JSON (channels, ``true_contribution``, ``true_roas``,
    and DGP ``notes``) or ``None`` when the example has no bundled key.
    """
    try:
        spec = EXAMPLES[name]
    except KeyError:
        raise ValueError(
            f"Unknown example {name!r}. Available: {sorted(EXAMPLES)}"
        ) from None
    if spec.truth is None:
        return None
    return json.loads(_resource_text(spec.truth))
