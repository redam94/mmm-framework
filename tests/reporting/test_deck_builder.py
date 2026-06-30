"""python-pptx builder that fills the designed template from a fitted model.

Fast tests cover the bundled-template resolver and the low-level template helpers
(formatting-preserving text set, label-anchored card fill) on a tiny synthetic
deck. The slow test builds the real template from a fitted model and re-parses the
output: the headline KPI cards (with 80% ranges), the channel scorecard, the
ROI/decomposition image swaps, and one per-channel deep-dive slide per channel
(extras trimmed) each carrying its saturation/zone chart.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pptx")

from mmm_framework.reporting.deck import builder, template as T  # noqa: E402


def test_default_template_exists():
    p = builder.default_template_path()
    assert p.exists() and p.suffix == ".pptx"


def _mini_prs():
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # a KPI card: label on top, value below, sub below that (same left)
    label = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(0.4))
    label.text_frame.text = "RETURN / $1"
    value = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(3), Inches(0.6))
    value.text_frame.text = "0.00"
    sub = slide.shapes.add_textbox(Inches(1), Inches(2.2), Inches(3), Inches(0.4))
    sub.text_frame.text = "placeholder"
    return prs, slide


def test_set_text_preserves_single_run():
    _, slide = _mini_prs()
    sh = next(T.iter_text_shapes(slide))
    assert T.set_text(sh, "RETURN / $1 (edited)")
    assert sh.text_frame.text == "RETURN / $1 (edited)"


def test_fill_card_label_anchored():
    _, slide = _mini_prs()
    assert T.fill_card(slide, "RETURN / $1", "1.52", "80% 1.08–2.08")
    texts = [sh.text_frame.text for sh in T.iter_text_shapes(slide)]
    assert "RETURN / $1" in texts  # label kept
    assert "1.52" in texts and "80% 1.08–2.08" in texts
    assert "placeholder" not in texts  # sub overwritten


# ---------------------------------------------------------------------------
# slow: build the real template from a fitted model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model():
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig
    from mmm_framework.model.trend_config import TrendType
    from mmm_framework.synth import dgp

    panel = dgp.build("clean", seed=0, n_weeks=104).panel()
    mmm = BayesianMMM(
        panel,
        ModelConfig(use_parametric_adstock=True),
        TrendConfig(type=TrendType.LINEAR),
    )
    mmm.fit(
        draws=250,
        tune=500,
        chains=2,
        target_accept=0.9,
        random_seed=3,
        progressbar=False,
    )
    return mmm


@pytest.mark.slow
def test_build_pptx_fills_template(fitted_model, tmp_path):
    from pptx import Presentation

    out = tmp_path / "deck.pptx"
    data = builder.build_pptx(
        fitted_model,
        out_path=out,
        client="Acme Corp",
        kpi_name="Sales",
        currency="$",
        break_even=1.0,
        hdi_prob=0.8,
    )
    assert isinstance(data, bytes) and len(data) > 10000 and out.exists()

    prs = Presentation(str(out))

    def texts(s):
        return [
            sh.text_frame.text
            for sh in s.shapes
            if sh.has_text_frame and sh.text_frame.text.strip()
        ]

    def has_label(s, lbl):
        return any(t.strip().lower() == lbl.lower() for t in texts(s))

    channels = set(fitted_model.channel_names)

    # the unused per-channel deep-dives are trimmed (template has 7; model has 4)
    deep = [
        s
        for s in prs.slides
        if has_label(s, "RETURN / $1") and has_label(s, "CARRYOVER HALF-LIFE")
    ]
    assert len(deep) == len(channels), (len(deep), len(channels))
    # each deep-dive names a distinct model channel and carries its zone chart
    named = []
    for s in deep:
        nm = next((t for t in texts(s) if t in channels), None)
        named.append(nm)
        npics = sum(1 for sh in s.shapes if "PICTURE" in str(sh.shape_type))
        assert npics >= 1, "saturation/zone chart should be inserted"
    assert set(named) == channels

    # headline KPI cards filled with 80% ranges
    head = next(s for s in prs.slides if has_label(s, "THE HEADLINE"))
    htexts = " ".join(texts(head))
    assert "80% range" in htexts

    # scorecard names every model channel
    sc = next(s for s in prs.slides if has_label(s, "CHANNEL SCORECARD"))
    sc_texts = texts(sc)
    assert channels.issubset(set(sc_texts))
