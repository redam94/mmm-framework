"""Model-defense report — the causal-rigor evidence, in plain English.

Turns the causal refutation suite (placebo / negative-control / random-common-
cause / data-subset), the sampler diagnostics, and any experiment calibration
into a one-click, CFO-facing "why you can trust this number" artifact. This is
the explainability collateral that makes the platform's rigor a sales asset.

``build_model_defense`` is a pure function over result dicts (testable);
``render_model_defense_html`` produces a self-contained, optionally-branded page.
"""

from __future__ import annotations

import html as _html
from typing import Any

_WHAT = {
    "vanish": "Feed the model a fake or irrelevant input; a trustworthy model "
    "should find essentially no effect.",
    "stable": "Perturb the data (a random extra factor, or a held-out subset); a "
    "trustworthy model's estimate should barely move.",
}

_LEVEL_COLOR = {
    "strong": ("#1f6f5c", "#e7f1ee"),
    "qualified": ("#a9741b", "#f7eed9"),
    "caution": ("#b3402a", "#f6e2dc"),
    "unknown": ("#5b5b66", "#eceef1"),
}


def _as_dict(obj: Any) -> dict:
    return obj.to_dict() if hasattr(obj, "to_dict") else (obj or {})


def _explain(test: dict) -> dict[str, Any]:
    kind = test.get("kind")
    passed = bool(test.get("passed"))
    if kind == "vanish":
        plain = (
            "the fake effect collapsed toward zero, so the real effect isn't an artifact."
            if passed
            else "the model still found an effect on the fake input — the result may be spurious."
        )
    else:  # stable
        plain = (
            "the estimate held steady under the perturbation."
            if passed
            else "the estimate shifted materially — the result looks fragile."
        )
    return {
        "name": test.get("name"),
        "kind": kind,
        "passed": passed,
        "channel": test.get("channel"),
        "what_it_checks": _WHAT.get(kind, ""),
        "verdict": ("On this test, " + plain),
        "original_effect": test.get("original_effect"),
        "refuted_effect": test.get("refuted_effect"),
    }


def build_model_defense(
    refutation: Any,
    *,
    convergence: Any | None = None,
    n_calibrated_experiments: int = 0,
    channel_label: str | None = None,
) -> dict[str, Any]:
    """Structured model-defense payload from a refutation result (+ optional
    convergence + calibration count). Accepts result objects or their dicts."""
    refutation = _as_dict(refutation)
    tests = refutation.get("tests") or []
    n_tests = len(tests)
    n_passed = int(refutation.get("n_passed", sum(1 for t in tests if t.get("passed"))))
    n_failed = int(refutation.get("n_failed", n_tests - n_passed))
    underpowered = bool(refutation.get("underpowered"))

    conv = _as_dict(convergence) if convergence is not None else None
    conv_ok: bool | None = None
    if conv:
        div = conv.get("divergences")
        rhat = conv.get("rhat_max")
        conv_ok = (div == 0) and (rhat is not None and rhat < 1.01)

    # Verdict ladder. Convergence failure dominates everything (a non-converged
    # fit can't be trusted regardless of refutation results).
    if conv_ok is False:
        verdict, level = (
            "Sampling did not converge — do not rely on this fit",
            "caution",
        )
    elif n_tests == 0:
        verdict, level = "Not yet assessed — run the refutation suite", "unknown"
    elif n_failed == 0 and not underpowered:
        verdict, level = "Robust — survived every refutation test", "strong"
    elif n_failed == 0:
        verdict, level = "Robust, with caveats (some tests underpowered)", "qualified"
    elif n_failed == 1:
        verdict, level = (
            "Mostly robust — one test flags a concern to review",
            "qualified",
        )
    else:
        verdict, level = (
            f"Needs scrutiny — {n_failed} refutation tests failed",
            "caution",
        )

    caveats = [
        "A passing refutation suite is evidence of robustness, not proof that every "
        "number is exactly right.",
        "These tests check causal robustness, not forecast accuracy — a model can "
        "predict well and attribute wrongly, and vice-versa.",
    ]
    if underpowered:
        caveats.append(
            "Some tests were underpowered; treat those passes as weaker evidence."
        )
    if n_calibrated_experiments > 0:
        caveats.append(
            f"This model is anchored to {n_calibrated_experiments} real experiment"
            f"{'s' if n_calibrated_experiments != 1 else ''}, so its causal claims "
            "don't rest on observational data alone."
        )
    else:
        caveats.append(
            "This model was not anchored to a randomized experiment — its causal "
            "claims rest on the modeling assumptions plus these refutations. A lift "
            "test would strengthen them."
        )

    return {
        "verdict": verdict,
        "level": level,
        "channel_label": channel_label,
        "n_tests": n_tests,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "underpowered": underpowered,
        "convergence": (
            {
                "divergences": conv.get("divergences"),
                "rhat_max": conv.get("rhat_max"),
                "ok": conv_ok,
            }
            if conv
            else None
        ),
        "n_calibrated_experiments": n_calibrated_experiments,
        "checks": [_explain(t) for t in tests],
        "caveats": caveats,
    }


def _esc(x: Any) -> str:
    return _html.escape("" if x is None else str(x))


def render_model_defense_html(
    payload: dict[str, Any],
    *,
    title: str = "Model Defense",
    brand: dict | None = None,
) -> str:
    """A self-contained, optionally-branded HTML page from a defense payload."""
    fg, bg = _LEVEL_COLOR.get(payload.get("level", "unknown"), _LEVEL_COLOR["unknown"])
    conv = payload.get("convergence")
    rows = []
    for c in payload.get("checks", []):
        mark = "✓ Pass" if c["passed"] else "✗ Fail"
        mcol = "#1f6f5c" if c["passed"] else "#b3402a"
        ch = (
            f" <span style='color:#6b7280'>({_esc(c['channel'])})</span>"
            if c.get("channel")
            else ""
        )
        rows.append(
            f"<tr><td><b>{_esc(c['name'])}</b>{ch}<div style='color:#5b5b66;font-size:.86em'>"
            f"{_esc(c['what_it_checks'])}</div></td>"
            f"<td style='color:{mcol};font-weight:600;white-space:nowrap'>{mark}</td>"
            f"<td style='color:#33333c'>{_esc(c['verdict'])}</td></tr>"
        )
    checks_html = (
        "\n".join(rows)
        or "<tr><td colspan='3' style='color:#5b5b66'>No refutation tests recorded.</td></tr>"
    )
    conv_html = ""
    if conv:
        ok = conv.get("ok")
        cm = "#1f6f5c" if ok else "#b3402a" if ok is False else "#5b5b66"
        conv_html = (
            "<div class='box'><b>Sampler convergence.</b> "
            f"<span style='color:{cm};font-weight:600'>{'converged' if ok else 'did NOT converge' if ok is False else 'reported'}</span> — "
            f"R&#770; max {_esc(round(conv['rhat_max'], 4)) if conv.get('rhat_max') is not None else '—'}, "
            f"{_esc(conv.get('divergences'))} divergence(s).</div>"
        )
    caveats_html = "".join(f"<li>{_esc(c)}</li>" for c in payload.get("caveats", []))
    sub = f"{payload.get('n_passed', 0)}/{payload.get('n_tests', 0)} causal refutation tests passed"
    if payload.get("channel_label"):
        title = f"{title} — {payload['channel_label']}"

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)}</title>
<style>
 body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#16161d;background:#faf9f6;line-height:1.6}}
 .wrap{{max-width:780px;margin:0 auto;padding:36px 28px}}
 h1{{font-family:Georgia,serif;font-size:1.7rem;margin:0 0 .2em}}
 .sub{{color:#5b5b66;margin:0 0 20px}}
 .verdict{{border-radius:12px;padding:16px 20px;margin:0 0 22px;font-weight:600;color:{fg};background:{bg};border:1px solid {fg}33}}
 table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:.93rem}}
 th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid #e7e3da;vertical-align:top}}
 th{{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:#5b5b66}}
 .box{{background:#fff;border:1px solid #e7e3da;border-radius:10px;padding:14px 18px;margin:14px 0;font-size:.92rem}}
 .caveats{{background:#fbf4ea;border:1px solid #ecdcc0;border-radius:10px;padding:14px 18px 14px 34px;margin:18px 0;font-size:.88rem;color:#4a4a40}}
 h2{{font-family:Georgia,serif;font-size:1.15rem;margin:1.4em 0 .3em}}
 .foot{{color:#8a8a93;font-size:.8rem;margin-top:26px}}
</style></head>
<body><div class="wrap">
 <h1>{_esc(title)}</h1>
 <p class="sub">{_esc(sub)} · the causal-rigor evidence behind this model.</p>
 <div class="verdict">{_esc(payload.get('verdict', ''))}</div>
 <h2>Refutation suite</h2>
 <p style="color:#33333c;font-size:.92rem;margin:.2em 0 0">Each test deliberately tries to <em>break</em> the model. A result you can defend survives them.</p>
 <table><thead><tr><th>Test</th><th>Result</th><th>What it means</th></tr></thead>
 <tbody>
{checks_html}
 </tbody></table>
 {conv_html}
 <h2>Read this honestly</h2>
 <ul class="caveats">{caveats_html}</ul>
 <p class="foot">Generated from the model's refutation suite, sampler diagnostics, and experiment calibration. Pre-registration + these refutations are how this platform reduces researcher degrees of freedom.</p>
</div></body></html>"""

    if brand:
        try:
            from mmm_framework.agents.report_builder import apply_branding_html

            page = apply_branding_html(page, brand)
        except Exception:
            pass
    return page


def model_defense_report(
    refutation: Any,
    *,
    convergence: Any | None = None,
    n_calibrated_experiments: int = 0,
    title: str = "Model Defense",
    channel_label: str | None = None,
    brand: dict | None = None,
) -> str:
    """One call: refutation (+ optional convergence/calibration) → branded HTML."""
    payload = build_model_defense(
        refutation,
        convergence=convergence,
        n_calibrated_experiments=n_calibrated_experiments,
        channel_label=channel_label,
    )
    return render_model_defense_html(payload, title=title, brand=brand)
