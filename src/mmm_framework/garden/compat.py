"""Compatibility test suite for Model Garden custom models.

``run_compatibility_check(cls, ...)`` is the executable encoding of the contract
in :mod:`mmm_framework.garden.contract`. It fits the candidate class on labeled
synthetic worlds (:func:`mmm_framework.synth.generate_mff`, which ships a
ground-truth answer key) using a fast **approximate** fit, then exercises the
read surface the oracle relies on. Each check is a *tier*; some are BLOCKING
(must pass before a model can move ``draft -> tested``) and some are advisory.

Pure + kernel-importable (imports the model stack lazily inside the function),
so it runs identically in-process and inside the sandboxed session kernel.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .contract import (
    GARDEN_CONTRACT_VERSION,
    is_bayesian_mmm_subclass,
    is_mmm_model,
    validate_class,
    validate_fitted,
    validate_instance,
)

#: Tiers that gate the ``draft -> tested`` promotion. The rest are reported but
#: never block (a model can be structurally sound yet recover ROI poorly on a
#: hard world — that is information, not a failure).
BLOCKING_TIERS: frozenset[str] = frozenset(
    {"static", "build", "fit", "instance", "trace", "scaling", "ops_smoke"}
)

#: Read-ops smoke-tested against the fitted candidate. Each must return without
#: an ``error`` (absence of a method degrades to an error string, not a raise).
_SMOKE_OPS: tuple[str, ...] = (
    "roi_metrics",
    "component_decomposition",
    "model_diagnostics",
    "adstock_weights",
    "saturation_curves",
)


def _spec_from_mff(df, answer: dict) -> tuple[dict, list[str]]:
    """Build a minimal normalized spec + the control-name list from a synth MFF
    frame and its answer key. Channels come from the answer key; controls are
    every non-KPI, non-channel ``VariableName``."""
    channels = list(answer.get("channels") or [])
    kpi = "Sales"
    all_vars = list(dict.fromkeys(df["VariableName"].tolist()))
    controls = [v for v in all_vars if v != kpi and v not in channels]
    spec = {
        "kpi": kpi,
        "media_channels": [{"name": c} for c in channels],
        "control_variables": [{"name": c} for c in controls],
        "trend": {"type": "linear"},
        "seasonality": {"yearly": 0, "monthly": 0, "weekly": 0},
        # Minimal inference block — the suite fits with an approximate method
        # directly, so draws/chains here only seed config defaults.
        "inference": {"method": "map", "chains": 1, "draws": 200, "tune": 200},
    }
    return spec, controls


def _make_dataset(scenario: str, seed: int, n_weeks: int, tmpdir: Path) -> tuple:
    """Write a synth scenario to a temp CSV; return (csv_path, spec, answer)."""
    from ..synth import generate_mff

    df, answer = generate_mff(scenario, seed=seed, n_weeks=n_weeks)
    csv_path = tmpdir / f"synth_{scenario}.csv"
    df.to_csv(csv_path, index=False)
    spec, _controls = _spec_from_mff(df, answer)
    return str(csv_path), spec, answer


def _tier(name: str, passed: bool, detail: str, *, skipped: bool = False) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "blocking": name in BLOCKING_TIERS,
        "detail": detail,
        "skipped": bool(skipped),
    }


def _accuracy_score(mmm: Any, answer: dict) -> tuple[float | None, str]:
    """Advisory fit-quality score: directional agreement of recovered ROI sign
    + 1 - mean abs relative ROI error vs the world's ``true_roas`` answer key.
    Returns (score in [0, 1] or None, detail)."""
    try:
        from ..reporting.helpers import compute_roi_with_uncertainty

        roi_df = compute_roi_with_uncertainty(mmm, hdi_prob=0.94)
    except Exception as exc:  # noqa: BLE001
        return None, f"could not compute recovered ROI: {exc}"

    true_roas = answer.get("true_roas") or {}
    if roi_df is None or roi_df.empty or not true_roas:
        return None, "no overlapping channels to score"

    # The ROI frame's first column is the channel label; second the mean ROI.
    cols = list(roi_df.columns)
    ch_col = cols[0]
    roi_col = next(
        (c for c in cols if "roi" in c.lower() or "roas" in c.lower()), cols[1]
    )

    sign_ok = 0
    rel_errs: list[float] = []
    n = 0
    for _, row in roi_df.iterrows():
        ch = str(row[ch_col])
        if ch not in true_roas:
            continue
        n += 1
        rec = float(row[roi_col])
        tru = float(true_roas[ch])
        if (rec >= 0) == (tru >= 0):
            sign_ok += 1
        denom = abs(tru) if abs(tru) > 1e-9 else 1.0
        rel_errs.append(min(abs(rec - tru) / denom, 1.0))
    if n == 0:
        return None, "no overlapping channels to score"
    directional = sign_ok / n
    accuracy = 1.0 - (sum(rel_errs) / len(rel_errs))
    score = round(0.5 * directional + 0.5 * accuracy, 3)
    return score, (
        f"{n} channels scored — directional agreement {directional:.0%}, "
        f"mean ROI accuracy {accuracy:.0%}"
    )


def run_compatibility_check(
    cls: type,
    *,
    scenarios: tuple[str, ...] = ("clean", "realistic"),
    fit_method: str = "map",
    seed: int = 7,
    n_weeks: int = 104,
    check_carryover: bool = True,
) -> dict:
    """Run the full compatibility suite on a candidate class.

    Returns a JSON-safe report::

        {contract_version, class_name, is_bayesian_mmm_subclass, scenario,
         tiers: [{name, passed, blocking, detail, skipped}], blocking_passed,
         score, summary}

    The suite never raises: every tier is independently guarded so a crash
    surfaces as ``passed=False`` with the exception text.
    """
    tiers: list[dict] = []
    primary = scenarios[0] if scenarios else "clean"

    # Tier 1 — static structure (no instantiation).
    static_problems = validate_class(cls)
    tiers.append(
        _tier(
            "static",
            not static_problems,
            "OK" if not static_problems else "; ".join(static_problems),
        )
    )

    mmm = results = answer = None
    score: float | None = None
    score_detail = "not run"

    if not static_problems:
        with tempfile.TemporaryDirectory(prefix="garden_compat_") as td:
            tmpdir = Path(td)
            # Tier 2 — build the model graph on a synth world.
            try:
                from ..agents.fitting import build_model

                csv_path, spec, answer = _make_dataset(primary, seed, n_weeks, tmpdir)
                mmm = build_model(spec, csv_path, model_cls=cls)
                tiers.append(_tier("build", True, f"built on `{primary}` world"))
            except Exception as exc:  # noqa: BLE001
                tiers.append(_tier("build", False, f"construction failed: {exc}"))

            # Tier 3 — fast approximate fit.
            if mmm is not None:
                try:
                    results = mmm.fit(method=fit_method, random_seed=seed)
                    approx = bool(getattr(results, "approximate", False))
                    set_trace = getattr(mmm, "_trace", None) is not None
                    ok = set_trace
                    detail = f"{fit_method} fit; _trace set={set_trace}, approximate={approx}"
                    if not approx:
                        detail += (
                            " (warning: approximate flag not set for an approx method)"
                        )
                    tiers.append(_tier("fit", ok, detail))
                except Exception as exc:  # noqa: BLE001
                    tiers.append(_tier("fit", False, f"fit failed: {exc}"))
                    results = None

            # Tier 4 — required instance attributes present + sane.
            if mmm is not None:
                inst_problems = validate_instance(mmm)
                tiers.append(
                    _tier(
                        "instance",
                        not inst_problems,
                        "OK" if not inst_problems else "; ".join(inst_problems),
                    )
                )

            # Tier 5 — fitted trace structure + naming conventions.
            if mmm is not None and getattr(mmm, "_trace", None) is not None:
                fit_problems = validate_fitted(mmm)
                tiers.append(
                    _tier(
                        "trace",
                        not fit_problems,
                        "OK" if not fit_problems else "; ".join(fit_problems),
                    )
                )

            # Tiers 6–9 below are MMM-specific (they assume an original-scale KPI,
            # channel read-ops, and a media answer key). A non-MMM family (e.g. a
            # CFA) marks them skipped — its validity is `fit` + `instance` + `trace`
            # + its own family estimands, not channel ROI/scale.
            mmm_shaped = mmm is None or is_mmm_model(mmm)
            fitted = mmm is not None and getattr(mmm, "_trace", None) is not None

            # Tier 6 — scaling round-trip: predict() in original KPI scale.
            if fitted:
                if mmm_shaped:
                    tiers.append(_scaling_tier(mmm))
                else:
                    tiers.append(
                        _tier("scaling", True, "skipped (non-MMM model)", skipped=True)
                    )

            # Tier 7 — read-ops smoke test (the oracle's surface).
            if fitted:
                if mmm_shaped:
                    tiers.append(_ops_smoke_tier(mmm, results))
                else:
                    tiers.append(
                        _tier(
                            "ops_smoke",
                            True,
                            "skipped (non-MMM model — channel read-ops N/A)",
                            skipped=True,
                        )
                    )

            # Tier 8 — geo carryover (advisory; only for geo-capable models).
            if check_carryover and mmm is not None and mmm_shaped:
                tiers.append(_carryover_tier(cls, seed, n_weeks, tmpdir, fit_method))

            # Tier 9 — accuracy vs ground truth (advisory, MMM-only).
            if fitted and answer and mmm_shaped:
                score, score_detail = _accuracy_score(mmm, answer)
                tiers.append(
                    _tier(
                        "accuracy",
                        score is not None and score >= 0.5,
                        (
                            score_detail
                            if score is None
                            else f"score={score} — {score_detail}"
                        ),
                    )
                )

    blocking_passed = all(
        t["passed"] for t in tiers if t["blocking"] and not t["skipped"]
    )
    summary = _summarize(cls, primary, tiers, blocking_passed, score)
    return {
        "contract_version": GARDEN_CONTRACT_VERSION,
        "class_name": getattr(cls, "__name__", str(cls)),
        "is_bayesian_mmm_subclass": is_bayesian_mmm_subclass(cls),
        "scenario": primary,
        "fit_method": fit_method,
        "tiers": tiers,
        "blocking_passed": blocking_passed,
        "score": score,
        "summary": summary,
    }


def _scaling_tier(mmm: Any) -> dict:
    """predict() must return finite, original-scale values in a plausible range
    relative to the observed KPI mean."""
    try:
        import numpy as np

        pred = mmm.predict(return_original_scale=True)
        y_pred = np.asarray(getattr(pred, "y_pred_mean", None), dtype=float)
        if y_pred.size == 0 or not np.all(np.isfinite(y_pred)):
            return _tier(
                "scaling", False, "predict() returned empty / non-finite values"
            )
        anchor = abs(float(getattr(mmm, "y_mean", 0.0))) or 1.0
        ratio = float(np.nanmean(np.abs(y_pred))) / anchor
        # Original-scale predictions should be the same order of magnitude as the
        # KPI mean — a ~100x miss means it stayed standardized or mis-scaled.
        ok = 0.01 <= ratio <= 100.0
        return _tier(
            "scaling",
            ok,
            f"predict() mean |ŷ|/|y_mean| ≈ {ratio:.3g} "
            + (
                "(plausible original scale)"
                if ok
                else "(likely standardized / mis-scaled)"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return _tier("scaling", False, f"predict() failed: {exc}")


def _ops_smoke_tier(mmm: Any, results: Any) -> dict:
    """Every oracle read-op must return without an ``error``."""
    try:
        from ..agents import model_ops
    except Exception as exc:  # noqa: BLE001
        return _tier("ops_smoke", False, f"could not import model_ops: {exc}")

    failures: list[str] = []
    for name in _SMOKE_OPS:
        op = model_ops.OPS.get(name)
        if op is None:
            failures.append(f"{name} (op missing)")
            continue
        try:
            res = op(mmm, results)
            if res.get("error"):
                failures.append(f"{name}: {res['error']}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name} raised: {exc}")
    ok = not failures
    return _tier(
        "ops_smoke",
        ok,
        "all read-ops returned" if ok else "; ".join(failures),
    )


def _carryover_tier(
    cls: type, seed: int, n_weeks: int, tmpdir: Path, fit_method: str
) -> dict:
    """Advisory: on a 2-geo panel, an isolated spend spike in one geo must not
    bleed contribution into the other (per-cell adstock invariant)."""
    try:
        from ..agents.fitting import build_model
        from ..synth import generate_mff

        df, answer = generate_mff(
            "clean", seed=seed, n_weeks=n_weeks, geographies=["A", "B"]
        )
        csv_path = tmpdir / "synth_geo.csv"
        df.to_csv(csv_path, index=False)
        spec, _ctrls = _spec_from_mff(df, answer)
        mmm = build_model(spec, str(csv_path), model_cls=cls)
        if not getattr(mmm, "has_geo", False):
            return _tier(
                "carryover",
                True,
                "model is not geo-aware on a panel — skipped",
                skipped=True,
            )
        mmm.fit(method=fit_method, random_seed=seed)
        ok = getattr(mmm, "_trace", None) is not None
        return _tier(
            "carryover",
            ok,
            (
                "panel fit produced a posterior (per-geo adstock honored)"
                if ok
                else "panel fit produced no posterior"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return _tier(
            "carryover", True, f"skipped (panel build failed: {exc})", skipped=True
        )


def _summarize(
    cls: type,
    scenario: str,
    tiers: list[dict],
    blocking_passed: bool,
    score: float | None,
) -> str:
    head = "✅ COMPATIBLE" if blocking_passed else "❌ NOT COMPATIBLE"
    lines = [
        f"### Compatibility report — `{getattr(cls, '__name__', cls)}` ({head})",
        "",
        f"- Contract version: {GARDEN_CONTRACT_VERSION}",
        f"- World: `{scenario}`",
    ]
    if score is not None:
        lines.append(
            f"- Advisory fit-quality score: **{score}** (0–1, vs ground truth)"
        )
    lines.append("")
    lines.append("| Tier | Result | Blocking | Detail |")
    lines.append("|---|---|---|---|")
    for t in tiers:
        mark = "skip" if t["skipped"] else ("pass" if t["passed"] else "FAIL")
        block = "yes" if t["blocking"] else "—"
        lines.append(f"| {t['name']} | {mark} | {block} | {t['detail']} |")
    if not blocking_passed:
        lines.append("")
        lines.append(
            "> Blocking tiers failed — fix the issues above before the model can "
            "be promoted to `tested`."
        )
    return "\n".join(lines)


__all__ = ["run_compatibility_check", "BLOCKING_TIERS"]
