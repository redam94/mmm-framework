"""
Turn outlier flags into concrete, applyable treatments.

The mapping encodes MMM-specific semantics:

* A media **isolated_spike** is treated as a recording error — winsorize it.
  Leaving it in corrupts the channel's max-normalization and flattens its
  saturation curve.
* A **kpi_shock** is treated as a real demand event — add an event dummy
  control so the model explains it instead of blaming/crediting media.
  Winsorizing the KPI would bias media coefficients.
* A **heavy_tail_member** cluster is a noise property: dummy out the largest
  shocks and note that the model's likelihood is Normal (a Student-t
  likelihood is not currently supported), so the residual tail risk should be
  recorded as an assumption.
* A **level_shift** is a structural changepoint: switch the trend to
  ``piecewise`` (supported by the model) or add a step dummy — never
  winsorize it away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OutlierConfig
from .loading import EDAPanel
from .results import OutlierFlag, RemediationAction

STUDENT_T_ADVISORY = (
    "the model's likelihood is Normal (Student-t is not supported), so heavy-"
    "tailed noise cannot be absorbed by the likelihood — record this as a "
    "modeling assumption (category: data)"
)


def _dummy_name(flag: OutlierFlag) -> str:
    date_part = flag.period.replace("-", "_")
    var_part = "".join(ch if ch.isalnum() else "_" for ch in flag.variable)
    return f"outlier_{var_part}_{date_part}"


def _winsorize_cap(panel: EDAPanel, flag: OutlierFlag) -> float:
    """Cap value: max(robust expected, p99 of the series excluding the point)."""
    series = panel.series(flag.variable, flag.dims or None).astype(float)
    others = series[series.index != pd.Timestamp(flag.period)]
    p99 = (
        float(np.nanpercentile(others.to_numpy(), 99)) if len(others) else flag.expected
    )
    return float(max(flag.expected, p99))


def recommend_treatments(
    panel: EDAPanel,
    flags: list[OutlierFlag],
    config: OutlierConfig | None = None,
) -> list[RemediationAction]:
    """Build remediation recommendations for a set of flags."""
    cfg = config or OutlierConfig()
    actions: list[RemediationAction] = []

    spikes = [f for f in flags if f.kind == "isolated_spike"]
    drops = [f for f in flags if f.kind == "isolated_drop"]
    shocks = [f for f in flags if f.kind == "kpi_shock"]
    heavy = [f for f in flags if f.kind == "heavy_tail_member"]
    shifts = [f for f in flags if f.kind == "level_shift"]

    for f in spikes:
        cap = _winsorize_cap(panel, f)
        damage = (panel.df_wide[f.variable].max() / cap) if cap > 0 else float("nan")
        actions.append(
            RemediationAction(
                action_id=f"winsorize:{f.flag_id}",
                flag_ids=[f.flag_id],
                strategy="winsorize",
                params={"cap_value": cap},
                rationale=(
                    f"{f.variable} on {f.period} is {f.value:.4g} vs an expected "
                    f"~{f.expected:.4g} — an isolated spike consistent with a "
                    f"data-entry error. It alone sets the channel's normalization "
                    f"scale (~{damage:.1f}x the capped max), which would flatten "
                    f"its saturation curve. Cap it at {cap:.4g}; alternative: an "
                    f"event dummy if the spend was real."
                ),
            )
        )

    for f in drops:
        impute_value = float(max(f.expected, 0.0))
        actions.append(
            RemediationAction(
                action_id=f"impute:{f.flag_id}",
                flag_ids=[f.flag_id],
                strategy="impute",
                params={"value": impute_value},
                rationale=(
                    f"{f.variable} on {f.period} is recorded as {f.value:.4g} "
                    f"while neighboring weeks spend ~{f.expected:.4g} — the "
                    "pattern of a missed data load on an always-on channel. "
                    f"Impute the local baseline ({impute_value:.4g}) after "
                    "confirming with the user the channel wasn't actually dark "
                    "that week."
                ),
            )
        )

    for f in shocks:
        name = _dummy_name(f)
        actions.append(
            RemediationAction(
                action_id=f"dummy:{f.flag_id}",
                flag_ids=[f.flag_id],
                strategy="dummy",
                params={"dummy_name": name, "periods": [f.period]},
                rationale=(
                    f"{f.variable} on {f.period} is {f.value:.4g} vs expected "
                    f"~{f.expected:.4g} — likely a real demand event (promo/PR). "
                    f"Add event dummy control `{name}` so the model explains it "
                    "instead of crediting media. Only winsorize if the user "
                    "confirms it's a data error."
                ),
                spec_change={"add_control": name},
            )
        )

    if heavy:
        top = sorted(heavy, key=lambda f: -abs(f.value - f.expected))[
            : cfg.heavy_tail_top_k
        ]
        for f in top:
            name = _dummy_name(f)
            actions.append(
                RemediationAction(
                    action_id=f"dummy:{f.flag_id}",
                    flag_ids=[f.flag_id],
                    strategy="dummy",
                    params={"dummy_name": name, "periods": [f.period]},
                    rationale=(
                        f"{f.variable} on {f.period} is one of the largest shocks "
                        "in a heavy-tailed residual pattern — dummy it out so it "
                        "doesn't distort coefficient estimates."
                    ),
                    spec_change={"add_control": name},
                )
            )
        var = heavy[0].variable
        actions.append(
            RemediationAction(
                action_id=f"note:heavy_tails:{var}",
                flag_ids=[f.flag_id for f in heavy],
                strategy="note",
                params={},
                rationale=(
                    f"`{var}` shows heavy-tailed residuals "
                    f"({len(heavy)} shock(s) beyond the dummied top "
                    f"{min(len(top), cfg.heavy_tail_top_k)}); "
                    + STUDENT_T_ADVISORY
                    + "."
                ),
            )
        )

    for f in shifts:
        actions.append(
            RemediationAction(
                action_id=f"trend:{f.flag_id}",
                flag_ids=[f.flag_id],
                strategy="note",
                params={"break_period": f.period},
                rationale=(
                    f"`{f.variable}` has a sustained level shift around {f.period} "
                    "— a structural changepoint, not a point outlier. Do NOT "
                    "winsorize it. Set the trend to `piecewise` (or add a step-"
                    "dummy control starting at the break) so the baseline can "
                    "absorb it; otherwise channels whose spend moves at the break "
                    "will absorb the misfit."
                ),
                spec_change={"setting_path": "trend.type", "value": "piecewise"},
            )
        )

    return actions


# ---------------------------------------------------------------------------
# application (pure — caller handles file I/O)
# ---------------------------------------------------------------------------


def apply_treatments(
    df_long: pd.DataFrame,
    actions: list[RemediationAction],
    flags: list[OutlierFlag],
    *,
    date_col: str = "Period",
    kpi: str | None = None,
) -> pd.DataFrame:
    """Apply winsorize / dummy / exclude actions to an MFF long frame.

    Returns a NEW frame; the input is not mutated. ``flags`` provides the
    (variable, period, dims) targeting for each action's ``flag_ids``.
    """
    flag_by_id = {f.flag_id: f for f in flags}
    out = df_long.copy()
    dates = pd.to_datetime(out[date_col])

    def _row_mask(flag: OutlierFlag) -> pd.Series:
        mask = (out["VariableName"] == flag.variable) & (
            dates == pd.Timestamp(flag.period)
        )
        for dim, val in (flag.dims or {}).items():
            if dim in out.columns:
                mask &= out[dim].astype(str) == val
        return mask

    new_rows: list[pd.DataFrame] = []
    excluded_periods: set[pd.Timestamp] = set()

    for action in actions:
        if action.strategy in ("winsorize", "impute"):
            new_value = float(
                action.params["cap_value"]
                if action.strategy == "winsorize"
                else action.params["value"]
            )
            for fid in action.flag_ids:
                flag = flag_by_id.get(fid)
                if flag is None:
                    continue
                out.loc[_row_mask(flag), "VariableValue"] = new_value

        elif action.strategy == "dummy":
            name = action.params["dummy_name"]
            periods = {pd.Timestamp(p) for p in action.params.get("periods", [])}
            # Build the dummy on the grain of the KPI (or the flagged variable):
            # one row per existing (period, dims) row of that variable.
            flag = next(
                (flag_by_id[fid] for fid in action.flag_ids if fid in flag_by_id),
                None,
            )
            grain_var = kpi or (flag.variable if flag else None)
            if grain_var is None:
                continue
            template = df_long[df_long["VariableName"] == grain_var].copy()
            template["VariableName"] = name
            template["VariableValue"] = (
                pd.to_datetime(template[date_col]).isin(periods).astype(float)
            )
            new_rows.append(template)

        elif action.strategy == "exclude_periods":
            excluded_periods.update(
                pd.Timestamp(p) for p in action.params.get("periods", [])
            )

        # "note" actions change the spec / record assumptions — no data edit.

    if new_rows:
        out = pd.concat([out, *new_rows], ignore_index=True)
    if excluded_periods:
        out = out[~pd.to_datetime(out[date_col]).isin(excluded_periods)].reset_index(
            drop=True
        )
    return out


__all__ = ["recommend_treatments", "apply_treatments", "STUDENT_T_ADVISORY"]
