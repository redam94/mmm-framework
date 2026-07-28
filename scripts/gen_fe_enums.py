"""Emit the frontend's mirror of the Python inference-method registry.

Run from the repo root::

    uv run python scripts/gen_fe_enums.py          # write the file
    uv run python scripts/gen_fe_enums.py --check  # exit 1 if it is stale

``tests/test_fe_enum_mirror.py`` runs the ``--check`` comparison in-process, so
CI fails when the two drift. That gate is the point of this script: the
frontend already hand-copied this enum twice, and both copies predated the
frequentist path, so a ``frequentist_ridge`` spec rendered an amber
"approximate" badge that contradicted the shipped rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mmm_framework.config.inference_methods import INFERENCE_METHODS

#: Written relative to the repo root.
TARGET = Path("frontend/src/api/generated/inferenceMethods.ts")

_HEADER = """// GENERATED — do not edit by hand.
//
// Source: src/mmm_framework/config/inference_methods.py
// Regenerate: uv run python scripts/gen_fe_enums.py
// Gated by: tests/test_fe_enum_mirror.py
//
// `inference.method` accepts a union of the Bayesian FitMethod members and the
// two frequentist InferenceMethod members. `approximate` is FALSE for the
// frequentist estimators: a penalized point estimate with bootstrap confidence
// intervals is not an approximation of a posterior, and labelling it one told
// users to "re-fit with NUTS" about a fit that never had a posterior to
// approximate.

export interface InferenceMethodInfo {
  value: string;
  label: string;
  paradigm: 'bayesian' | 'frequentist';
  approximate: boolean;
  intervalKind: 'credible' | 'confidence';
  caveat: string | null;
}

"""

_FOOTER = """
const BY_VALUE: Record<string, InferenceMethodInfo> = Object.fromEntries(
  INFERENCE_METHODS.map((m) => [m.value, m]),
);

/** Descriptor for a method value, or `undefined` when it is not recognized.
 *
 *  Returns `undefined` rather than guessing — the guess is what broke: a
 *  `!(nuts|smc)` fallback classified every unknown value as approximate. */
export function methodInfo(value: string | null | undefined): InferenceMethodInfo | undefined {
  if (!value) return undefined;
  return BY_VALUE[String(value).trim().toLowerCase()];
}

/** Human label, falling back to the raw value for an unrecognized method. */
export function methodLabel(value: string | null | undefined): string {
  return methodInfo(value)?.label ?? String(value ?? '');
}
"""


def render() -> str:
    rows = [
        {
            "value": m.value,
            "label": m.label,
            "paradigm": m.paradigm,
            "approximate": m.approximate,
            "intervalKind": m.interval_kind,
            "caveat": m.caveat,
        }
        for m in INFERENCE_METHODS
    ]
    body = json.dumps(rows, indent=2, ensure_ascii=False)
    # JSON is valid TS object-literal syntax; the quoted keys are intentional so
    # the diff of a regeneration stays minimal.
    return (
        f"{_HEADER}export const INFERENCE_METHODS: ReadonlyArray<InferenceMethodInfo> "
        f"= {body};\n{_FOOTER}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the generated file is missing or stale",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    target = root / TARGET
    expected = render()

    if args.check:
        actual = target.read_text() if target.exists() else None
        if actual == expected:
            return 0
        print(
            f"{TARGET} is stale — run: uv run python scripts/gen_fe_enums.py",
            file=sys.stderr,
        )
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(expected)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
