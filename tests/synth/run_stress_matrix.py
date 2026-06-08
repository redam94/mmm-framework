"""CLI: fit the MMM across structural-violation scenarios and emit the matrix.

Examples
--------
    uv run python -m tests.synth.run_stress_matrix --quick
    uv run python -m tests.synth.run_stress_matrix --all --draws 800
    uv run python -m tests.synth.run_stress_matrix clean adstock_misspec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dgp import PRIORITY, SCENARIOS
from .harness import matrix_frame, run_matrix, to_markdown

# Control + the highest-prevalence silent-failure cases.
QUICK = [
    "clean",
    "unobserved_confounding",
    "reverse_causality",
    "multicollinearity",
    "adstock_misspec",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenarios", nargs="*", help="scenario names (default: --quick)")
    p.add_argument("--all", action="store_true", help="run every scenario")
    p.add_argument("--quick", action="store_true", help="control + top-4 violations")
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--tune", type=int, default=500)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--ref-draws", type=int, default=120)
    p.add_argument(
        "--no-refutation", action="store_true", help="skip refit-based suite"
    )
    p.add_argument(
        "--legacy-adstock", action="store_true", help="default 2-point blend"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="tests/synth/results", help="output dir")
    args = p.parse_args()

    if args.all:
        names = list(PRIORITY)
    elif args.scenarios:
        names = args.scenarios
    else:
        names = QUICK
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        p.error(f"unknown scenarios: {unknown}\nchoose from: {list(SCENARIOS)}")

    results = run_matrix(
        names,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        ref_draws=args.ref_draws,
        run_refutation=not args.no_refutation,
        parametric=not args.legacy_adstock,
        seed=args.seed,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(results)
    (out / "stress_matrix.md").write_text(md)
    matrix_frame(results).to_csv(out / "stress_matrix.csv", index=False)
    (out / "stress_matrix.json").write_text(
        json.dumps([r.to_row() for r in results], indent=2, default=str)
    )

    print("\n" + md)
    n_silent = sum(r.silent_failure for r in results)
    print(
        f"\nWrote {out}/stress_matrix.(md|csv|json) — "
        f"{n_silent} silent failure(s) across {len(results)} scenario(s)."
    )


if __name__ == "__main__":
    main()
