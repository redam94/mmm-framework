# Docs-freshness automation — the contract it has to meet

An external schedule (not a workflow in this repo — see "Where the driver
lives") pushes branches named `docs-freshness/<date>` containing fixes for stale
documentation. Between 2026-07-01 and 2026-07-24 it produced **21 branches, zero
PRs, and zero merges**. This file records what went wrong and the rules any
future version of that automation must follow.

## What the failure cost

The wasted runs were the cheap part. Because nothing landed:

- The same staleness was re-detected daily. `README.md` was "fixed" in **15
  separate branches**, none of which reached `main`.
- The errors stayed live the whole time — including a real behavior bug, not a
  docs fix, sitting unmerged in `docs-freshness/2026-07-22`: `compute_cross_effects`
  probed for `get_cross_effect_summary` (singular) when both model classes define
  `get_cross_effects_summary`, so every report fell through to a manual branch
  that reported undeclared, structurally-zero outcome pairs as *estimated*
  cross-effects. That shipped to users for three weeks while the fix sat on a
  branch nobody could see.
- Nothing alerted. A branch with no PR is invisible in every view a maintainer
  actually looks at.

The output also needed **review, not blind merging**: two of the automation's
targets were false positives (`docs/glossary.html` already documented
`nuts_sampler` correctly, and `docs/causal-03-structural-mediation.html` calls an
extension model whose `fit()` legitimately accepts it).

## Rules

1. **Every run opens a PR.** A branch with no PR does not exist. If the diff is
   docs-only and the gates are green, pushing straight to `main` is also fine —
   what is not fine is a branch that terminates in nothing.
2. **One long-lived branch, rebased.** Accumulate onto a single
   `docs-freshness` branch rather than minting a dated one per run. One PR that
   updates daily is reviewable; 21 are not.
3. **The backlog is alarmed.** `.github/workflows/branch-backlog.yml` runs daily
   and **fails** when more than 3 `docs-freshness/*` branches are unmerged with
   no PR, so silent accumulation surfaces on day 3 instead of day 21.
4. **Mechanical drift belongs in CI, not in the automation.** Anything a static
   check can catch should fail a test instead of being re-discovered daily by an
   LLM. `tests/test_docs_snippets.py` now covers `README.md`, `CLAUDE.md` and
   `technical-docs/*.md` in addition to `docs/*.html` (#172) — that is the root
   cause of this backlog closed. What is left for the automation is the
   genuinely semantic drift: prose that is still true-shaped but no longer true.
5. **Its output is reviewed.** Verify each claimed staleness against the live
   API before landing it. Roughly 2 in 90 of the audited findings were wrong.

## Where the driver lives

Unknown from the repo alone. There is no workflow in `.github/workflows/` that
produces these branches, and the branches carry no PRs, so the driver is an
external schedule — a cron job or a scheduled agent on someone's machine or
account. **If you are looking for it, check your scheduled agents / routines
list**, not this repository. Whatever it is, it pushes a branch and stops.

## Cleaning up the existing backlog

The 21 branches were audited on 2026-07-25 rather than replayed (`main` had
moved and the branches overlap heavily). Method: extract every removed line from
all 23 commits, keep those whose text still exists in `main` (91 candidates
across 21 files), verify each against the live API. Everything worth keeping
landed in #168; their SHAs are recorded in that PR's thread, so the originals
stay recoverable after the branches are deleted.

## Related

- #172 — the snippet gate that now covers Markdown (root cause).
- #173 — this process issue.
- `technical-docs/doc-snippet-testing.md` — what the gate checks and how to opt
  a block out of it.
