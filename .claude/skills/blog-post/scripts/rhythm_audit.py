#!/usr/bin/env python3
"""Report the rhythm signals the style checker cannot see.

``check_blog_style.py`` gates punctuation and structure. It says nothing about the part of
``technical-docs/writing-guide.md`` that matters most and is easiest to fake: sentence-length
variation and sentence openers.

This does not gate anything and there is no target score. It points at paragraphs worth rereading.
A clump is not automatically wrong; three short sentences in a row can be doing real work. What you
are looking for is a page where clumping is everywhere, which is what prose assembled to a shape
rather than to an argument looks like.

    python3 .claude/skills/blog-post/scripts/rhythm_audit.py docs/blog-my-post.html
    python3 .claude/skills/blog-post/scripts/rhythm_audit.py docs/blog-my-post.html --verbose
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def _load_checker():
    """Reuse the checker's prose extractor so both tools agree on what counts as prose."""
    path = REPO / "docs" / "tools" / "check_blog_style.py"
    spec = importlib.util.spec_from_file_location("check_blog_style", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'“])")


def paragraphs(doc: str) -> list[str]:
    """Body <p> text, minus the byline and figure captions, which are not argument prose."""
    main = re.search(r"(?s)<main.*</main>", re.sub(r"(?s)<!--.*?-->", " ", doc))
    if not main:
        main = re.search(r"(?s)<body.*</body>", doc)
    if not main:
        return []
    out = []
    for tag, body in re.findall(r"(?s)<p([^>]*)>(.*?)</p>", main.group(0)):
        if any(c in tag for c in ("post-meta", "chart-caption", "arc-kicker")):
            continue
        text = re.sub(r"(?s)<[^>]+>", "", body)
        text = re.sub(r"&[a-zA-Z]+\d*;|&#\d+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) >= 12:
            out.append(text)
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def audit(path: Path, verbose: bool = False) -> None:
    doc = path.read_text(encoding="utf-8")
    paras = paragraphs(doc)
    if not paras:
        print(f"{path.name}: no body paragraphs found")
        return

    all_sents: list[str] = []
    clumpy: list[tuple[int, list[int]]] = []
    for i, para in enumerate(paras, 1):
        sents = sentences(para)
        all_sents.extend(sents)
        lengths = [len(s.split()) for s in sents]
        # The guide: "Never let three consecutive sentences fall within five words of each other."
        runs = [
            j
            for j in range(len(lengths) - 2)
            if max(lengths[j : j + 3]) - min(lengths[j : j + 3]) <= 5
        ]
        if runs:
            clumpy.append((i, lengths))

    lengths = [len(s.split()) for s in all_sents]
    openers = [s.split()[0].strip("\"'“(").lower() for s in all_sents if s.split()]
    adjacent = [
        (i, openers[i]) for i in range(len(openers) - 1) if openers[i] == openers[i + 1]
    ]
    para_lengths = [len(sentences(p)) for p in paras]

    print(f"{path.name}")
    print(f"  {len(paras)} paragraphs, {len(all_sents)} sentences")
    print(
        f"  sentence words: min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, "
        f"max {max(lengths)}, mean {sum(lengths) / len(lengths):.1f}"
    )
    print(
        f"  under 8 words: {sum(1 for x in lengths if x < 8)}   "
        f"over 35: {sum(1 for x in lengths if x > 35)}"
    )
    print(
        f"  paragraph sentences: min {min(para_lengths)}, max {max(para_lengths)}, "
        f"one-sentence paragraphs {sum(1 for x in para_lengths if x == 1)}"
    )
    print(
        f"  clumped paragraphs (3 sentences within 5 words): "
        f"{len(clumpy)} of {len(paras)}"
    )
    print(f"  adjacent repeated openers: {len(adjacent)}")

    common = [(w, n) for w, n in Counter(openers).most_common(5) if n > 1]
    if common:
        print("  most frequent openers: " + ", ".join(f"{w!r} x{n}" for w, n in common))

    if verbose:
        if clumpy:
            print("\n  clumped paragraphs, by sentence length:")
            for i, lengths_ in clumpy:
                print(f"    para {i:>3}: {lengths_}")
        if adjacent:
            print("\n  repeated openers:")
            for i, word in adjacent:
                print(f"    {word!r}: {all_sents[i][:70]!r}")
                print(f"    {' ' * len(repr(word))}  {all_sents[i + 1][:70]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="list the clumped paragraphs"
    )
    args = ap.parse_args()
    _load_checker()  # fail early if the repo layout moved
    for f in args.files:
        audit(Path(f).resolve(), verbose=args.verbose)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
