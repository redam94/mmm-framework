#!/usr/bin/env python3
"""Build the client-side search index + glossary tooltip data for the docs site.

Run from the docs/ directory (or any cwd — paths resolve relative to docs/):

    python3 tools/build_search_index.py

Idempotent; re-run after adding or editing pages. Two outputs, both consumed
lazily by shared/components.js (the Cmd-K search palette and the glossary
tooltips are progressive enhancements — the site works without these files):

  shared/search-index.json  [{p: page, t: title, d: description,
                              h: [[id, heading], ...], b: body-text excerpt}]
  shared/glossary.json      [{id, term, def}] from glossary.html's <dt>/<dd>
"""
import glob
import json
import os
import re
from html.parser import HTMLParser

DOCS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(DOCS)

EXCLUDE = {"TEMPLATE.html", "TEMPLATE-SIDEBAR.html", "404.html",
           "mmm-example-report.html"}  # the example report is a rendered artifact
BODY_CHARS = 4500      # per-page body excerpt cap (keeps the index ~small)
DEF_CHARS = 220        # glossary tooltip length cap


class PageExtractor(HTMLParser):
    """Pull title, meta description, h2/h3 headings (with ids) and body text."""

    SKIP = {"script", "style", "nav", "footer", "svg", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.description = ""
        self.headings = []      # [(id, text)]
        self.body_parts = []
        self._skip_depth = 0
        self._in_title = False
        self._heading = None    # (id, [text parts]) while inside h2/h3
        self._in_body = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "meta" and attrs.get("name") == "description":
            self.description = attrs.get("content", "")
        elif tag == "body":
            self._in_body = True
        elif tag in ("h2", "h3") and self._in_body:
            self._heading = (attrs.get("id", ""), [])

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        elif tag in ("h2", "h3") and self._heading is not None:
            text = " ".join("".join(self._heading[1]).split())
            if text:
                self.headings.append((self._heading[0], text))
            self._heading = None

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        elif self._heading is not None:
            self._heading[1].append(data)
        elif self._in_body:
            self.body_parts.append(data)


class GlossaryExtractor(HTMLParser):
    """Pull (id, term, definition) triples from glossary.html <dt>/<dd> pairs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.entries = []       # [{id, term, def}]
        self._dt = None         # (id, [parts]) while inside <dt>
        self._dd = None         # [parts] while inside <dd>
        self._pending = None    # last finished dt awaiting its dd

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "dt":
            self._dt = (attrs.get("id", ""), [])
        elif tag == "dd" and self._pending:
            self._dd = []

    def handle_endtag(self, tag):
        if tag == "dt" and self._dt is not None:
            term = " ".join("".join(self._dt[1]).split())
            self._pending = (self._dt[0], term)
            self._dt = None
        elif tag == "dd" and self._dd is not None and self._pending:
            definition = " ".join("".join(self._dd).split())
            gid, term = self._pending
            if gid and term and definition:
                if len(definition) > DEF_CHARS:
                    definition = definition[: DEF_CHARS - 1].rsplit(" ", 1)[0] + "…"
                self.entries.append({"id": gid, "term": term, "def": definition})
            self._dd = None
            self._pending = None

    def handle_data(self, data):
        if self._dt is not None:
            self._dt[1].append(data)
        elif self._dd is not None:
            self._dd.append(data)


def clean_glossary_term(term):
    """'ROI / mROI' → primary lookup form 'ROI'; strip parentheticals."""
    term = re.sub(r"\s*\(.*?\)", "", term)
    term = term.split("/")[0].strip()
    return term


def build_search_index():
    index = []
    for path in sorted(glob.glob("*.html")):
        if path in EXCLUDE:
            continue
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        parser = PageExtractor()
        parser.feed(raw)

        title = parser.title.split("|")[0].strip() or path
        body = " ".join(" ".join(parser.body_parts).split())
        noise = {"on this page", "the math series", "the workshop series",
                 "the stress series", "the aurora series", "interactive",
                 "getting started", "quick start", "core concepts"}
        headings = [[hid, text] for hid, text in parser.headings
                    if text.lower() not in noise]
        index.append({
            "p": path,
            "t": title,
            "d": parser.description.strip(),
            "h": headings[:40],
            "b": body[:BODY_CHARS],
        })

    out = os.path.join("shared", "search-index.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out}: {len(index)} pages, "
          f"{os.path.getsize(out) / 1024:.0f} KiB")


def build_glossary():
    with open("glossary.html", encoding="utf-8") as fh:
        raw = fh.read()
    parser = GlossaryExtractor()
    parser.feed(raw)

    entries = []
    seen = set()
    for entry in parser.entries:
        term = clean_glossary_term(entry["term"])
        # Tooltip-linkable terms: multi-char words, no single letters/symbols.
        if len(term) < 3 or term.lower() in seen:
            continue
        seen.add(term.lower())
        entries.append({"id": entry["id"], "term": term, "def": entry["def"]})

    out = os.path.join("shared", "glossary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out}: {len(entries)} terms")


if __name__ == "__main__":
    build_search_index()
    build_glossary()
