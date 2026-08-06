#!/usr/bin/env python3
"""Build the REST API reference page (docs/rest-api.html).

Reads the OpenAPI 3.x schema exported from the live FastAPI app
(docs/shared/openapi.json), groups every operation into a functional area,
renders static HTML for each operation (method badge, path, summary, and a
collapsible detail block with parameters / request body / response shapes),
and injects the result into docs/rest-api.html between the
``<!-- REST-DOCS:BEGIN -->`` / ``<!-- REST-DOCS:END -->`` markers.  The
sidebar table of contents is injected between ``<!-- REST-TOC:BEGIN -->`` /
``<!-- REST-TOC:END -->``.

Run from docs/ (or anywhere -- paths resolve relative to this file):

    python3 tools/build_rest_docs.py

Stdlib only.  The script hard-asserts that every operation in the spec is
rendered exactly once, that the operation count matches EXPECTED_OPS, that
the resulting HTML has balanced tags, and that the page stays under
MAX_PAGE_BYTES.  When the API surface changes, re-export openapi.json,
update EXPECTED_OPS, and re-run.
"""

from __future__ import annotations

import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
SPEC_PATH = DOCS / "shared" / "openapi.json"
PAGE_PATH = DOCS / "rest-api.html"

EXPECTED_OPS = 213  # update when openapi.json is re-exported
MAX_PAGE_BYTES = 2_500_000

HTTP_METHODS = ("get", "post", "put", "patch", "delete")

DOCS_BEGIN = "<!-- REST-DOCS:BEGIN -->"
DOCS_END = "<!-- REST-DOCS:END -->"
TOC_BEGIN = "<!-- REST-TOC:BEGIN -->"
TOC_END = "<!-- REST-TOC:END -->"

# ---------------------------------------------------------------------------
# Functional-area grouping
# ---------------------------------------------------------------------------
# Classification rules, checked in order (most specific first).  Display
# order on the page is defined separately by GROUPS below.
RULES: list[tuple[str, str]] = [
    ("health", r"^/(health|metrics|observability|model-config|lmstudio-models|vertex-models|integrations/catalog)$"),
    ("auth", r"^/(auth|users)(/|$)"),
    ("files", r"^/(files|tables|plots|workspace)/"),
    ("data", r"^/upload$|^/(dataset/preview|outliers|data-studio)/"),
    ("garden", r"^/model-garden(/|$)"),
    ("models", r"^/(models|runs|portfolio|portfolio-benchmark)(/|$)"),
    ("reports", r"^/(report|results-report|prefit-report|client-report|client-slides|project-report|project-slides|model-defense|slide-deck|artifacts)(/|$)"
                r"|^/projects/[^/]+/generate-deck"),
    ("experiments", r"^/(experiments|analysis-plans)(/|$)"
                    r"|^/projects/[^/]+/(experiment-design|experiment-priorities|ghost-ads|calibration-coverage)"),
    ("validation", r"^/projects/[^/]+/(validate|validations|spec-curve)"),
    ("learning", r"^/projects/[^/]+/learning-programs"),
    ("planner", r"^/budget-plans(/|$)|^/projects/[^/]+/planner/"),
    ("branding", r"^/preferences$|^/projects/[^/]+/branding"),
    ("sessions", r"^/chat$|^/sessions(/|$)"
                 r"|^/(history|state|rewind|workflow|spec|dag|assumption_history|assumptions|assumption)/"),
    ("projects", r"^/projects(/|$)|^/kb/"),
]

# Display order: (slug, title, one-line blurb).
GROUPS: list[tuple[str, str, str]] = [
    ("sessions", "Sessions &amp; Chat",
     "The Oracle: the streaming <code>/chat</code> loop plus per-session lifecycle and state &mdash; "
     "model spec, causal DAG, assumptions, workflow tracker, history, rewind, and export."),
    ("data", "Data: Upload, EDA &amp; Data Studio",
     "Getting data into a session: raw upload, dataset preview, outlier-treatment actions, and the "
     "staged Data Studio pipeline (upload &rarr; interactive EDA &rarr; clean &rarr; commit as the working dataset)."),
    ("projects", "Projects, Team &amp; Knowledge Base",
     "Project CRUD and everything scoped to a project: members, onboarding, the guide session, "
     "knowledge-base ingest/search, data connections, deliverables, scorecards, sign-offs, pacing, and history."),
    ("experiments", "Experiments &amp; Design",
     "The experiment registry with its lifecycle transitions, the Design Studio "
     "(design / identify / optimize / simulate jobs), ghost-ads power, and pre-registered analysis plans."),
    ("validation", "Validation &amp; Model QA",
     "Non-blocking validation jobs on the project&rsquo;s latest saved model (validate / PPC / residuals / "
     "channels / refutation / cross-validation / SBC / coverage), persisted run history, and the specification curve."),
    ("planner", "Planner &amp; Budget",
     "Budget plans (create, list, fetch, delete, CSV export) and the non-blocking planner "
     "optimize / scenario jobs behind the Almanac page."),
    ("learning", "Learning Programs (Sextant)",
     "Model-free continuous-learning programs: geo response-surface experiments run in waves &mdash; "
     "create programs, ingest waves, fit posteriors, and design the next wave."),
    ("reports", "Reports, Decks &amp; Artifacts",
     "Every generated document &mdash; fit report, pre-fit design readout, interactive results report, client "
     "report and slides, project report and slides, model defense, slide decks &mdash; plus stored session artifacts."),
    ("models", "Models, Runs &amp; Portfolio",
     "Saved fitted models and their dashboards, run history and cross-run comparison, and the "
     "portfolio / benchmark views."),
    ("garden", "Model Garden &amp; Atelier",
     "The versioned custom-model registry (register, promote, test, fetch source) and the Atelier IDE "
     "services: lint, format, copilot, and the per-model notebook."),
    ("branding", "Preferences &amp; Branding",
     "Global operator preferences and per-project client branding, including SSRF-guarded "
     "extraction of brand colors from a client website."),
    ("files", "Files, Tables &amp; Plots",
     "Content-addressed session resources: uploaded files, streamed dashboard tables, captured plots, "
     "and workspace file listings."),
    ("auth", "Auth, Orgs &amp; Users",
     "The built-in organization auth layer (signup, login, JWT refresh, invites, members, password reset, "
     "usage, audit export) and the project-team user roster."),
    ("health", "Health &amp; Platform Meta",
     "Liveness and observability probes plus platform discovery: model configuration and the "
     "LM&nbsp;Studio / Vertex model lists and the integrations catalog."),
]

GROUP_TITLES = {slug: title for slug, title, _ in GROUPS}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def esc(value: object) -> str:
    """html.escape everything that comes out of the spec (defense in depth)."""
    return html.escape(str(value), quote=True)


def code_spans(escaped_text: str) -> str:
    """Convert reST/markdown ``code`` and `code` spans (post-escape) to <code>."""
    out = re.sub(r"``([^`]+)``", r"<code>\1</code>", escaped_text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def slugify(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def classify(path: str) -> str | None:
    for slug, pattern in RULES:
        if re.search(pattern, path):
            return slug
    return None


# ---------------------------------------------------------------------------
# Schema rendering
# ---------------------------------------------------------------------------


def type_str(schema: object) -> str:
    """Human-readable type for a schema node ($ref resolved by NAME only)."""
    if not isinstance(schema, dict) or not schema:
        return "any"
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            parts = [type_str(s) for s in schema[key]]
            non_null = list(dict.fromkeys(p for p in parts if p != "null"))
            label = " | ".join(non_null) if non_null else "null"
            if "null" in parts and non_null:
                label += ", nullable"
            return label
    if "allOf" in schema:
        parts = list(dict.fromkeys(type_str(s) for s in schema["allOf"]))
        return " & ".join(parts)
    if "enum" in schema:
        vals = [json.dumps(v) for v in schema["enum"]]
        if len(vals) > 6:
            vals = vals[:5] + ["…"]
        return "enum: " + " | ".join(vals)
    if "const" in schema:
        return "const " + json.dumps(schema["const"])
    t = schema.get("type")
    if isinstance(t, list):  # OpenAPI 3.1 type arrays
        non_null = [x for x in t if x != "null"]
        label = " | ".join(non_null) if non_null else "null"
        if "null" in t and non_null:
            label += ", nullable"
        return label
    if t == "array":
        return "array of " + type_str(schema.get("items", {}))
    if t:
        fmt = schema.get("format")
        return f"{t} ({fmt})" if fmt else str(t)
    if "properties" in schema:
        return "object"
    return "any"


def resolve_ref(schema: object, schemas: dict) -> tuple[str | None, dict]:
    """Resolve a top-level $ref one level; returns (name, schema_dict)."""
    if isinstance(schema, dict) and "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return name, schemas.get(name, {}) or {}
    return None, schema if isinstance(schema, dict) else {}


def default_str(prop: dict) -> str:
    if "default" not in prop:
        return ""
    try:
        rendered = json.dumps(prop["default"])
    except (TypeError, ValueError):
        rendered = str(prop["default"])
    if len(rendered) > 40:
        rendered = rendered[:37] + "…"
    return rendered


MAX_FIELD_ROWS = 16


def fields_table(schema: dict, schemas: dict, caption: str = "") -> str:
    """Render the top-level fields of an (already resolved) object schema."""
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return ""
    required = set(schema.get("required", []) or [])
    rows: list[str] = []
    items = list(props.items())
    for name, prop in items[:MAX_FIELD_ROWS]:
        if not isinstance(prop, dict):
            prop = {}
        req = '<span class="api-req">required</span>' if name in required else "optional"
        dflt = default_str(prop)
        type_cell = esc(type_str(prop))
        if dflt:
            type_cell += f' <span class="api-default">= {esc(dflt)}</span>'
        desc = prop.get("description", "")
        if desc and len(desc) > 140:
            desc = desc[:137] + "…"
        desc_cell = f"<td>{code_spans(esc(desc))}</td>" if desc else "<td></td>"
        rows.append(
            f"<tr><td><code>{esc(name)}</code></td>"
            f"<td>{type_cell}</td><td>{req}</td>{desc_cell}</tr>"
        )
    more = ""
    if len(items) > MAX_FIELD_ROWS:
        more = (
            f'<tr><td colspan="4" class="api-more">… '
            f"{len(items) - MAX_FIELD_ROWS} more fields (see "
            f"<a href=\"shared/openapi.json\">openapi.json</a>)</td></tr>"
        )
    cap = f'<div class="api-schema-name">{caption}</div>' if caption else ""
    return (
        f"{cap}<table class=\"api-table\"><thead><tr>"
        f"<th>Field</th><th>Type</th><th>Required</th><th>Notes</th>"
        f"</tr></thead><tbody>{''.join(rows)}{more}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Operation rendering
# ---------------------------------------------------------------------------

PARAM_ORDER = {"path": 0, "query": 1, "header": 2, "cookie": 3}


def render_parameters(params: list[dict]) -> str:
    if not params:
        return ""
    params = sorted(params, key=lambda p: (PARAM_ORDER.get(p.get("in", ""), 9), str(p.get("name", ""))))
    rows = []
    for p in params:
        req = '<span class="api-req">required</span>' if p.get("required") else "optional"
        rows.append(
            f"<tr><td><code>{esc(p.get('name', ''))}</code></td>"
            f"<td>{esc(p.get('in', ''))}</td>"
            f"<td>{esc(type_str(p.get('schema', {})))}</td>"
            f"<td>{req}</td></tr>"
        )
    return (
        '<h4 class="api-h">Parameters</h4>'
        '<table class="api-table"><thead><tr>'
        "<th>Name</th><th>In</th><th>Type</th><th>Required</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render_request_body(op: dict, schemas: dict) -> str:
    rb = op.get("requestBody")
    if not isinstance(rb, dict):
        return ""
    parts = ['<h4 class="api-h">Request body</h4>']
    required = " (required)" if rb.get("required") else ""
    for ctype, media in (rb.get("content") or {}).items():
        schema = media.get("schema", {}) if isinstance(media, dict) else {}
        name, resolved = resolve_ref(schema, schemas)
        label = f"<code>{esc(ctype)}</code>{required}"
        if name:
            label += f" &mdash; <code>{esc(name)}</code>"
        parts.append(f'<p class="api-meta">{label}</p>')
        table = fields_table(resolved, schemas)
        if table:
            parts.append(table)
        elif not name:
            parts.append(f'<p class="api-meta">Schema: {esc(type_str(schema))}</p>')
    return "".join(parts)


SUCCESS_CODES = ("200", "201", "202", "204")


def render_responses(op: dict, schemas: dict) -> str:
    resps = op.get("responses") or {}
    parts = ['<h4 class="api-h">Responses</h4>']
    ok = next((c for c in SUCCESS_CODES if c in resps), None)
    if ok == "204":
        parts.append('<p class="api-meta"><strong>204</strong> &mdash; no content.</p>')
    elif ok is not None:
        resp = resps[ok] if isinstance(resps[ok], dict) else {}
        desc = esc(resp.get("description", ""))
        content = resp.get("content") or {}
        rendered_schema = False
        for ctype, media in content.items():
            schema = media.get("schema", {}) if isinstance(media, dict) else {}
            name, resolved = resolve_ref(schema, schemas)
            shape = f"<code>{esc(name)}</code>" if name else esc(type_str(schema))
            if not name and shape == "any":
                shape = "JSON (shape not declared in the schema)"
            parts.append(
                f'<p class="api-meta"><strong>{esc(ok)}</strong> {desc} &mdash; '
                f"<code>{esc(ctype)}</code>, {shape}</p>"
            )
            table = fields_table(resolved, schemas)
            if table:
                parts.append(table)
            rendered_schema = True
        if not rendered_schema:
            parts.append(f'<p class="api-meta"><strong>{esc(ok)}</strong> {desc}</p>')
    other = [c for c in resps if c != ok]
    if other:
        labels = []
        for c in sorted(other):
            r = resps[c] if isinstance(resps[c], dict) else {}
            d = r.get("description", "")
            labels.append(f"<strong>{esc(c)}</strong> ({esc(d)})" if d else f"<strong>{esc(c)}</strong>")
        parts.append(f'<p class="api-meta api-also">Also: {", ".join(labels)}.</p>')
    return "".join(parts)


def render_notes(op: dict) -> str:
    desc = (op.get("description") or "").strip()
    if not desc:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", desc) if p.strip()]
    rendered = "".join(
        f"<p>{code_spans(esc(re.sub(chr(10), ' ', p)))}</p>" for p in paragraphs
    )
    return f'<div class="api-notes">{rendered}</div>'


def render_operation(method: str, path: str, op: dict, path_params: list[dict], schemas: dict) -> str:
    summary = op.get("summary") or op.get("operationId") or ""
    op_id = f"op-{method}-{slugify(path) or 'root'}"
    deprecated = (
        ' <span class="api-deprecated">deprecated</span>' if op.get("deprecated") else ""
    )
    # merge path-item-level parameters with op-level ones (dedup by name+in)
    seen = set()
    params: list[dict] = []
    for p in list(op.get("parameters") or []) + list(path_params):
        if not isinstance(p, dict):
            continue
        key = (p.get("name"), p.get("in"))
        if key in seen:
            continue
        seen.add(key)
        params.append(p)

    body = (
        render_notes(op)
        + render_parameters(params)
        + render_request_body(op, schemas)
        + render_responses(op, schemas)
    )
    return (
        f'<details class="api-op" id="{esc(op_id)}">'
        f"<summary>"
        f'<span class="method method-{esc(method)}">{esc(method.upper())}</span>'
        f'<code class="api-path">{esc(path)}</code>'
        f'<span class="api-sum">{esc(summary)}{deprecated}</span>'
        f"</summary>"
        f'<div class="api-body">{body}</div>'
        f"</details>"
    )


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------


def build(spec: dict) -> tuple[str, str, dict[str, int], int]:
    schemas = (spec.get("components") or {}).get("schemas") or {}
    grouped: dict[str, list[str]] = {slug: [] for slug, _, _ in GROUPS}
    counts: dict[str, int] = {slug: 0 for slug, _, _ in GROUPS}
    total_ops = 0
    unmatched: list[str] = []

    for path in sorted(spec.get("paths", {})):
        item = spec["paths"][path]
        slug = classify(path)
        path_params = list(item.get("parameters") or [])
        ops_here = [(m, item[m]) for m in HTTP_METHODS if m in item]
        if not ops_here:
            continue
        if slug is None or slug not in grouped:
            unmatched.append(path)
            continue
        for method, op in ops_here:
            grouped[slug].append(render_operation(method, path, op, path_params, schemas))
            counts[slug] += 1
            total_ops += 1

    if unmatched:
        raise SystemExit(
            "ERROR: paths not covered by any group (add a rule):\n  "
            + "\n  ".join(unmatched)
        )

    info = spec.get("info") or {}
    version = esc(info.get("version", ""))
    n_paths = len(spec.get("paths", {}))

    sections: list[str] = [
        f'<p class="api-provenance">Generated from '
        f'<a href="shared/openapi.json"><code>shared/openapi.json</code></a> &mdash; '
        f"<strong>{esc(info.get('title', 'API'))}</strong> v{version} &mdash; "
        f"{total_ops} operations across {n_paths} paths. "
        f"Do not edit this section by hand; re-run "
        f"<code>python3 tools/build_rest_docs.py</code>.</p>"
    ]
    toc_items: list[str] = []
    for slug, title, blurb in GROUPS:
        n = counts[slug]
        if n == 0:
            continue
        sections.append(
            f'<section class="api-group">'
            f'<h2 id="grp-{slug}">{title} <span class="api-count">{n}</span></h2>'
            f'<p class="api-blurb">{blurb}</p>'
            f"{''.join(grouped[slug])}"
            f"</section>"
        )
        toc_items.append(
            f'<li><a href="#grp-{slug}">{title} '
            f'<span class="toc-count">{n}</span></a></li>'
        )

    toc = f'<ul class="sidebar-nav">{"".join(toc_items)}</ul>'
    return "\n".join(sections), toc, counts, total_ops


def inject(text: str, begin: str, end: str, payload: str) -> str:
    i = text.index(begin)
    j = text.index(end)
    if j < i:
        raise SystemExit(f"ERROR: marker {end!r} appears before {begin!r}")
    return text[: i + len(begin)] + "\n" + payload + "\n" + text[j:]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class TagBalanceChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001
        pass  # self-closing: nothing to track

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_TAGS:
            return
        if not self.stack:
            self.errors.append(f"stray closing </{tag}> with no open tags")
            return
        if self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            self.errors.append(
                f"mismatched </{tag}>: innermost open tag is <{self.stack[-1]}>"
            )
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            if self.stack:
                self.stack.pop()
        else:
            self.errors.append(f"closing </{tag}> that was never opened")


def check_html(text: str) -> list[str]:
    checker = TagBalanceChecker()
    checker.feed(text)
    checker.close()
    errors = list(checker.errors)
    if checker.stack:
        errors.append("unclosed tags at EOF: " + ", ".join(checker.stack))
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    sections_html, toc_html, counts, total_ops = build(spec)

    assert total_ops == EXPECTED_OPS, (
        f"operation count changed: rendered {total_ops}, expected {EXPECTED_OPS}. "
        f"Nothing was dropped by grouping (coverage is asserted separately) -- "
        f"the spec itself changed; update EXPECTED_OPS in {Path(__file__).name}."
    )

    page = PAGE_PATH.read_text(encoding="utf-8")
    page = inject(page, DOCS_BEGIN, DOCS_END, sections_html)
    page = inject(page, TOC_BEGIN, TOC_END, toc_html)
    PAGE_PATH.write_text(page, encoding="utf-8")

    errors = check_html(page)
    if errors:
        print("HTML balance errors:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    size = PAGE_PATH.stat().st_size
    assert size < MAX_PAGE_BYTES, f"page too large: {size} bytes >= {MAX_PAGE_BYTES}"

    print(f"Rendered {total_ops} operations across {len(spec['paths'])} paths:")
    for slug, title, _ in GROUPS:
        plain = re.sub(r"<[^>]+>", "", title).replace("&amp;", "&").replace("&nbsp;", " ")
        print(f"  {counts[slug]:4d}  {plain}")
    print(f"Wrote {PAGE_PATH} ({size:,} bytes); HTML tags balanced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
