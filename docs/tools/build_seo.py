#!/usr/bin/env python3
"""Build / refresh the SEO + LLM-friendliness layer for the docs site.

Run from the docs/ directory (or any cwd — paths resolve relative to docs/):

    python3 tools/build_seo.py

It is idempotent and safe to re-run after adding or editing pages. Three jobs:

  1. augment_pages()  — inject Twitter Card meta + a schema.org JSON-LD @graph
                        (Organization + WebSite + per-page WebPage/TechArticle/
                        FAQPage + BreadcrumbList) into each content page's <head>,
                        just before </head>. Pages already carrying the sentinel
                        are skipped, so existing structured data is never doubled.
  2. build_sitemap()  — regenerate sitemap.xml from the real files (top-level pages
                        minus TEMPLATE*, plus artifacts/*.html) with git lastmod.
  3. build_llms_txt() — regenerate llms.txt (the LLM-discovery index, llmstxt.org)
                        grouped by the site's information architecture.

After adding a NEW page: also add it to the relevant group in GROUPS/SERIES below
(for llms.txt) and, if it is deep-series content, to DEMO_PAGES (for breadcrumbs).
"""
import os, re, glob, json, html, subprocess
from html.parser import HTMLParser

DOCS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(DOCS)

SITE = "https://redam94.github.io/mmm-framework"
ORG_ID = f"{SITE}/#organization"
SITE_ID = f"{SITE}/#website"
SENTINEL = "<!-- seo:augmented (structured data + social cards) -->"
DEFAULT_IMG = f"{SITE}/assets/mmm-framework-social-preview.png"
TODAY = "2026-07-23"
EXCLUDE = {"TEMPLATE.html", "TEMPLATE-SIDEBAR.html", "404.html"}
BLOG_PREFIX = "blog-"  # blog-*.html posts emit BlogPosting nodes (index: blog.html)

DEMO_PAGES = {
    "demos.html", "scientific-workflow-demo.html", "scientific-workflow-simple.html",
    "workflow-budget-optimization.html", "workflow-channel-effectiveness.html",
    "workflow-forecasting.html", "workflow-calibration-decisions.html", "mmm-example-report.html",
    "pressure-testing.html", "stress-00-rosy-picture.html", "stress-01-carryover-shape.html",
    "stress-02-time-structure.html", "stress-03-confounding-selection.html",
    "stress-04-extension-traps.html", "stress-05-gauntlet.html", "stress-06-geo-hierarchy.html",
    "mmm-walkthrough.html", "aurora-00-overview.html", "aurora-01-causality.html",
    "aurora-02-base-mmm.html", "aurora-03-extended-mmm.html", "aurora-04-reporting.html",
    "aurora-05-unified-workflow.html", "causal-features-showcase.html",
    "workshop-00-thinking-in-distributions.html", "workshop-01-priors.html",
    "workshop-02-sampling.html", "workshop-03-first-mmm.html",
    "workshop-04-reading-the-posterior.html", "workshop-05-from-draws-to-decisions.html",
    "math-00-overview.html", "math-01-adstock.html", "math-02-saturation.html",
    "math-03-seasonality-trend.html", "math-04-bayesian-model.html",
    "math-05-calibration.html", "math-06-extensions.html",
    "causal-00-the-ladder.html", "causal-01-confounding-adjustment.html",
    "causal-02-mmm-as-causal-model.html", "causal-03-structural-mediation.html",
    "causal-04-latent-confounders.html", "causal-05-measuring-one-experiment.html",
    "causal-06-calibrating-the-model.html", "causal-07-many-experiments.html",
    "causal-08-designing-next-experiment.html", "causal-09-measurement-program.html",
    "causal-10-closed-loop.html",
}
TECH_PREFIXES = ("math-", "stress-", "workshop-", "aurora-", "causal-0", "causal-1", "scientific-workflow-", "workflow-")
TECH_EXPLICIT = {
    "getting-started.html", "modeling-guide.html", "real-data-guide.html",
    "interpreting-results.html", "causal-inference.html", "bayesian-workflow.html",
    "variable-selection.html", "scientific-modeling.html", "measurement-calibration.html",
    "identification-assumptions.html", "technical-guide.html", "data-requirements.html",
    "data-prep-cookbook.html", "migration-guide.html",
    "mmm-walkthrough.html", "pressure-testing.html", "causal-features-showcase.html",
    "business-readiness-report.html",
}
NAV_LABEL = {
    "index.html": "Home", "getting-started.html": "Getting Started",
    "modeling-guide.html": "Modeling Guide", "real-data-guide.html": "Real-Data Guide",
    "interpreting-results.html": "Interpreting Results", "reading-the-report.html": "Reading the Report",
    "data-prep-cookbook.html": "Data-Prep Cookbook", "migration-guide.html": "Migration Guide",
    "business-stakeholders.html": "For Business",
    "glossary.html": "Glossary", "faq.html": "FAQ", "troubleshooting.html": "Troubleshooting",
    "causal-inference.html": "Causal Inference",
    "bayesian-workflow.html": "Bayesian Workflow", "variable-selection.html": "Variable Selection",
    "scientific-modeling.html": "Scientific Modeling", "measurement-calibration.html": "Calibration Loop",
    "identification-assumptions.html": "Identification Assumptions", "technical-guide.html": "Technical Guide",
    "platform-overview.html": "Platform Overview", "model-garden.html": "Model Garden & Atelier",
    "pricing.html": "Pricing", "data-requirements.html": "Data Requirements & Runtime",
    "data-connections.html": "Data Connections", "trust.html": "Trust & Security",
    "security.html": "Security & AI Governance", "responsible-disclosure.html": "Responsible Disclosure",
    "demos.html": "Demos & Reports", "pressure-testing.html": "Pressure Testing",
    "mmm-walkthrough.html": "MMM Walkthrough", "mmm-example-report.html": "Example Report",
    "about.html": "About", "evaluator.html": "For Evaluators", "changelog.html": "Changelog",
    "blog.html": "Research",
}

ORG_NODE = {
    "@type": "Organization", "@id": ORG_ID, "name": "MMM Framework",
    "url": f"{SITE}/",
    "description": "Open-source Bayesian Marketing Mix Modeling framework and the Augur measurement platform.",
    "logo": {"@type": "ImageObject", "url": f"{SITE}/assets/augur-mark.svg"},
    "sameAs": ["https://github.com/redam94/mmm-framework"],
}
WEBSITE_NODE = {
    "@type": "WebSite", "@id": SITE_ID, "name": "MMM Framework",
    "url": f"{SITE}/",
    "description": "Documentation for the MMM Framework — a Bayesian Marketing Mix Modeling library and the Augur causal-measurement platform.",
    "inLanguage": "en-US", "publisher": {"@id": ORG_ID},
}


def git_date(p):
    d = subprocess.run(["git", "log", "-1", "--format=%cs", "--", p],
                       capture_output=True, text=True).stdout.strip()
    return d or TODAY


def byline_date(s):
    """Publication date from a blog post's visible <time datetime="…"> byline."""
    m = re.search(r'<time\s+datetime="(\d{4}-\d{2}-\d{2})"', s)
    return m.group(1) if m else None


def head_of(s):
    return s.split("</head>")[0] if "</head>" in s else s


def _name(head, n):
    m = re.search(rf'<meta\s+name="{re.escape(n)}"\s+content="(.*?)"', head, re.S)
    return m.group(1).strip() if m else None


def _prop(head, p):
    m = re.search(rf'<meta\s+property="{re.escape(p)}"\s+content="(.*?)"', head, re.S)
    return m.group(1).strip() if m else None


def _canon(head):
    m = re.search(r'<link\s+rel="canonical"\s+href="(.*?)"', head, re.S)
    return m.group(1).strip() if m else None


def _title(head, f):
    m = re.search(r"<title>(.*?)</title>", head, re.S)
    return m.group(1).strip() if m else f


def nice_title(title):
    """First non-brand segment of a page <title> (handles 'Brand — Page' too)."""
    parts = re.split(r"\s*[|—–]\s*", title)
    for p in parts:
        if p.strip() and p.strip().lower() != "mmm framework":
            return p.strip()
    return title.strip()


def strip_brand(title):
    """Article headline = the title without the trailing ' | Site Name' segment."""
    return title.rsplit(" | ", 1)[0].strip() if " | " in title else title.strip()


def clean_leaf(f, title):
    if f in NAV_LABEL:
        return NAV_LABEL[f]
    return nice_title(title)


class FaqParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.mode, self.depth, self.buf, self.pairs, self._q = None, 0, [], [], None

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "")
        if self.mode is None:
            if "faq-question" in cls:
                self.mode, self.depth, self.buf = "q", 1, []
            elif "faq-answer" in cls:
                self.mode, self.depth, self.buf = "a", 1, []
        elif tag == "div":
            self.depth += 1

    def handle_endtag(self, tag):
        if self.mode and tag == "div":
            self.depth -= 1
            if self.depth == 0:
                text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
                if self.mode == "q":
                    self._q = text
                elif self._q:
                    self.pairs.append([self._q, text]); self._q = None
                self.mode, self.buf = None, []

    def handle_data(self, d):
        if self.mode: self.buf.append(d)

    def handle_entityref(self, n):
        if self.mode: self.buf.append(html.unescape("&%s;" % n))

    def handle_charref(self, n):
        if self.mode: self.buf.append(html.unescape("&#%s;" % n))


def page_node(f, title, desc, canon, img, date, published=None):
    is_tech = f in TECH_EXPLICIT or f.startswith(TECH_PREFIXES)
    is_faq = f == "faq.html"
    node = {
        "url": canon, "name": title, "description": desc, "inLanguage": "en-US",
        "isPartOf": {"@id": SITE_ID}, "primaryImageOfPage": img, "dateModified": date,
        "breadcrumb": {"@id": f"{canon}#breadcrumb"}, "@id": f"{canon}#webpage",
    }
    if is_faq:
        node["@type"] = "FAQPage"
    elif f.startswith(BLOG_PREFIX):
        node["@type"] = "BlogPosting"
        node.update({"headline": strip_brand(title), "datePublished": published or date,
                     "image": img, "author": {"@type": "Person", "name": "Matthew Reda"},
                     "publisher": {"@id": ORG_ID}, "mainEntityOfPage": canon})
    elif is_tech:
        node["@type"] = "TechArticle"
        node.update({"headline": strip_brand(title), "datePublished": date, "image": img,
                     "author": {"@type": "Person", "name": "Matthew Reda"},
                     "publisher": {"@id": ORG_ID}, "mainEntityOfPage": canon})
    else:
        node["@type"] = "WebPage"
        node["publisher"] = {"@id": ORG_ID}
    return node, is_faq


def breadcrumb_node(f, leaf, canon):
    items = [("Home", f"{SITE}/index.html")]
    if f in DEMO_PAGES and f != "demos.html":
        items.append(("Demos & Reports", f"{SITE}/demos.html"))
    if f.startswith(BLOG_PREFIX):
        items.append(("Research", f"{SITE}/blog.html"))
    items.append((leaf, canon))
    # Dedupe by URL (the homepage's self-crumb collapses into the Home crumb).
    seen, deduped = set(), []
    for n, u in items:
        if u not in seen:
            seen.add(u); deduped.append((n, u))
    if len(deduped) < 2:
        return None  # a lone Home crumb carries no information; omit the breadcrumb
    return {"@type": "BreadcrumbList", "@id": f"{canon}#breadcrumb",
            "itemListElement": [{"@type": "ListItem", "position": i + 1, "name": n, "item": u}
                                for i, (n, u) in enumerate(deduped)]}


def build_block(f, s):
    head = head_of(s)
    title = _title(head, f)
    desc = _name(head, "description") or _prop(head, "og:description") or title
    canon = _canon(head) or f"{SITE}/{f}"
    img = _prop(head, "og:image") or DEFAULT_IMG
    og_title = _prop(head, "og:title") or title
    og_desc = _prop(head, "og:description") or desc
    has_og = head.count('property="og:') > 0

    lines = ["    " + SENTINEL]
    if _canon(head) is None:
        lines.append(f'    <link rel="canonical" href="{canon}">')
    if not has_og:
        lines += [f'    <meta property="og:title" content="{og_title}">',
                  f'    <meta property="og:description" content="{og_desc}">',
                  f'    <meta property="og:url" content="{canon}">',
                  f'    <meta property="og:image" content="{img}">',
                  '    <meta property="og:type" content="website">']
    if _prop(head, "og:site_name") is None:
        lines.append('    <meta property="og:site_name" content="MMM Framework">')
    if _prop(head, "og:locale") is None:
        lines.append('    <meta property="og:locale" content="en_US">')
    if _name(head, "twitter:card") is None:
        lines += ['    <meta name="twitter:card" content="summary_large_image">',
                  f'    <meta name="twitter:title" content="{og_title}">',
                  f'    <meta name="twitter:description" content="{og_desc}">',
                  f'    <meta name="twitter:image" content="{img}">']
    if _name(head, "author") is None:
        lines.append('    <meta name="author" content="Matthew Reda">')

    node, is_faq = page_node(f, html.unescape(title), html.unescape(desc), canon, img, git_date(f),
                             published=byline_date(s))
    bc = breadcrumb_node(f, html.unescape(clean_leaf(f, title)), canon)
    if bc is None:
        node.pop("breadcrumb", None)
    graph = [ORG_NODE, WEBSITE_NODE, node] + ([bc] if bc else [])
    if is_faq:
        p = FaqParser(); p.feed(s)
        if p.pairs:
            node["mainEntity"] = [{"@type": "Question", "name": q,
                                   "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in p.pairs]
    payload = json.dumps({"@context": "https://schema.org", "@graph": graph}, indent=2, ensure_ascii=False)
    payload = "\n".join("    " + ln for ln in payload.splitlines())
    lines += ['    <script type="application/ld+json">', payload, '    </script>']
    return "\n".join(lines) + "\n"


def strip_block(s):
    """Remove a previously-injected SEO block so it can be rebuilt (refresh)."""
    return re.sub(r"[ \t]*" + re.escape(SENTINEL) + r".*?</script>\n?", "", s, flags=re.S)


def augment_pages():
    added = refreshed = 0
    for f in sorted(glob.glob("*.html")):
        if f in EXCLUDE:
            continue
        s = open(f, encoding="utf-8").read()
        if "</head>" not in s:
            continue
        had = SENTINEL in s
        if had:
            s = strip_block(s)  # rebuild from current <head>/<body> so edits propagate
        open(f, "w", encoding="utf-8").write(s.replace("</head>", build_block(f, s) + "</head>", 1))
        refreshed += had
        added += not had
    print(f"augment_pages: {added} newly augmented, {refreshed} refreshed")


def build_sitemap():
    pages = [f for f in sorted(glob.glob("*.html")) if f not in EXCLUDE]
    artifacts = sorted(glob.glob("artifacts/*.html"))
    key = {"platform-overview.html", "model-garden.html", "getting-started.html", "faq.html", "pricing.html"}

    def url(loc, mod, pri=None):
        e = ['    <url>', f'        <loc>{loc}</loc>', f'        <lastmod>{mod}</lastmod>',
             '        <changefreq>monthly</changefreq>']
        if pri: e.append(f'        <priority>{pri}</priority>')
        e.append('    </url>'); return "\n".join(e)

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">', '',
           url(f"{SITE}/index.html", git_date("index.html"), "1.0")]
    for f in pages:
        if f == "index.html": continue
        out.append(url(f"{SITE}/{f}", git_date(f), "0.8" if f in key else None))
    for a in artifacts:
        out.append(url(f"{SITE}/{a}", git_date(a)))
    out.append("</urlset>")
    open("sitemap.xml", "w").write("\n".join(out) + "\n")
    print(f"build_sitemap: {len(pages) + len(artifacts)} urls")


# llms.txt information architecture — keep in sync when adding pages.
GROUPS = [
    ("Start here", ["index.html", "getting-started.html", "platform-overview.html", "faq.html", "troubleshooting.html", "glossary.html"]),
    ("Learn the framework", ["modeling-guide.html", "real-data-guide.html", "interpreting-results.html", "reading-the-report.html", "data-prep-cookbook.html", "migration-guide.html", "business-stakeholders.html", "data-requirements.html"]),
    ("Methodology & causal foundations", ["causal-inference.html", "bayesian-workflow.html", "scientific-modeling.html", "variable-selection.html", "measurement-calibration.html", "experiment-playbook.html", "continuous-learning.html", "continuous-learning-math.html", "ltv-modeling.html", "identification-assumptions.html", "technical-guide.html"]),
    ("Platform (Augur)", ["platform-overview.html", "model-garden.html", "pricing.html", "data-connections.html", "trust.html", "security.html", "responsible-disclosure.html", "evaluator.html"]),
    ("Proof, demos & worked examples", ["demos.html", "mmm-walkthrough.html", "mmm-example-report.html", "causal-features-showcase.html", "pressure-testing.html", "scientific-workflow-demo.html", "scientific-workflow-simple.html", "business-readiness-report.html"]),
    ("Task workflows", ["workflow-budget-optimization.html", "workflow-channel-effectiveness.html", "workflow-forecasting.html", "workflow-calibration-decisions.html"]),
    ("Project", ["about.html", "changelog.html"]),
]
SERIES = [
    ("Mathematics series", ["math-00-overview.html", "math-01-adstock.html", "math-02-saturation.html", "math-03-seasonality-trend.html", "math-04-bayesian-model.html", "math-05-calibration.html", "math-06-extensions.html"]),
    ("Pressure-testing / stress series", ["stress-00-rosy-picture.html", "stress-01-carryover-shape.html", "stress-02-time-structure.html", "stress-03-confounding-selection.html", "stress-04-extension-traps.html", "stress-05-gauntlet.html", "stress-06-geo-hierarchy.html"]),
    ("Bayesian workshop series (beginner)", ["workshop-00-thinking-in-distributions.html", "workshop-01-priors.html", "workshop-02-sampling.html", "workshop-03-first-mmm.html", "workshop-04-reading-the-posterior.html", "workshop-05-from-draws-to-decisions.html"]),
    ("Aurora framework tour", ["aurora-00-overview.html", "aurora-01-causality.html", "aurora-02-base-mmm.html", "aurora-03-extended-mmm.html", "aurora-04-reporting.html", "aurora-05-unified-workflow.html"]),
    ("Causal inference series", ["causal-00-the-ladder.html", "causal-01-confounding-adjustment.html", "causal-02-mmm-as-causal-model.html", "causal-03-structural-mediation.html", "causal-04-latent-confounders.html", "causal-05-measuring-one-experiment.html", "causal-06-calibrating-the-model.html", "causal-07-many-experiments.html", "causal-08-designing-next-experiment.html", "causal-09-measurement-program.html", "causal-10-closed-loop.html"]),
    ("Modern measurement research (blog)", ["blog.html",
        "blog-activity-bias.html", "blog-causal-estimates-observational.html",
        "blog-table-2-fallacy.html", "blog-table-2-fallacy-mmm.html",
        "blog-attribution-incrementality.html",
        "blog-geo-experiments-tbr.html", "blog-switchback-experiments.html",
        "blog-synthetic-control.html",
        "blog-staggered-did.html", "blog-pretrends-testing.html",
        "blog-causalimpact-bsts.html",
        "blog-bayesian-mmm-carryover-shape.html", "blog-calibrating-mmm-experiments.html",
        "blog-carryover-experiment-timing.html",
        "blog-surrogate-outcomes.html",
        "blog-modeling-pitfalls.html", "blog-multiple-comparisons.html",
        "blog-p-hacking-evidence.html",
        "blog-p-values-across-models.html",
        "blog-testing-no-difference.html",
        "blog-prior-predictive-sbc.html",
        "blog-when-sampling-fails.html",
        "blog-lindley-to-dad.html", "blog-geo-holdout-eig.html",
        "blog-bed-bo-bandits.html", "blog-thompson-sampling.html",
        "blog-continuous-learning-interactions.html"]),
    ("Consultant artifacts (templates)", ["artifacts/index.html"]),
]


def _meta_for_llms(f):
    s = open(f, encoding="utf-8").read()
    head = head_of(s)
    t = nice_title(html.unescape(_title(head, f)))
    d = _name(head, "description")
    d = re.sub(r"\s+", " ", html.unescape(d)).strip() if d else ""
    if len(d) > 400:  # these are curated one-paragraph summaries; cut only if very long
        d = d[:400].rsplit(" ", 1)[0].rstrip(",;:—– ") + "…"  # never mid-word
    return t, d


def build_llms_txt():
    def line(f):
        t, d = _meta_for_llms(f)
        return f"- [{t}]({SITE}/{f})" + (f": {d}" if d else "")

    out = ["# MMM Framework — Bayesian Marketing Mix Modeling & the Augur platform", "",
           "> MMM Framework is a production-ready, open-source Bayesian Marketing Mix Modeling (MMM) library built on PyMC, and Augur, the application that runs the whole causal measurement loop on top of it. It emphasizes genuine uncertainty quantification, pre-registered analysis to limit researcher degrees of freedom, and calibration of model estimates against real-world experiments (geo lift, matched-market tests).", "",
           "Marketing Mix Modeling answers a causal question — what *incremental* sales did each marketing channel cause — not merely what moved together. The docs below cover the modeling method (adstock, saturation, hierarchical Bayesian inference), the identification assumptions it rests on, and the Augur platform (the Oracle agent workspace, the Auspices experiment planner, and the Atelier for authoring bespoke model families). Key concepts: adstock/carryover, saturation/diminishing returns, ROI and marginal ROAS, posterior credible intervals, experiment calibration, and the EIG/EVOI experiment-prioritization loop.", ""]
    for name, files in GROUPS:
        out.append(f"## {name}")
        out += [line(f) for f in files]
        out.append("")
    out += ["## Optional", "",
            "These are deep, multi-part series and downloadable templates — read on demand; skip if context is limited.", ""]
    for name, files in SERIES:
        out.append(f"### {name}")
        out += [line(f) for f in files]
        out.append("")
    open("llms.txt", "w", encoding="utf-8").write("\n".join(out).rstrip() + "\n")
    n = sum(1 for l in out if l.startswith("- ["))
    print(f"build_llms_txt: {n} links")


if __name__ == "__main__":
    augment_pages()
    build_sitemap()
    build_llms_txt()
    print("done.")
