"""Sphinx configuration for mmm-framework documentation."""

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add the src directory to sys.path for autodoc
sys.path.insert(0, os.path.abspath("../../../src"))

# -- Project information -----------------------------------------------------
project = "MMM Framework"
copyright = f"{datetime.now().year}, Matthew Reda"
author = "Matthew Reda"
version = "1.2.0"
release = "1.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    # Third-party extensions
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "sphinxcontrib.mermaid",  # architecture flow diagrams
]

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Type hints configuration
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_type_aliases = {
    "NDArray": "numpy.ndarray",
}

# Generate autosummary stubs
autosummary_generate = True

# -- Napoleon settings (Google-style docstrings) -----------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- Mock imports for heavy dependencies -------------------------------------
# RTD installs the real (lean-core) package, but autodoc still mocks the heavy
# numerical stack so importing every module for signatures never triggers
# pytensor/jax initialization or compilation. Optional-stack packages
# (fastapi/redis/httpx — server + [agents] extras, not installed on RTD) stay
# mocked defensively; nothing in the documented tree imports them at module
# level (tests/test_lean_imports.py pins that).
autodoc_mock_imports = [
    # PyMC ecosystem
    "pymc",
    "pytensor",
    "pytensor.tensor",
    "arviz",
    # JAX/NumPyro ecosystem
    "jax",
    "jaxlib",
    "numpyro",
    "nutpie",
    # Heavy numerical packages
    "numba",
    # Optional-stack packages (server package / [agents] extra)
    "redis",
    "fastapi",
    "uvicorn",
    "plotly",
    "httpx",
]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# Theme: Furo. Chosen over sphinx_rtd_theme because it is themed *entirely*
# through CSS custom properties, which lets this site share one palette with the
# hand-authored GitHub Pages site (docs/shared/styles.css) instead of
# maintaining two unrelated looks. Furo also ships a light/dark toggle that
# mirrors that site's [data-theme="dark"] token swap.
html_theme = "furo"
html_title = f"MMM Framework {release}"

DOCS_SITE = "https://redam94.github.io/mmm-framework/"

# -- Brand palette -----------------------------------------------------------
# MIRROR of the design tokens in docs/shared/styles.css (:root and
# [data-theme="dark"]). Keep the two in lockstep: when a token changes there,
# change it here. Furo injects these as `body { --name: value }` in a <style>
# block that is emitted AFTER html_css_files, so the palette must live here —
# a :root block in custom.css would lose. custom.css carries structure and
# typography only.
#
# sage green = primary/brand, dusty blue = accent/prose links, warm off-white
# page. Text-bearing roles use the *dark* variants on light backgrounds (and
# the light variants on dark) so contrast holds both ways.
_BRAND_LIGHT = {
    # Typography
    "font-stack": "'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "font-stack--monospace": "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
    "font-stack--headings": "'DM Serif Display', Georgia, serif",
    # Brand
    "color-brand-primary": "#6d8a4a",
    "color-brand-content": "#4a6d8a",
    "color-brand-visited": "#4a6d8a",
    # Surfaces
    "color-background-primary": "#fafbf9",
    "color-background-secondary": "#f0f2ed",
    "color-background-hover": "#eaeee3",
    "color-background-hover--transparent": "rgba(143, 168, 106, 0.10)",
    "color-background-border": "#d4ddd4",
    "color-background-item": "#e6ebe0",
    # Foreground
    "color-foreground-primary": "#2d3a2d",
    "color-foreground-secondary": "#5a6b5a",
    "color-foreground-muted": "#6d7d6d",
    "color-foreground-border": "#d4ddd4",
    # Announcement strip. Furo applies this token via `background-color`, which
    # cannot take a gradient — the brand gradient rides on a separate token that
    # custom.css applies as `background-image` over this solid fallback.
    "color-announcement-background": "#6d8a4a",
    "color-announcement-text": "#ffffff",
    "mmm-announcement-gradient": "linear-gradient(135deg, #8fa86a, #5f7d3f)",
    # Sidebar
    "color-sidebar-background": "#f0f2ed",
    "color-sidebar-background-border": "#d4ddd4",
    "color-sidebar-brand-text": "#2d3a2d",
    "color-sidebar-caption-text": "#5a6b5a",
    "color-sidebar-link-text": "#5a6b5a",
    "color-sidebar-link-text--top-level": "#2d3a2d",
    "color-sidebar-item-background--current": "rgba(143, 168, 106, 0.12)",
    "color-sidebar-item-background--hover": "rgba(143, 168, 106, 0.08)",
    "color-sidebar-item-expander-background--hover": "rgba(143, 168, 106, 0.16)",
    "color-sidebar-search-background": "#ffffff",
    "color-sidebar-search-background--focus": "#ffffff",
    "color-sidebar-search-border": "#d4ddd4",
    "color-sidebar-search-foreground": "#2d3a2d",
    "color-sidebar-search-icon": "#6d7d6d",
    # Right-hand "On this page"
    "color-toc-title-text": "#6d7d6d",
    "color-toc-item-text": "#5a6b5a",
    "color-toc-item-text--hover": "#2d3a2d",
    "color-toc-item-text--active": "#6d8a4a",
    # Code
    "color-code-background": "#f4f6f1",
    "color-code-foreground": "#2d3a2d",
    "color-inline-code-background": "#eef1e8",
    # API signatures
    "color-api-background": "rgba(143, 168, 106, 0.07)",
    "color-api-background-hover": "rgba(143, 168, 106, 0.14)",
    "color-api-name": "#4a6d8a",
    "color-api-pre-name": "#6d7d6d",
    "color-api-paren": "#6d7d6d",
    "color-api-keyword": "#6d8a4a",
    "color-api-overall": "#5a6b5a",
    "color-highlight-on-target": "rgba(212, 168, 106, 0.22)",
    # Admonitions — three semantic families only (informational = accent,
    # advisory = primary, caution = warning, failure = danger), so a page of
    # mixed notes reads as one palette rather than a traffic light.
    "color-admonition-background": "transparent",
    "color-admonition-title--note": "#4a6d8a",
    "color-admonition-title-background--note": "rgba(106, 143, 168, 0.11)",
    "color-admonition-title--seealso": "#4a6d8a",
    "color-admonition-title-background--seealso": "rgba(106, 143, 168, 0.11)",
    "color-admonition-title--important": "#4a6d8a",
    "color-admonition-title-background--important": "rgba(106, 143, 168, 0.11)",
    "color-admonition-title--tip": "#6d8a4a",
    "color-admonition-title-background--tip": "rgba(143, 168, 106, 0.13)",
    "color-admonition-title--hint": "#6d8a4a",
    "color-admonition-title-background--hint": "rgba(143, 168, 106, 0.13)",
    "color-admonition-title--admonition-todo": "#6d7d6d",
    "color-admonition-title-background--admonition-todo": "rgba(90, 107, 90, 0.11)",
    "color-admonition-title--warning": "#a07c3d",
    "color-admonition-title-background--warning": "rgba(212, 168, 106, 0.16)",
    "color-admonition-title--caution": "#a07c3d",
    "color-admonition-title-background--caution": "rgba(212, 168, 106, 0.16)",
    "color-admonition-title--attention": "#a07c3d",
    "color-admonition-title-background--attention": "rgba(212, 168, 106, 0.16)",
    "color-admonition-title--danger": "#b1554b",
    "color-admonition-title-background--danger": "rgba(201, 112, 103, 0.14)",
    "color-admonition-title--error": "#b1554b",
    "color-admonition-title-background--error": "rgba(201, 112, 103, 0.14)",
    # Cards / tables / misc
    "color-card-background": "#ffffff",
    "color-card-border": "#d4ddd4",
    "color-card-marginals-background": "#f0f2ed",
    "color-table-header-background": "#f0f2ed",
    "color-problematic": "#c97067",
    # Named accents reused by custom.css
    "mmm-primary": "#8fa86a",
    "mmm-primary-dark": "#6d8a4a",
    "mmm-accent": "#6a8fa8",
    "mmm-accent-dark": "#4a6d8a",
    "mmm-surface": "#ffffff",
    "mmm-success": "#6abf8a",
    "mmm-warning": "#d4a86a",
    "mmm-danger": "#c97067",
    "mmm-shadow-sm": "0 1px 2px rgba(45, 58, 45, 0.04), 0 2px 8px rgba(45, 58, 45, 0.05)",
    "mmm-shadow-md": "0 2px 4px rgba(45, 58, 45, 0.04), 0 10px 28px rgba(45, 58, 45, 0.08)",
}

_BRAND_DARK = {
    "color-brand-primary": "#b5cb8d",
    "color-brand-content": "#9ec3da",
    "color-brand-visited": "#9ec3da",
    "color-background-primary": "#151913",
    "color-background-secondary": "#1c211a",
    "color-background-hover": "#222821",
    "color-background-hover--transparent": "rgba(157, 184, 119, 0.14)",
    "color-background-border": "#394236",
    "color-background-item": "#283027",
    "color-foreground-primary": "#e4e9df",
    "color-foreground-secondary": "#a6b3a0",
    "color-foreground-muted": "#8f9c8a",
    "color-foreground-border": "#394236",
    "color-announcement-background": "#3d5227",
    "color-announcement-text": "#e9f0df",
    "mmm-announcement-gradient": "linear-gradient(135deg, #475e2d, #33441f)",
    "color-sidebar-background": "#1c211a",
    "color-sidebar-background-border": "#394236",
    "color-sidebar-brand-text": "#e4e9df",
    "color-sidebar-caption-text": "#a6b3a0",
    "color-sidebar-link-text": "#a6b3a0",
    "color-sidebar-link-text--top-level": "#e4e9df",
    "color-sidebar-item-background--current": "rgba(157, 184, 119, 0.16)",
    "color-sidebar-item-background--hover": "rgba(157, 184, 119, 0.10)",
    "color-sidebar-item-expander-background--hover": "rgba(157, 184, 119, 0.20)",
    "color-sidebar-search-background": "#222821",
    "color-sidebar-search-background--focus": "#222821",
    "color-sidebar-search-border": "#394236",
    "color-sidebar-search-foreground": "#e4e9df",
    "color-sidebar-search-icon": "#8f9c8a",
    "color-toc-title-text": "#8f9c8a",
    "color-toc-item-text": "#a6b3a0",
    "color-toc-item-text--hover": "#e4e9df",
    "color-toc-item-text--active": "#b5cb8d",
    "color-code-background": "#1c211a",
    "color-code-foreground": "#e4e9df",
    "color-inline-code-background": "#222821",
    "color-api-background": "rgba(157, 184, 119, 0.08)",
    "color-api-background-hover": "rgba(157, 184, 119, 0.16)",
    "color-api-name": "#9ec3da",
    "color-api-pre-name": "#8f9c8a",
    "color-api-paren": "#8f9c8a",
    "color-api-keyword": "#b5cb8d",
    "color-api-overall": "#a6b3a0",
    "color-highlight-on-target": "rgba(221, 184, 126, 0.20)",
    "color-admonition-background": "transparent",
    "color-admonition-title--note": "#9ec3da",
    "color-admonition-title-background--note": "rgba(127, 168, 192, 0.16)",
    "color-admonition-title--seealso": "#9ec3da",
    "color-admonition-title-background--seealso": "rgba(127, 168, 192, 0.16)",
    "color-admonition-title--important": "#9ec3da",
    "color-admonition-title-background--important": "rgba(127, 168, 192, 0.16)",
    "color-admonition-title--tip": "#b5cb8d",
    "color-admonition-title-background--tip": "rgba(157, 184, 119, 0.16)",
    "color-admonition-title--hint": "#b5cb8d",
    "color-admonition-title-background--hint": "rgba(157, 184, 119, 0.16)",
    "color-admonition-title--admonition-todo": "#a6b3a0",
    "color-admonition-title-background--admonition-todo": "rgba(166, 179, 160, 0.14)",
    "color-admonition-title--warning": "#ddb87e",
    "color-admonition-title-background--warning": "rgba(221, 184, 126, 0.16)",
    "color-admonition-title--caution": "#ddb87e",
    "color-admonition-title-background--caution": "rgba(221, 184, 126, 0.16)",
    "color-admonition-title--attention": "#ddb87e",
    "color-admonition-title-background--attention": "rgba(221, 184, 126, 0.16)",
    "color-admonition-title--danger": "#d98d84",
    "color-admonition-title-background--danger": "rgba(217, 141, 132, 0.16)",
    "color-admonition-title--error": "#d98d84",
    "color-admonition-title-background--error": "rgba(217, 141, 132, 0.16)",
    "color-card-background": "#222821",
    "color-card-border": "#394236",
    "color-card-marginals-background": "#283027",
    "color-table-header-background": "#1c211a",
    "color-problematic": "#d98d84",
    "mmm-primary": "#9db877",
    "mmm-primary-dark": "#b5cb8d",
    "mmm-accent": "#7fa8c0",
    "mmm-accent-dark": "#9ec3da",
    "mmm-surface": "#222821",
    "mmm-success": "#7ecb9c",
    "mmm-warning": "#ddb87e",
    "mmm-danger": "#d98d84",
    "mmm-shadow-sm": "0 1px 2px rgba(0, 0, 0, 0.30), 0 2px 8px rgba(0, 0, 0, 0.22)",
    "mmm-shadow-md": "0 2px 4px rgba(0, 0, 0, 0.30), 0 10px 28px rgba(0, 0, 0, 0.28)",
}

html_theme_options = {
    "light_css_variables": _BRAND_LIGHT,
    "dark_css_variables": _BRAND_DARK,
    "light_logo": "logo.svg",
    "dark_logo": "logo.svg",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "announcement": (
        "Library reference for the Bayesian MMM engine. "
        f'<a href="{DOCS_SITE}">Tutorials, methodology and the research blog &rarr;</a>'
    ),
    # "View source" / "Edit on GitHub" buttons above each page.
    "source_repository": "https://github.com/redam94/mmm-framework/",
    "source_branch": "main",
    "source_directory": "docs/api/source/",
    "top_of_page_buttons": ["view", "edit"],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/redam94/mmm-framework",
            "class": "",
            # Octicon mark-github, inlined so the footer needs no remote asset.
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                "8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01."
                "37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01"
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-."
                "89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.8"
                "2.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1."
                "92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1."
                '48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mmm-framework/",
            "class": "",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 24 24"><path d="M12 2 3 6.5v11L12 22l9-4.5v-11L12 2zm0 2.2 '
                '6.6 3.3L12 10.8 5.4 7.5 12 4.2zM5 9.2l6 3v7.4l-6-3V9.2zm14 0v7.4l-6 3v-7.4l6-3z"></path></svg>'
            ),
        },
    ],
}

html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_css_files = [
    # Brand typefaces, matching the GitHub Pages site.
    "https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap",
    "custom.css",
]

# Extra sidebar panel linking back to the main documentation site, appended to
# Furo's default sidebar stack (theme.conf `sidebars`).
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/mmm-links.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

# -- Source suffix configuration ---------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Exclude patterns --------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Templates path ----------------------------------------------------------
templates_path = ["_templates"]

# -- Suppress warnings from docstrings ---------------------------------------
# Some docstrings have RST formatting issues that are not critical
suppress_warnings = [
    "autodoc.import_object",
    "ref.python",
]

# Ignore duplicate object warnings (from module-level vs class-level docs)
# These occur when documenting both the module and its contents
nitpicky = False
