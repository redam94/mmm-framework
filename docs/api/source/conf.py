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
html_theme = "sphinx_rtd_theme"
html_title = f"MMM Framework {release}"
html_theme_options = {
    "logo_only": False,
    # sphinx_rtd_theme >= 3 renamed display_version -> version_selector
    "version_selector": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

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
