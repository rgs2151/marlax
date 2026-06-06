import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "MARLAX"
author = "MARLAX contributors"
release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "pydata_sphinx_theme"
html_title = "MARLAX"
html_static_path = ["_static"]
html_css_files = ["css/site.css"]
html_js_files = ["js/cooperative_grid.js"]
html_theme_options = {
    "github_url": "https://github.com/rgs2151/marlax",
    "navbar_align": "left",
    "navbar_center": [],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "show_toc_level": 2,
    "show_nav_level": 2,
    "show_prev_next": False,
    "secondary_sidebar_items": ["page-toc"],
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}

autosummary_generate = True
