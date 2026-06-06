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

html_theme = "sphinx_book_theme"
html_title = "MARLAX"
html_theme_options = {
    "repository_url": "https://github.com/rgs2151/marlax",
    "use_repository_button": True,
    "use_download_button": False,
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}

autosummary_generate = True
