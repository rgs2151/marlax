project = "MARLAX"
author = "MARLAX contributors"
release = "0.1.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "myst_nb"]
html_theme = "pydata_sphinx_theme"
html_title = "MARLAX"
html_context = {"default_mode": "light"}
html_theme_options = {
    "github_url": "https://github.com/rgs2151/marlax",
    "logo": {"text": "MARLAX"},
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "search_bar_text": "Search documentation",
    "search_as_you_type": True,
    "show_nav_level": 2,
}
autodoc_member_order = "bysource"
autodoc_typehints = "none"
exclude_patterns = ["**/.ipynb_checkpoints/**"]
nb_execution_mode = "off"
myst_title_to_header = True
