import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import envodat  # noqa: E402

project = "EnvoDat"
author = "Linus Nwankwo, Björn Ellensohn, Vedant Dave, Peter Hofer, Jan Forstner, Marlene Villneuve, Robert Galler, Elmar Rueckert"
copyright = "2024-2026, Chair of Cyber-Physical Systems, Montanuniversität Leoben"
release = envodat.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

html_theme = "sphinx_rtd_theme"
html_title = "EnvoDat Dataset Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_context = {
    "display_github": True,
    "github_user": "LinusNEP",
    "github_repo": "EnvoDat",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
