"""Sphinx configuration for the gdmbayes documentation.

Build locally with::

    sphinx-build -b html docs docs/_build/html
"""

from importlib.metadata import version as _pkg_version

# -- Project information -----------------------------------------------------
project = "gdmbayes"
author = "Harold Horsley"
copyright = "2026, Harold Horsley"
release = _pkg_version("gdmbayes")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # pull docstrings from the package
    "sphinx.ext.autosummary",  # generate one stub page per API object
    "sphinx.ext.intersphinx",  # cross-link to upstream library docs
    "numpydoc",  # parse NumPy-style docstrings
    "myst_parser",  # Markdown sources (README, developer notes)
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# -- autodoc / autosummary / numpydoc ----------------------------------------
autosummary_generate = True  # write stub pages under docs/generated/
autodoc_typehints = "description"  # type hints go into the parameter list, not the signature
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_member_order = "bysource"
numpydoc_show_class_members = False  # autodoc already lists members; avoid duplicate tables
numpydoc_class_members_toctree = False

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3  # auto-generate #anchors for h1-h3 so README links resolve

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "gdmbayes"
html_static_path = []
html_theme_options = {
    "github_url": "https://github.com/butterman0/gdmbayes",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
html_context = {
    "github_user": "butterman0",
    "github_repo": "gdmbayes",
    "github_version": "main",
    "doc_path": "docs",
}
