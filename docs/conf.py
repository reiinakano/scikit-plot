# -*- coding: utf-8 -*-
#
# Scikit-plot documentation build configuration file
#
# This file is execfile()d with the current directory set to its
# containing dir.

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'Scikit-plot'
copyright = f'{datetime.now().year}, Reiichiro S. Nakano'
author = 'Reiichiro S. Nakano'

# The full version, including alpha/beta/rc tags
try:
    import scikitplot
    version = scikitplot.__version__
    release = version
except (ImportError, AttributeError):
    version = '0.3.7'
    release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'nbsphinx',
]

# Add any paths that contain templates here
templates_path = ['_templates']

# The suffix(es) of source filenames
source_suffix = ['.rst', '.md']

# The master toctree document
master_doc = 'index'

# List of patterns to ignore
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
]

# The name of the Pygments style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# Output file base name
htmlhelp_basename = 'Scikitplotdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'Scikit-plot.tex', 'Scikit-plot Documentation',
     'Reiichiro S. Nakano', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'scikit-plot', 'Scikit-plot Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'Scikit-plot', 'Scikit-plot Documentation',
     author, 'Scikit-plot', 'An intuitive library to add plotting functionality to scikit-learn objects.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# nbsphinx settings
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_timeout = 300

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# -- ReadTheDocs configuration -----------------------------------------------

# On Read the Docs we need to use a different approach
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    html_theme = 'sphinx_rtd_theme'
