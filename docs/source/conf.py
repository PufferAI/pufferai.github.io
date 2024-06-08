# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = []
autodoc_mock_imports = ["torch", "pufferlib.pytorch"]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

autodoc_inherit_docstrings = False

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
# Currently import is broken, copy version manually for now
import pufferlib
release = pufferlib.__version__

project = f'PufferLib v{release}'
copyright = '2022, Joseph Suarez'
author = 'Joseph Suarez'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
      'sphinx.ext.autodoc', 
      'sphinx.ext.coverage', 
      'sphinx.ext.napoleon',
      'sphinx_design',
      'sphinxcontrib.youtube',
      'sphinx.ext.autosectionlabel',
   ]

#Don't sort method names
autodoc_member_order = 'bysource'

#Include __init__
autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PufferLib'
copyright = '2023, Joseph Suarez'
author = 'Joseph Suarez'

import pufferlib
release = pufferlib.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
      'sphinx.ext.autodoc', 
      'sphinx.ext.coverage', 
      'sphinx.ext.napoleon',
      'sphinx_design',
      'sphinxcontrib.youtube',
      'sphinx.ext.autosectionlabel',
   ]

#Don't sort method names
autodoc_member_order = 'bysource'

#Include __init__
autoclass_content = 'both'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']

text = '#f1f1f1'
background = '#061a1a'
foreground = '#000000'
highlight = '#00bbbb'
muted = '#005050'


html_theme_options = {
    "light_css_variables": {
        "color-foreground-primary": "black",
        "color-foreground-secondary": muted,
        "color-foreground-muted": muted,
        "color-foreground-border": "#878787",
        "color-background-primary": "white",
        "color-background-secondary": "#bbcccc",
        "color-background-hover": "#efeff4ff",
        "color-background-hover--transparent": "#efeff400",
        "color-background-border": muted,
        "color-background-item": "#ccc",
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#eeebee",
        "color-brand-primary": "black",
        "color-brand-content": "black",
        "color-inline-code-background": "#f8f9fb",
        "color-highlighted-background": "#ddeeff",
        "color-guilabel-background": "#ddeeff80",
        "color-guilabel-border": "#bedaf580",
        "color-card-background": "#bbcccc",
    },
    "dark_css_variables": {
        "color-problematic": "#ee5151",
        "color-foreground-primary": text, # Text
        "color-foreground-secondary": highlight, # Some icons
        "color-foreground-muted": highlight, #Some headings and icons
        "color-foreground-border": "#666666",
        "color-background-primary": background, # Main Background
        "color-background-secondary": foreground, # Sidebar
        "color-background-hover": "#1e2124ff",
        "color-background-hover--transparent": "#1e212400",
        "color-background-border": "#303335", # Sidebar border
        "color-background-item": "#444",
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#eeebee",
        "color-brand-primary": highlight, # Sidebar Items
        "color-brand-content": highlight, # Embedded Links
        "color-highlighted-background": "#083563",
        "color-guilabel-background": "#08356380",
        "color-guilabel-border": "#13395f80",
        "color-admonition-background": "#18181a",
        "color-card-border": "#1a1c1e",
        "color-card-background": foreground,
        "color-card-marginals-background": "#1e2124ff",
        "color-inline-code-background": "#00000000", # Download background
    }
}

pygments_dark_style = "monokai"
