import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Fuse"
copyright = "2026, Ryan R. <pwnfo@proton.me>"
author = "Ryan R."
release = "4.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "icon.png",
    "dark_logo": "icon.png",
    "source_repository": "https://github.com/pwnfo/fuse",
    "source_branch": "main",
    "source_directory": "docs/",
}
