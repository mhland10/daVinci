import os
import sys
sys.path.insert(0, os.path.abspath("."))  # Ensure Python files in your project can be found

extensions = [
    "sphinx.ext.autodoc",  # Enables documentation from docstrings
    "sphinx.ext.napoleon",  # Enables support for Google/NumPy docstrings
]
