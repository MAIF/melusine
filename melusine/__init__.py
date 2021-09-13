"""Top-level package for melusine."""

__author__ = """Sacha Samama & Tom Stringer & Hugo Perrier"""
__email__ = "ssamama@quantmetry.com, tstringer@quantmetry.com, hperrier@quantmetry.com"
__version__ = "2.3.1"

from .data.data_loader import load_email_data
from .config.config import config

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "load_email_data",
    "config",
]
