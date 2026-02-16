"""The `melusine.backend` module provides an interface to the active backend for Melusine.
The active backend is determined by the user's configuration.
Supported backends : dict or pandas.
"""
from .active_backend import backend

__all__ = ["backend"]
