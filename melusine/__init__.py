"""
Top-level package.
"""
import logging
from ctypes import CDLL, cdll
from typing import Any, Optional

import pandas as pd

from melusine._config import config

__all__ = ["config"]

VERSION = (3, 0, 0)
__version__ = ".".join(map(str, VERSION))

# ------------------------------- #
#             LOGGING
# ------------------------------- #
logging.getLogger(__name__).addHandler(logging.NullHandler())


# ------------------------------- #
#           MONKEY PATCH
# ------------------------------- #

# Monkey patch for pandas DataFrame memory leaking on linux OS (pandas issue #2659)
try:
    # Try executing linux malloc_trim function (release free memory)
    cdll.LoadLibrary("libc.so.6")
    libc: Optional[CDLL] = CDLL("libc.so.6")
    if libc is not None:
        libc.malloc_trim(0)
except (OSError, AttributeError):  # pragma: no cover
    # Incompatible OS: this monkey patch is not needed
    libc = None

# Store the standard pandas method
__std_del: Optional[Any] = getattr(pd.DataFrame, "__del__", None)


# Prepare a new __del__ method
def __fixed_del(self: Any) -> None:  # pragma: no cover
    """Override DataFrame's __del__ method: call the standard method + release free memory with malloc_trim."""
    if __std_del is not None:
        __std_del(self)
    if libc is not None:
        libc.malloc_trim(0)


# Override standard pandas method if needed
if libc is not None:
    pd.DataFrame.__del__ = __fixed_del
