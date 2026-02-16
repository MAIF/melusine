"""Top-level package.
"""

import logging

from melusine._config import config

__all__ = ["config"]

VERSION = (3, 3, 0)
__version__ = ".".join(map(str, VERSION))

# ------------------------------- #
#             LOGGING
# ------------------------------- #
logging.getLogger(__name__).addHandler(logging.NullHandler())
