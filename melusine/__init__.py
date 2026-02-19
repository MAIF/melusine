"""Top-level package."""

import logging

from melusine._config import config
from melusine.pipeline import MelusinePipeline

__all__ = ["config", "MelusinePipeline"]

VERSION = (3, 3, 1)
__version__ = ".".join(map(str, VERSION))

# ------------------------------- #
#             LOGGING
# ------------------------------- #
logging.getLogger(__name__).addHandler(logging.NullHandler())
