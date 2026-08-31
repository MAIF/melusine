"""Top-level package."""

import logging

from melusine._config import config
from melusine.base import MelusineDetector, MelusineRegex, MelusineRegexFullMatch, MelusineTransformer
from melusine.pipeline import MelusinePipeline

__all__ = [
    "config",
    "MelusineDetector",
    "MelusineRegex",
    "MelusineRegexFullMatch",
    "MelusineTransformer",
    "MelusinePipeline"
]

VERSION = (3, 4, 0)
__version__ = ".".join(map(str, VERSION))

# ------------------------------- #
#             LOGGING
# ------------------------------- #
logging.getLogger(__name__).addHandler(logging.NullHandler())
