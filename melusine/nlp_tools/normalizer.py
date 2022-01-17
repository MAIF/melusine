import logging

from typing import Sequence
import unicodedata

logger = logging.getLogger(__name__)


class Normalizer:
    FILENAME = "normalizer.json"
    """
    Normalizer
    """

    def __init__(
        self,
        form: str = "NFKD",
        lowercase: bool = True,
    ):
        """
        Parameters
        ----------
        form: str
            Normalization method
        """
        # Normalization form
        self.form = form

        # Lower casing
        self.lowercase = lowercase

    def _normalize_text(self, text: str) -> str:
        if self.form:
            text = (
                unicodedata.normalize(self.form, text)
                .encode("ASCII", "ignore")
                .decode("utf-8")
            )
        return text

    def normalize(self, text: str) -> str:
        """
        Apply Normalization to the text.

        Parameters
        ----------
        text: str
            Input text to be normalized

        Returns
        -------
        tokens: Sequence[str]
            List of tokens
        """
        # Text splitting
        text = self._normalize_text(text)

        # Lowercasing
        if self.lowercase:
            text = text.lower()

        return text
