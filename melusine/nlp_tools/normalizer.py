import logging

from typing import Sequence
import unicodedata

from melusine.utils.io_utils import load_pkl_generic, save_pkl_generic

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
        input_columns=("text",),
        output_columns=("text",),
    ):
        """
        Parameters
        ----------
        form: str
            Normalization method
        """
        if isinstance(input_columns, str):
            self.input_columns = (input_columns,)
        else:
            self.input_columns = input_columns

        if isinstance(output_columns, str):
            self.output_columns = (output_columns,)
        else:
            self.output_columns = output_columns

        # Normalization form
        self.form = form

        # Lower casing
        self.lowercase = lowercase

    def _normalize_text(self, text):
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

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        input_col = self.input_columns[0]
        output_col = self.output_columns[0]

        data[output_col] = data[input_col].apply(self.normalize)
        return data

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save object instance to a pickle file.

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            Prefix of the saved object
        """
        save_pkl_generic(self, self.FILENAME, path, filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load object instance from a pickle file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
            Prefix of the saved object

        Returns
        -------
        _: DummyLemmatizer
            DummyLemmatizer instance
        """
        return load_pkl_generic(cls.FILENAME, path, filename_prefix=filename_prefix)
