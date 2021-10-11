import logging

from typing import Sequence
import unicodedata

from melusine.core.melusine_transformer import MelusineTransformer

logger = logging.getLogger(__name__)


class Normalizer(MelusineTransformer):
    FILENAME = "normalizer.json"
    """
    Normalizer
    """

    def __init__(
        self,
        form: str = "NFKD",
        lowercase: bool = True,
        input_columns="text",
        output_columns="text",
    ):
        """
        Parameters
        ----------
        form: str
            Normalization method
        """

        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.normalize,
        )
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

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the Instance into a json file.

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            A prefix to add to the names of the files saved.
        """
        d = self.__dict__.copy()

        # Save Normalizer
        self.save_json(save_dict=d, path=path, filename_prefix=filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the Instance from a json file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
        Returns
        -------
        _: Normalizer
            Normalizer instance
        """
        # Load the Tokenizer config file
        config_dict = cls.load_json(path, filename_prefix=filename_prefix)

        return cls(**config_dict)
