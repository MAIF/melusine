import logging

from typing import Dict, Sequence, Union, List
from abc import abstractmethod
import unicodedata

from melusine.nlp_tools.base_melusine_class import BaseMelusineClass
from melusine.nlp_tools.pipeline import MelusineTransformer

logger = logging.getLogger(__name__)


class MelusineNormalizer(MelusineTransformer):
    CONFIG_KEY = "normalizer"
    FILENAME = "normalizer.json"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def normalize(self, text: str):
        raise NotImplementedError()


class Normalizer(MelusineNormalizer):
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
        super().__init__()

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
        Save the Normalizer into a json file.

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            A prefix to add to the names of the files saved.
        """
        d = self.__dict__.copy()

        # Save Tokenizer
        self.save_json(save_dict=d, path=path, filename_prefix=filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the Normalizer from a json file.

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

    def transform(self, df):
        df["text"] = df["text"].apply(self.normalize)
        return df
