import re
import logging

from typing import Dict, Sequence, Union, List
from abc import abstractmethod

from melusine.nlp_tools.base_melusine_class import BaseMelusineClass
from melusine.nlp_tools.pipeline import MelusineTransformer
from melusine.nlp_tools.transformer_backend import backend

logger = logging.getLogger(__name__)


class MelusineTokenizer(MelusineTransformer):
    CONFIG_KEY = "tokenizer"
    FILENAME = "tokenizer.json"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def tokenize(self, text: str):
        raise NotImplementedError()


class RegexTokenizer(MelusineTransformer):
    FILENAME = "tokenizer.json"
    """
    RegexTokenizer which does the following:
    - General flagging (using regex)
    - Join collocations (deterministic phrasing)
    - Name flagging (using FlashText)
    - Text splitting
    - Stopwords removal
    """

    def __init__(
        self,
        tokenizer_regex: str = r"\w+(?:[\?\-\"_]\w+)*",
        stopwords: List[str] = None,
        input_columns=("text",),
        output_columns=("tokens",),
    ):
        """
        Parameters
        ----------
        tokenizer_regex: str
            Regex used to split the text into tokens
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.tokenize,
        )
        # Tokenizer regex
        if not tokenizer_regex:
            raise AttributeError(
                "You should specify a tokenizer_regex or use the default one"
            )
        self.tokenizer_regex = tokenizer_regex

        # Stopwords
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords) or None

    def _text_to_tokens(self, text: str) -> Sequence[str]:
        """
        Split a text into values
        Parameters
        ----------
        text: Text to be split
        Returns
        -------
        tokens: Sequence[str]
            List of tokens
        """
        if isinstance(text, str):
            tokens = re.findall(self.tokenizer_regex, text, re.M + re.DOTALL)
        else:
            tokens = []
        return tokens

    def _remove_stopwords(self, tokens: Sequence[str]) -> Sequence[str]:
        """
        Remove stopwords from tokens.
        Parameters
        ----------
        tokens: Sequence[str]
            List of tokens
        Returns
        -------
        tokens: Sequence[str]
            List of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stopwords]

    def tokenize(self, text: str) -> Sequence[str]:
        """
        Apply the full tokenization pipeline on a text.
        Parameters
        ----------
        text: str
            Input text to be tokenized

        Returns
        -------
        tokens: Sequence[str]
            List of tokens
        """
        # Text splitting
        tokens = self._text_to_tokens(text)

        # Stopwords removal
        tokens = self._remove_stopwords(tokens)

        return tokens

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the TextProcessor into a json file
        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            A prefix to add to the names of the files saved by the tokenizer.
        """
        d = self.__dict__.copy()

        # Save Tokenizer
        self.save_json(save_dict=d, path=path, filename_prefix=filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the tokenizer from a json file
        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
        Returns
        -------
        _: TextProcessor
            TextProcessor instance
        """
        # Load the Tokenizer config file
        config_dict = cls.load_json(path, filename_prefix=filename_prefix)

        return cls(**config_dict)
