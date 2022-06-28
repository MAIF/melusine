import logging
import re
import pandas as pd
from typing import Sequence, Union, List
from sklearn.base import BaseEstimator, TransformerMixin
from melusine import config
from melusine.nlp_tools.normalizer import Normalizer
from melusine.nlp_tools.token_flagger import FlashtextTokenFlagger

logger = logging.getLogger(__name__)


class RegexTokenizer:
    FILENAME = "tokenizer.json"
    """
    Tokenize text using a regex split pattern.
    """

    def __init__(
        self,
        tokenizer_regex: str = r"\w+(?:[\?\-\"_]\w+)*",
        stopwords: List[str] = None,
    ):
        """
        Parameters
        ----------
        tokenizer_regex: str
            Regex used to split the text into tokens
        """
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


class Tokenizer(BaseEstimator, TransformerMixin):
    """
    Tokenizer class to split text into tokens.
    """

    def __init__(
        self,
        input_column="text",
        output_column="tokens",
        stop_removal=True,
    ):
        self.input_column = input_column
        self.output_column = output_column

        self.normalizer = Normalizer(
            form=config["normalizer"]["form"],
            lowercase=config["normalizer"]["lowercase"],
        )

        if stop_removal:
            stopwords = config["tokenizer"]["stopwords"]
        else:
            stopwords = []
        self.regex_tokenizer = RegexTokenizer(
            stopwords=stopwords,
            tokenizer_regex=config["tokenizer"]["tokenizer_regex"],
        )
        self.token_flagger = FlashtextTokenFlagger(
            token_flags=config["token_flagger"]["token_flags"]
        )

    def fit(self, X, y=None):
        """Unused method. Defined only for compatibility with scikit-learn API."""
        return self

    def transform(self, data):
        """Applies tokenize method on pd.Dataframe.

        Parameters
        ----------
        data : pandas.DataFrame,
            Data on which transformations are applied.

        Returns
        -------
        pandas.DataFrame
        """
        logger.debug("Start tokenizer transform")
        text_col = self.input_column
        token_col = self.output_column

        if isinstance(data, dict):
            if pd.isna(data[text_col]):
                data[token_col] = ""
            data[token_col] = self.regex_tokenizer.tokenize(data[text_col])
            data[token_col] = self.token_flagger.flag_tokens(data[token_col])
        else:
            data[token_col] = (
                data[text_col].fillna("").apply(self.regex_tokenizer.tokenize)
            )
            data[token_col] = data[token_col].apply(self.token_flagger.flag_tokens)

        logger.debug("Finished tokenizer transform")
        return data
