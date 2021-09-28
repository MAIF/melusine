import re
import logging
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Dict, Sequence, Union
from abc import abstractmethod

from melusine.nlp_tools.base_melusine_class import BaseMelusineClass
from melusine.nlp_tools.lemmatizer import MelusineLemmatizer
from melusine.nlp_tools.phraser_new import MelusinePhraser, DeterministicPhraser
from melusine.nlp_tools.text_flagger import (
    MelusineTextFlagger,
    DeterministicTextFlagger,
)
from melusine.nlp_tools.token_flagger import MelusineTokenFlagger, FlashtextTokenFlagger

logger = logging.getLogger(__name__)


class BaseMelusineTokenizer(BaseMelusineClass):
    CONFIG_KEY = "tokenizer"
    FILENAME = "tokenizer.json"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def tokenize(self, text: str):
        raise NotImplementedError()


class WordLevelTokenizer(BaseMelusineTokenizer):
    """
    Tokenizer which does the following:
    - General flagging (using regex)
    - Join collocations (deterministic phrasing)
    - Name flagging (using FlashText)
    - Text splitting
    - Stopwords removal
    """

    # Attributes excluded from the default save methods
    # Ex: pickle.dump or json.dump
    EXCLUDE_LIST = ["keyword_processor"]

    def __init__(
        self,
        tokenizer_regex: str = r"\w+(?:[\?\-\"_]\w+)*",
        normalization: str = "NFKD",
        lowercase: bool = True,
        stopwords: Sequence[str] = None,
        remove_stopwords: bool = False,
        text_flagger: MelusineTextFlagger = None,
        phraser: MelusinePhraser = None,
        token_flagger: MelusineTokenFlagger = None,
        text_flags: Dict = None,
        token_flags: Dict = None,
        collocations: Dict = None,
        lemmatizer=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tokenizer_regex: str
            Regex used to split the text into tokens
        normalization: Union[str, None]
            Type of normalization to apply to the text
        lowercase: bool
            If True, lowercase the text
        stopwords: Sequence[str]
            List of words to be removed
        remove_stopwords: bool
            If True, stopwords removal is enabled
        flag_dict: Dict[str, str]
            Flagging dict with regex as key and replace_text as value
        collocations_dict: Dict[str, str]
            Dict with expressions to be grouped into one unit of sens
        flashtext_separators: Sequence[Str]
            List of separator words for FlashText
        names: Sequence[Str]
            List of names to be flagged
        name_flag: str
            Replace value for names
        """
        super().__init__()

        # Tokenizer regex
        if not tokenizer_regex:
            raise AttributeError(
                "You should specify a tokenizer_regex or use the default one"
            )
        self.tokenizer_regex = tokenizer_regex

        # Normalization
        self.normalization = normalization

        # Lower casing
        self.lowercase = lowercase

        # Stopwords
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords) or None
        self.remove_stopwords = remove_stopwords

        # Text Flagger
        if text_flags:
            if text_flagger:
                raise AttributeError(
                    f"You should specify only one of 'text_flags' and 'text_flagger'"
                )
            else:
                self.text_flagger = DeterministicTextFlagger(text_flags=text_flags)
        else:
            self.text_flagger = text_flagger

        # Phraser
        if collocations:
            if phraser:
                raise AttributeError(
                    f"You should specify only one of 'collocations' and 'phraser'"
                )
            else:
                self.phraser = DeterministicPhraser(collocations=collocations)
        else:
            self.phraser = phraser

        # Token Flagger
        if token_flags:
            if token_flagger:
                raise AttributeError(
                    f"You should specify only one of 'token_flags' and 'token_flagger'"
                )
            else:
                self.token_flagger = FlashtextTokenFlagger(token_flags=token_flags)
        else:
            self.token_flagger = token_flagger

        # Lemmatizer
        self.lemmatizer = lemmatizer

    def _normalize_text(self, text):
        if self.normalization:
            text = (
                unicodedata.normalize(self.normalization, text)
                .encode("ASCII", "ignore")
                .decode("utf-8")
            )
        return text

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
        # Normalize text
        text = self._normalize_text(text)

        if self.lowercase:
            text = text.lower()

        # Text flagging
        if self.text_flagger is not None:
            text = self.text_flagger.flag_text(text)

        # Phraser
        if self.phraser is not None:
            text = self.phraser.phrase(text)

        # Text splitting
        tokens = self._text_to_tokens(text)

        # Lemmatizer
        if self.lemmatizer is not None:
            tokens = self.lemmatizer.lemmatize(tokens)

        # Token Flagging
        if self.token_flagger is not None:
            tokens = self.token_flagger.flag_tokens(tokens)

        # Stopwords removal
        tokens = self._remove_stopwords(tokens)

        return tokens

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the tokenizer into a json file
        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            A prefix to add to the names of the files saved by the tokenizer.
        """
        d = self.__dict__.copy()

        # Convert sets to lists (json compatibility)
        for key, val in d.items():
            if isinstance(val, set):
                d[key] = list(val)

        # Save Phraser
        self.save_obj(d, path, filename_prefix, MelusinePhraser.CONFIG_KEY)

        # Save Text Flagger
        self.save_obj(d, path, filename_prefix, MelusineTextFlagger.CONFIG_KEY)

        # Save Lemmatizer
        self.save_obj(d, path, filename_prefix, MelusineLemmatizer.CONFIG_KEY)

        # Save Token Flagger
        self.save_obj(d, path, filename_prefix, MelusineTokenFlagger.CONFIG_KEY)

        # Save Tokenizer
        d = {self.CONFIG_KEY: d}
        self.save_json(
            save_dict=d,
            path=path,
            filename_prefix=filename_prefix,
        )

    @classmethod
    def load(cls, path: str):
        """
        Load the tokenizer from a json file
        Parameters
        ----------
        path: str
            Load path
        Returns
        -------
        _: WordLevelTokenizer
            Tokenizer instance
        """
        # Load the Tokenizer config file
        config_dict = cls.load_json(path)

        if cls.CONFIG_KEY in config_dict:
            config_dict = config_dict[cls.CONFIG_KEY]

        # Load phraser
        phraser = cls.load_obj(
            config_dict, path=path, obj_key=MelusinePhraser.CONFIG_KEY
        )

        # Load text flagger
        text_flagger = cls.load_obj(
            config_dict, path=path, obj_key=MelusineTextFlagger.CONFIG_KEY
        )

        # Load lemmatizer
        lemmatizer = cls.load_obj(
            config_dict, path=path, obj_key=MelusineLemmatizer.CONFIG_KEY
        )

        # Load token flagger
        token_flagger = cls.load_obj(
            config_dict, path=path, obj_key=MelusineTokenFlagger.CONFIG_KEY
        )

        return cls(
            phraser=phraser,
            token_flagger=token_flagger,
            text_flagger=text_flagger,
            lemmatizer=lemmatizer,
            **config_dict,
        )


class Tokenizer(BaseEstimator, TransformerMixin):
    """
    This class is deprecated and should not be used.
    Is is kept to ensure retro-compatibility with earlier Melusine versions.
    It will be removed eventually.

    """

    def __init__(self, input_column, tokenizer=None):
        if not tokenizer:
            self.tokenizer = WordLevelTokenizer()
        else:
            self.tokenizer = tokenizer

        self.input_column = input_column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.copy()
        x["tokens"] = x[self.input_column].apply(self.tokenizer.tokenize)
        return x
