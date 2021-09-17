import logging
import re
import json
import unicodedata

from flashtext import KeywordProcessor
from melusine import config
from typing import List, Dict, Sequence, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseMelusineTokenizer(ABC):
    INDENT = 4
    SORT_KEYS = True

    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, text: str):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError()


class WordLevelTokenizer(BaseMelusineTokenizer):
    """
    Tokenizer which does the following:
    - General flagging (using regex)
    - Name flagging (using FlashText)
    - Text splitting
    - Stopwords removal
    """

    # Attributes excluded from the default save methods
    # Ex: pickle.dump or json.dump
    EXCLUDE_LIST = ["keyword_processor"]

    def __init__(
        self,
        tokenizer_regex: str = config["tokenizer"]["tokenizer_regex"],
        normalization: Union[str, None] = config["tokenizer"]["normalization"],
        lowercase: bool = config["tokenizer"]["lowercase"],
        stopwords: Sequence[str] = config["tokenizer"]["stopwords"],
        remove_stopwords: bool = config["tokenizer"]["remove_stopwords"],
        flag_dict: Dict[str, str] = config["tokenizer"]["flag_dict"],
        collocations_dict: Dict[str, str] = config["tokenizer"]["collocations_dict"],
        flashtext_separators: Sequence[str] = config["tokenizer"][
            "flashtext_separators"
        ],
        names: Sequence[str] = config["tokenizer"]["names"],
        name_flag: str = config["tokenizer"]["name_flag"],
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

        # Flags
        if not flag_dict:
            self.flag_dict = {}
        else:
            self.flag_dict = flag_dict

        # Collocations
        if not flag_dict:
            self.collocations_dict = {}
        else:
            self.collocations_dict = collocations_dict

        # Flashtext parameters
        self.flashtext_separators = flashtext_separators

        # Collocations
        if not names:
            self.names = {}
        else:
            self.names = names

        # Name flag
        if not name_flag:
            raise AttributeError(
                "You should specify a name_flag or use the default one"
            )
        self.name_flag = name_flag

        self.keyword_processor = None
        self.init_flashtext()

    def init_flashtext(self) -> None:
        """
        Initialize the flashtext KeywordProcessor object
        """
        self.keyword_processor = KeywordProcessor()

        for x in self.flashtext_separators:
            self.keyword_processor.add_non_word_boundary(x)
            self.keyword_processor.add_keywords_from_dict({"flag_name_": self.names})

    def _normalize_text(self, text):
        if self.normalization:
            text = (
                unicodedata.normalize(self.normalization, text)
                .encode("ASCII", "ignore")
                .decode("utf-8")
            )
        return text

    def _flag_text(self, text: str, flag_dict: Dict[str, str] = None) -> str:
        """
        General flagging: replace remarkable expressions by a flag
        Ex: 0123456789 => flag_phone_
        Parameters
        ----------
        flag_dict: Dict[str, str]
            Flagging dict with regex as key and replace_text as value
        text: str
            Text to be flagged
        Returns
        -------
        text: str
            Flagged text
        """
        if not flag_dict:
            flag_dict = self.flag_dict

        for key, value in flag_dict.items():
            if isinstance(value, dict):
                text = self._flag_text(text, value)
            else:
                text = re.sub(key, value, text, flags=re.I)

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

    def _flag_names(self, tokens: Sequence[str]) -> Sequence[str]:
        """
        Replace name tokens with a flag
        Parameters
        ----------
        tokens: Sequence[str]
            List of tokens
        Returns
        -------
        tokens: Sequence[str]
            List of tokens with names flagged
        """
        return [self.keyword_processor.replace_keywords(token) for token in tokens]

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
        text = self._flag_text(text, self.flag_dict)

        # Join collocations
        text = self._flag_text(text, self.collocations_dict)

        # Text splitting
        tokens = self._text_to_tokens(text)

        # Stopwords removal
        tokens = self._remove_stopwords(tokens)

        # Flagging
        tokens = self._flag_names(tokens)

        return tokens

    def save(self, path: str) -> None:
        """
        Save the tokenizer into a json file
        Parameters
        ----------
        path: str
            Save path
        """
        d = self.__dict__.copy()
        for key in self.EXCLUDE_LIST:
            _ = d.pop(key, None)

        # Convert sets to lists
        for key, val in d.items():
            if isinstance(val, set):
                d[key] = list(val)

        # Wrap into a tokenizer key to comply with the Melusine configurations
        save_dict = {"tokenizer": d}

        with open(path, "w") as f:
            json.dump(save_dict, f, sort_keys=self.SORT_KEYS, indent=self.INDENT)

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
        with open(path, "r") as f:
            tokenizer_config = json.load(f)

        return cls(**tokenizer_config["tokenizer"])
