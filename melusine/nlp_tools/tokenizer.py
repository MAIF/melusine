import logging
import re
import json
from flashtext import KeywordProcessor
from melusine import config
from typing import List, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseMelusineTokenizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path):
        raise NotImplementedError()

    @abstractmethod
    def load(self, path):
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
        stopwords: List[str] = config["tokenizer"]["stopwords"],
        remove_stopwords: bool = config["tokenizer"]["remove_stopwords"],
        flag_dict: Dict[str, str] = config["tokenizer"]["flag_dict"],
        flashtext_separators: List[str] = config["tokenizer"]["flashtext_separators"],
        flashtext_names: List[str] = config["tokenizer"]["flashtext_names"],
        name_flag: str = config["tokenizer"]["name_flag"],
    ):
        """
        Parameters
        ----------
        tokenizer_regex: str
            Regex used to split the text into tokkens
        stopwords: List[str]
            List of words to be removed
        remove_stopwords: bool
            If True, stopwords removal is enabled
        flag_dict: Dict[str, str]
            Flagging dict with regex as key and replace_text as value
        flashtext_separators: List[Str]
            List of separator words for FlashText
        flashtext_names: List[Str]
            List of names to be flagged
        name_flag: str
            Replace value for names
        """
        super().__init__()

        # Text splitting
        self.tokenizer_regex = tokenizer_regex

        # Stopwords
        self.stopwords = set(stopwords) or None
        self.remove_stopwords = remove_stopwords
        self.flag_dict = flag_dict
        self.flashtext_separators = flashtext_separators
        self.flashtext_names = flashtext_names
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
            self.keyword_processor.add_keywords_from_dict(
                {"flag_name_": self.flashtext_names}
            )

    @staticmethod
    def _flag_text(flag_dict: Dict[str, str], text: str) -> str:
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
        for key, value in flag_dict.items():
            if isinstance(value, dict):
                text = WordLevelTokenizer._flag_text(value, text)
            else:
                text = re.sub(key, value, text, flags=re.I)

        return text

    def _text_to_tokens(self, text: str) -> List[str]:
        """
        Split a text into values
        Parameters
        ----------
        text: Text to be split
        Returns
        -------
        tokens: List[str]
            List of tokens
        """
        if isinstance(text, str):
            tokens = re.findall(self.tokenizer_regex, text, re.M + re.DOTALL)
        else:
            tokens = []
        return tokens

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        Parameters
        ----------
        tokens: List[str]
            List of tokens
        Returns
        -------
        tokens: List[str]
            List of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stopwords]

    def _flag_names(self, tokens: List[str]) -> List[str]:
        """
        Replace name tokens with a flag
        Parameters
        ----------
        tokens: List[str]
            List of tokens
        Returns
        -------
        tokens: List[str]
            List of tokens with names flagged
        """
        return [self.keyword_processor.replace_keywords(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        """
        Apply the full tokenization pipeline on a text.
        Parameters
        ----------
        text: str
            Input text to be tokenized
        Returns
        -------
        tokens: List[str]
            List of tokens
        """
        # Text flagging
        text = self._flag_text(self.flag_dict, text)

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

        with open(path, "w") as f:
            json.dump(d, f)

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
            params = json.load(f)

        return cls(**params)


class Tokenizer:
    pass
