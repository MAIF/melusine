import logging
import re
import json
from flashtext import KeywordProcessor
from sklearn.base import BaseEstimator, TransformerMixin
from melusine import config

logger = logging.getLogger(__name__)


class WordLevelTokenizer(BaseEstimator, TransformerMixin):

    # Attributes excluded from the default save methods
    # Ex: pickle.dump or json.dump
    EXCLUDE_LIST = ["keyword_processor"]

    def __init__(
        self,
        tokenizer_regex=config["tokenizer"]["tokenizer_regex"],
        stopwords=config["tokenizer"]["stopwords"],
        remove_stopwords=config["tokenizer"]["remove_stopwords"],
        flag_dict=config["tokenizer"]["flag_dict"],
        flashtext_separators=config["tokenizer"]["flashtext_separators"],
        flashtext_names=config["tokenizer"]["flashtext_names"],
        name_flag=config["tokenizer"]["name_flag"],
        **kwargs
    ):

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

    def init_flashtext(self):
        self.keyword_processor = KeywordProcessor()

        for x in self.flashtext_separators:
            self.keyword_processor.add_non_word_boundary(x)
            self.keyword_processor.add_keywords_from_dict(
                {"flag_name_": self.flashtext_names}
            )

    @staticmethod
    def _flag_text(flag_dict, text):
        for key, value in flag_dict.items():
            if isinstance(value, dict):
                text = WordLevelTokenizer._flag_text(value, text)
            else:
                text = re.sub(key, value, text, flags=re.I)

        return text

    def _text_to_tokens(self, text):
        if isinstance(text, str):
            tokens = re.findall(self.tokenizer_regex, text, re.M + re.DOTALL)
        else:
            tokens = []
        return tokens

    def _remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stopwords]

    def _flag_names(self, tokens):
        return [self.keyword_processor.replace_keywords(token) for token in tokens]

    def tokenize(self, text):
        # Text flagging
        text = self._flag_text(self.flag_dict, text)

        # Text splitting
        tokens = self._text_to_tokens(text)

        # Stopwords removal
        tokens = self._remove_stopwords(tokens)

        # Flagging
        tokens = self._flag_names(tokens)

        return tokens

    def save(self, path):
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
    def load(cls, path):
        with open(path, "r") as f:
            params = json.load(f)

        return cls(**params)
