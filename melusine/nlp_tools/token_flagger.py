import logging
from flashtext import KeywordProcessor
from typing import Dict, Sequence, Union
from abc import abstractmethod

from melusine.nlp_tools.pipeline import MelusineTransformer
from melusine.nlp_tools.base_melusine_class import BaseMelusineClass
from melusine.nlp_tools.transformer_backend import backend

logger = logging.getLogger(__name__)


class MelusineTokenFlagger(MelusineTransformer):
    FILENAME = "token_flagger.json"
    EXCLUDE_LIST = list()
    CONFIG_KEY = "token_flagger"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def flag_tokens(self, tokens: Sequence[str]) -> Sequence[str]:
        raise NotImplementedError()


class FlashtextTokenFlagger(MelusineTransformer):
    FILENAME = "token_flagger.json"

    def __init__(
        self,
        token_flags=None,
        flashtext_separators: Sequence[str] = ("-", "_", "/"),
        input_columns=("tokens",),
        output_columns=("tokens",),
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.flag_tokens,
        )

        self.token_flags = token_flags

        # Flashtext parameters
        self.flashtext_separators = flashtext_separators
        self.keyword_processor = KeywordProcessor()
        for x in self.flashtext_separators:
            self.keyword_processor.add_non_word_boundary(x)
        self.add_token_flags(token_flags)

        self.EXCLUDE_LIST.append("keyword_processor")

    def add_token_flags(self, token_flags_dict):
        for key, value in token_flags_dict.items():
            if isinstance(value, dict):
                self.add_token_flags(value)
            else:
                self.keyword_processor.add_keywords_from_dict({key: value})

    def flag_tokens(self, tokens: Sequence[str]) -> Sequence[str]:
        """
        Replace tokens by a flag.

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

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the Token Flagger into a json file

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            Prefix for saved files.
        """
        d = self.__dict__.copy()

        self.save_json(
            save_dict=d,
            path=path,
            filename_prefix=filename_prefix,
        )

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the FlashtextTokenFlagger from a json file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
        Returns
        -------
        _: FlashtextTokenFlagger
            FlashtextTokenFlagger instance
        """
        # Load json file
        config_dict = cls.load_json(path, filename_prefix=filename_prefix)

        return cls(**config_dict)
