import logging
from flashtext import KeywordProcessor
from typing import Sequence

from melusine.utils.io_utils import save_pkl_generic, load_pkl_generic

logger = logging.getLogger(__name__)


class FlashtextTokenFlagger:
    FILENAME = "token_flagger.json"

    def __init__(
        self,
        token_flags=None,
        flashtext_separators: Sequence[str] = ("-", "_", "/"),
        input_columns=("tokens",),
        output_columns=("tokens",),
    ):
        if isinstance(input_columns, str):
            self.input_columns = (input_columns,)
        else:
            self.input_columns = input_columns

        if isinstance(output_columns, str):
            self.output_columns = (output_columns,)
        else:
            self.output_columns = output_columns

        self.token_flags = token_flags

        # Flashtext parameters
        self.flashtext_separators = flashtext_separators
        self.keyword_processor = KeywordProcessor()
        for x in self.flashtext_separators:
            self.keyword_processor.add_non_word_boundary(x)
        self.add_token_flags(token_flags)

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

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        input_col = self.input_columns[0]
        output_col = self.output_columns[0]

        data[output_col] = data[input_col].apply(self.flag_tokens)
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
