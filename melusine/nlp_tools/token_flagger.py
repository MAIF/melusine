import logging
from flashtext import KeywordProcessor
from typing import Sequence

from melusine.core.melusine_transformer import MelusineTransformer

logger = logging.getLogger(__name__)


class FlashtextTokenFlagger(MelusineTransformer):
    FILENAME = "token_flagger.json"

    def __init__(
        self,
        token_flags=None,
        flashtext_separators: Sequence[str] = ("-", "_", "/"),
        input_columns="tokens",
        output_columns="tokens",
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
        # Load parameters from json file
        json_data = cls.load_json(path, filename_prefix=filename_prefix)

        return cls.from_json(**json_data)
