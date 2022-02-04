import logging
from flashtext import KeywordProcessor
from typing import Sequence

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
