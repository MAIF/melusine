import re
import logging
from typing import Dict

from melusine.core.melusine_transformer import MelusineTransformer

logger = logging.getLogger(__name__)


class DeterministicTextFlagger(MelusineTransformer):
    FILENAME = "text_flagger.json"

    def __init__(
        self,
        text_flags: Dict[str, str] = None,
        input_columns="text",
        output_columns="text",
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.flag_text,
        )

        # Collocations
        if not text_flags:
            self.text_flags = {}
        else:
            self.text_flags = text_flags

    @staticmethod
    def _flag_text(text: str, flag_dict: Dict[str, str]) -> str:
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
                text = DeterministicTextFlagger._flag_text(text, value)
            else:
                text = re.sub(key, value, text, flags=re.I)

        return text

    def flag_text(self, text):
        # Join collocations
        text = self._flag_text(text, self.text_flags)

        return text

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
        Load the DeterministicTextFlagger from a json file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
        Returns
        -------
        _: DeterministicTextFlagger
            DeterministicTextFlagger instance
        """
        # Load parameters from json file
        json_data = cls.load_json(path, filename_prefix=filename_prefix)

        return cls.from_json(**json_data)
