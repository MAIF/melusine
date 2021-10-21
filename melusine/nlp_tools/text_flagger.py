import re
import logging
from typing import Dict

from melusine.utils.io_utils import load_pkl_generic, save_pkl_generic

logger = logging.getLogger(__name__)


class DeterministicTextFlagger:
    FILENAME = "text_flagger.json"

    def __init__(
        self,
        text_flags: Dict[str, str] = None,
        input_columns=("text",),
        output_columns=("text",),
    ):
        if isinstance(input_columns, str):
            self.input_columns = (input_columns,)
        else:
            self.input_columns = input_columns

        if isinstance(output_columns, str):
            self.output_columns = (output_columns,)
        else:
            self.output_columns = output_columns

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

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        input_col = self.input_columns[0]
        output_col = self.output_columns[0]

        data[output_col] = data[input_col].apply(self.flag_text)
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
