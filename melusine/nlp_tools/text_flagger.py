import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class DeterministicTextFlagger:
    FILENAME = "text_flagger.json"

    def __init__(self, text_flags: Dict[str, str] = None):
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
