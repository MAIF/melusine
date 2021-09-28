import logging
from abc import abstractmethod
from typing import Dict

from melusine.nlp_tools.base_melusine_class import BaseMelusineClass
from melusine.nlp_tools.nlp_tools_utils import _flag_text


logger = logging.getLogger(__name__)


class MelusineTextFlagger(BaseMelusineClass):
    FILENAME = "text_flagger.json"
    CONFIG_KEY = "text_flagger"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def flag_text(self, text):
        raise NotImplementedError()


class DeterministicTextFlagger(MelusineTextFlagger):
    def __init__(
        self,
        text_flags: Dict[str, str] = None,
    ):
        super().__init__()
        # Collocations
        if not text_flags:
            self.text_flags = {}
        else:
            self.text_flags = text_flags

    def flag_text(self, text):
        # Join collocations
        text = _flag_text(text, self.text_flags)

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
    def load(cls, path: str):
        """
        Load the DeterministicTextFlagger from a json file.

        Parameters
        ----------
        path: str
            Load path
        Returns
        -------
        _: DeterministicTextFlagger
            DeterministicTextFlagger instance
        """
        # Load json file
        config_dict = cls.load_json(path)

        return cls(**config_dict)
