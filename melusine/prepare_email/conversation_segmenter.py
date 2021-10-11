import logging
from typing import Sequence

from melusine.core.melusine_transformer import MelusineTransformer

logger = logging.getLogger(__name__)


class Conversation_Segmenter(MelusineTransformer):
    FILENAME = "conversation_segmenter.json"

    def __init__(
        self,
        input_columns="body",
        output_columns="body",
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.segment_conversation,
        )

    def segment_conversation(self, text: str) -> Sequence[str]:
        """ """
        return [text]

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
