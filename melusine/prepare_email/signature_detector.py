import re
from time import time

from melusine.core.melusine_transformer import MelusineTransformer
from melusine.prepare_email.message import Message


class SignatureDetector(MelusineTransformer):
    FILENAME = "email_segmenter.json"

    def __init__(
        self,
        signature_patterns=None,
        input_columns="email",
        output_columns="email",
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.detect_signature_email,
        )

        self.signature_patterns = signature_patterns

    def detect_signature_email(self, email):
        for m in email.messages:
            self.detect_signature_message(m)
        return email

    def detect_signature_message(self, message):

        sig = self.detect_signature_text(self, message.text)

        return sig

    def transform(self, df):
        df["email"] = df["email"].apply(self.segment_email)

        return df

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the Instance into a json file.

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            A prefix to add to the names of the files saved.
        """
        d = self.__dict__.copy()

        # Save Normalizer
        self.save_json(save_dict=d, path=path, filename_prefix=filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the Instance from a json file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str

        Returns
        -------
        _: EmailSegmenter
            EmailSegmenter instance
        """
        # Load parameters from json file
        json_data = cls.load_json(path, filename_prefix=filename_prefix)

        return cls.from_json(**json_data)

    def detect_signature_text(self, self1, text):
        pass
